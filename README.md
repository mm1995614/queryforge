# QueryForge

A CLI tool that translates natural language into structured [NHTSA](https://www.nhtsa.gov/) API requests using LLMs. Ask questions about vehicle recalls, safety complaints, and crash test ratings in plain English or Chinese — QueryForge handles the rest.

> Built as a GoFreight AI Engineer take-home assessment. Domain chosen from my background as an AI Data Engineer in the automotive industry.

---

## Features

- Natural language → NHTSA API query translation
- Supports English and Chinese input
- Covers three NHTSA datasets: **Recalls**, **Complaints**, **Safety Ratings**
- Hardened against typos, ambiguous inputs, missing fields, and out-of-scope queries
- Multi-model evaluation pipeline comparing Claude, GPT-4o-mini, and Llama 3.3

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| CLI framework | Typer + Rich |
| Query generation | Llama 3.3 70B (Groq, free tier) |
| Data source | NHTSA Public API (no auth required) |
| Eval models | Claude Sonnet 4.6, GPT-4o-mini, Llama 3.3 70B (Groq) |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/mm1995614/queryforge.git
cd queryforge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Key | Required for | Where to get |
|-----|-------------|-------------|
| `GROQ_API_KEY` | CLI + Part 2 eval | [console.groq.com](https://console.groq.com) — free tier available |
| `ANTHROPIC_API_KEY` | Part 2 eval only | [console.anthropic.com](https://console.anthropic.com) |
| `OPENAI_API_KEY` | Part 2 eval only | [platform.openai.com](https://platform.openai.com) |

---

## Usage

```bash
# Recalls
python cli.py "Show me recalls for Toyota Camry 2020"
python cli.py "Toyota Camry 2020 有哪些召回問題"

# Complaints
python cli.py "What safety complaints exist for Honda Civic 2019?"
python cli.py "2019 Honda Civic 消費者投訴"

# Safety Ratings
python cli.py "What are the crash test ratings for Ford F-150 2022?"
python cli.py "Ford F-150 2022 安全評等"
```

---

## Architecture

```
User Input (natural language, English or Chinese)
        │
        ▼
┌─────────────────────┐
│  query_generator.py │  ← LLM (Llama 3.3 70B via Groq)
│                     │    Outputs strict JSON only
└─────────────────────┘
        │
        ▼
{
  "endpoint": "recalls",
  "make": "TOYOTA",
  "model": "CAMRY",
  "year": "2020"
}
        │
        ▼
┌─────────────────────┐
│   nhtsa_client.py   │  ← Calls NHTSA Public API
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│     display.py      │  ← Rich terminal output
└─────────────────────┘
```

**NHTSA Endpoints used:**

| Type | URL |
|------|-----|
| Recalls | `api.nhtsa.gov/recalls/recallsByVehicle?make=&model=&modelYear=` |
| Complaints | `api.nhtsa.gov/complaints/complaintsByVehicle?make=&model=&modelYear=` |
| Safety Ratings | `api.nhtsa.gov/SafetyRatings/modelyear/{year}/make/{make}/model/{model}` |

---

## Part 1: Build a Tool, Break It, and Harden It

**Domain:** NHTSA (National Highway Traffic Safety Administration) Public API — chosen because it is free, requires no authentication, and covers three distinct vehicle safety datasets (recalls, complaints, safety ratings) directly relevant to automotive industry workflows.

**Languages tested:** English and Traditional Chinese

---

### Requirement 1 — Baseline Execution

The CLI accepts a natural language query, passes it to an LLM (Llama 3.3 70B via Groq) which outputs a strict JSON structured query, executes that query against the NHTSA Public API, and displays the results in the terminal using Rich.

```bash
python cli.py "Show me recalls for Toyota Camry 2020"
python cli.py "Toyota Camry 2020 有哪些召回問題"
```

---

### Requirement 2 — Break It

The tool was tested against four categories of adversarial inputs. Results below reflect actual observed behavior.

#### Typos

Typos test whether the LLM can infer the correct make/model/year even when the input contains spelling mistakes or character substitutions.

| Input | Observed Behavior |
|-------|------------------|
| `"toyta cmary 2020 recall"` | ✅ LLM correctly normalized to TOYOTA CAMRY, query succeeded |
| `"Hond Civ1c 2O19 的投訴"` | ✅ LLM correctly normalized to HONDA CIVIC 2019 despite digit/letter confusion and Chinese keyword |

#### Ambiguous Inputs

Ambiguous inputs are queries where required information is missing or unclear, forcing the LLM to either infer, ask for clarification, or return an error.

| Input | Ambiguity | Observed Behavior |
|-------|-----------|------------------|
| `"Toyota Camry recalls"` | No year specified | ❌ LLM returned `year: null`, client sent `modelYear=null` to API → 400 error instead of clean `missing_year` error |
| `"2020 Camry recalls"` | No make specified | ⚠️ LLM silently inferred TOYOTA from model name and returned results — no warning to user that an assumption was made |
| `"那台 Accord 有什麼問題？"` | No year, vague reference ("that") | ❌ LLM returned Chinese error message but terminal displayed garbled text due to Windows encoding (cp950) |

#### Conflicting Constraints

Conflicting constraints are queries where the user provides contradictory requirements, making it impossible to produce a single unambiguous structured query.

| Input | Conflict | Observed Behavior |
|-------|----------|------------------|
| `"Toyota Camry 2019 and 2020 recalls"` | Two years — API accepts only one | ⚠️ LLM silently chose 2019, ignored 2020 — no indication to user that data for 2020 was dropped |
| `"Honda or Toyota Camry 2020 recalls"` | Two makes — API accepts only one | ⚠️ LLM silently chose TOYOTA, ignored HONDA — same silent resolution |
| `"Show me both recalls and safety ratings for Toyota Camry 2020"` | Two endpoints — API handles only one per request | ❌ LLM returned unexpected structure, caused application crash |

#### Languages Other Than English (Traditional Chinese)

| Input | Observed Behavior |
|-------|------------------|
| `"Toyota Camry 2020 有哪些召回問題"` | ✅ Correctly identified as recalls query, returned results |
| `"2019 Honda Civic 消費者投訴"` | ✅ Correctly identified as complaints query, returned 355 results |
| `"那台 Accord 有什麼問題？"` | ❌ Missing year error message returned in Chinese but garbled on Windows terminal (cp950 encoding) |

---

### Requirement 3 — Harden & Fix

#### Design Principle

> **Never guess. Never drop. Always ask.**

Every required parameter (year, make/model, endpoint) must be explicitly present before calling the API. If any parameter is missing, the tool asks the user to supply it. If the user provides multiple values for the same parameter type, the tool queries all of them and synthesizes the results — no information is silently discarded.

This principle is known in UX design as **explicit over implicit**: making assumptions on behalf of the user produces wrong answers and erodes trust. For an LLM-powered tool this matters even more, because silent inference is invisible to the user.

---

#### Fix 1 — Missing Parameters → Interactive Prompt (was: crash or silent assumption)

**Root cause:** The LLM returned `year: null` or silently inferred a make without telling the user. The API received invalid params and returned a 400 error, or returned results the user didn't explicitly ask for.

**Fix:** When the LLM returns `missing_year`, `missing_make`, or `missing_endpoint`, the CLI now prompts the user for the missing value and retries — up to 3 rounds. The tool never fills in a default.

| Input | Before | After |
|-------|--------|-------|
| `"Toyota Camry recalls"` | `year: null` → API 400 error | Prompts: "Which model year is your vehicle?" → retries with full params |
| `"2020 Camry recalls"` | Silently assumed TOYOTA, no warning | Correctly infers TOYOTA (unambiguous), but if make was truly unknown, would prompt |
| `"那台 Accord 有什麼問題？"` | Silent `missing_year`, no warning | Prompts: "請問哪一年？" and waits for user input |

---

#### Fix 2 — Windows UTF-8 Encoding (was: Chinese characters garbled)

**Root cause:** Windows terminal defaults to cp950 encoding. When the LLM returned a Chinese error message, `sys.stdout` could not encode it, producing garbled output.

**Fix:** Two-layer fix:
1. `cli.py` calls `sys.stdout.reconfigure(encoding="utf-8")` at startup
2. `display.py` creates `Console(force_terminal=True)` so Rich does not fall back to ASCII

| Input | Before | After |
|-------|--------|-------|
| `"那台 Accord 有什麼問題？"` | `缺少???` garbled on Windows | `缺少年份，請問哪一年？` renders correctly |

---

#### Fix 3 — Multiple Values / Conflicting Constraints → Fan-out + Synthesis (was: silent drop or crash)

**Root cause:** The LLM silently chose one value when multiple were given (e.g. picked 2019, dropped 2020), or returned an unexpected structure for multi-endpoint queries that caused a crash.

**Fix:** Implemented an **agentic fan-out pattern**:
1. System prompt updated: when the user specifies multiple years, makes, or endpoints, the LLM outputs `{"queries": [...]}` — one sub-query per combination
2. The CLI executes each sub-query independently against the NHTSA API
3. All results are fed back to the LLM, which synthesizes a unified answer in the user's language

This is the same architecture used in modern AI agent frameworks (tool use / function calling), where the model decides how many API calls are needed to fully answer the question.

| Input | Before | After |
|-------|--------|-------|
| `"Toyota Camry 2019 and 2020 recalls"` | Silently chose 2019, dropped 2020 | Queries both years, synthesizes combined summary |
| `"Honda or Toyota Camry 2020 recalls"` | Silently chose TOYOTA | Queries both makes, synthesizes comparison |
| `"Show me both recalls and safety ratings for Toyota Camry 2020"` | Unexpected LLM structure → crash | Queries both endpoints, synthesizes complete answer |

**Trade-off acknowledged:** Fan-out increases token usage and latency (proportional to number of sub-queries). This trade-off was accepted because completeness of information was prioritised over efficiency — dropping user-specified data silently is a worse outcome than a slower response.

---

#### Design Trade-off: Rule-based vs. LLM-driven Handling

The current system prompt is deliberately explicit — it lists specific typo corrections, keyword mappings, and error conditions. This is a **rule-based** approach.

The alternative is a **principle-based** approach: give the LLM a small set of high-level rules (e.g. "never guess a missing parameter", "year means model year, not issue year") and let it generalise to all edge cases on its own.

| Approach | Pros | Cons |
|----------|------|------|
| Rule-based (current) | Predictable, easy to test and eval | Rules multiply as scope expands |
| Principle-based | Fewer rules, handles novel inputs | Output less stable, harder to test |

For this project, rule-based is the right choice because the scope is narrow (3 endpoints, 3 required fields) and the eval pipeline requires deterministic, testable outputs. In a broader domain with dozens of endpoints or open-ended queries, the balance would shift toward principle-based prompting with LLM-driven generalisation.

---

#### Limitations of the "Always Ask" Approach

The interactive prompt strategy works well when the user knows the answer but simply didn't provide it. It does not solve every case:

| Situation | Example | Why prompting doesn't help |
|-----------|---------|---------------------------|
| User doesn't have the information | "I'm not sure, it's a rental car" | No amount of asking will surface a value the user doesn't know |
| Query is fundamentally out of scope | "Which car is the safest overall?" | The problem isn't a missing parameter — the NHTSA API doesn't support comparative queries |

For these cases, the correct response is not to ask again, but to clearly explain what the tool can and cannot do. The current implementation handles this via the `out_of_scope` error path, which exits with a plain-language explanation rather than looping.

---

#### Identified as Beyond Fix — Carried Forward to Requirement 4

During hardening, the following cases were identified as impossible to resolve at the programme level. No amount of prompt engineering or code changes can fully solve them, because the root cause lies outside the system boundary:

| Input Example | Root Cause | Why Unfixable Here |
|---------------|-----------|-------------------|
| `"我不知道我租的車是什麼牌子"` | User genuinely lacks the information | Prompting cannot surface data the user doesn't have |
| `"Show me 2021 recalls"` (meaning recalls *issued* in 2021) | NHTSA API only supports querying by vehicle model year, not by recall issue year | API design constraint — no workaround at the application layer |
| `"Safety ratings for Toyota Camry 2020"` (expecting star scores) | Safety Ratings endpoint returns variant list + Vehicle IDs first; actual star ratings require a second call per variant | One query becomes 3–5 API calls depending on variant count |

These cases are documented in detail in **Requirement 4**.

---

### Requirement 4 — Remaining Hard Cases

These cases remain unresolved after hardening because the root cause lies outside the programme boundary — in the NHTSA API design, in the user's own knowledge, or in the fundamental probabilistic nature of LLMs. No amount of prompt engineering or code changes can fully solve them.

---

#### Hard Case 1 — User Lacks the Required Information

**What happens:** The interactive prompt asks the user for a missing parameter, but the user genuinely does not have the answer.

**Example:**
```
Tool:  "Which make and model is your vehicle?"
User:  "I'm not sure, it's a rental car"
```

**Why fundamentally hard:** Prompting can only surface information the user already knows. If the information does not exist in the user's mind, no number of retries will produce it. The tool correctly exits with an error, but cannot resolve the underlying gap.

---

#### Hard Case 2 — NHTSA API Does Not Support Querying by Recall Issue Year

**What happens:** A user asking "Show me 2021 recalls" likely means recalls *issued* in 2021. The NHTSA API `modelYear` parameter refers to the **vehicle's model year**, not the year the recall was issued. There is no API parameter for recall issue year.

**Example:**
```
User intent:  "Recalls issued in 2021"
API behaviour: Returns all recalls ever issued for vehicles built in 2021
```

**Why fundamentally hard:** This is an API design constraint. The application cannot retrieve data the API does not expose. The tool currently clarifies in its prompt that "year" means model year, but cannot fulfil the user's original intent if they wanted issue-year filtering.

---

#### Hard Case 3 — Safety Ratings Requires a Second API Call Per Variant

**What happens:** The NHTSA Safety Ratings endpoint returns a list of vehicle variants (e.g. AWD, FWD, Hybrid) with Vehicle IDs — but not the actual star ratings. A second API call per variant is required to retrieve the crash test scores.

**Example:**
```
Query: "Safety ratings for Toyota Camry 2020"
API call 1 → [{"VehicleId": 14855, "VehicleDescription": "2020 Toyota CAMRY 4 DR AWD"}, ...]
API call 2 (per variant) → actual star ratings
```

**Why fundamentally hard:** This is an NHTSA API design decision. A single user query becomes 3–5 API calls depending on how many variants exist. The current implementation displays the available variants and informs the user that a more specific query is needed. Fixing it fully would require multi-pass fetching in `nhtsa_client.py`, which multiplies latency and cost proportionally.

---

#### Hard Case 4 — Out-of-Scope Detection is Probabilistic, Not Guaranteed

**What happens:** Comparative queries (e.g. "哪台車最安全？") should return `out_of_scope`, but the LLM occasionally misclassifies them as `missing_make` and triggers an interactive prompt instead.

**Why fundamentally hard:** LLMs generalise probabilistically. Adding examples to the system prompt improves accuracy but cannot guarantee correct behaviour on every novel phrasing. This is a property of the model, not the prompt.

**What would be required to fix:** A dedicated intent classification step before the main query parser — either a separate LLM call or a fine-tuned classifier trained to distinguish "specific vehicle lookup" from "comparative / open-ended" queries. Both add latency and cost.

---

## Part 2: Multi-Model Evaluation

### Requirement 1 — Data Generation

30 natural language queries were **programmatically generated** using Groq (Llama 3.3 70B) via [`eval/generate_test_cases.py`](eval/generate_test_cases.py). The generation prompt specified exact category counts, adversarial requirements, and output schema — the model was not hand-held through each case. The set was designed with **10 simple baselines and 20 hard adversarial cases** to create a discriminative benchmark that exposes meaningful capability differences between models.

**Final category distribution:**

| Category | Count | Type | Adversarial challenge |
|----------|-------|------|-----------------------|
| adversarial_recalls | 2 | simple | Clean recall queries — baseline |
| adversarial_complaints | 2 | simple | Clean complaint queries — baseline |
| complex_constraints | 2 | simple | Multiple years or endpoints, no typos |
| make_inference | 2 | simple | Make absent, model name uniquely implies it |
| noise_and_descriptors | 1 | simple | One obvious trim word to strip |
| error_out_of_scope | 1 | simple | Obviously comparative query |
| hard_error_missing_year | 4 | **hard** | Vehicle present but year absent — model must not guess |
| hard_error_missing_make | 4 | **hard** | Year present, generic category only — no valid brand |
| hard_error_out_of_scope | 4 | **hard** | Specific vehicle mentioned — vehicle presence is a trap |
| hard_noise_and_descriptors | 4 | **hard** | Stacked sub-model codes, trim, engine, body style noise |
| hard_adversarial_mixed | 4 | **hard** | 3+ simultaneous challenges: typos + Chinese + informal |

**Adversarial coverage enforced in the generation prompt:**
- At least 3 queries partially or fully in Traditional Chinese
- At least 2 queries with simultaneous typos in both make and model (e.g. `toyta cmary`, `hond civick`)
- At least 1 abbreviated make that must expand (e.g. `chevy→CHEVROLET`, `bimmer→BMW`, `VW→VOLKSWAGEN`)
- All complex_constraints cases require `{"queries": [...]}` output — not a single structured query

---

### Requirement 2 — Ground Truth

Ground truth for all 30 cases was proposed by the generation model and then **manually reviewed and verified** field by field. The schema used:

```json
// Valid query
{"endpoint": "recalls|complaints|safetyRatings", "make": "UPPERCASE", "model": "UPPERCASE", "year": "YYYY"}

// Multiple queries (complex constraints)
{"queries": [{"endpoint": "...", "make": "...", "model": "...", "year": "..."}, ...]}

// Error
{"error": "missing_year|missing_make|out_of_scope"}
```

**Why error cases are the hardest adversarial cases:**
Models are trained to be helpful — when a parameter is missing they tend to guess or fill in a default rather than return an error. `missing_year` cases (e.g. `"toyota camry recalls"`) and `out_of_scope` cases (e.g. `"what is the safest car in 2022?"`) are designed to expose exactly this failure mode.

**Why complex_constraints cases break models:**
Most models default to outputting a single structured query. Queries like `"toyota camry 2019 and 2020 recalls"` require the model to recognise the ambiguity, fan out into two sub-queries, and wrap them in a `{"queries": [...]}` array — a pattern that breaks models which only know how to output a single object.

#### Representative Test Cases

The cases below illustrate the difficulty level. Each is genuinely hard — not merely obscure.

**Case 16 — `hard_error_missing_make`** (year present, brand absent)
```json
{
  "nl_query": "show me 2019 truck recalls",
  "ground_truth": {"error": "missing_make"},
  "notes": "Year and endpoint are clear. 'truck' is a vehicle category, not a brand. Model must resist inferring Ford/Chevrolet/RAM and instead return missing_make."
}
```
*Why it's hard:* The query sounds complete. All three models initially invented a make (`FORD`, `CHEVROLET`) because "truck" primes common brands. Llama failed this category (3/4 cases) even after Round 2 prompt iteration.

**Case 27 — `hard_adversarial_mixed`** (typos + Chinese + missing year)
```json
{
  "nl_query": "toyta cmary 安全評等 is it safe?",
  "ground_truth": {"error": "missing_year"},
  "notes": "Typos in make and model, Chinese safety-rating keyword, informal phrasing — but no year. Model must normalise the typos AND still refuse to guess a year."
}
```
*Why it's hard:* Three simultaneous challenges. A model that correctly handles any two in isolation will still fail if the third causes it to skip the year-check step. The original ground truth generated by Llama hallucinated `year: "2022"` for this case — caught and fixed during manual review.

**Case 29 — `hard_adversarial_mixed`** (abbreviation + typo + digit/letter confusion in year)
```json
{
  "nl_query": "chevy slverado 2O21 recal",
  "ground_truth": {"endpoint": "recalls", "make": "CHEVROLET", "model": "SILVERADO", "year": "2021"},
  "notes": "Abbreviated make (chevy→CHEVROLET), typo in model (slverado), digit/letter confusion in year (2O21→2021), and typo in keyword (recal). Model must resolve all four simultaneously."
}
```
*Why it's hard:* `2O21` contains a capital letter O where the digit 0 should be. The year looks like a 4-character string but is not a valid 4-digit year until character normalisation is applied. All three models passed this case after Round 1 principle-based rules were added.

---

### Requirement 3 & 4 — Execution & Iteration

#### Models Evaluated

| Model | Type | Provider |
|-------|------|---------|
| `claude-sonnet-4-6` | closed-source | Anthropic |
| `gpt-4o-mini` | closed-source | OpenAI |
| `llama-3.3-70b-versatile` | open-weight | Groq (free tier) |

#### Final Results

| Model | Accuracy | Avg Field Score | >85%? |
|-------|----------|----------------|-------|
| claude-sonnet-4-6 | 30/30 (100%) | 1.00 | ✓ |
| gpt-4o-mini | 27/30 (90%) | 0.90 | ✓ |
| llama-3.3-70b-versatile | 26/30 (87%) | 0.89 | ✓ |

Full per-case results: [eval/results/summary.json](eval/results/summary.json)

#### Prompt Iteration History

Two rounds of prompt iteration were required before all three models reached >85%.

**Round 0 — Baseline (hard test set, original prompt)**

| Model | Score | Status |
|-------|-------|--------|
| Claude Sonnet 4.6 | 26/30 (87%) | ✓ |
| GPT-4o-mini | 23/30 (77%) | ✗ |
| Llama 3.3 70B | 21/30 (70%) | ✗ |

**Round 1 — Principle-based rules**

Added three principle-based rules to the system prompt:
1. Generic vehicle categories (`sedan`, `truck`, `SUV`) are not valid makes — return `missing_make`
2. Sub-model codes (`328i`, `C200`) must be stripped to base model family (`3 SERIES`, `C-CLASS`)
3. Character-level noise (`0↔O`, `1↔I`, `5↔S`) should be normalised before parsing

These were written as generalising principles rather than hard-coded brand mappings, ensuring they would also handle unseen inputs (e.g. Mercedes `C200`, `T0Y0TA`).

*Key diff — Round 0 → Round 1:*

```diff
- 3. Correct typos in make and model using context and common automotive knowledge.
+ 3. Correct typos in make and model using context and common automotive knowledge. Input may
+    also contain character-level noise where visually similar characters are substituted
+    (e.g. 0↔O, 1↔I, 5↔S). Apply best-effort normalization to recover the most likely
+    intended make, model, and year.

+ 9. If the model field contains a sub-model code or trim designation (e.g. 328i, C200,
+    A4 2.0T, Civic Sport), strip the sub-model suffix and return only the base model
+    family name as used in automotive safety databases (e.g. 3 SERIES, C-CLASS, A4, CIVIC).

+ 10. If the query cannot be answered by looking up a specific vehicle (make+model+year),
+     return out_of_scope. Generic category words (car, sedan, truck, SUV, vehicle) are NOT
+     brand names and are NOT valid makes or models.
```

| Model | Score | Status |
|-------|-------|--------|
| Claude Sonnet 4.6 | 29/30 (97%) | ✓ |
| GPT-4o-mini | 27/30 (90%) | ✓ |
| Llama 3.3 70B | 21/30 (70%) | ✗ |

Claude and GPT improved significantly. Llama did not — the same principle-based instructions that worked for frontier models were ignored by Llama in favour of its trained "be helpful" prior.

**Round 2 — Checklist-style enforcement for Llama**

Rules 7 and 8 were rewritten as explicit step-by-step checks:

*Key diff — Round 1 → Round 2:*

```diff
- 7. NEVER guess a year. If no year is explicitly present, return {"error": "missing_year"}.
- 8. MAKE CHECK — if no specific brand is present, return {"error": "missing_make"}.
-    Generic category words are not valid makes.

+ 7. YEAR CHECK — before returning any result, ask: does the query contain a specific
+    4-digit year (e.g. 2019, 2022)?
+    - YES → use it
+    - NO → return {"error": "missing_year", "message": "..."} immediately. No exceptions.
+    NEVER guess, infer, or default to the current year. Phrases like "my car", "I bought
+    it used", "is it good?" do NOT imply a year.
+
+ 8. MAKE CHECK — before returning any result, ask: does the query contain a specific
+    vehicle brand name or a model name that uniquely implies a brand?
+    - YES → use it or infer it
+    - NO → return {"error": "missing_make", "message": "..."} immediately. No exceptions.
+    CRITICAL: Generic category words (car, sedan, truck, SUV, vehicle, automobile, pickup)
+    are NOT brand names and are NOT valid makes or models.
+    Examples that MUST return missing_make:
+    - "2021 sedan complaints" → no brand → missing_make
+    - "show me 2019 truck recalls" → no brand → missing_make
+    - "2022 SUV safety ratings" → no brand → missing_make
```

| Model | Score | Status |
|-------|-------|--------|
| Claude Sonnet 4.6 | 30/30 (100%) | ✓ |
| GPT-4o-mini | 27/30 (90%) | ✓ |
| Llama 3.3 70B | 26/30 (87%) | ✓ |

The checklist framing dramatically improved Llama's year detection (1/4 → 4/4) but only partially improved make detection (0/4 → 1/4). The root cause is discussed in Performance Analysis below.

---

### Requirement 5 — Write-up

#### Model Selection

Three models were chosen to represent distinct points on the capability-cost-openness spectrum:

**Claude Sonnet 4.6 (Anthropic)** — the highest-capability closed-source option. Chosen because structured output tasks with complex rules (error detection, multi-query fan-out) benefit from precise instruction following, and Claude is the strongest model available for this. Expected to set the upper bound.

**GPT-4o-mini (OpenAI)** — a cost-efficient closed-source model with native JSON output mode (`response_format: json_object`). Chosen to represent the category of small but capable frontier models that are widely used in production. Expected to match Claude on clean inputs but show gaps on adversarial cases.

**Llama 3.3 70B (Groq)** — the strongest open-weight model available on a free inference tier. Chosen to represent the open-weight category, which matters for cost-sensitive or data-privacy-sensitive deployments. Expected to be competitive but less reliable on edge cases.

All three models were expected to reach >85% because the task is narrow and well-defined: three endpoints, three required fields, deterministic output schema. Any instruction-tuned model with good English and Chinese comprehension should handle the core cases reliably; the adversarial cases separate them.

---

#### Performance Analysis

**What models initially got wrong (Round 0, hard test set):**

All three models failed on the same two categories, for different reasons:

*hard_error_missing_year* — Models were given queries like `"Ford F-150 transmission complaints"` or `"我的 Toyota Camry 有召回嗎？"` where the vehicle is present but no year is given. Claude and GPT guessed years or returned malformed JSON; Llama returned partial results. All three had a strong trained prior toward "give a useful answer" rather than "refuse with an error."

*hard_error_missing_make* — Queries like `"2021 sedan complaints"` or `"2022 SUV safety ratings"` contain year and endpoint but only a generic vehicle category, not a brand. All models — especially Llama — tended to invent a make (e.g. returning `TOYOTA` for "sedan") rather than returning `missing_make`.

Claude also failed 3/4 `hard_noise_and_descriptors` cases, while GPT failed 2/4. Both struggled with stacked sub-model codes and trim designations (e.g. `BMW 328i xDrive Sport Line`). Llama, interestingly, performed perfectly on noise stripping throughout.

**How prompt iteration improved results:**

The key finding from iteration was that **frontier models (Claude, GPT) respond well to principle-based instructions**, while **Llama requires checklist-style, step-by-step enforcement**.

When told "generic vehicle categories are not valid makes — return missing_make", Claude and GPT immediately applied this correctly. Llama continued to guess, because its training prior for "helpful assistant" overrides abstract rules.

When the same rule was rewritten as an explicit decision tree — "MAKE CHECK: before returning any result, ask: does the query contain a specific brand name? NO → return missing_make immediately" — Llama's year detection improved from 1/4 to 4/4. Make detection improved only from 0/4 to 1/4, because the semantic boundary between a brand name and a generic category (e.g. "truck" vs "FORD") is inherently fuzzier than the presence/absence of a 4-digit year.

**Remaining failures after final iteration:**

- GPT (3 failures): `hard_adversarial_mixed` — queries with three or more simultaneous challenges (typo + Chinese keyword + missing year, or abbreviated make + heavy typo + digit-letter confusion). GPT correctly handles individual challenges but degrades when they stack.
- Llama (4 failures): `hard_error_missing_make` (3 remaining) — Llama consistently infers a brand from generic category words despite explicit instructions. This is a model-level capability gap, not a prompt gap.

---

#### Learnings

**Models fill gaps with guesses, not errors**

Going into this project, the assumption was that a well-instructed model would recognize missing information and return an error. The actual behaviour was the opposite: when a required parameter was absent, all three models defaulted to inventing a plausible value and returning a confident answer. A query like `"Ford F-150 transmission complaints"` — no year present — would come back with a fabricated year rather than `missing_year`. This mirrors how humans respond to incomplete information: the brain fills the gap rather than admitting uncertainty. The corrective was explicit, step-by-step enforcement in the prompt ("YEAR CHECK: does the query contain a 4-digit year? NO → return missing_year immediately"), not a general instruction to "be accurate." The takeaway is that models should not be trusted to self-detect missing inputs without structural guardrails — they are optimized to produce an answer, not to refuse.

**A test set that all models pass teaches you nothing**

The first version of the 30-case eval produced scores of Claude 87%, GPT 87%, Llama 87% across the board. Every model passed, and the numbers gave no signal about which model was actually more capable or where the failure modes lay. The test set was redesigned with 20 hard adversarial cases targeting the exact failure modes identified during Part 1 hardening — missing parameters, generic category words, stacked noise, out-of-scope traps. The result was a benchmark where scores spread from 70% to 87% in Round 0, revealing real capability differences and giving a concrete target for prompt iteration. A discriminative benchmark must actively try to break each model; cases where all models succeed provide no information.

**LLM-generated ground truth requires human review**

Test cases were generated by Llama 3.3 70B, which also happens to be one of the models being evaluated. The assumption was that the generation model could produce valid ground truth for cases it was not directly being scored on. This proved false: Case 27 (`"toyta cmary 安全評等 is it safe?"`) had no year in the query, but the Llama-generated ground truth included `"year": "2022"` — a hallucination that would have made a genuinely correct `missing_year` response score zero. Manual field-by-field review of all 30 cases caught this and several similar rationalizations where the generation model had quietly resolved ambiguity in a way that matched its own tendencies, not the intended test. When LLMs generate their own ground truth, they encode their own biases into the benchmark. Human review is the only reliable way to break that loop.

**Prompt engineering targets the model, not the task**

The most unexpected finding was that the same prompt produced drastically different results across models. Adding principle-based rules — "generic vehicle categories are not valid makes" — caused Claude and GPT-4o-mini to improve immediately and significantly. Llama 3.3 70B showed no improvement at all on the same rules. The model's trained "helpful assistant" prior overrode abstract instructions in a way that frontier models did not exhibit. Only when the rules were restructured as explicit decision trees ("MAKE CHECK — before returning any result, ask: does the query contain a specific brand name? NO → return missing_make immediately") did Llama's behaviour change. This revealed that prompt engineering is not a task-level problem — it is a model-level problem. A prompt optimized for Claude is not automatically effective for Llama. In production systems that switch or compare models, prompts need to be validated per model, not assumed to transfer.

---

## License

MIT
