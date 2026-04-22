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
git clone https://github.com/<your-username>/queryforge.git
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

| Key | Where to get |
|-----|-------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier available |

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
│  query_generator.py │  ← LLM (Claude Sonnet 4.6)
│                     │    Outputs strict JSON only
└─────────────────────┘
        │
        ▼
{
  "endpoint": "recalls",
  "params": {
    "make": "TOYOTA",
    "model": "CAMRY",
    "year": "2020"
  }
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

| Input | Observed Behavior |
|-------|------------------|
| `"toyta cmary 2020 recall"` | ✅ LLM correctly normalized to TOYOTA CAMRY, query succeeded |
| `"Hond Civ1c 2O19 的投訴"` | ✅ LLM correctly normalized to HONDA CIVIC 2019, query succeeded |

#### Ambiguous Inputs

| Input | Observed Behavior |
|-------|------------------|
| `"Toyota Camry recalls"` (no year) | ❌ LLM returned `year: null`, client sent `modelYear=null` to API → 400 error |
| `"2020 Camry recalls"` (no make) | ⚠️ LLM inferred TOYOTA from model name and returned results — silent assumption, no warning to user |
| `"Toyota Camry 2027 recalls"` (future year) | ❌ API returned 400 error — no graceful handling |
| `"1965 Ford Mustang safety rating"` (pre-API era) | ✅ API returned empty results, displayed gracefully |
| `"Toyota Camry 2020 brake recalls only"` (component filter) | ⚠️ Returned all recalls without filtering — component-level filtering not supported by NHTSA API |
| `"那台 Accord 有什麼問題？"` (no year, Chinese) | ❌ LLM returned Chinese error message, but terminal displayed garbled text due to Windows encoding (cp950) |

#### Conflicting Constraints

| Input | Observed Behavior |
|-------|------------------|
| `"Which SUVs have the most recalls in 2022?"` | ❌ LLM produced empty make/model, API returned 400 error instead of a clean out_of_scope message |
| `"Best and worst safety rated car in 2022"` | ✅ Returned no results — silent failure, no out_of_scope error |
| `"2022年最安全的車是哪台？"` | ✅ Returned no results — same silent failure |

#### Languages Other Than English (Traditional Chinese)

| Input | Observed Behavior |
|-------|------------------|
| `"Toyota Camry 2020 有哪些召回問題"` | ✅ Correctly identified as recalls query, returned results |
| `"Toyota Camry 2020 煞車召回"` | ⚠️ Returned all recalls — component filter not supported |
| `"那台 Accord 有什麼問題？"` | ❌ Error message garbled on Windows terminal |
| `"2022年最安全的車是哪台？"` | ⚠️ Silent failure — no results, no explanation |
| `"Hond Civ1c 2O19 的投訴"` | ✅ Typo corrected, correct results returned |

---

### Requirement 3 — Harden & Fix

_To be completed after fixing identified bugs._

---

### Requirement 4 — Remaining Hard Cases

_To be completed after hardening._

---

## Part 2: Multi-Model Evaluation

### Evaluation Design

- **30 test cases** covering recalls, complaints, and safety ratings, with a mix of clean inputs, typos, missing fields, multilingual queries, and edge cases
- **Ground truth** manually written as exact JSON for each case
- **Scoring** is field-level: each correct field (endpoint, make, model, year) scores 1 point; full accuracy requires all fields correct
- **Target:** all three models > 85% accuracy (≥ 26/30 fully correct)

### Models Evaluated

| Model | Type | Provider |
|-------|------|---------|
| `claude-sonnet-4-6` | closed-source | Anthropic |
| `gpt-4o-mini` | closed-source | OpenAI |
| `llama-3.3-70b-versatile` | open-weight | Groq (free tier) |

### Results

<!-- Fill in after running eval/evaluator.py -->

| Model | Accuracy | Avg Field Score |
|-------|----------|----------------|
| claude-sonnet-4-6 | — | — |
| gpt-4o-mini | — | — |
| llama-3.3-70b-versatile | — | — |

### Model Selection Rationale

<!-- Fill in after eval -->

### Performance Analysis

<!-- Fill in after eval -->

### Learnings

<!-- Fill in after eval -->

---

## License

MIT
