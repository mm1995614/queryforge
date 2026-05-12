# Prompt Version History

Each version is a standalone `.txt` file. The evaluator (`eval/evaluator.py`) reads
the active prompt from the file named in `PROMPT_VERSION` in `src/query_generator.py`.
Each eval run produces a corresponding result file in `eval/results/`.

---

## v2_checklist_enforcement (current) — Round 2

**File:** [`v2_checklist_enforcement.txt`](v2_checklist_enforcement.txt)
**Results:** [`eval/results/v2_checklist_enforcement.json`](../eval/results/v2_checklist_enforcement.json)

| Model | Score | Pass? |
|-------|-------|-------|
| Claude Sonnet 4.6 | 30/30 (100%) | ✓ |
| GPT-4o-mini | 27/30 (90%) | ✓ |
| Llama 3.3 70B | 26/30 (87%) | ✓ |

**Changes from v1:**

Rules 7 and 8 were rewritten from abstract principle statements into explicit
step-by-step decision trees with YES/NO branches:

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

**Why:** Llama 3.3 70B ignored principle-based rules (v1 scored 21/30) because its
trained "helpful assistant" prior overrides abstract instructions. The checklist framing
with explicit YES/NO branches is harder to bypass than a general statement.

**Effect:** Llama year detection improved from 1/4 → 4/4. Make detection improved
from 0/4 → 1/4 (the brand/category boundary is semantically fuzzier than year presence).

---

## v1_principle_rules — Round 1

**File:** [`v1_principle_rules.txt`](v1_principle_rules.txt)
**Results:** [`eval/results/v1_principle_rules.json`](../eval/results/v1_principle_rules.json)

| Model | Score | Pass? |
|-------|-------|-------|
| Claude Sonnet 4.6 | 29/30 (97%) | ✓ |
| GPT-4o-mini | 27/30 (90%) | ✓ |
| Llama 3.3 70B | 21/30 (70%) | ✗ |

**Changes from v0:**

Three new principle-based rules were added in response to Round 0 failure analysis:

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

**Why:** Round 0 analysis showed three distinct failure modes: (1) Claude and GPT both failed
`hard_noise_and_descriptors` because `2O21`-style digit/letter confusion wasn't handled;
(2) all models failed sub-model stripping (`BMW 328i xDrive` → should return `3 SERIES`);
(3) all models invented makes for generic category words instead of returning `missing_make`.
Principle-based rules were chosen over hard-coded mappings so they would generalise to
unseen inputs (e.g. `T0Y0TA`, Mercedes `A180`).

**Effect:** Claude and GPT improved significantly. Llama did not — see v2 for root cause.

---

## v0_baseline — Round 0

**File:** [`v0_baseline.txt`](v0_baseline.txt)
**Results:** [`eval/results/v0_baseline.json`](../eval/results/v0_baseline.json)

| Model | Score | Pass? |
|-------|-------|-------|
| Claude Sonnet 4.6 | 26/30 (87%) | ✓ |
| GPT-4o-mini | 23/30 (77%) | ✗ |
| Llama 3.3 70B | 21/30 (70%) | ✗ |

**Prompt design:** Basic structured-output rules covering endpoint inference, typo
correction, and multi-query fan-out. No character noise handling. Year and make checks
stated as simple prohibitions rather than enforced decision trees.

**Key failures (informed Round 1 iteration):**
- `hard_noise_and_descriptors`: All models struggled with stacked sub-model codes
  (e.g. `BMW 328i xDrive Sport Line`) — base model stripping was not covered
- `hard_error_missing_year`: Models guessed years rather than returning `missing_year`
  — the prohibition "NEVER guess a year" was not strong enough to override the
  models' trained "give a useful answer" prior
- `hard_error_missing_make`: All models invented makes from generic category words
  (`truck`→`FORD`, `sedan`→`TOYOTA`) — no explicit rule covered this case
