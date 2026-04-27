"""
Part 2 Requirement 1 — Programmatic test case generation.

Uses Groq (Llama 3.3 70B) to generate 30 diverse NL queries + ground truth
covering recalls, complaints, safetyRatings, edge cases, multilingual inputs,
typos, missing fields, and out-of-scope queries.

Usage:
    python eval/generate_test_cases.py
"""

import json
import sys
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq()

# ── Generation prompt ──────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are building an adversarial evaluation dataset for a NHTSA vehicle safety query parser.

The parser accepts natural language queries and outputs structured JSON with this schema:
- Single valid query: {"endpoint": "recalls|complaints|safetyRatings", "make": "UPPERCASE", "model": "UPPERCASE", "year": "YYYY"}
- Multiple queries (when user specifies multiple years, makes, or endpoints): {"queries": [{"endpoint": "...", "make": "...", "model": "...", "year": "..."}, ...]}
- Error: {"error": "missing_year|missing_make|out_of_scope"}

Generate exactly 30 test cases as a JSON array. Each element must have:
- "id": integer 1–30
- "category": one of the categories below
- "nl_query": the natural language input string
- "ground_truth": the exact correct JSON the parser should output
- "notes": one sentence explaining the adversarial challenge

Category distribution (must follow exactly):

SIMPLE BASELINE — 10 cases total (2 per category, straightforward inputs, no tricks):
1. adversarial_recalls — 2 cases (clean English, make+model+year all present, clear recall keyword)
2. adversarial_complaints — 2 cases (clean English, all fields present, clear complaint keyword)
3. complex_constraints — 2 cases (multiple years or endpoints, no typos)
4. make_inference — 2 cases (make absent, model uniquely implies make, no other tricks)
5. noise_and_descriptors — 1 case (one obvious trim word to strip, e.g. SE or AWD)
   error_out_of_scope — 1 case (obviously comparative, e.g. "which car is safest?")

HARD ADVERSARIAL — 20 cases total (4 per category, designed to actively break models):
6. hard_error_missing_year — 4 cases
   These must be TRICKY. The vehicle is present and the query sounds complete, but year is absent.
   Models tend to guess a year — these cases punish that.
   Examples of hard patterns:
   - "Any recalls for my Honda Civic? I bought it used." (no year, sounds complete)
   - "Ford F-150 transmission complaints" (model present, no year, sounds like a valid query)
   - "我的 Toyota Camry 有召回嗎？" (Chinese, no year)
   - "Jeep Wrangler safety rating — is it good?" (informal, no year)

7. hard_error_missing_make — 4 cases
   Year is present. No make. No model name that uniquely implies a make.
   Models tend to invent a make — these cases punish that.
   Examples:
   - "2021 sedan complaints" (year present, generic vehicle type not a specific make/model)
   - "show me 2019 truck recalls" (generic category, not a specific vehicle)
   - "2022 SUV safety ratings" (year present, no make/model)
   - "any 2020 car recalls?" (year present, no identifiable vehicle)

8. hard_error_out_of_scope — 4 cases
   TRAP: a real, specific vehicle is mentioned but the question CANNOT be answered by NHTSA API.
   The vehicle presence is a trap — models may try to generate a query instead of returning out_of_scope.
   Examples:
   - "Is the 2021 Toyota Camry more reliable than the Honda Accord?" (comparative, vehicle present)
   - "Which year of the Ford F-150 had the most recalls?" (aggregate across years, no single year)
   - "How do I reset the oil light on my 2019 Honda Civic?" (maintenance, not safety data)
   - "What's the resale value of a 2020 Jeep Wrangler?" (non-safety question, vehicle present)

9. hard_noise_and_descriptors — 4 cases
   Multiple noise words stacked together. The model must strip ALL noise and return the base model only.
   Sub-model numbers (e.g. 328i→3 SERIES), engine specs (2.0T), trim (Sport), body style (Hatchback),
   drivetrain (AWD), colour, and package names all count as noise.
   Examples:
   - "2021 BMW 328i xDrive Sport Line recalls" → model: 3 SERIES
   - "2020 Honda Civic Sport 1.5T CVT Hatchback complaints" → model: CIVIC
   - "Toyota Camry XSE V6 AWD 2022 safety rating" → model: CAMRY
   - "Ford F-150 XLT SuperCrew 4x4 2019 recalls" → model: F-150

10. hard_adversarial_mixed — 4 cases
    Each query combines 3+ adversarial signals simultaneously:
    typo in make + typo in model + Chinese keyword + informal phrasing
    or: abbreviated make + missing year + partial Chinese
    or: wrong capitalization + noise descriptor + out-of-order fields
    These are the hardest cases and should look like real messy user input.
    Examples:
    - "toyta cmary 安全評等 is it safe?" (typos + Chinese + informal, safety ratings)
    - "hond civick 2019 投訴 any issues?" (typos + Chinese + informal, complaints)
    - "chevy slverado 2O21 recal" (abbreviation + typo in model + digit/letter confusion in year, recalls)
    - "2022 vw gof safty ratng" (abbreviated make + typos in model + typo in keyword)

For hard_adversarial_mixed cases: year "2O21" means the digit zero was replaced with letter O — year should still resolve to "2021".

Ground truth rules:
- make and model must be UPPERCASE
- year must be a 4-digit string
- Use real vehicles and real model years (2015–2024)
- For make_inference cases, make MUST be absent from the nl_query
- For noise_and_descriptors cases, noise words MUST appear in nl_query but NOT in ground_truth model field
- For all error cases, ground_truth must be exactly {"error": "missing_year"}, {"error": "missing_make"}, or {"error": "out_of_scope"} — no other fields
- For hard_error_missing_make, nl_query must NOT contain any model name that uniquely implies a make
- For hard_error_out_of_scope, a specific vehicle MUST be mentioned in the query (this is what makes it a trap)
- For complex_constraints, ground_truth must use {"queries": [...]} format

Output ONLY a valid JSON array with exactly 30 elements. No markdown, no explanation."""

# ── Generate ───────────────────────────────────────────────────────────────────

def generate_cases() -> list:
    print("Calling Groq (Llama 3.3 70B) to generate 30 test cases...")
    resp = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4096,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": GENERATION_PROMPT},
        ],
    )
    raw = resp.choices[0].message.content
    data = json.loads(raw)

    # Model may wrap the array in a key
    if isinstance(data, list):
        return data
    for val in data.values():
        if isinstance(val, list):
            return val
    raise ValueError(f"Unexpected response shape: {list(data.keys())}")


# ── Validate ───────────────────────────────────────────────────────────────────

VALID_ENDPOINTS = {"recalls", "complaints", "safetyRatings"}
VALID_ERRORS = {"missing_year", "missing_make", "missing_endpoint", "out_of_scope", "ambiguous_input"}
REQUIRED_KEYS = {"id", "category", "nl_query", "ground_truth", "notes"}

def validate_single_query(gt: dict, prefix: str) -> list[str]:
    issues = []
    for field in ("endpoint", "make", "model", "year"):
        if field not in gt:
            issues.append(f"{prefix}: ground_truth missing '{field}'")
    if gt.get("endpoint") not in VALID_ENDPOINTS:
        issues.append(f"{prefix}: invalid endpoint '{gt.get('endpoint')}'")
    if gt.get("make") != gt.get("make", "").upper():
        issues.append(f"{prefix}: make not uppercase")
    if gt.get("model") != gt.get("model", "").upper():
        issues.append(f"{prefix}: model not uppercase")
    year = gt.get("year", "")
    if not (isinstance(year, str) and len(year) == 4 and year.isdigit()):
        issues.append(f"{prefix}: year '{year}' is not a 4-digit string")
    return issues


def validate(cases: list) -> list[str]:
    issues = []
    ids_seen = set()

    for i, c in enumerate(cases):
        prefix = f"Case {c.get('id', f'[index {i}]')}"

        missing = REQUIRED_KEYS - c.keys()
        if missing:
            issues.append(f"{prefix}: missing keys {missing}")
            continue

        cid = c["id"]
        if cid in ids_seen:
            issues.append(f"{prefix}: duplicate id")
        ids_seen.add(cid)

        gt = c["ground_truth"]

        if "error" in gt:
            if gt["error"] not in VALID_ERRORS:
                issues.append(f"{prefix}: unknown error type '{gt['error']}'")
        elif "queries" in gt:
            # complex_constraints: validate each sub-query
            for j, sub in enumerate(gt["queries"]):
                issues.extend(validate_single_query(sub, f"{prefix} sub-query {j+1}"))
        else:
            issues.extend(validate_single_query(gt, prefix))

    if len(cases) != 30:
        issues.append(f"Expected 30 cases, got {len(cases)}")

    return issues


# ── Write test_cases.py ────────────────────────────────────────────────────────

HEADER = '''"""
30 adversarial test cases for QueryForge multi-model evaluation.

Auto-generated by eval/generate_test_cases.py using Groq (Llama 3.3 70B).

Each case has:
  - nl_query    : natural language input fed to the model
  - ground_truth: the exact JSON the model should output
  - category    : for analysis grouping
  - notes       : the adversarial challenge this case tests
"""

TEST_CASES = '''

def write_test_cases(cases: list, out_path: Path) -> None:
    content = HEADER + json.dumps(cases, indent=4, ensure_ascii=False) + "\n"
    out_path.write_text(content, encoding="utf-8")
    print(f"Written to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    out_path = Path(__file__).parent / "test_cases.py"

    cases = generate_cases()
    print(f"Received {len(cases)} cases from model.")

    issues = validate(cases)
    if issues:
        print("\nValidation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nAttempting to auto-fix...")

        # Auto-fix: uppercase make/model, coerce year to string
        for c in cases:
            gt = c.get("ground_truth", {})
            if "error" not in gt:
                if "make" in gt:
                    gt["make"] = str(gt["make"]).upper()
                if "model" in gt:
                    gt["model"] = str(gt["model"]).upper()
                if "year" in gt:
                    gt["year"] = str(gt["year"])

        issues_after = validate(cases)
        if issues_after:
            print("Remaining issues after auto-fix:")
            for issue in issues_after:
                print(f"  - {issue}")
            print("\nSaving anyway — review manually before running eval.")
        else:
            print("All issues resolved by auto-fix.")

    write_test_cases(cases, out_path)

    print("\nCategory breakdown:")
    from collections import Counter
    counts = Counter(c["category"] for c in cases)
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
