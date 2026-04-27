"""
Part 2 Requirement 1 — Programmatic test case generation.

Uses Groq (Llama 3.3 70B) to generate 30 diverse NL queries + ground truth
covering recalls, complaints, safetyRatings, edge cases, multilingual inputs,
typos, missing fields, and out-of-scope queries.

Usage:
    python eval/generate_test_cases.py
"""

import ast
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EXPECTED_CASE_COUNT = 30
# Groq Llama 3.3 70B is used because it's available on the free tier,
# which is sufficient for one-time test case generation.
GENERATOR_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

VALID_ENDPOINTS = {"recalls", "complaints", "safetyRatings"}
VALID_ERRORS = {"missing_year", "missing_make", "missing_endpoint", "out_of_scope", "ambiguous_input"}
REQUIRED_KEYS = {"id", "category", "nl_query", "ground_truth", "notes"}

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

def _call_with_retry(client: Groq) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GENERATOR_MODEL,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": GENERATION_PROMPT}],
            )
            return response.choices[0].message.content
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "Attempt %d/%d failed (%s) — retrying in %ds...",
                attempt, MAX_RETRIES, exc, RETRY_DELAY_SECONDS,
            )
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("unreachable")  # satisfies type checkers; loop always raises or returns


def generate_cases(client: Groq) -> list[dict]:
    logger.info("Calling Groq (%s) to generate %d test cases...", GENERATOR_MODEL, EXPECTED_CASE_COUNT)
    raw = _call_with_retry(client)
    data = json.loads(raw)

    # Model may wrap the array in a key
    if isinstance(data, list):
        cases = data
    else:
        cases = next((value for value in data.values() if isinstance(value, list)), None)
        if cases is None:
            raise ValueError(f"Unexpected response shape: {list(data.keys())}")

    logger.info("Received %d cases from model.", len(cases))
    return cases


# ── Validate ───────────────────────────────────────────────────────────────────

def _validate_single_query(ground_truth: dict, prefix: str) -> list[str]:
    issues = []
    for field in ("endpoint", "make", "model", "year"):
        if field not in ground_truth:
            issues.append(f"{prefix}: ground_truth missing '{field}'")

    # Only check validity when the field is present to avoid duplicate error messages
    endpoint = ground_truth.get("endpoint")
    if endpoint is not None and endpoint not in VALID_ENDPOINTS:
        issues.append(f"{prefix}: invalid endpoint '{endpoint}'")

    make = ground_truth.get("make")
    if make is not None and make != make.upper():
        issues.append(f"{prefix}: make not uppercase")

    model = ground_truth.get("model")
    if model is not None and model != model.upper():
        issues.append(f"{prefix}: model not uppercase")

    year = ground_truth.get("year")
    if year is not None and not (isinstance(year, str) and len(year) == 4 and year.isdigit()):
        issues.append(f"{prefix}: year '{year}' is not a 4-digit string")

    return issues


def validate(cases: list[dict]) -> list[str]:
    issues = []
    ids_seen = set()

    for index, case in enumerate(cases):
        prefix = f"Case {case.get('id', f'[index {index}]')}"

        missing_keys = REQUIRED_KEYS - case.keys()
        if missing_keys:
            issues.append(f"{prefix}: missing keys {missing_keys}")
            continue

        case_id = case["id"]
        if case_id in ids_seen:
            issues.append(f"{prefix}: duplicate id")
        ids_seen.add(case_id)

        ground_truth = case["ground_truth"]

        if "error" in ground_truth:
            if ground_truth["error"] not in VALID_ERRORS:
                issues.append(f"{prefix}: unknown error type '{ground_truth['error']}'")
        elif "queries" in ground_truth:
            for sub_index, sub_query in enumerate(ground_truth["queries"]):
                issues.extend(_validate_single_query(sub_query, f"{prefix} sub-query {sub_index + 1}"))
        else:
            issues.extend(_validate_single_query(ground_truth, prefix))

    if len(cases) != EXPECTED_CASE_COUNT:
        issues.append(f"Expected {EXPECTED_CASE_COUNT} cases, got {len(cases)}")

    return issues


# ── Auto-fix ───────────────────────────────────────────────────────────────────

def _fix_single_query_fields(query: dict) -> None:
    if "make" in query:
        query["make"] = str(query["make"]).upper()
    if "model" in query:
        query["model"] = str(query["model"]).upper()
    if "year" in query:
        query["year"] = str(query["year"])


def _fix_ground_truth_fields(case: dict) -> None:
    ground_truth = case.get("ground_truth", {})
    if "error" in ground_truth:
        return
    if "queries" in ground_truth:
        for sub_query in ground_truth["queries"]:
            _fix_single_query_fields(sub_query)
    else:
        _fix_single_query_fields(ground_truth)


def _report_issues(issues: list[str], header: str) -> None:
    logger.warning("%s:", header)
    for issue in issues:
        logger.warning("  - %s", issue)


def validate_and_repair(cases: list[dict]) -> list[dict]:
    issues = validate(cases)
    if not issues:
        return cases

    _report_issues(issues, "Validation issues found")
    logger.info("Attempting to auto-fix...")

    for case in cases:
        _fix_ground_truth_fields(case)

    remaining_issues = validate(cases)
    if remaining_issues:
        _report_issues(remaining_issues, "Remaining issues after auto-fix")
        logger.error("Unfixable validation issues — aborting to prevent corrupt test_cases.py.")
        sys.exit(1)

    logger.info("All issues resolved by auto-fix.")
    return cases


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


def write_test_cases(cases: list[dict], out_path: Path) -> None:
    content = HEADER + json.dumps(cases, indent=4, ensure_ascii=False) + "\n"
    try:
        ast.parse(content)
    except SyntaxError as exc:
        raise RuntimeError(f"Generated content is not valid Python syntax: {exc}") from exc
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written to %s", out_path)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_category_breakdown(cases: list[dict]) -> None:
    logger.info("Category breakdown:")
    counts = Counter(case["category"] for case in cases)
    for category, count in sorted(counts.items()):
        logger.info("  %s: %d", category, count)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    client = Groq()
    out_path = Path(__file__).parent / "test_cases.py"
    cases = generate_cases(client)
    cases = validate_and_repair(cases)
    write_test_cases(cases, out_path)
    print_category_breakdown(cases)


if __name__ == "__main__":
    main()
