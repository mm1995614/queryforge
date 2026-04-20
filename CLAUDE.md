# QueryForge — CLAUDE.md

## Project Overview

QueryForge is a CLI tool that translates natural language queries into structured NHTSA (National Highway Traffic Safety Administration) API requests using LLMs. Users ask questions about vehicle recalls, safety complaints, and crash test ratings in plain English or Chinese — the tool handles translation and execution.

**Assessment context:** GoFreight AI Engineer take-home assessment (Part 1 + Part 2).
**Domain rationale:** Developer works as AI Data Engineer at a car dealership. NHTSA is a free, public, no-auth API with rich automotive data directly relevant to dealership after-sales workflows.

---

## Project Structure

```
queryforge/
├── cli.py                  # Typer CLI entry point
├── src/
│   ├── __init__.py
│   ├── query_generator.py  # NL → structured JSON via LLM
│   ├── nhtsa_client.py     # Execute structured JSON against NHTSA API
│   └── display.py          # Rich terminal output
├── eval/
│   ├── __init__.py
│   ├── test_cases.py       # 30 NL queries + ground truth JSON
│   ├── evaluator.py        # Multi-model eval pipeline
│   └── results/            # Eval output JSON files (gitignored except summary)
├── .env                    # API keys — never commit
├── .env.example
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Structured Query Schema

The LLM must output **only** this JSON format, with no extra text or markdown:

```json
{
  "endpoint": "recalls | complaints | safetyRatings",
  "params": {
    "make": "string (uppercase, e.g. TOYOTA)",
    "model": "string (uppercase, e.g. CAMRY)",
    "year": "string (4-digit, e.g. 2020)"
  }
}
```

Error schema (when input is invalid or out of scope):

```json
{
  "error": "missing_year | missing_make | out_of_scope | ambiguous_input",
  "message": "human-readable explanation"
}
```

---

## NHTSA API Endpoints

| Endpoint key   | URL pattern |
|----------------|-------------|
| recalls        | `GET https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}` |
| complaints     | `GET https://api.nhtsa.gov/complaints/complaintsByVehicle?make={make}&model={model}&modelYear={year}` |
| safetyRatings  | `GET https://api.nhtsa.gov/SafetyRatings/modelyear/{year}/make/{make}/model/{model}` |

- No authentication required
- All params should be URL-encoded
- make/model should be passed as uppercase to NHTSA for consistent results

---

## Models

| Model                      | Provider  | Type         | SDK     |
|----------------------------|-----------|--------------|---------|
| claude-sonnet-4-6          | Anthropic | closed-source | anthropic |
| gpt-4o-mini                | OpenAI    | closed-source | openai  |
| llama-3.3-70b-versatile    | Groq      | open-weight  | groq    |

---

## Environment Variables

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GROQ_API_KEY=
```

---

## Running the Project

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Fill in API keys in .env

# Run CLI
python cli.py "Show me recalls for Toyota Camry 2020"
python cli.py "Toyota Camry 2020 有哪些召回問題"

# Run eval pipeline
python eval/evaluator.py
```

---

## Key Design Decisions

1. **JSON-only LLM output** — System prompt enforces strict JSON with no surrounding text. This makes eval scoring deterministic and avoids brittle string parsing.

2. **Field-level partial scoring in eval** — Each correct field (endpoint, make, model, year) scores independently. Binary exact match would undercount near-correct outputs and give less signal for prompt iteration.

3. **Chinese + English input** — System prompt instructs the LLM to extract intent regardless of input language. No pre-translation step needed.

4. **Three endpoints only** — Scoped to recalls/complaints/safetyRatings to keep ground truth deterministic. Open-ended queries ("tell me everything about this car") are explicitly out of scope.

5. **Uppercase normalization** — NHTSA API is case-sensitive for make/model. The client layer normalizes all params to uppercase before calling the API, so the LLM doesn't have to get case exactly right.

---

## Hardening Approach

| Failure type          | Handling strategy |
|-----------------------|-------------------|
| Typo in make/model    | LLM corrects naturally; system prompt explicitly instructs correction |
| Missing year          | Returns `{"error": "missing_year"}` with explanation |
| Missing make          | Returns `{"error": "missing_make"}` with explanation |
| Ambiguous model       | LLM picks best match and includes `"note"` field flagging uncertainty |
| Future/invalid year   | Passes through to API; API returns empty results |
| Non-automotive query  | Returns `{"error": "out_of_scope"}` |
| Other languages       | Handled natively by LLM |
| Conflicting constraints | LLM resolves by prioritizing most specific constraint |

---

## Eval Scoring

Each test case scored on 4 dimensions:

| Field    | Weight |
|----------|--------|
| endpoint | 1 pt   |
| make     | 1 pt   |
| model    | 1 pt   |
| year     | 1 pt   |

- **Field score** = correct_fields / expected_fields
- **Case accuracy** = 1.0 only when all fields match exactly
- **Model accuracy** = cases_with_score_1.0 / 30

Target: all three models > 85% (≥ 26/30 cases fully correct).

---

## Common Pitfalls

- NHTSA make/model values are strict — always uppercase in API calls
- Safety ratings endpoint has different URL structure (path params, not query params)
- Groq free tier has rate limits — add `time.sleep(1)` between eval calls
- Some NHTSA endpoints return 200 with empty results array, not 404, when vehicle not found
- `.env` must never be committed — double-check `.gitignore` before push
