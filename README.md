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
| Query generation | Claude Sonnet 4.6 (Anthropic) |
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

## Part 1: Failure Case Analysis

### Documented Failure Cases

| Category | Example Input | Observed Behavior | Severity |
|----------|--------------|-------------------|----------|
| Typo in make | `"Toyata Camry 2020 recalls"` | LLM sometimes corrects, occasionally hallucinates wrong make | Low |
| Missing year | `"Toyota Camry recalls"` | Returns `error: missing_year` — handled | Low |
| Missing make | `"2020 Camry recalls"` | LLM infers Toyota in most cases; sometimes ambiguous | Medium |
| Ambiguous model name | `"Civic complaints 2019"` | Correct — Honda is strongly implied; may fail for less-known brands | Low |
| Future model year | `"Toyota Camry 2027 recalls"` | Query executes, NHTSA returns empty results — no crash | Low |
| Non-automotive query | `"What is the weather in Tokyo?"` | Returns `error: out_of_scope` — handled | Low |
| Very old vehicles (pre-1980) | `"1965 Ford Mustang safety rating"` | API returns no data; gracefully returns empty | Low |
| Conflicting constraints | `"Best and worst safety rated car in 2022"` | Out of scope — NHTSA API requires specific make/model | High |
| Vague component filter | `"Toyota Camry 2020 engine problems"` | Passes as complaints query; component filtering not supported by API | Medium |

### Fixed Cases

- **Typo correction** — System prompt explicitly instructs the LLM to normalize common make/model typos before generating the query.
- **Missing year** — Explicit `missing_year` error returned with a human-readable message; no silent failure.
- **Non-English input** — LLM handles natively; no pre-translation step needed.
- **Casing issues** — Client layer normalizes all make/model params to uppercase before calling NHTSA, decoupling LLM output from API casing requirements.
- **Out-of-scope queries** — Explicit `out_of_scope` error with explanation rather than a malformed API call.

### Remaining Hard Cases

**1. Ambiguous make inference from model name alone**

> Input: `"What are the issues with the Accord?"` (no make specified)

The LLM correctly infers Honda in this case because "Accord" is strongly associated with one brand. However, for models shared across brands (e.g., "Ranger" — Ford and Mitsubishi), inference is unreliable. This is fundamentally hard because it requires world knowledge about which brand is most likely given regional context, and that distribution is not static.

**2. Comparative or aggregate queries**

> Input: `"Which SUVs have the most recalls in 2022?"`

NHTSA's API is lookup-based, not query-based. It requires a specific make and model. Answering comparative questions would require iterating over all makes and models — potentially thousands of API calls. This is an architectural limitation of the data source, not the LLM.

**3. Component-level filtering**

> Input: `"Toyota Camry 2020 brake recalls only"`

NHTSA returns all recalls for a vehicle; there is no server-side component filter. Client-side filtering is possible but brittle — the LLM would need to classify recall descriptions, which introduces another layer of potential error.

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
