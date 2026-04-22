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
