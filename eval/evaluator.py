"""
Multi-model evaluation pipeline for QueryForge.

Runs all 30 test cases against 3 models and scores field-level accuracy.
Results are saved to eval/results/summary.json and printed to the terminal.

Usage:
    python eval/evaluator.py
"""

import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import openai
from groq import Groq
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.test_cases import TEST_CASES
from src.query_generator import (
    SYSTEM_PROMPT, MINIMAL_PROMPT, PROMPT_VERSION, MINIMAL_PROMPT_VERSION,
)

# Prompt variants for the before/after (no-harness vs with-harness) comparison.
# Each maps to a versioned prompt file under prompts/ (see prompts/CHANGELOG.md).
PROMPTS = {"full": SYSTEM_PROMPT, "minimal": MINIMAL_PROMPT}
PROMPT_VERSIONS = {"full": PROMPT_VERSION, "minimal": MINIMAL_PROMPT_VERSION}

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
console = Console(force_terminal=True)

MODELS = [
    {
        "id": "claude-sonnet-4-6",
        "provider": "anthropic",
        "display": "Claude Sonnet 4.6",
        "type": "closed-source",
    },
    {
        "id": "gpt-4o-mini",
        "provider": "openai",
        "display": "GPT-4o-mini",
        "type": "closed-source",
    },
    {
        "id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "display": "Llama 3.3 70B (Groq)",
        "type": "open-weight",
    },
    {
        "id": "qwen2.5:7b-instruct",
        "provider": "ollama",
        "display": "Qwen2.5 7B (Local)",
        "type": "local",
    },
    {
        "id": "llama3.1:8b",
        "provider": "ollama",
        "display": "Llama 3.1 8B (Local)",
        "type": "local",
    },
    {
        "id": "gemma2:9b",
        "provider": "ollama",
        "display": "Gemma 2 9B (Local)",
        "type": "local",
    },
]

# ── API clients ────────────────────────────────────────────────────────────────
# Lazily constructed so a local-only run (--provider ollama) doesn't require the
# cloud API keys to be present in .env. Each client is built on first use.

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

_clients: dict = {}


def _client(provider: str):
    if provider not in _clients:
        if provider == "anthropic":
            _clients[provider] = anthropic.Anthropic()
        elif provider == "openai":
            _clients[provider] = openai.OpenAI()
        elif provider == "groq":
            _clients[provider] = Groq()
        elif provider == "ollama":
            # Ollama exposes an OpenAI-compatible endpoint; no real key needed.
            _clients[provider] = openai.OpenAI(
                base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _clients[provider]


# ── Inference ──────────────────────────────────────────────────────────────────

# Greedy decoding (temperature=0) so the eval is deterministic and reproducible —
# scores don't drift run-to-run. Important for results used in decision-making.
TEMPERATURE = 0


def call_model(model: dict, nl_query: str, system_prompt: str = SYSTEM_PROMPT) -> dict:
    provider = model["provider"]
    model_id = model["id"]

    try:
        if provider == "anthropic":
            resp = _client(provider).messages.create(
                model=model_id,
                max_tokens=256,
                temperature=TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": nl_query}],
            )
            text = next(b.text for b in resp.content if b.type == "text")
            return json.loads(text)

        elif provider in ("openai", "groq", "ollama"):
            # All three speak the OpenAI chat-completions API.
            # For ollama, response_format json_object maps to Ollama's
            # `format=json`, which forces syntactically valid JSON output.
            resp = _client(provider).chat.completions.create(
                model=model_id,
                max_tokens=256,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": nl_query},
                ],
            )
            return json.loads(resp.choices[0].message.content)

    except json.JSONDecodeError:
        return {"error": "json_parse_error", "message": "Model output was not valid JSON"}
    except Exception as e:
        return {"error": "call_failed", "message": str(e)}


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_single_query(pred: dict, gt: dict) -> dict:
    fields = ["endpoint", "make", "model", "year"]
    results = {}
    for f in fields:
        gt_val = gt.get(f, "").upper()
        pred_val = str(pred.get(f, "")).upper()
        results[f] = gt_val == pred_val
    field_score = sum(results.values()) / len(fields)
    return {"score": field_score, "fields": results, "correct": all(results.values())}


def score(prediction: dict, ground_truth: dict) -> dict:
    """
    Field-level scoring.

    For error cases: correct if prediction also has 'error' with the same type.
    For complex_constraints cases: ground_truth has 'queries' key; prediction must
      also have 'queries' with matching sub-queries (order-based).
    For valid cases: 1 point per correct field (endpoint, make, model, year).
    Returns {"score": float 0–1, "fields": {...}, "correct": bool}
    """
    if "error" in ground_truth:
        correct = (
            "error" in prediction
            and prediction.get("error") == ground_truth.get("error")
        )
        return {"score": 1.0 if correct else 0.0, "fields": {"error": correct}, "correct": correct}

    if "queries" in ground_truth:
        gt_queries = ground_truth["queries"]
        pred_queries = prediction.get("queries", [])
        if not isinstance(pred_queries, list) or len(pred_queries) == 0:
            return {"score": 0.0, "fields": {"queries": False}, "correct": False}
        sub_scores = []
        for i, gt_q in enumerate(gt_queries):
            pred_q = pred_queries[i] if i < len(pred_queries) else {}
            sub_scores.append(score_single_query(pred_q, gt_q))
        avg_score = sum(s["score"] for s in sub_scores) / len(sub_scores)
        fully_correct = all(s["correct"] for s in sub_scores)
        return {"score": avg_score, "fields": {"queries": fully_correct}, "correct": fully_correct}

    return score_single_query(prediction, ground_truth)


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_eval(models=None, system_prompt: str = SYSTEM_PROMPT, prompt_variant: str = "full") -> dict:
    active = models if models is not None else MODELS
    results = {m["id"]: {"model": m, "cases": [], "correct": 0, "total": 30} for m in active}

    total = len(TEST_CASES) * len(active)
    done = 0

    console.print(
        f"\n[bold]Running evaluation: {len(TEST_CASES)} cases × {len(active)} models "
        f"(prompt: {prompt_variant})[/bold]\n"
    )

    # Model-outer loop: each model runs all 30 cases before moving on. This keeps
    # a local (ollama) model resident in VRAM for its whole pass instead of being
    # swapped in/out on every case — critical on a small GPU that can hold only
    # one ~5GB model at a time.
    for model in active:
        for case in TEST_CASES:
            console.print(
                f"[dim][{done + 1}/{total}] {model['display']} — Case {case['id']}: {case['nl_query'][:50]}[/dim]"
            )

            output = call_model(model, case["nl_query"], system_prompt)
            result = score(output, case["ground_truth"])

            if result["correct"]:
                results[model["id"]]["correct"] += 1

            results[model["id"]]["cases"].append({
                "id": case["id"],
                "category": case["category"],
                "nl_query": case["nl_query"],
                "ground_truth": case["ground_truth"],
                "prediction": output,
                **result,
            })

            done += 1
            # Groq free tier rate limit: stay under 30 req/min.
            # Local (ollama) models have no rate limit — don't sleep.
            if model["provider"] == "groq":
                time.sleep(2)
            elif model["provider"] == "ollama":
                pass
            else:
                time.sleep(0.3)

    return results


# ── Output ─────────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    table = Table(title="Evaluation Results", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Model", width=28)
    table.add_column("Type", width=14)
    table.add_column("Accuracy", width=10)
    table.add_column("Score", width=8)
    table.add_column(">85%?", width=7)

    for model_id, data in results.items():
        m = data["model"]
        correct = data["correct"]
        total = data["total"]
        accuracy = correct / total
        avg_score = sum(c["score"] for c in data["cases"]) / total
        passes = "[green]✓[/green]" if accuracy >= 0.85 else "[red]✗[/red]"
        table.add_row(
            m["display"],
            m["type"],
            f"{correct}/{total} ({accuracy:.0%})",
            f"{avg_score:.2f}",
            passes,
        )

    console.print("\n")
    console.print(table)

    # Per-category breakdown — use the models actually present in results so
    # columns and rows stay aligned when running a subset (e.g. --provider ollama).
    console.print("\n[bold]Accuracy by category:[/bold]\n")
    categories = sorted({c["category"] for c in TEST_CASES})
    cat_table = Table(box=box.SIMPLE, header_style="bold")
    cat_table.add_column("Category", width=25)
    for data in results.values():
        cat_table.add_column(data["model"]["display"][:18], width=20)

    for cat in categories:
        cat_cases = [c for c in TEST_CASES if c["category"] == cat]
        row = [cat]
        for model_id, data in results.items():
            correct = sum(
                1 for c in data["cases"]
                if c["category"] == cat and c["correct"]
            )
            row.append(f"{correct}/{len(cat_cases)}")
        cat_table.add_row(*row)

    console.print(cat_table)


def _read_json_safe(path: Path):
    """Read a JSON file, tolerating legacy non-UTF-8 files.

    Older result files were written without an explicit encoding and ended up in
    the Windows default (cp950) when they contained Chinese text. Fall back to
    cp950, then to a lossy read, so merging never crashes on a stale file.
    """
    for enc in ("utf-8", "cp950"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
        except (json.JSONDecodeError, OSError):
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None


def save_results(results: dict, prompt_variant: str = "full") -> None:
    # "full" (with-harness) is the canonical results file that holds all models
    # (cloud + local). "minimal" (no-harness baseline) is kept in a separate file
    # so the before/after comparison doesn't overwrite itself.
    suffix = "" if prompt_variant == "full" else f"_{prompt_variant}"
    results_dir = Path(__file__).parent / "results"
    out_path = results_dir / f"summary{suffix}.json"

    # Merge with any existing summary so a partial run (e.g. --provider ollama)
    # appends to / updates prior results rather than discarding the other models.
    existing_models: list = []
    if out_path.exists():
        prior = _read_json_safe(out_path)
        if prior:
            existing_models = prior.get("models", [])

    by_id = {m["id"]: m for m in existing_models}
    # Preserve the original ordering from MODELS so the table reads cloud → local.
    order = [m["id"] for m in MODELS]

    for model_id, data in results.items():
        m = data["model"]
        correct = data["correct"]
        total = data["total"]
        by_id[model_id] = {
            "id": model_id,
            "display": m["display"],
            "type": m["type"],
            "accuracy": round(correct / total, 4),
            "correct": correct,
            "total": total,
            "avg_field_score": round(
                sum(c["score"] for c in data["cases"]) / total, 4
            ),
        }

    merged = [by_id[i] for i in order if i in by_id]
    merged += [m for mid, m in by_id.items() if mid not in order]  # any unknown ids last

    # Record which prompt produced these results (matches the prompts/ versioning).
    version = PROMPT_VERSIONS.get(prompt_variant, prompt_variant)
    summary = {
        "prompt_version": version,
        "prompt_file": f"prompts/{version}.txt",
        "timestamp": datetime.now().isoformat(),
        "models": merged,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Results saved to {out_path}[/green]")

    # Full per-case results: merge by model id too (keep prior models' detail).
    full_path = results_dir / f"full_results{suffix}.json"
    full_existing: dict = {}
    if full_path.exists():
        prior_full = _read_json_safe(full_path)
        if isinstance(prior_full, dict):
            full_existing = prior_full
    full_existing.update(results)
    full_path.write_text(
        json.dumps(full_existing, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", nargs="+", help="Run only these providers (e.g. groq anthropic openai ollama)")
    parser.add_argument(
        "--prompt", choices=["full", "minimal"], default="full",
        help="Which system prompt to use: 'full' (with-harness, engineered) or "
             "'minimal' (no-harness baseline). Results save to separate files.",
    )
    args = parser.parse_args()

    active_models = [m for m in MODELS if m["provider"] in args.provider] if args.provider else MODELS
    system_prompt = PROMPTS[args.prompt]
    results = run_eval(active_models, system_prompt=system_prompt, prompt_variant=args.prompt)
    print_summary(results)
    save_results(results, prompt_variant=args.prompt)
