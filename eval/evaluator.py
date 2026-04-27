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
from src.query_generator import SYSTEM_PROMPT

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
]

# ── API clients ────────────────────────────────────────────────────────────────

_anthropic = anthropic.Anthropic()
_openai = openai.OpenAI()
_groq = Groq()


# ── Inference ──────────────────────────────────────────────────────────────────

def call_model(model: dict, nl_query: str) -> dict:
    provider = model["provider"]
    model_id = model["id"]

    try:
        if provider == "anthropic":
            resp = _anthropic.messages.create(
                model=model_id,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": nl_query}],
            )
            text = next(b.text for b in resp.content if b.type == "text")
            return json.loads(text)

        elif provider == "openai":
            resp = _openai.chat.completions.create(
                model=model_id,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": nl_query},
                ],
            )
            return json.loads(resp.choices[0].message.content)

        elif provider == "groq":
            resp = _groq.chat.completions.create(
                model=model_id,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
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

def run_eval(models=None) -> dict:
    active = models if models is not None else MODELS
    results = {m["id"]: {"model": m, "cases": [], "correct": 0, "total": 30} for m in active}

    total = len(TEST_CASES) * len(active)
    done = 0

    console.print(f"\n[bold]Running evaluation: {len(TEST_CASES)} cases × {len(active)} models[/bold]\n")

    for case in TEST_CASES:
        for model in active:
            console.print(
                f"[dim][{done + 1}/{total}] {model['display']} — Case {case['id']}: {case['nl_query'][:50]}[/dim]"
            )

            output = call_model(model, case["nl_query"])
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
            # Groq free tier rate limit: stay under 30 req/min
            if model["provider"] == "groq":
                time.sleep(2)
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

    # Per-category breakdown
    console.print("\n[bold]Accuracy by category:[/bold]\n")
    categories = sorted({c["category"] for c in TEST_CASES})
    cat_table = Table(box=box.SIMPLE, header_style="bold")
    cat_table.add_column("Category", width=25)
    for m in MODELS:
        cat_table.add_column(m["display"][:18], width=20)

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


def save_results(results: dict) -> None:
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": [],
    }
    for model_id, data in results.items():
        m = data["model"]
        correct = data["correct"]
        total = data["total"]
        summary["models"].append({
            "id": model_id,
            "display": m["display"],
            "type": m["type"],
            "accuracy": round(correct / total, 4),
            "correct": correct,
            "total": total,
            "avg_field_score": round(
                sum(c["score"] for c in data["cases"]) / total, 4
            ),
        })

    out_path = Path(__file__).parent / "results" / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(f"\n[green]Results saved to {out_path}[/green]")

    # Also save full per-case results
    full_path = Path(__file__).parent / "results" / "full_results.json"
    full_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", nargs="+", help="Run only these providers (e.g. groq anthropic openai)")
    args = parser.parse_args()

    active_models = [m for m in MODELS if m["provider"] in args.provider] if args.provider else MODELS
    results = run_eval(active_models)
    print_summary(results)
    save_results(results)
