import sys
import typer
from rich.console import Console
from src.query_generator import generate_query, synthesize_results
from src.nhtsa_client import execute_query
from src import display

# Fix Windows cp950 encoding — force UTF-8 for stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

app = typer.Typer(help="Query NHTSA vehicle safety data using natural language.")
console = Console()

MISSING_PROMPTS = {
    "missing_year": "Which model year is your vehicle? (e.g. 2021 = a 2021 Toyota Camry) / 請問車輛出廠年份？(例如 2021 代表 2021 年出廠的車): ",
    "missing_make": "Which make and model? (e.g. Toyota Camry) / 請問車廠和車型？: ",
    "missing_endpoint": "Query type — recalls / complaints / safetyRatings? / 查詢類型？: ",
}

MISSING_MESSAGES = {
    "missing_year": "Year is required. Please specify the model year.",
    "missing_make": "Make/model is required. Please specify the vehicle.",
    "missing_endpoint": "Query type is required. Please specify recalls, complaints, or safetyRatings.",
}


def _check_missing_fields(structured: dict) -> str | None:
    """Catch cases where LLM returns a query with empty/null required fields instead of an error JSON."""
    if "error" in structured or "queries" in structured:
        return None
    if not structured.get("year") or str(structured.get("year")).lower() in ("null", "none", ""):
        return "missing_year"
    if not structured.get("make") or str(structured.get("make")).lower() in ("null", "none", ""):
        return "missing_make"
    if not structured.get("endpoint") or str(structured.get("endpoint")).lower() in ("null", "none", ""):
        return "missing_endpoint"
    return None


@app.command()
def query(
    nl_query: str = typer.Argument(..., help="Natural language query about vehicle safety"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show generated structured query"),
):
    """
    Ask a question about vehicle safety in plain English or Chinese.

    Examples:\n
      python cli.py "Show me recalls for Toyota Camry 2020"\n
      python cli.py "Toyota Camry 2020 有哪些召回問題"\n
      python cli.py "Show me both recalls and safety ratings for Ford F-150 2022"\n
      python cli.py "Toyota Camry 2019 and 2020 recalls"
    """
    current_query = nl_query

    # Interactive loop: keep asking until all required params are present (max 3 rounds)
    for attempt in range(3):
        with console.status("[bold green]Generating query..."):
            structured = generate_query(current_query)

        if verbose:
            display.show_query(structured)

        # LLM may return empty fields instead of a proper error JSON — catch both cases
        error = structured.get("error") or _check_missing_fields(structured)
        if error and error not in MISSING_PROMPTS:
            structured["error"] = error
            structured.setdefault("message", MISSING_MESSAGES.get(error, error))

        if error in MISSING_PROMPTS:
            # Principle: never guess — ask the user for the missing value
            console.print(f"\n[yellow]{structured.get('message', '')}[/yellow]")
            supplement = typer.prompt(MISSING_PROMPTS[error])
            current_query = f"{current_query} {supplement}"
            continue

        if error:
            display.show_query(structured)
            raise typer.Exit(1)

        break
    else:
        console.print("[red]Could not complete the query after multiple attempts.[/red]")
        raise typer.Exit(1)

    # Multi-query fan-out: user specified multiple years / makes / endpoints
    if "queries" in structured:
        queries = structured["queries"]
        console.print(f"\n[cyan]Detected {len(queries)} sub-queries — fetching all to give you a complete answer.[/cyan]\n")

        all_results = []
        for q in queries:
            label = f"{q.get('endpoint')} · {q.get('make')} {q.get('model')} {q.get('year')}"
            if verbose:
                display.show_query(q)
            with console.status(f"[bold green]Fetching {label}..."):
                result = execute_query(q)
            all_results.append({"query": q, "result": result})

        with console.status("[bold green]Synthesizing results..."):
            synthesis = synthesize_results(nl_query, all_results)

        display.show_synthesis(synthesis)
        return

    # Single query
    with console.status("[bold green]Fetching NHTSA data..."):
        result = execute_query(structured)

    display.show_results(structured["endpoint"], result)


if __name__ == "__main__":
    app()
