import typer
from rich.console import Console
from src.query_generator import generate_query
from src.nhtsa_client import execute_query
from src import display

app = typer.Typer(help="Query NHTSA vehicle safety data using natural language.")
console = Console()


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
      python cli.py "Ford F-150 2022 安全評等" --verbose
    """
    with console.status("[bold green]Generating query..."):
        structured = generate_query(nl_query)

    if verbose:
        display.show_query(structured)

    if "error" in structured:
        display.show_query(structured)
        raise typer.Exit(1)

    with console.status("[bold green]Fetching NHTSA data..."):
        result = execute_query(structured)

    display.show_results(structured["endpoint"], result)


if __name__ == "__main__":
    app()
