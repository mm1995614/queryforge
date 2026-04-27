from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# force_terminal ensures Rich doesn't fall back to plain ASCII on Windows
console = Console(force_terminal=True)


def show_query(query: dict) -> None:
    if "error" in query:
        console.print(Panel(
            f"[yellow]{query.get('message', query['error'])}[/yellow]",
            title="[red]Query Error[/red]",
            border_style="red",
        ))
        return
    console.print(Panel(
        f"Endpoint : [cyan]{query.get('endpoint')}[/cyan]\n"
        f"Make     : [cyan]{query.get('make')}[/cyan]\n"
        f"Model    : [cyan]{query.get('model')}[/cyan]\n"
        f"Year     : [cyan]{query.get('year')}[/cyan]",
        title="[green]Generated Query[/green]",
        border_style="green",
    ))


def show_results(endpoint: str, data: dict) -> None:
    if "error" in data:
        console.print(f"[red]Error: {data.get('message', data['error'])}[/red]")
        return

    results = data.get("results", [])
    if not results:
        console.print("[yellow]No results found for this vehicle.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(results)} result(s)[/bold]\n")

    if endpoint == "recalls":
        _show_recalls(results)
    elif endpoint == "complaints":
        _show_complaints(results)
    elif endpoint == "safetyRatings":
        _show_safety_ratings(results)


def show_synthesis(text: str) -> None:
    console.print(Panel(
        text,
        title="[green]Summary[/green]",
        border_style="green",
    ))


def _show_recalls(results: list) -> None:
    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=True)
    table.add_column("Campaign #", width=18)
    table.add_column("Component", width=22)
    table.add_column("Summary", width=55)
    table.add_column("Date", width=12)

    for r in results[:10]:
        summary = (r.get("Summary") or "")[:120]
        table.add_row(
            r.get("NHTSACampaignNumber", "N/A"),
            r.get("Component", "N/A"),
            summary,
            (r.get("ReportReceivedDate") or "")[:10],
        )

    console.print(table)
    if len(results) > 10:
        console.print(f"[dim]...and {len(results) - 10} more[/dim]")


def _show_complaints(results: list) -> None:
    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=True)
    table.add_column("ODI #", width=12)
    table.add_column("Component", width=22)
    table.add_column("Summary", width=55)
    table.add_column("Date", width=12)

    for r in results[:10]:
        summary = (r.get("summary") or "")[:120]
        table.add_row(
            str(r.get("odiNumber", "N/A")),
            r.get("components", "N/A"),
            summary,
            str(r.get("dateOfIncident", "N/A")),
        )

    console.print(table)
    if len(results) > 10:
        console.print(f"[dim]...and {len(results) - 10} more[/dim]")


def _show_safety_ratings(results: list) -> None:
    table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=True)
    table.add_column("Vehicle", width=35)
    table.add_column("Overall", width=10)
    table.add_column("Frontal", width=10)
    table.add_column("Side", width=10)
    table.add_column("Rollover", width=10)

    for r in results[:10]:
        table.add_row(
            r.get("VehicleDescription", "N/A"),
            str(r.get("OverallRating", "N/A")),
            str(r.get("OverallFrontCrashRating", "N/A")),
            str(r.get("OverallSideCrashRating", "N/A")),
            str(r.get("RolloverRating", "N/A")),
        )

    console.print(table)
