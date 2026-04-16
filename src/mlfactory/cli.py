"""MLFactory CLI entry point.

Phase 0: stub. Commands land in later phases.
"""

from __future__ import annotations

import typer
from rich.console import Console

from mlfactory import __version__

app = typer.Typer(
    name="mlfactory",
    help="MLFactory: a self-improving automated games strategy research factory.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed MLFactory version."""
    console.print(f"mlfactory {__version__}")


@app.command()
def doctor() -> None:
    """Check the environment: python, torch, MPS."""
    import sys

    console.print(f"[bold]Python[/bold]: {sys.version.split()[0]}")
    try:
        import torch

        console.print(f"[bold]Torch[/bold]: {torch.__version__}")
        console.print(f"[bold]MPS available[/bold]: {torch.backends.mps.is_available()}")
        console.print(f"[bold]MPS built[/bold]: {torch.backends.mps.is_built()}")
    except ImportError:
        console.print("[red]Torch not installed[/red]")


if __name__ == "__main__":
    app()
