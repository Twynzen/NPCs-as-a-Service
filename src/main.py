"""NPC Service CLI - Server and Simulator."""

import json
import sys
from pathlib import Path
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

from src.characters.loader import load_character, list_characters
from src.customers.generator import CustomerGenerator
from src.simulation.conversation import ConversationSimulator
from src.simulation.experience_extractor import ExperienceExtractor
from src.simulation.evaluator import ConversationEvaluator
from src.llm.ollama_client import OllamaClient
from src.config import settings

app = typer.Typer(
    name="npc-service",
    help="NPC Service - AI-powered NPCs with persistent memory",
    add_completion=False,
)
console = Console()


# === Server Commands ===

@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host", "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port to bind to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload", "-r",
        help="Enable auto-reload for development",
    ),
):
    """Start the NPC Service API server."""
    import uvicorn

    console.print(Panel.fit(
        "[bold blue]NPC Service[/bold blue]\n"
        f"Starting server on [green]http://{host}:{port}[/green]\n"
        f"Playground: [cyan]http://{host}:{port}/playground[/cyan]\n"
        f"API Docs: [cyan]http://{host}:{port}/docs[/cyan]",
        border_style="blue",
    ))

    uvicorn.run(
        "src.server.app:app",
        host=host,
        port=port,
        reload=reload,
    )


def get_output_dir() -> Path:
    """Get the output directory, creating if needed."""
    output_dir = Path(__file__).parent.parent / "output"
    (output_dir / "conversations").mkdir(parents=True, exist_ok=True)
    (output_dir / "experiences").mkdir(parents=True, exist_ok=True)
    return output_dir


@app.command()
def simulate(
    character: str = typer.Option(
        "zamir",
        "--character", "-c",
        help="Character name (filename without extension)",
    ),
    count: int = typer.Option(
        10,
        "--count", "-n",
        help="Number of conversations to generate",
    ),
    model: str = typer.Option(
        "phi3.5",
        "--model", "-m",
        help="Ollama model to use",
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434",
        "--ollama-url",
        help="Ollama API URL",
    ),
    max_turns: int = typer.Option(
        20,
        "--max-turns",
        help="Maximum turns per conversation",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature", "-t",
        help="LLM sampling temperature",
    ),
    output_file: str = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (default: output/conversations/CHARACTER_TIMESTAMP.json)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show conversation details during generation",
    ),
):
    """Generate synthetic conversations between an NPC and random customers."""

    # Header
    console.print(Panel.fit(
        "[bold blue]NPC Conversation Simulator[/bold blue]\n"
        f"Character: [green]{character}[/green] | Count: [yellow]{count}[/yellow] | Model: [cyan]{model}[/cyan]",
        border_style="blue",
    ))

    # Initialize LLM client
    console.print("\n[dim]Connecting to Ollama...[/dim]")
    llm = OllamaClient(base_url=ollama_url, model=model)

    if not llm.is_available():
        console.print("[red]Error: Ollama server not available at {ollama_url}[/red]")
        console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
        raise typer.Exit(1)

    if not llm.model_exists(model):
        available = llm.list_models()
        console.print(f"[red]Error: Model '{model}' not found[/red]")
        if available:
            console.print(f"[dim]Available models: {', '.join(available)}[/dim]")
        console.print(f"[dim]Try: ollama pull {model}[/dim]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Connected to Ollama, using model: [cyan]{model}[/cyan]")

    # Load character
    try:
        npc = load_character(character)
        console.print(f"[green]✓[/green] Loaded character: [bold]{npc.name}[/bold] - {npc.role}")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        available = list_characters()
        if available:
            console.print(f"[dim]Available characters: {', '.join(available)}[/dim]")
        raise typer.Exit(1)

    # Initialize components
    customer_gen = CustomerGenerator()
    simulator = ConversationSimulator(llm, max_turns=max_turns, temperature=temperature)
    extractor = ExperienceExtractor(llm)
    evaluator = ConversationEvaluator(llm)

    # Storage for results
    all_conversations = []
    all_experiences = []
    metrics_summary = {
        "total": count,
        "completed": 0,
        "avg_turns": 0,
        "avg_consistency": 0,
        "natural_endings": 0,
    }

    # Generate conversations
    console.print(f"\n[bold]Generating {count} conversations...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Simulating...", total=count)

        for i in range(count):
            # Generate customer
            customer = customer_gen.generate()

            progress.update(
                task,
                description=f"[{i+1}/{count}] {customer.archetype} ({customer.emotional_state})",
            )

            try:
                # Run simulation
                conversation = simulator.simulate(npc, customer)

                # Extract experiences
                extraction = extractor.extract(conversation, npc.name)

                # Evaluate
                metrics = evaluator.evaluate(conversation, npc)

                # Store results
                result = {
                    "id": conversation.id,
                    "character": conversation.character,
                    "customer_profile": {
                        "archetype": customer.archetype,
                        "faction": customer.faction,
                        "emotional_state": customer.emotional_state,
                        "objective": customer.objective,
                        "trust_level": customer.trust_level,
                    },
                    "turns": [
                        {"speaker": t.speaker, "message": t.message}
                        for t in conversation.turns
                    ],
                    "extracted_experiences": [
                        {
                            "description": exp.description,
                            "importance": exp.importance,
                            "topics": exp.topics,
                        }
                        for exp in extraction.experiences
                    ],
                    "metrics": {
                        "num_turns": metrics.num_turns,
                        "character_consistency": round(metrics.character_consistency, 2),
                        "conversation_naturalness": round(metrics.conversation_naturalness, 2),
                        "conversation_ended_naturally": metrics.ended_naturally,
                    },
                }

                all_conversations.append(result)
                all_experiences.extend(extraction.experiences)

                # Update summary
                metrics_summary["completed"] += 1
                metrics_summary["avg_turns"] += metrics.num_turns
                metrics_summary["avg_consistency"] += metrics.character_consistency
                if metrics.ended_naturally:
                    metrics_summary["natural_endings"] += 1

                # Verbose output
                if verbose:
                    console.print(f"\n[dim]--- Conversation {conversation.id} ---[/dim]")
                    for turn in conversation.turns[:4]:  # Show first 4 turns
                        speaker = "[cyan]Customer[/cyan]" if turn.speaker == "customer" else f"[green]{npc.name}[/green]"
                        console.print(f"{speaker}: {turn.message[:100]}...")
                    if len(conversation.turns) > 4:
                        console.print(f"[dim]... and {len(conversation.turns) - 4} more turns[/dim]")

            except Exception as e:
                console.print(f"[red]Error in conversation {i+1}: {e}[/red]")
                if verbose:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

            progress.advance(task)

    # Calculate final averages
    if metrics_summary["completed"] > 0:
        metrics_summary["avg_turns"] = round(
            metrics_summary["avg_turns"] / metrics_summary["completed"], 1
        )
        metrics_summary["avg_consistency"] = round(
            metrics_summary["avg_consistency"] / metrics_summary["completed"], 2
        )

    # Save results
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_file:
        conv_path = Path(output_file)
    else:
        conv_path = output_dir / "conversations" / f"{character}_{timestamp}.json"

    exp_path = output_dir / "experiences" / f"{character}_{timestamp}_experiences.json"

    # Write conversations
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "character": character,
                "model": model,
                "generated_at": datetime.now().isoformat(),
                "count": metrics_summary["completed"],
                "summary": metrics_summary,
            },
            "conversations": all_conversations,
        }, f, indent=2, ensure_ascii=False)

    # Write experiences
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "character": character,
                "total_experiences": len(all_experiences),
                "generated_at": datetime.now().isoformat(),
            },
            "experiences": [
                {
                    "description": exp.description,
                    "importance": exp.importance,
                    "topics": exp.topics,
                    "emotional_context": exp.emotional_context,
                    "customer_type": exp.customer_type,
                    "conversation_id": exp.conversation_id,
                }
                for exp in all_experiences
            ],
        }, f, indent=2, ensure_ascii=False)

    # Summary table
    console.print("\n")
    table = Table(title="Generation Summary", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Conversations Generated", str(metrics_summary["completed"]))
    table.add_row("Average Turns", str(metrics_summary["avg_turns"]))
    table.add_row("Average Consistency", f"{metrics_summary['avg_consistency']:.0%}")
    table.add_row("Natural Endings", f"{metrics_summary['natural_endings']}/{metrics_summary['completed']}")
    table.add_row("Total Experiences", str(len(all_experiences)))

    console.print(table)

    console.print(f"\n[green]✓[/green] Conversations saved to: [blue]{conv_path}[/blue]")
    console.print(f"[green]✓[/green] Experiences saved to: [blue]{exp_path}[/blue]")

    llm.close()


@app.command()
def list_npcs():
    """List available NPC characters."""
    characters = list_characters()

    if not characters:
        console.print("[yellow]No character templates found.[/yellow]")
        console.print("[dim]Add YAML or JSON files to src/characters/templates/[/dim]")
        return

    console.print("\n[bold]Available Characters:[/bold]\n")

    for name in characters:
        try:
            char = load_character(name)
            console.print(f"  [green]{name}[/green] - {char.name}: {char.role}")
        except Exception:
            console.print(f"  [yellow]{name}[/yellow] - (error loading)")

    console.print()


@app.command()
def check():
    """Check system requirements (Ollama connection, models, etc.)."""
    console.print("\n[bold]System Check[/bold]\n")

    # Check Ollama
    llm = OllamaClient()
    if llm.is_available():
        console.print("[green]✓[/green] Ollama server is running")

        models = llm.list_models()
        if models:
            console.print(f"[green]✓[/green] Available models: {', '.join(models)}")

            # Check for recommended models
            recommended = ["phi3.5", "llama3.2", "phi3"]
            has_recommended = any(
                any(r in m for r in recommended)
                for m in models
            )
            if has_recommended:
                console.print("[green]✓[/green] Recommended model available")
            else:
                console.print("[yellow]![/yellow] Consider pulling a recommended model:")
                console.print("[dim]  ollama pull phi3.5[/dim]")
        else:
            console.print("[yellow]![/yellow] No models found")
            console.print("[dim]  ollama pull phi3.5[/dim]")
    else:
        console.print("[red]✗[/red] Ollama server not available")
        console.print("[dim]  Start with: ollama serve[/dim]")

    llm.close()

    # Check characters
    characters = list_characters()
    if characters:
        console.print(f"[green]✓[/green] {len(characters)} character(s) available: {', '.join(characters)}")
    else:
        console.print("[yellow]![/yellow] No character templates found")

    # Check output directories
    output_dir = get_output_dir()
    console.print(f"[green]✓[/green] Output directory: {output_dir}")

    console.print()


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
