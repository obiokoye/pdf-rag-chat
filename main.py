"""Chat with your PDFs, with page citations.

Usage:
    python main.py serve                         # web app on http://localhost:8000
    python main.py ingest my.pdf folder/         # add PDFs to the index
    python main.py ask "How many days of annual leave do I get?"
    python main.py chat                          # terminal chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ragchat import Answer, RAGChat, UnreadablePDF
from ragchat.library import Library, open_library

console = Console(highlight=False)


def banner(lib: Library) -> None:
    if lib.llm.demo:
        console.print(
            Panel(
                f"[bold]Demo mode: scripted AI replies[/bold] [dim]({lib.llm.demo_reason}). "
                "Retrieval and citations are real; the answers are scripted for the sample PDFs.[/dim]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            f"[bold green]pdf-rag-chat[/bold green] using [cyan]{lib.llm.model}[/cyan] ({lib.llm.provider}), "
            f"embeddings: {lib.store.embedder.name}"
        )


def show_answer(answer: Answer) -> None:
    style = "green" if answer.grounded else "yellow"
    console.print(Panel(answer.text, title="answer", title_align="left", border_style=style))
    if answer.citations:
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("#", justify="right")
        table.add_column("Source", no_wrap=True)
        table.add_column("Page", justify="right")
        table.add_column("Match", justify="right")
        for c in answer.citations:
            table.add_row(f"[{c.n}]", c.doc, str(c.page), f"{c.score:.2f}")
        console.print(table)
    elif not answer.called_llm:
        console.print("  [dim]No chunk was relevant enough, so the model was not called.[/dim]")
    elif not answer.grounded:
        console.print("  [yellow]⚠ This answer cites no source. Treat it with care.[/yellow]")
    console.print()


def cmd_ingest(lib: Library, paths: list[str]) -> int:
    files = [p for raw in paths for p in (sorted(Path(raw).glob("*.pdf")) if Path(raw).is_dir() else [Path(raw)])]
    ok = True
    for path in files:
        try:
            pages, chunks = lib.ingest(path)
            console.print(f"  [green]✓[/green] {path.name}  [dim]{pages} pages → {chunks} chunks[/dim]")
        except (UnreadablePDF, FileNotFoundError) as exc:
            console.print(f"  [red]✗ {exc}[/red]")
            ok = False
    return 0 if ok else 1


def cmd_chat(lib: Library, question: str | None) -> int:
    lib.ingest_samples_if_empty()
    docs = ", ".join(f"{d.name} ({d.pages}p)" for d in lib.store.documents())
    console.print(f"[dim]Indexed: {docs}[/dim]\n")
    chat = RAGChat(lib.store, lib.llm.client, lib.llm.model)
    if question:
        show_answer(chat.ask(question))
        return 0
    console.print("[dim]Ask about your PDFs. 'exit' to quit.[/dim]")
    while True:
        try:
            text = console.input("[bold]you › [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if text.lower() in {"exit", "quit"}:
            return 0
        if text:
            show_answer(chat.ask(text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with your PDFs, with page citations.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    ingest = sub.add_parser("ingest", help="add PDFs (files or folders) to the index")
    ingest.add_argument("paths", nargs="+")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    sub.add_parser("chat", help="interactive terminal chat")
    args = parser.parse_args(argv)

    load_dotenv()
    if args.command == "serve":
        import uvicorn

        console.print(f"[bold]Open http://localhost:{args.port}[/bold]")
        uvicorn.run("app:create_app", factory=True, host=args.host, port=args.port)
        return 0

    lib = open_library()
    banner(lib)
    if args.command == "ingest":
        return cmd_ingest(lib, args.paths)
    return cmd_chat(lib, getattr(args, "question", None))


if __name__ == "__main__":
    sys.exit(main())
