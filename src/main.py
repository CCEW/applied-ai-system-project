"""
Command-line entry point for the RAG Music Assistant.

The app now behaves as a natural-language music assistant: you describe what
you want in plain English, it RETRIEVES matching songs from the catalog
(local, 0 tokens), a Groq LLM writes a grounded recommendation, and a guardrail
verifies the answer before showing it (falling back to a rule-based answer if
anything goes wrong).

Usage:
    python -m src.main                      # interactive prompt
    python -m src.main "chill study music"  # one-shot query
"""
import logging
import sys
from typing import List

from src.assistant import recommend
from src.recommender import load_songs

SONGS_PATH = "data/songs.csv"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Your request: {result['query']}")
    print(f"Detected preferences: {result['prefs']}")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    conf = result["confidence"]
    print(f"[source: {result['source']}]  [confidence: {conf['score']:.2f}]")
    print(f"[why: {conf['reason']}]")
    if result["source"] == "llm" and result["validation"]:
        print(f"[grounded in: {', '.join(result['validation']['mentioned'])}]")
    elif result["fallback_reason"]:
        print(f"[fallback: {result['fallback_reason']}]")
    print("=" * 60)


def run_once(query: str, songs: List[dict]) -> None:
    result = recommend(query, songs, k=5)
    print_result(result)


def run_interactive(songs: List[dict]) -> None:
    print("\n🎵 Music Assistant — describe what you'd like to hear.")
    print("   Examples: 'high energy pop for the gym', 'rainy day study music'")
    print("   Type 'quit' or press Ctrl+C to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 🎶")
            return
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye! 🎶")
            return
        run_once(query, songs)
        print()


def main() -> None:
    configure_logging()
    songs = load_songs(SONGS_PATH)

    query = " ".join(sys.argv[1:]).strip()
    if query:
        run_once(query, songs)
    else:
        run_interactive(songs)


if __name__ == "__main__":
    main()
