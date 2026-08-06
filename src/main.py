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
import argparse
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


def print_candidates(result: dict) -> None:
    candidates = result.get("candidates", [])
    if not candidates:
        print("No candidates retrieved.")
        return
    print("\nRetrieved candidates:")
    for i, c in enumerate(candidates, start=1):
        print(
            f"{i}. {c['title']} — artist: {c['artist']}, genre: {c['genre']}, "
            f"mood: {c['mood']}, energy: {c['energy']:.2f}, score: {c['score']:.2f}"
        )


def run_once(query: str, songs: List[dict], k: int = 5) -> None:
    result = recommend(query, songs, k=k)
    print_result(result)
    return result


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

    parser = argparse.ArgumentParser(description="RAG Music Assistant CLI")
    parser.add_argument("query", nargs="*", help="Text query for recommendation")
    parser.add_argument("-k", "--k", type=int, default=5, help="Number of candidates to retrieve")
    parser.add_argument("--show-candidates", action="store_true", help="Print retrieved candidates and scores")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if query:
        result = run_once(query, songs, k=args.k)
        if args.show_candidates:
            print_candidates(result)
    else:
        run_interactive(songs)


if __name__ == "__main__":
    main()
