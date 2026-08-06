"""
Generation layer for the RAG music assistant (Task 2).

This module is the "G" in RAG. It takes the grounded candidate songs produced
by `src/retrieval.py` and asks a Groq-hosted LLM (llama-3.1-8b-instant, free
tier) to write a short, friendly recommendation.

Grounding is enforced by the prompt: the model is told it may ONLY recommend
songs from the provided candidate list and must refer to them by their exact
title. The catalog-validation guardrail in Task 3 double-checks this.

Token use is kept low: retrieval is local (0 tokens); only this short final
response costs tokens, and `max_tokens` is capped.

Requires the environment variable GROQ_API_KEY. Get a free key at
https://console.groq.com/keys
"""
import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 400

SYSTEM_PROMPT = (
    "You are a helpful music recommendation assistant for a small song catalog. "
    "You will be given a user's request and a list of CANDIDATE songs that were "
    "already retrieved from the catalog. Follow these rules strictly:\n"
    "1. Recommend ONLY songs from the candidate list. Never invent or mention any "
    "song, artist, or album that is not in the list.\n"
    "2. Refer to each song by its EXACT title as written in the candidates.\n"
    "3. Recommend the 2-3 best fits for the request and briefly say why, using the "
    "song's real features (genre, mood, energy, acousticness).\n"
    "4. If none of the candidates fit well, say so honestly.\n"
    "Keep the whole reply under 120 words and friendly."
)


class GenerationError(RuntimeError):
    """Raised when the LLM call cannot be completed (missing key, API error)."""


def _format_candidates(candidates: List[Dict]) -> str:
    """Render candidates as a compact, token-light context block."""
    lines = []
    for c in candidates:
        lines.append(
            f"- \"{c['title']}\" by {c['artist']} "
            f"[genre={c['genre']}, mood={c['mood']}, "
            f"energy={c['energy']:.2f}, acousticness={c['acousticness']:.2f}, "
            f"match_score={c['score']:.2f}]"
        )
    return "\n".join(lines)


def build_prompt(query: str, candidates: List[Dict]) -> str:
    """Build the grounded user prompt from the request and retrieved candidates."""
    return (
        f"User request: {query}\n\n"
        f"CANDIDATE songs (the only songs you may recommend):\n"
        f"{_format_candidates(candidates)}\n\n"
        f"Write the recommendation now."
    )


def generate_recommendation(
    retrieval_result: Dict,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> str:
    """Generate a grounded, natural-language recommendation from retrieved candidates.

    Args:
        retrieval_result: the dict returned by `src.retrieval.retrieve`.
        model: Groq model id (default: llama-3.1-8b-instant).
        api_key: optional override; otherwise read from GROQ_API_KEY.

    Returns:
        The model's recommendation text.

    Raises:
        GenerationError: if the key is missing, the SDK is not installed, or the
            API call fails. The caller (Task 3) catches this and falls back to
            the rule-based result.
    """
    candidates = retrieval_result.get("candidates", [])
    query = retrieval_result.get("query", "")
    if not candidates:
        raise GenerationError("no candidates to generate from")

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise GenerationError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and set it in your environment."
        )

    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - depends on install
        raise GenerationError(
            "The 'groq' package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    prompt = build_prompt(query, candidates)
    logger.info("Calling Groq model %s (%d candidates)", model, len(candidates))

    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.4,
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK/network error uniformly
        logger.error("Groq API call failed: %s", exc)
        raise GenerationError(f"Groq API call failed: {exc}") from exc

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise GenerationError("Groq returned an empty response")

    logger.info("Groq returned %d characters", len(text))
    return text


if __name__ == "__main__":
    # Manual end-to-end check (needs GROQ_API_KEY). Retrieval alone is free;
    # this script makes one small paid-tier-free API call.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        from src.recommender import load_songs
        from src.retrieval import retrieve
    except Exception:
        from recommender import load_songs
        from retrieval import retrieve

    catalog = load_songs("data/songs.csv")
    result = retrieve("high energy pop for the gym", catalog, k=5)
    try:
        print("\n" + generate_recommendation(result))
    except GenerationError as err:
        print(f"\n[generation unavailable] {err}")
