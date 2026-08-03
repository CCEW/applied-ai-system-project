"""
Retrieval layer for the RAG music assistant (Task 1).

This module is the "R" in RAG. It turns a free-text request like
"rainy day study music that isn't too sleepy" into structured taste
preferences, then retrieves the most relevant songs *from the catalog*
using the existing rule-based scorer (`score_song`).

It is fully local and consumes ZERO tokens. The candidates it returns are
the grounding context that the generation layer (Task 2) must build its
answer from -- the LLM is never allowed to invent songs outside this set.
"""
import logging
from typing import Dict, List, Optional

from src.recommender import recommend_songs

logger = logging.getLogger(__name__)

# --- Query parsing vocabulary -------------------------------------------------
# Energy cues map words in the request to a target energy value in [0, 1].
# The scorer rewards *closeness* to this target, so we pick representative points.
HIGH_ENERGY_WORDS = {
    "energetic", "intense", "hype", "hyped", "workout", "gym", "pump",
    "pumped", "party", "dance", "dancing", "upbeat", "fast", "high energy",
    "powerful", "loud", "banger", "run", "running", "cardio",
}
LOW_ENERGY_WORDS = {
    "chill", "calm", "relaxed", "relaxing", "sleepy", "sleep", "study",
    "studying", "focus", "focused", "mellow", "quiet", "soft", "slow",
    "rainy", "rain", "cozy", "gentle", "background", "ambient", "lofi",
}
ACOUSTIC_WORDS = {
    "acoustic", "unplugged", "organic", "guitar", "piano", "folk",
    "stripped", "singer-songwriter",
}

HIGH_ENERGY_TARGET = 0.9
LOW_ENERGY_TARGET = 0.25


def _catalog_vocabulary(songs: List[Dict], field: str) -> List[str]:
    """Collect the distinct values of a field (e.g. genre, mood) from the catalog.

    Parsing against the real catalog keeps the assistant data-driven: if new
    genres are added to songs.csv, they become recognizable automatically.
    """
    seen = []
    for song in songs:
        value = str(song.get(field, "")).strip().lower()
        if value and value not in seen:
            seen.append(value)
    return seen


def parse_query(query: str, songs: List[Dict]) -> Dict:
    """Parse a free-text request into a `user_prefs` dict for `score_song`.

    Returns keys: genre, mood, energy, likes_acoustic. Any signal that is not
    present in the request is left as None (energy) or omitted, so the scorer
    simply skips it rather than guessing.
    """
    text = f" {query.lower().strip()} "

    genres = _catalog_vocabulary(songs, "genre")
    moods = _catalog_vocabulary(songs, "mood")

    matched_genre: Optional[str] = next(
        (g for g in genres if f" {g} " in text or g in query.lower()), None
    )
    matched_mood: Optional[str] = next(
        (m for m in moods if f" {m} " in text or m in query.lower()), None
    )

    def _mentions(words: set) -> bool:
        return any(w in text for w in words)

    energy: Optional[float] = None
    if _mentions(HIGH_ENERGY_WORDS) and not _mentions(LOW_ENERGY_WORDS):
        energy = HIGH_ENERGY_TARGET
    elif _mentions(LOW_ENERGY_WORDS) and not _mentions(HIGH_ENERGY_WORDS):
        energy = LOW_ENERGY_TARGET
    elif _mentions(HIGH_ENERGY_WORDS) and _mentions(LOW_ENERGY_WORDS):
        # Conflicting cues (e.g. "sad but energetic") -> stay neutral, let genre/mood decide.
        energy = None

    likes_acoustic = _mentions(ACOUSTIC_WORDS)

    prefs: Dict = {
        "genre": matched_genre,
        "mood": matched_mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
    }
    logger.info("Parsed query %r -> %s", query, prefs)
    return prefs


def retrieve(query: str, songs: List[Dict], k: int = 5) -> Dict:
    """Retrieve the top-k grounded candidate songs for a free-text request.

    Returns a dict with the parsed preferences and a list of candidates, each
    carrying the song's real feature data, its rule-based score, and the reason
    it matched. This is the grounding context handed to the generation layer.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not songs:
        raise ValueError("no songs available to recommend from")

    prefs = parse_query(query, songs)
    ranked = recommend_songs(prefs, songs, k=k)

    candidates = [
        {
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "energy": song["energy"],
            "acousticness": song["acousticness"],
            "score": score,
            "reason": explanation,
        }
        for song, score, explanation in ranked
    ]
    logger.info("Retrieved %d candidates for query %r", len(candidates), query)
    return {"query": query, "prefs": prefs, "candidates": candidates}


if __name__ == "__main__":
    # Quick manual smoke test (zero tokens, no API needed).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.recommender import load_songs

    catalog = load_songs("data/songs.csv")
    for q in [
        "rainy day study music that isn't too sleepy",
        "high energy pop for the gym",
        "chill acoustic folk for a cozy evening",
    ]:
        result = retrieve(q, catalog, k=3)
        print(f"\nQuery: {q}")
        print(f"Parsed: {result['prefs']}")
        for c in result["candidates"]:
            print(f"  {c['title']} ({c['genre']}/{c['mood']}) score={c['score']:.2f}")
