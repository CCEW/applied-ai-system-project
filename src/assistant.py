"""
Assistant orchestrator with guardrails (Task 3).

This is the piece that makes the RAG pipeline a *reliability* system. It runs
the full flow:

    retrieve (local, 0 tokens)  ->  generate (Groq)  ->  VALIDATE  ->  answer

The guardrail (`validate_answer`) verifies that the LLM's reply is actually
grounded in the retrieved catalog:

  * grounding check   -- the reply must name at least one retrieved candidate.
  * hallucination check -- any song title the reply puts in quotes that is NOT
    in the catalog is flagged as an off-catalog (invented) mention.

If generation fails (missing key, API error) OR the guardrail fails, the
assistant falls back to a deterministic, rule-based recommendation so the user
always gets a safe, catalog-only answer. Everything is logged.
"""
import logging
import re
from typing import Dict, List, Optional, Set

try:
    from src.generation import GenerationError, generate_recommendation
    from src.retrieval import retrieve
except Exception:
    # Support running the scripts directly (e.g. `streamlit run src/app.py`)
    # where the `src` package may not be importable; fall back to local imports.
    from generation import GenerationError, generate_recommendation
    from retrieval import retrieve

logger = logging.getLogger(__name__)

# Matches text inside straight or curly double quotes -> candidate "song titles".
_QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")

# Maximum score score_song can award: genre(+2) + mood(+1) + energy(+2) + acoustic(+1).
MAX_SCORE = 6.0


def _catalog_titles(songs: List[Dict]) -> Set[str]:
    return {str(s["title"]).strip().lower() for s in songs}


def validate_answer(answer: str, candidates: List[Dict], songs: List[Dict]) -> Dict:
    """Guardrail: check the LLM answer is grounded in the retrieved catalog.

    Returns a report dict:
        grounded      -- at least one retrieved candidate title appears in answer
        mentioned     -- retrieved candidate titles found in the answer
        off_catalog   -- quoted titles that exist nowhere in the catalog (invented)
        ok            -- grounded AND no off-catalog mentions
    """
    text_lower = answer.lower()
    candidate_titles = {str(c["title"]).strip().lower() for c in candidates}
    catalog_titles = _catalog_titles(songs)

    mentioned = sorted(
        {c["title"] for c in candidates if str(c["title"]).strip().lower() in text_lower}
    )

    off_catalog = sorted(
        {
            phrase.strip()
            for phrase in _QUOTED.findall(answer)
            if phrase.strip().lower() not in catalog_titles
            # ignore short quoted fragments that are clearly not song titles
            and len(phrase.strip()) > 3
        }
    )

    grounded = len(mentioned) > 0
    ok = grounded and not off_catalog

    report = {
        "grounded": grounded,
        "mentioned": mentioned,
        "off_catalog": off_catalog,
        "ok": ok,
    }
    if not ok:
        logger.warning("Guardrail rejected LLM answer: %s", report)
    else:
        logger.info("Guardrail passed: %d grounded mention(s)", len(mentioned))
    return report


def compute_confidence(
    candidates: List[Dict], source: str, validation: Optional[Dict]
) -> Dict:
    """Rate how confident the assistant is in the recommendation, in [0, 1].

    Confidence combines two signals:
      * retrieval match strength -- how strongly the top song matched the request
        (top score relative to the maximum possible score).
      * source -- an LLM answer that passed the guardrail is trusted more than a
        fallback, because the fallback ran only because something went wrong.

    Returns {"score": float, "reason": str}.
    """
    if not candidates:
        return {"score": 0.0, "reason": "no matching songs were found"}

    top_score = float(candidates[0]["score"])
    match_strength = max(0.0, min(1.0, top_score / MAX_SCORE))

    # Weight confidence mostly on how well the top song actually matched the
    # request, so a weak/irrelevant match scores low even if the LLM replied
    # fluently. Source only shifts the floor: a guardrail-passing LLM answer is
    # trusted a little more than a fallback.
    if source == "llm" and validation and validation.get("ok"):
        # Increase base trust for LLM answers while keeping match strength
        # influence. This makes confident LLM-passed answers score higher
        # without removing the importance of retrieval match quality.
        score = 0.45 + 0.50 * match_strength
        reason = (
            f"LLM answer passed the guardrail (grounded in "
            f"{len(validation['mentioned'])} song(s)); top match strength "
            f"{match_strength:.2f}"
        )
    else:
        score = 0.20 + 0.50 * match_strength
        reason = (
            f"rule-based fallback used; recommendation is catalog-grounded but "
            f"unvalidated by the LLM; top match strength {match_strength:.2f}"
        )

    return {"score": round(score, 2), "reason": reason}


def rule_based_answer(candidates: List[Dict]) -> str:
    """Deterministic, catalog-only fallback recommendation (no LLM, 0 tokens)."""
    if not candidates:
        return "No songs in the catalog matched your request."
    top = candidates[:3]
    lines = ["Here are the best matches from the catalog:"]
    for c in top:
        reason = c["reason"].replace("Matched because ", "").rstrip(".")
        lines.append(
            f'- "{c["title"]}" by {c["artist"]} '
            f"({c['genre']}, {c['mood']}) — {reason}."
        )
    return "\n".join(lines)


def recommend(
    query: str,
    songs: List[Dict],
    k: int = 5,
    api_key: Optional[str] = None,
) -> Dict:
    """Full grounded recommendation flow with guardrails and safe fallback.

    Returns:
        {
          query, prefs, candidates,   # from retrieval
          answer,                      # final text shown to the user
          source: "llm" | "fallback", # where `answer` came from
          validation,                 # guardrail report (None if LLM never ran)
          fallback_reason,            # why we fell back (None if source == "llm")
        }
    """
    retrieval_result = retrieve(query, songs, k=k)
    candidates = retrieval_result["candidates"]

    source = "fallback"
    validation: Optional[Dict] = None
    fallback_reason: Optional[str] = None

    try:
        llm_answer = generate_recommendation(retrieval_result, api_key=api_key)
        validation = validate_answer(llm_answer, candidates, songs)
        if validation["ok"]:
            answer = llm_answer
            source = "llm"
        else:
            answer = rule_based_answer(candidates)
            fallback_reason = "guardrail rejected the model's answer (not grounded)"
    except GenerationError as err:
        answer = rule_based_answer(candidates)
        fallback_reason = str(err)
        logger.warning("Falling back to rule-based answer: %s", err)

    confidence = compute_confidence(candidates, source, validation)
    logger.info(
        "Recommendation source=%s confidence=%.2f for query %r",
        source,
        confidence["score"],
        query,
    )
    return {
        **retrieval_result,
        "answer": answer,
        "source": source,
        "validation": validation,
        "fallback_reason": fallback_reason,
        "confidence": confidence,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        from src.recommender import load_songs
    except Exception:
        from recommender import load_songs

    catalog = load_songs("data/songs.csv")
    result = recommend("high energy pop for the gym", catalog, k=5)
    print(f"\n[source: {result['source']}]")
    if result["fallback_reason"]:
        print(f"[fallback reason: {result['fallback_reason']}]")
    print("\n" + result["answer"])
