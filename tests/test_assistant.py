"""
Tests for the RAG assistant: retrieval, guardrails, and safe fallback.

These tests are fully OFFLINE and FREE — the LLM call is monkeypatched, so no
Groq API key or network access is needed to run `pytest`.
"""
import src.assistant as assistant
from src.assistant import (
    compute_confidence,
    recommend,
    rule_based_answer,
    validate_answer,
)
from src.generation import GenerationError
from src.recommender import load_songs
from src.retrieval import parse_query, retrieve

SONGS = load_songs("data/songs.csv")


# --- Retrieval ---------------------------------------------------------------

def test_parse_query_detects_genre_and_high_energy():
    prefs = parse_query("high energy pop for the gym", SONGS)
    assert prefs["genre"] == "pop"
    assert prefs["energy"] == 0.9  # high-energy cue


def test_parse_query_detects_low_energy_and_acoustic():
    prefs = parse_query("chill acoustic music for studying", SONGS)
    assert prefs["energy"] == 0.25  # low-energy cue
    assert prefs["likes_acoustic"] is True


def test_parse_query_does_not_match_pop_inside_kpop():
    # "k-pop" and "pop" are different genres; the parser must not match "pop"
    # as a substring of "k-pop" (regression test for hyphen-boundary matching).
    assert parse_query("some k-pop bangers", SONGS)["genre"] is None
    # ...but real pop and hyphenated/compound genres still resolve.
    assert parse_query("high energy pop for the gym", SONGS)["genre"] == "pop"
    assert parse_query("hip-hop beats", SONGS)["genre"] == "hip-hop"


def test_negation_excludes_genre_from_results():
    # "anything but country" -> country is excluded and absent from candidates.
    prefs = parse_query("anything but country music", SONGS)
    assert "country" in prefs["exclude_genres"]
    assert prefs["genre"] is None
    candidates = retrieve("anything but country music", SONGS, k=8)["candidates"]
    assert all(c["genre"] != "country" for c in candidates)


def test_negation_keeps_positive_genre_and_drops_excluded():
    # "something pop but no rock" -> keep pop, drop rock.
    prefs = parse_query("something pop but no rock music", SONGS)
    assert prefs["genre"] == "pop"
    assert "rock" in prefs["exclude_genres"]
    candidates = retrieve("something pop but no rock music", SONGS, k=8)["candidates"]
    assert all(c["genre"] != "rock" for c in candidates)


def test_no_negation_leaves_exclude_empty():
    prefs = parse_query("high energy pop for the gym", SONGS)
    assert prefs["exclude_genres"] == []


def test_parse_query_conflicting_energy_stays_neutral():
    # Both a high and a low cue -> the parser refuses to guess.
    prefs = parse_query("something sad but energetic and intense yet chill", SONGS)
    assert prefs["energy"] is None


def test_retrieve_returns_grounded_candidates_from_catalog():
    result = retrieve("high energy pop for the gym", SONGS, k=5)
    catalog_titles = {s["title"] for s in SONGS}
    assert len(result["candidates"]) == 5
    # Every retrieved candidate must be a real song from the catalog.
    for c in result["candidates"]:
        assert c["title"] in catalog_titles


# --- Guardrail: validate_answer ---------------------------------------------

def _candidates():
    return retrieve("high energy pop for the gym", SONGS, k=5)["candidates"]


def test_validate_answer_passes_when_grounded():
    candidates = _candidates()
    answer = f'I recommend "{candidates[0]["title"]}" — it fits perfectly.'
    report = validate_answer(answer, candidates, SONGS)
    assert report["ok"] is True
    assert candidates[0]["title"] in report["mentioned"]
    assert report["off_catalog"] == []


def test_validate_answer_flags_hallucinated_song():
    candidates = _candidates()
    # Mentions a real candidate BUT also an invented song in quotes.
    answer = f'Try "{candidates[0]["title"]}" and also "Totally Fake Song".'
    report = validate_answer(answer, candidates, SONGS)
    assert "Totally Fake Song" in report["off_catalog"]
    assert report["ok"] is False


def test_validate_answer_rejects_when_not_grounded():
    candidates = _candidates()
    answer = "Here is some generic advice with no song titles at all."
    report = validate_answer(answer, candidates, SONGS)
    assert report["grounded"] is False
    assert report["ok"] is False


# --- Fallback ----------------------------------------------------------------

def test_rule_based_answer_only_uses_catalog_songs():
    candidates = _candidates()
    text = rule_based_answer(candidates)
    catalog_titles = {s["title"] for s in SONGS}
    # At least the top candidate should appear, and it must be a real song.
    assert candidates[0]["title"] in text
    assert any(title in text for title in catalog_titles)


# --- Orchestration: recommend (LLM mocked) ----------------------------------

def test_recommend_uses_llm_when_answer_is_grounded(monkeypatch):
    def fake_generate(retrieval_result, **kwargs):
        top = retrieval_result["candidates"][0]["title"]
        return f'You should listen to "{top}".'

    monkeypatch.setattr(assistant, "generate_recommendation", fake_generate)
    result = recommend("high energy pop for the gym", SONGS, k=5)
    assert result["source"] == "llm"
    assert result["validation"]["ok"] is True
    assert result["fallback_reason"] is None


def test_recommend_falls_back_when_generation_errors(monkeypatch):
    def boom(retrieval_result, **kwargs):
        raise GenerationError("simulated API failure")

    monkeypatch.setattr(assistant, "generate_recommendation", boom)
    result = recommend("high energy pop for the gym", SONGS, k=5)
    assert result["source"] == "fallback"
    assert "simulated API failure" in result["fallback_reason"]
    # The fallback answer is still a real, catalog-grounded recommendation.
    assert result["candidates"][0]["title"] in result["answer"]


def test_recommend_falls_back_when_guardrail_rejects(monkeypatch):
    def hallucinate(retrieval_result, **kwargs):
        return 'Listen to "An Invented Song That Does Not Exist Anywhere".'

    monkeypatch.setattr(assistant, "generate_recommendation", hallucinate)
    result = recommend("high energy pop for the gym", SONGS, k=5)
    assert result["source"] == "fallback"
    assert result["validation"]["ok"] is False


# --- Confidence scoring ------------------------------------------------------

def test_confidence_is_valid_probability():
    result = recommend("high energy pop for the gym", SONGS, k=5)
    score = result["confidence"]["score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert result["confidence"]["reason"]


def test_confidence_higher_for_passing_llm_than_fallback():
    candidates = _candidates()
    passing = {"ok": True, "mentioned": [candidates[0]["title"]], "off_catalog": []}
    llm_conf = compute_confidence(candidates, "llm", passing)
    fallback_conf = compute_confidence(candidates, "fallback", None)
    # Same candidates, but a guardrail-passing LLM answer is trusted more.
    assert llm_conf["score"] > fallback_conf["score"]


def test_confidence_zero_when_no_candidates():
    assert compute_confidence([], "fallback", None)["score"] == 0.0
