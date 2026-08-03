"""
Reliability evaluation harness for the RAG Music Assistant.

Runs a fixed set of labeled cases that exercise the reliability-critical parts of
the system -- query parsing, grounded retrieval, the guardrail, and the safe
fallback -- and records a PASS/FAIL for each against an explicit criterion.

This harness is OFFLINE and FREE: it never calls the Groq API (the fallback case
simulates an LLM failure), so results are reproducible with no key or network.

Run:
    python -m evaluation.run_eval

Writes machine-readable results to:
    evaluation/results.json   (structured JSON)
    evaluation/results.md     (markdown table + summary line)
"""
import json
import logging
from typing import Callable, Dict, List

import src.assistant as assistant
from src.assistant import recommend, validate_answer
from src.generation import GenerationError
from src.recommender import load_songs
from src.retrieval import parse_query, retrieve

logging.disable(logging.CRITICAL)  # keep eval output clean

SONGS = load_songs("data/songs.csv")
CATALOG_TITLES = {s["title"] for s in SONGS}


def _case(input_desc: str, criteria: str, check: Callable[[], bool], detail: str = "") -> Dict:
    """Run a single check and capture the result, guarding against exceptions."""
    try:
        passed = bool(check())
        note = detail
    except Exception as exc:  # noqa: BLE001 - a crash is itself a failed reliability check
        passed = False
        note = f"raised {type(exc).__name__}: {exc}"
    return {
        "input": input_desc,
        "criteria": criteria,
        "result": "Pass" if passed else "Fail",
        "detail": note,
    }


def _fallback_result_for(query: str) -> Dict:
    """Force the fallback path by simulating an LLM failure (no API call)."""
    original = assistant.generate_recommendation

    def boom(retrieval_result, **kwargs):
        raise GenerationError("simulated LLM failure (offline eval)")

    assistant.generate_recommendation = boom
    try:
        return recommend(query, SONGS, k=5)
    finally:
        assistant.generate_recommendation = original


def build_cases() -> List[Dict]:
    cases: List[Dict] = []

    # 1. Query parsing: genre + energy detection.
    prefs = parse_query("high energy pop for the gym", SONGS)
    cases.append(_case(
        '"high energy pop for the gym"',
        "Detects genre=pop and high energy",
        lambda: prefs["genre"] == "pop" and prefs["energy"] == 0.9,
        detail=str(prefs),
    ))

    # 2. Query parsing: acoustic + low energy.
    prefs2 = parse_query("chill acoustic music for studying", SONGS)
    cases.append(_case(
        '"chill acoustic music for studying"',
        "Detects low energy and acoustic preference",
        lambda: prefs2["energy"] == 0.25 and prefs2["likes_acoustic"] is True,
        detail=str(prefs2),
    ))

    # 3. Query parsing: conflicting energy cues stay neutral (no guessing).
    prefs3 = parse_query("something sad but energetic and intense yet chill", SONGS)
    cases.append(_case(
        '"sad but energetic ... yet chill"',
        "Conflicting energy cues -> energy left neutral (None)",
        lambda: prefs3["energy"] is None,
        detail=str(prefs3),
    ))

    # 4. Retrieval grounding: every candidate is a real catalog song.
    retrieved = retrieve("high energy pop for the gym", SONGS, k=5)["candidates"]
    cases.append(_case(
        '"high energy pop for the gym"',
        "All retrieved candidates exist in the catalog",
        lambda: all(c["title"] in CATALOG_TITLES for c in retrieved),
        detail=f"{len(retrieved)} candidates checked",
    ))

    # 5. Guardrail rejects a hallucinated song title.
    hallucinated = f'Try "{retrieved[0]["title"]}" and also "Totally Fake Song".'
    report_bad = validate_answer(hallucinated, retrieved, SONGS)
    cases.append(_case(
        "LLM answer citing an invented song",
        "Guardrail flags off-catalog title and rejects (ok=False)",
        lambda: report_bad["ok"] is False and "Totally Fake Song" in report_bad["off_catalog"],
        detail=str(report_bad),
    ))

    # 6. Guardrail passes a properly grounded answer.
    good = f'I recommend "{retrieved[0]["title"]}" for your workout.'
    report_ok = validate_answer(good, retrieved, SONGS)
    cases.append(_case(
        "LLM answer citing a retrieved song",
        "Guardrail accepts grounded answer (ok=True)",
        lambda: report_ok["ok"] is True,
        detail=str(report_ok),
    ))

    # 7. Empty input handled gracefully (no crash; raises a clear error).
    def empty_raises_valueerror():
        try:
            retrieve("", SONGS, k=5)
            return False
        except ValueError:
            return True

    cases.append(_case(
        "Empty request string",
        "Handled gracefully with a clear ValueError (no crash)",
        empty_raises_valueerror,
    ))

    # 8. Safe fallback on LLM failure returns a catalog-grounded answer.
    fb = _fallback_result_for("something moody for a night drive")
    cases.append(_case(
        '"something moody for a night drive" (LLM fails)',
        "Falls back to a catalog-grounded rule-based answer",
        lambda: fb["source"] == "fallback" and fb["candidates"][0]["title"] in fb["answer"],
        detail=f"source={fb['source']}, confidence={fb['confidence']['score']}",
    ))

    # 9. Confidence score is a valid probability in [0, 1].
    conf = fb["confidence"]["score"]
    cases.append(_case(
        "Confidence score output",
        "Confidence is a number within [0, 1]",
        lambda: isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0,
        detail=f"confidence={conf}",
    ))

    return cases


def write_results(cases: List[Dict]) -> str:
    passed = sum(1 for c in cases if c["result"] == "Pass")
    total = len(cases)
    summary = f"{passed} out of {total} reliability checks passed."

    # JSON (machine-readable)
    with open("evaluation/results.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "passed": passed, "total": total, "cases": cases},
                  fh, indent=2)

    # Markdown table (human/parser-readable)
    lines = [
        "# Reliability Evaluation Results",
        "",
        f"**Summary:** {summary}",
        "",
        "| # | Test Input | Evaluation Criteria | Result | Detail |",
        "|---|------------|---------------------|--------|--------|",
    ]
    for i, c in enumerate(cases, start=1):
        detail = c["detail"].replace("|", "\\|")
        lines.append(
            f"| {i} | {c['input']} | {c['criteria']} | {c['result']} | {detail} |"
        )
    lines.append("")
    with open("evaluation/results.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return summary


def main() -> None:
    cases = build_cases()
    summary = write_results(cases)
    print(summary)
    for i, c in enumerate(cases, start=1):
        print(f"  [{c['result']:>4}] {i}. {c['criteria']}")
    print("\nWrote evaluation/results.json and evaluation/results.md")


if __name__ == "__main__":
    main()
