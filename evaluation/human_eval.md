# Human Evaluation

Manual review of real assistant outputs by the author. Each row is one request I
ran, the criteria I judged it against, and the verdict. This complements the
automated harness (`evaluation/results.md`) with human judgment of answer quality.

Reviewer: project author · Method: ran each query with `python -m src.main "<query>"`
and read the answer, its `source`, and its `confidence`.

| # | Test Input | Evaluation Criteria | Confidence | Result |
|---|------------|---------------------|-----------|--------|
| 1 | "chill acoustic folk for a cozy evening" | Recommends calm/acoustic songs; only real catalog songs; reason cites real features | 0.80 | **Pass** — Velvet Harbor (folk), Spacewalk Thoughts (ambient), Library Rain (lofi); all low-energy/acoustic and in-catalog |
| 2 | "high energy pop for the gym" | Recommends high-energy songs; catalog-only; grounded | 0.84 | **Pass** — Gym Hero, Storm Runner, Sunrise City. Minor: Storm Runner is *rock*, but high-energy and the model disclosed the genre mismatch |
| 3 | "something moody for a night drive" (no API key) | Handles missing key gracefully; still returns a valid catalog answer | 0.35 | **Pass** — fell back to rule-based; top pick "Night Drive Loop" (moody) is a strong fit; no crash |
| 4 | "some k-pop bangers" | Correctly interpret genre; avoid claiming a genre the catalog lacks | 0.84 | **Fail** — parser matched "pop" as a substring of "k-pop" and recommended plain pop with high confidence. The catalog has no k-pop; the assistant should have signaled low confidence |
| 5 | "asdfghjkl zzz" (nonsense) | Should not present an irrelevant answer as confident | 0.28 | **Pass (partial)** — still returns generic songs, but low confidence (0.28) correctly signals a weak match after the confidence formula was re-weighted |

**Summary:** 4 of 5 human-reviewed cases passed. The system is strong on clear,
well-formed requests and degrades safely when the LLM is unavailable. The one
failure (case 4) is a query-parsing limitation: substring matching lets "k-pop"
register as "pop." Case 5 confirmed the value of re-weighting the confidence score
toward retrieval match strength so irrelevant queries no longer look confident.
