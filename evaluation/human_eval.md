# Human Evaluation

Manual review of real assistant outputs by the author. Each row is one request I
ran, the criteria I judged it against, and the verdict. This complements the
automated harness (`evaluation/results.md`) with human judgment of answer quality.

Reviewer: project author · Method: ran each query with `python -m src.main "<query>"`
and read the answer, its `source`, and its `confidence`.

| # | Test Input | Evaluation Criteria | Confidence | Result |
|---|------------|---------------------|-----------|--------|
| 1 | "chill acoustic folk for a cozy evening" | Recommends calm/acoustic songs; only real catalog songs; reason cites real features | 0.76 | **Pass** — Velvet Harbor (folk), Spacewalk Thoughts (ambient), Library Rain (lofi); all low-energy/acoustic and in-catalog |
| 2 | "high energy pop for the gym" | Recommends high-energy songs; catalog-only; grounded | 0.84 | **Pass** — Gym Hero, Greedy (Tate McRae), Sunrise City; all energetic pop and in-catalog |
| 3 | "something moody for a night drive" (no API key) | Handles missing key gracefully; still returns a valid catalog answer | 0.36 | **Pass** — fell back to rule-based; top picks "Type Shit", "Prada", "Nowhere" (all moody) are strong fits; no crash |
| 4 | "something pop but no rock music" | Respect a negated preference | 0.65 | **Pass** — detected `exclude_genres: ['rock']`, kept pop, and no rock songs appeared in the results |
| 5 | "asdfghjkl zzz" (nonsense) | Should not present an irrelevant answer as confident | 0.28 | **Pass (partial)** — still returns generic songs, but low confidence (0.28) correctly signals a weak match after the confidence formula was re-weighted |

**Summary:** all 5 human-reviewed cases behaved correctly (case 5 only partially, on a
nonsense input). The system is strong on clear, well-formed requests, honors negation, and
degrades safely when the LLM is unavailable. Case 5 confirmed the value of re-weighting the
confidence score toward retrieval match strength so irrelevant queries no longer look
confident.
