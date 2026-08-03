# Reliability Evaluation Results

**Summary:** 9 out of 9 reliability checks passed.

| # | Test Input | Evaluation Criteria | Result | Detail |
|---|------------|---------------------|--------|--------|
| 1 | "high energy pop for the gym" | Detects genre=pop and high energy | Pass | {'genre': 'pop', 'mood': None, 'energy': 0.9, 'likes_acoustic': False} |
| 2 | "chill acoustic music for studying" | Detects low energy and acoustic preference | Pass | {'genre': None, 'mood': 'chill', 'energy': 0.25, 'likes_acoustic': True} |
| 3 | "sad but energetic ... yet chill" | Conflicting energy cues -> energy left neutral (None) | Pass | {'genre': None, 'mood': 'chill', 'energy': None, 'likes_acoustic': False} |
| 4 | "high energy pop for the gym" | All retrieved candidates exist in the catalog | Pass | 5 candidates checked |
| 5 | LLM answer citing an invented song | Guardrail flags off-catalog title and rejects (ok=False) | Pass | {'grounded': True, 'mentioned': ['Gym Hero'], 'off_catalog': ['Totally Fake Song'], 'ok': False} |
| 6 | LLM answer citing a retrieved song | Guardrail accepts grounded answer (ok=True) | Pass | {'grounded': True, 'mentioned': ['Gym Hero'], 'off_catalog': [], 'ok': True} |
| 7 | Empty request string | Handled gracefully with a clear ValueError (no crash) | Pass |  |
| 8 | "something moody for a night drive" (LLM fails) | Falls back to a catalog-grounded rule-based answer | Pass | source=fallback, confidence=0.35 |
| 9 | Confidence score output | Confidence is a number within [0, 1] | Pass | confidence=0.35 |
