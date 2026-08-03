# 🎵 Music Recommender + Natural-Language Assistant (RAG)

## Summary

**Original project (CodePath Modules 1–3): _Music Recommender Simulation_.** The original
project represented songs and a user "taste profile" as structured data and designed a
transparent, rule-based scoring function that ranked songs by how closely they matched a
user's preferred genre, mood, energy level, and acoustic preference. Its core capability
was explainability — every recommendation came with a plain-English reason for why it
scored the way it did.

**This project extends that recommender into a natural-language music assistant.** Instead
of filling out a rigid profile, a user describes what they want in plain English — *"chill
acoustic folk for a cozy evening"* — and the system retrieves matching songs from the
catalog, uses a language model to write a grounded recommendation, and validates that
answer with a guardrail before showing it. It matters because it demonstrates a safe,
low-cost pattern for putting an LLM in front of real data: retrieval keeps the model
grounded in the actual catalog, and a validation layer plus a rule-based fallback keep the
system reliable even when the model errors or hallucinates.

---

## How The System Works

Each song is described with features such as genre, mood, energy, tempo, valence, danceability, and acousticness. The user profile stores the user’s favorite genre, favorite mood, target energy, and whether they tend to like acoustic music.

The recommender uses a simple two-step process:

- Scoring Rule: score one song based on how well it matches the user profile.
- Ranking Rule: sort all scored songs and return the best ones first.

### Algorithm Recipe

A good starting point is to give stronger weight to genre than mood, because genre is often a broader signal of taste while mood is more specific. For example:

- +2.0 points for a genre match
- +1.0 point for a mood match
- additional similarity points for energy, based on how close the song’s energy is to the user’s target

For the energy feature, the system rewards closeness rather than just “higher” or “lower” values. If a user prefers energy around 0.8, a song at 0.75 should score better than a song at 0.20 because it is closer to the target. This setup helps the system distinguish between very different tastes, such as an intense rock song and a chill lofi song.

### Potential Biases

This system may over prioritize genre and overlook a song that is a great mood match but a different genre. It also uses a small set of rules, so it may not capture more subtle personal taste as well as a more advanced recommender.

---

## AI Extension: Natural-Language Assistant (RAG + Guardrails)

The base recommender needs a rigid profile (`genre`, `mood`, `energy`...). The
extension adds a **natural-language assistant** so you can just say what you
want — *"chill acoustic folk for a cozy evening"* — and get a grounded answer.

It uses **Retrieval-Augmented Generation (RAG)** with a **reliability guardrail**:

1. **Retrieve** (`src/retrieval.py`, local, **0 tokens**) — parses your request
   into taste preferences and ranks the catalog with the existing `score_song`
   rule. The top songs become the *grounding context*.
2. **Generate** (`src/generation.py`) — a free Groq LLM (`llama-3.1-8b-instant`)
   writes a short recommendation using **only** those retrieved songs.
3. **Validate** (`src/assistant.py`) — a guardrail checks the answer is grounded
   (names a retrieved song) and free of hallucinated titles. If generation fails
   or the guardrail rejects the answer, the app **falls back to a safe,
   rule-based recommendation**. Every step is logged.

This makes the AI feature part of the main app flow: the response is *generated
from* retrieved catalog data, not printed beside a canned answer. Retrieval is
free; only the short final reply spends (a small number of) tokens.

---

## Architecture Overview

The system is a linear RAG pipeline with a validation gate. The full diagram lives in
[`diagrams/system_architecture.mmd`](diagrams/system_architecture.mmd) — open it in
[mermaid.live](https://mermaid.live) or a Mermaid-enabled editor.

Data flows in one direction: **input → retrieve → generate → validate → output.**

| Stage | Component | What it does |
|---|---|---|
| **Input** | user + `data/songs.csv` | A free-text request plus the song catalog. |
| **Retriever** | `src/retrieval.py` | Parses the request into taste preferences and ranks the catalog with the original `score_song` rule. Top-k songs become the grounding context. Local, 0 tokens. |
| **Generator** | `src/generation.py` | A Groq LLM writes a recommendation using **only** the retrieved songs. |
| **Guardrail / Evaluator** | `src/assistant.py` | Validates the answer is grounded and free of hallucinated titles (catalog = ground truth). On failure or API error, falls back to a rule-based answer. |
| **Output** | `src/main.py`, `src/app.py` | The recommendation plus its `source` (`llm` or `fallback`) and what it was grounded in. |
| **Testing** | `tests/` | Offline tests (LLM mocked) assert the retriever, guardrail, and fallback behave. |

**Where AI results get checked:** the guardrail verifies every model answer against the
catalog before it reaches the user; automated tests assert that hallucinated answers are
rejected and the system falls back safely; and the CLI/Streamlit UI surface the detected
preferences and grounding so a human can see *why* each answer was produced.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add a free Groq API key (for the LLM step). Get one at
   https://console.groq.com/keys — free, no credit card. Copy `.env.example`
   to `.env` and paste your key:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=your_key_here
```

> Without a key the app still runs — it just uses the safe rule-based fallback
> for the answer instead of the LLM.

4. Run the app:

```bash
python -m src.main                          # interactive: type your requests
python -m src.main "chill study music"      # one-shot query
streamlit run src/app.py                    # optional web UI
```

### Running Tests

Run the tests with:

```bash
pytest
```

The suite covers the base recommender (`tests/test_recommender.py`) and the AI
extension — retrieval, the grounding/hallucination guardrail, and the safe
fallback (`tests/test_assistant.py`). **All tests are offline and free**: the LLM
call is mocked, so no Groq API key or network access is needed to run `pytest`.

---

## Sample Interactions

Four real runs of the assistant. Note the `[source: ...]` tag on each — it shows whether
the answer came from the LLM (guardrail passed) or the safe rule-based fallback.

**Example 1 — grounded LLM answer:**

```
$ python -m src.main "chill acoustic folk for a cozy evening"

============================================================
Your request: chill acoustic folk for a cozy evening
Detected preferences: {'genre': 'folk', 'mood': 'chill', 'energy': 0.25, 'likes_acoustic': True, 'exclude_genres': []}
------------------------------------------------------------
For a cozy evening, I'd recommend the following chill acoustic folk songs:

1. "Velvet Harbor" by Marina Vale - Its nostalgic mood and high acousticness make it perfect for a relaxing evening.
2. "Spacewalk Thoughts" by Orbit Bloom - Although more ambient, its high acousticness and chill mood fit the bill.
3. "Library Rain" by Paper Lanterns - This lofi track has a soothing atmosphere and a high acousticness score, ideal for unwinding.
------------------------------------------------------------
[source: llm]  [confidence: 0.76]
[why: LLM answer passed the guardrail (grounded in 3 song(s)); top match strength 0.68]
[grounded in: Library Rain, Spacewalk Thoughts, Velvet Harbor]
============================================================
```

**Example 2 — grounded LLM answer (detects genre + high energy):**

```
$ python -m src.main "high energy pop for the gym"

============================================================
Your request: high energy pop for the gym
Detected preferences: {'genre': 'pop', 'mood': None, 'energy': 0.9, 'likes_acoustic': False, 'exclude_genres': []}
------------------------------------------------------------
Based on your request for high energy pop for the gym, I'd recommend the following:

1. "Gym Hero" by Max Pulse - This song is specifically titled for the gym and has a high energy level (0.93) to match your intense workout.
2. "Greedy" by Tate McRae - With an energetic mood and moderate energy level (0.75), this song is perfect for a high-intensity gym session.
3. "Sunrise City" by Neon Echo - Although it's not as energetic as the first two, its happy mood and decent energy level (0.82) make it a great fit for a more upbeat gym playlist.
------------------------------------------------------------
[source: llm]  [confidence: 0.84]
[why: LLM answer passed the guardrail (grounded in 3 song(s)); top match strength 0.81]
[grounded in: Greedy, Gym Hero, Sunrise City]
============================================================
```

**Example 3 — safe fallback (no API key / guardrail failure):** when the LLM is unavailable
or its answer fails validation, the app falls back to a rule-based recommendation drawn
only from the catalog, so the user always gets a valid answer.

```
$ python -m src.main "something moody for a night drive"

============================================================
Your request: something moody for a night drive
Detected preferences: {'genre': None, 'mood': 'moody', 'energy': None, 'likes_acoustic': False, 'exclude_genres': []}
------------------------------------------------------------
Here are the best matches from the catalog:
- "Type Shit" by Metro Boomin (hip-hop, moody) — mood match (+1.0); acoustic preference.
- "Prada" by Casso (electronic, moody) — mood match (+1.0); acoustic preference.
- "Nowhere" by The Black Keys (rock, moody) — mood match (+1.0); acoustic preference.
------------------------------------------------------------
[source: fallback]  [confidence: 0.36]
[why: rule-based fallback used; recommendation is catalog-grounded but unvalidated by the LLM; top match strength 0.33]
[fallback: GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys and set it in your environment.]
============================================================
```

**Example 4 — negation handling:** the parser detects excluded genres and filters them out of
retrieval, so a request like "no rock" never surfaces rock songs.

```
$ python -m src.main "something pop but no rock music"

============================================================
Your request: something pop but no rock music
Detected preferences: {'genre': 'pop', 'mood': None, 'energy': None, 'likes_acoustic': False, 'exclude_genres': ['rock']}
------------------------------------------------------------
Based on your request for pop music without rock, I recommend the following three songs:

- "Flowers" by Miley Cyrus: This upbeat pop song has a happy mood and moderate energy, making it a great fit.
- "Espresso" by Sabrina Carpenter: This song shares a similar happy mood and moderate energy, perfect for a pop playlist.
- "Gym Hero" by Max Pulse: Although intense, this song is still a great fit for pop fans, with a high energy level and minimal acoustic elements.
------------------------------------------------------------
[source: llm]  [confidence: 0.65]
[why: LLM answer passed the guardrail (grounded in 3 song(s)); top match strength 0.49]
[grounded in: Espresso, Flowers, Gym Hero]
============================================================
```



**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## What the Sample Interactions Show

Read together, the [Sample Interactions](#sample-interactions) above point to three patterns
(the underlying numbers and edge cases are in the [Testing Summary](#testing-summary), not
repeated here):

- **Clear requests map to sensible, on-theme songs.** Energetic and calm/acoustic queries
  pull visibly different results, and an answer only reaches the user once it passes the
  guardrail.
- **Conflicting or vague requests expose the base scorer's bias.** When cues conflict the
  parser stays neutral, but the underlying rule still leans on energy and acousticness — the
  main limitation carried over from the original recommender.
- **Failure is safe, not silent.** A rejected or unavailable LLM answer falls back to a
  catalog-only recommendation with a lower confidence score — never a crash or an invented
  song.

---

## Design Decisions

- **RAG instead of a bigger model or fine-tuning.** The original rule-based scorer already
  ranks the catalog well and is fully explainable, so I reused it as the *retriever* rather
  than throwing it away. The LLM only rewrites the top candidates into natural language.
  Trade-off: the quality of the final answer is capped by the retriever — if scoring misses
  a good song, the LLM never sees it.
- **Retrieval runs locally; only generation costs tokens.** Parsing and ranking happen with
  plain Python over `songs.csv`, so the token cost is one short prompt per request. Trade-off:
  local keyword parsing is simpler and cheaper than an embedding model, but it can miss
  phrasings not in its vocabulary.
- **Groq `llama-3.1-8b-instant`.** Chosen for a free tier, low latency, and small token use.
  Trade-off: an 8B open model is less capable than a frontier model, but grounding + the
  guardrail make that acceptable for this task.
- **Guardrail + rule-based fallback over trusting the model.** Every answer is validated
  against the catalog, and any failure degrades to a deterministic rule-based answer instead
  of an error. Trade-off: the guardrail is a pragmatic heuristic (title/quote matching), not
  a perfect hallucination detector, and it can occasionally reject a valid but oddly-phrased
  answer.
- **Two entry points (CLI + Streamlit).** The CLI keeps the app scriptable and easy to test;
  the Streamlit UI adds a transparency panel for a non-technical audience.

---

## Testing Summary

The system proves its reliability four ways: **automated tests**, a **confidence score**,
**logging + error handling**, and **human evaluation**.

> **In short:** 21/21 unit tests pass and 9/9 automated reliability checks pass, all offline
> with no API key. In human review, all reviewed queries behaved correctly (one only
> partially, on a nonsense input). Testing drove real fixes: a substring bug (the genre
> "pop" matching inside "k-pop") is fixed with hyphen-aware boundary matching; negation is
> now handled (e.g. "something pop but no rock" excludes rock); and an over-generous first
> confidence formula was re-weighted, so a nonsense query drops from 0.62 to ~0.28 while a
> strong match sits at ~0.84.

Run the reliability artifacts yourself:

```bash
pytest                         # 21 unit tests (LLM mocked, offline)
python -m evaluation.run_eval  # writes evaluation/results.md + results.json
```

### Reproducible reliability & guardrail results (captured logs)


**1. Unit tests — `pytest`:**

```text
$ pytest -q
.................                                                        [100%]
17 passed in 1.17s
```

**2. Automated reliability harness — `python -m evaluation.run_eval`:**

```text
$ python -m evaluation.run_eval
9 out of 9 reliability checks passed.
  [Pass] 1. Detects genre=pop and high energy
  [Pass] 2. Detects low energy and acoustic preference
  [Pass] 3. Conflicting energy cues -> energy left neutral (None)
  [Pass] 4. All retrieved candidates exist in the catalog
  [Pass] 5. Guardrail flags off-catalog title and rejects (ok=False)
  [Pass] 6. Guardrail accepts grounded answer (ok=True)
  [Pass] 7. Handled gracefully with a clear ValueError (no crash)
  [Pass] 8. Falls back to a catalog-grounded rule-based answer
  [Pass] 9. Confidence is a number within [0, 1]

Wrote evaluation/results.json and evaluation/results.md
```

Full results table (also in [`evaluation/results.md`](evaluation/results.md)):

| # | Test Input | Evaluation Criteria | Result |
|---|------------|---------------------|--------|
| 1 | "high energy pop for the gym" | Detects genre=pop and high energy | Pass |
| 2 | "chill acoustic music for studying" | Detects low energy and acoustic preference | Pass |
| 3 | "sad but energetic ... yet chill" | Conflicting energy cues → energy left neutral (None) | Pass |
| 4 | "high energy pop for the gym" | All retrieved candidates exist in the catalog | Pass |
| 5 | LLM answer citing an invented song | Guardrail flags off-catalog title and rejects | Pass |
| 6 | LLM answer citing a retrieved song | Guardrail accepts grounded answer | Pass |
| 7 | Empty request string | Handled gracefully (clear error, no crash) | Pass |
| 8 | LLM fails mid-request | Falls back to a catalog-grounded answer | Pass |
| 9 | Confidence score output | Confidence is a number within [0, 1] | Pass |

**3. Guardrail catching a hallucination and falling back** (the LLM is forced to invent a
song; the guardrail rejects it and the user still gets a safe, catalog-only answer):

```text
LLM answer: You should listen to "Totally Fake Song" by Nobody.
Guardrail report: {'grounded': False, 'mentioned': [], 'off_catalog': ['Totally Fake Song'], 'ok': False}
Final source: fallback
Confidence: 0.61
Answer shown to user:
Here are the best matches from the catalog:
- "Gym Hero" by Max Pulse (pop, intense) — genre match (+2.0); energy similarity (+1.9); acoustic preference.
- "Sunrise City" by Neon Echo (pop, happy) — genre match (+2.0); energy similarity (+1.8); acoustic preference.
- "Greedy" by Tate McRae (pop, energetic) — genre match (+2.0); energy similarity (+1.7); acoustic preference.
```

**4. Human evaluation** (full table in [`evaluation/human_eval.md`](evaluation/human_eval.md)):

| # | Test Input | Criteria | Confidence | Result |
|---|------------|----------|-----------|--------|
| 1 | "chill acoustic folk for a cozy evening" | Calm/acoustic, catalog-only | 0.76 | Pass |
| 2 | "high energy pop for the gym" | High-energy, grounded | 0.84 | Pass |
| 3 | "something moody for a night drive" (no key) | Graceful fallback | 0.36 | Pass |
| 4 | "something pop but no rock music" | Respect negation in the request | 0.65 | **Pass** (excludes rock, keeps pop) |
| 5 | "asdfghjkl zzz" (nonsense) | Don't look confident on junk | 0.28 | Pass (partial) |

- **Automated tests** — `tests/test_recommender.py` + `tests/test_assistant.py` cover parsing,
  retrieval, negation handling, the grounding/hallucination guardrail, both fallback paths,
  and confidence.
- **Automated reliability harness** — [`evaluation/results.md`](evaluation/results.md) (and
  `results.json`) records each check as a parseable Test Input / Criteria / Result row.
- **Confidence scoring** — every answer reports a `[confidence: 0.00–1.00]` based on retrieval
  match strength and whether the guardrail passed.
- **Logging + error handling** — every step is logged; API/key errors and rejected answers
  degrade safely to the rule-based fallback instead of crashing.
- **Human evaluation** — [`evaluation/human_eval.md`](evaluation/human_eval.md) documents a
  manual review of real outputs in a markdown table.

**What worked / what didn't / what I learned:** clear requests produced grounded, on-catalog
answers and the guardrail correctly rejected a synthetic hallucinated title. Testing surfaced
two parsing weaknesses that I then fixed: a substring false-positive (genre "pop" matched
inside "k-pop"), resolved with hyphen-aware boundary matching, and missing negation handling,
now supported so "no rock" excludes rock. Confidence scoring also started too generous and was
re-weighted toward retrieval match strength. I learned to test the *scaffolding* around the
model (retrieval, negation, validation, fallback, confidence) deterministically and mock the
model itself — keeping the
suite fast, free, and reproducible while still proving the reliability behavior.

---

## Limitations and Risks

This recommender is simple and works best on a small, controlled dataset. It can miss important details about real musical taste and may make recommendations that feel reasonable but are not truly personal.
There is also a risk of bias. The system can over-focus on a few features, such as energy or acousticness, and may repeat similar kinds of songs instead of offering a broader range of options. The catalog was expanded to 68 songs across many genres (pop, country, rock, hip-hop, reggae, R&B, latin, electronic, and more) with recent tracks to reduce this bias, but it is still a fixed, local dataset whose audio-feature values are approximate, so it will not generalize to tastes or genres it has no data for.

---

## Reflection

Building this taught me that the hard part of an AI feature usually isn't the model call it's everything around it. Turning a rule-based recommender into an LLM assistant was mostly about keeping the model honest: retrieving the right context, forcing the answer to stay grounded in real data, and deciding what should happen when the model fails. Framing the problem as "retrieve → generate → verify → fall back" made a an unreliable idea into a system I could actually test and trust.




