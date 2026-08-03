# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

musicFinder 4

---

## 2. Intended Use  

This recommender is meant to suggest songs that fit a simple user taste profile. It is designed for classroom use and small experiments, not for a real music app that needs very personal or careful recommendations.

Its goal is to match songs to a user who says what genre, mood, energy level, and acoustic preference they want.

---

## 3. How the Model Works  

The system looks at a few song features and compares them to the user’s preferences. It checks genre, mood, energy level, and how acoustic the song sounds. A song gets a higher score when it matches these features well.

In plain terms, the model tries to answer: “Which songs feel like the kind of music this user seems to want?”

**Extended version (RAG assistant).** The project now wraps this rule-based scorer in a
natural-language assistant. A user types a free-text request; the scorer *retrieves* the
best-matching catalog songs; a Groq LLM (`llama-3.1-8b-instant`) writes a recommendation
grounded only in those songs; and a guardrail validates the answer against the catalog
before it is shown, falling back to a rule-based answer if the LLM fails or hallucinates.
Every answer also carries a confidence score. So the extended system has two models working
together: a transparent rule-based retriever and a language model that only rephrases what
the retriever already found.

---

## 4. Data  

The system uses a dataset of 68 songs spanning many genres — pop, country, rock, hip-hop, reggae, R&B, latin, electronic, and more — including recent popular tracks. Each song includes information like title, artist, genre, mood, energy, and acousticness. The dataset is broader and less biased than the original 20-song version, but it is still limited: the audio-feature values are approximate, and it does not include lyrics, artist history, or personal listening habits.

---

## 5. Strengths  

The recommender works well when a user has clear and simple preferences. For example, it does a decent job when someone wants happy pop songs or calm acoustic songs.

It also gives sensible results when the user clearly prefers one style over another. In those cases, the top songs usually line up with the user’s stated mood and energy level.

---

## 6. Limitations and Bias 

One weakness is that the system can over-focus on energy and acousticness. A song with the right energy level may rank very high even if its genre or mood does not really fit the user.

This can create a filter bubble. The recommender may keep showing songs that are similar in a narrow way, instead of exploring a wider range of music.

It also struggles with conflicting preferences. If a user says they want something sad but very energetic, the system may still favor songs that match the energy part more strongly.

---

## 7. Evaluation  

I tested the system with a few different user profiles. I compared a happy pop profile, an acoustic profile, and a conflicting profile that wanted a sad mood but high energy.

The happy pop profile mostly favored bright, energetic songs like “Gym Hero” and “Sunrise City.” The acoustic profile shifted toward calmer songs like “Spacewalk Thoughts” and “Midnight Waltz.” This makes sense because the system is strongly rewarding energy and acousticness.

The conflicting profile showed that the recommender can still pick songs based mostly on one feature. That was useful to see because it helped reveal where the system is strongest and where it is too simple.

---

## 8. Future Work  

If I kept developing this, I would add more features such as tempo, danceability, and valence. I would also make the scoring less dependent on one or two signals so it feels more balanced. I would also improve how the system handles conflicting or unusual preferences. A better version could mix different signals more fairly and show more variety in the final recommendations.

---

## 9. Personal Reflection  

My biggest learning moment was realizing that a recommender can look smart even when it is using very simple rules. I thought the system would need a lot of complex logic, but a few clear signals like genre, mood, energy, and acousticness were enough to produce accurate recommendations.

Using AI tools helped me move faster, especially when I was writing code, and explaining what I was seeing. They were especially helpful for turning raw thoughts into a working structure.

I was surprised by how simple algorithms can still feel like real recommendations. A basic scoring system can make suggestions that seem personal, even though it is really just matching a obvious features.

If I extended this project, I would try adding more features like tempo, danceability, etc. I would also want to make the recommendations more balanced and less biased so the system feels more thoughtful and less repetitive.

---

## 10. Responsible AI Reflection

### What are the limitations or biases in your system?

The system inherits the biases of its rule-based retriever and adds a few of its own. The
scorer over-weights energy and acousticness, so it can form a filter bubble and recommend a
narrow band of similar songs. Because the LLM is only allowed to pick from what the retriever
surfaces, **any bias in retrieval is passed straight through to the final answer** — if a good
song is never retrieved, the language model never gets a chance to suggest it. The query
parser is keyword-based and English-only: it recognizes only the genres and moods that exist
in the catalog, so it cannot serve a genre the catalog lacks. (It does handle simple negation
— "no rock", "anything but country" — but not more complex phrasing.) The catalog (68 songs)
is broader than the original 20 and now spans many
genres with recent tracks, but it is still fixed and its audio-feature values are approximate,
so the assistant cannot serve tastes it has no data for, and the 8B language model is less
capable than a frontier model. Finally, the guardrail is a heuristic (title/quote matching), not a perfect
hallucination detector.

### Could your AI be misused, and how would you prevent that?

The direct risk is low — it recommends songs from a fixed, local catalog and cannot take real
actions. The realistic misuse is *over-trust*: presenting an LLM-written answer as an
authoritative or "personalized" recommendation when it is really matching a few surface
features, which could mislead a user or, at scale, quietly push everyone toward the same
popular tracks. I mitigate this by (1) grounding every answer in retrieved catalog data and
validating it with a guardrail, (2) attaching a **confidence score** so weak matches are
visibly weak, (3) exposing the detected preferences and grounding in the UI so the reasoning
is transparent rather than a black box, and (4) documenting in this card that the system is a
classroom demo, not a production recommender. A larger deployment would also need rate
limiting and no logging of personal listening data.

### What surprised you while testing your AI's reliability?

Two things. First, how easily a fluent answer can *look* trustworthy while being weakly
grounded: my first confidence formula rated a nonsense query ("asdfghjkl zzz") at **0.62**,
because it gave too much credit simply for the guardrail passing. Re-weighting confidence
toward the retriever's actual match strength dropped that same query to about **0.28** while a
strong match stayed near **0.84** — testing, not intuition, is what exposed the gap. Second,
how much of the reliability work lives *around* the model rather than in it: the parts that
made the system trustworthy (retrieval, the guardrail, the fallback, confidence) are all
deterministic code I could unit-test offline, and mocking the LLM kept the whole suite fast,
free, and reproducible.

### Describe your collaboration with AI during this project.

I built this project in collaboration with an AI coding assistant (Claude Code), working
task-by-task: choosing the extension, then implementing retrieval, generation, guardrails, the
interface, tests, and docs. The AI wrote most of the scaffolding while I made the key
decisions — provider choice (a free, low-token model), what the guardrail should check, and
when to stop — and verified each step by running the code and reading the output.

**One helpful suggestion:** the AI proposed reusing the original `score_song` rule as the RAG
*retriever* instead of discarding it, and pairing the LLM with a catalog-validation guardrail
plus a rule-based fallback. This "retrieve → generate → verify → fall back" framing is what
turned a fuzzy idea into a system I could actually test and trust, and it reused work I had
already done rather than throwing it away.

**One flawed suggestion:** the AI's first confidence formula floored guardrail-passing LLM
answers at 0.55, which made irrelevant queries look confident (the 0.62 nonsense case above).
That was a genuinely misleading reliability signal. I caught it by testing edge cases, and we
corrected the formula to weight retrieval match strength more heavily. It was a good reminder
that AI suggestions need to be checked against real behavior, not accepted because they sound
reasonable.