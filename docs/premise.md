# The premise and task card

Every world poses the same question:

> *Here is a cohort with an outcome. Which measurements drive it? Return an ordered list, most likely first, or an empty list if nothing can be found.*

The question is simple and the difficulty lies in the worlds. A world is a cohort of patients with a binary outcome and a set of measurements. Hidden among the measurements is a planted mechanism, or no mechanism at all. The world's answer key records what was planted and how much of it can be recovered from the data the world provides.

## What an agent receives

A **world card** (`WorldCard`). It lists:
- the world's identifier, tier and mode;
- the number of patients that can be revealed;
- every feature, with its data type, its timing relative to the outcome and its assay price;
- the recruitment strata, the price list and the budget;
- the premise text itself.

The data then arrive in one of two ways:

| Mode | How data arrive | What is measured |
|---|---|---|
| `full_access` | All patients and all features are revealed at reset. | Inference under correlation, confounding and leaks. |
| `sequential` | Nothing is revealed at reset. The agent spends budget to `recruit` patients (revealing their outcome) and to `assay` features on them. | All of the above, plus what to measure and when to stop. |

## What an agent returns

One `submit` action carrying an ordered `ranking` of feature identifiers, most likely driver first. An empty ranking means "nothing can be found". A ranking may not repeat a feature, and every identifier must exist in the world.

## Rules every agent should know

1. **Post-outcome features are traps.** A feature whose timing is `post_outcome` was measured after the outcome occurred. Listing any of them, anywhere in the ranking, scores that world 0 and counts towards the Leak rate.
2. **Order matters, and so does stopping.** Only the first R distinct answers count, where R is the number of recoverable slots in the world's answer key. R is not revealed. Padding the list with guesses does not help. Recovery is normalised against chance for the same list, so a random addition gains nothing in expectation, and every guess risks a leak.
3. **Correlated substitutes count once.** Features that are near-duplicates of each other (|r| ≥ 0.8, complete linkage) form one cluster. Only the first listed member of a cluster is kept.
4. **Equivalent answers earn credit.** If the data cannot tell a planted feature apart from a substitute, the answer key lists both in that slot's equivalence set, and either earns the credit.
5. **Saying "nothing" is a skill.** About one world in five has no signal. Abstaining there earns Restraint, and abstaining on a world that has signal costs it. Random abstention scores zero.
6. **Data cost counts in sequential mode.** Spending more than the reference cost (what a well-designed study would spend) scales the world's credit down.
7. **There is no score feedback during evaluation.** Public-train worlds ship with answer keys for development. Evaluation tiers show only aggregate scorecards after the scorecard closes.

[scoring.md](scoring.md) gives the exact scoring. [interface.md](interface.md) gives the exact payloads.
