# Primitives catalogue

Benchmark worlds are built from a small set of causal **roles** and **mechanics**. Each one targets a specific shortcut that a careless discovery method takes, and each has a documented answer-key rule. This catalogue describes what each role means for scoring. It does not describe how worlds are generated. The generator is private, as in ARC; fairness comes from publishing these rules and the task distribution.

| Role or mechanic | What is planted | Shortcut it defeats | Answer-key rule | Toy fixture |
|---|---|---|---|---|
| Generating feature | A measured feature drives the outcome | — (the base case) | `single`; the feature and any equivalents | `toy-driver` |
| Stand-in | The driver has a near-duplicate the data cannot separate from it | Insisting on one exact feature | `single`; both are in the equivalence set, but exact credit needs the driver | `toy-stand-in` |
| Cause in another data type | A copy-number change drives both an expression feature and the outcome | Ranking the downstream expression correlate | `single`; the copy-number feature | `toy-wrong-type` |
| Observed confounder | A clinical variable drives the outcome and shifts other features | Ranking the features the confounder shifts | `single`; the clinical indicator | `toy-confounder` |
| Leak | A post-outcome measurement tracks the outcome | Ranking by association without reading timing | The feature is in the **reject set**; listing it zeroes the world | `toy-leak` |
| Hidden cause | An unmeasured cause drives the outcome; measured proxies carry its signal | Treating the best proxy as the cause | `single`; credit goes to the proxies the data cannot separate from the cause | — |
| No signal | Nothing drives the outcome | Always claiming a discovery | R = 0; only an empty list is restrained | `toy-null-a`, `toy-null-b` |
| Interaction | Two features matter only jointly | Marginal screening | `joint`; both members are needed | `toy-interaction` |
| Module | Several features contribute with different weights | All-or-nothing selection | `weighted_coverage` | `toy-module` |
| Effect modifier | A feature matters only within a subgroup | Pooled-only analysis | `joint` (feature and modifier) | — |
| Mediator | A cause acts through an intermediate measurement | Ranking the upstream correlate above the direct driver | `single`; the mediator is the direct driver, and the upstream cause earns credit only through the equivalence set | — |
| Collider and selection | Inclusion depends on the outcome and another feature | Conditioning on a collider | Collider-selected features never earn credit | — |
| Contradictory mixture | Subgroups have opposite associations (Simpson's paradox) | Trusting the pooled direction | Recoverable causes, or certified **underdetermined** (then only Restraint is scored) | — |
| Cross-cohort shift | A cohort or batch effect touches many features | Mistaking batch for biology | The batch indicator is a neutral nuisance | — |
| Neutral group | A planted effect too small to recover from this world's data | — (fairness) | Neutral: listing it neither helps nor hurts | `toy-neutral` |

Other rules apply in every world:
- No world's shape reveals its role. Every world has the same number of post-outcome columns and the same mix of column types.
- Feature names are fake unless the world says otherwise.
- No-signal worlds keep the shape of a signal world.
- **Difficulty** varies with:
  - effect size;
  - outcome rate;
  - sample size and number of features;
  - substitute correlation;
  - confounding strength;
  - assay noise and missingness;
  - the source cohort.

  Worlds are grouped into difficulty tiers by how recoverable their mechanisms are.
