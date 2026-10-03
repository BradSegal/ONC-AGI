# ONC-AGI scoring specification (scorer-1.0)

This document defines how a submission is scored. It is executable. Every block tagged `scoring-world`, `scoring-example`, `scoring-chance` or `scoring-aggregate` is run against the real scorer by `tests/runtime/unit/test_scoring_spec.py`, which checks that the stated expectations hold.

## 1. One world, one list

The agent submits an ordered list of feature ids, most likely driver first, or an empty list. The world's answer key, which is private, holds:

- the **recoverable groups**. Each has one or more *parts*, and each part has a true feature and an **oracle-equivalence set**: the features the Monte Carlo oracle certified as detectable in its place.
- the **neutral groups**: planted but not recoverable from the data, such as underdetermined or decoy terms.
- the **reject set**: post-outcome leaks.
- the truth-independent **clusters**: features correlated at |r| ≥ 0.8.
- the **strata**: data type × correlation bin × hub bin, used for matched chance.

The **depth** R is the number of recoverable parts.

Scoring takes six steps.

1. **Validation.** An unknown feature id or a repeated feature is refused with a typed error.
2. **Leak reject.** If any listed feature is in the reject set, the world scores Find 0 wherever the leak appears in the list, and the world counts toward Leak rate.
3. **Representatives.** The list is reduced to its first-listed feature per cluster. Near-duplicates therefore neither earn extra credit nor use extra depth.
4. **Neutral removal.** Members of neutral groups' equivalence sets are dropped. They neither earn credit nor use depth.
5. **Credit at depth R.** The top R remaining representatives earn credit one to one, in list order. Each credits the first uncredited part whose equivalence set contains it. The credit is *exact* if it is the true feature itself, or if the part is not exactly recoverable. Each group's credit rule then applies:
   - `single`: one part, credited by any member of its equivalence set;
   - `joint` (interactions, modifiers, mixtures): all parts, or nothing;
   - `weighted_coverage` (modules): the covered share of absolute weights, times the number of parts.

   Raw recovery is the credited slots divided by R.
6. **Chance normalisation.** The *matched* chance c is the expected raw recovery of the same list with each feature replaced by a random feature from its own stratum, pushed through steps 3–5. Then:
   - `find_signed = (raw − c) / (1 − c)`, which is zero in expectation for any outcome-blind list and drives statistics and gates;
   - `find = clip(find_signed, 0, 1)`, which is displayed.

   Strict uses exact credit in the same way.

### The worked world

World `spec-w` has seven features in six clusters (`a` and `a2` are near-duplicates) and depth R = 2:

- `a` generates the outcome; its near-duplicate `a2` is an equivalent substitute.
- `b` generates the outcome; `c`, in a *different* cluster, is oracle-equivalent to it.
- `d` is a neutral decoy.
- `L` is measured after the outcome.

```json scoring-world name=spec-w
{
  "world_id": "spec-w",
  "difficulty_tier": 1,
  "groups": [
    {"group_id": "g1", "role": "direct", "label": "recoverable", "credit_rule": "single",
     "parts": [{"true_feature": "a", "equivalence_set": ["a", "a2"], "exact_recoverable": true}]},
    {"group_id": "g2", "role": "stand_in", "label": "recoverable", "credit_rule": "single",
     "parts": [{"true_feature": "b", "equivalence_set": ["b", "c"], "exact_recoverable": true}]},
    {"group_id": "g3", "role": "decoy", "label": "neutral", "credit_rule": "single",
     "parts": [{"true_feature": "d", "equivalence_set": ["d"], "exact_recoverable": false}]}
  ],
  "reject_set": ["L"],
  "clusters": {"a": 0, "a2": 0, "b": 1, "c": 2, "d": 3, "e": 4, "L": 5},
  "strata": {"a": "s", "a2": "s", "b": "s", "c": "s", "d": "s", "e": "s", "L": "s"},
  "reference_cost": 100.0,
  "detection_threshold": 3.0,
  "oracle_version": "spec"
}
```

In the examples below the matched chance is given as `(c_raw, c_exact) = (0.1, 0.05)`, so the arithmetic is visible. Section 2 shows how chance itself is computed.

**The oracle list** recovers both parts exactly: `find = (1 − 0.1) / 0.9 = 1`.

```json scoring-example world=spec-w
{"ranking": ["a", "b"], "chance": [0.1, 0.05],
 "expect": {"raw_recovery": 1.0, "find": 1.0, "find_exact": 1.0, "leaked": false, "abstained": false}}
```

**A near-duplicate listed first** takes the cluster's single slot: `a2` represents cluster 0, and `a` is deduplicated away. Find is full, but Strict credits only `b`: `find_exact = (0.5 − 0.05) / 0.95`.

```json scoring-example world=spec-w
{"ranking": ["a2", "a", "b"], "chance": [0.1, 0.05],
 "expect": {"raw_recovery": 1.0, "find": 1.0, "find_exact": 0.47368421052631576}}
```

**An oracle-equivalent feature outside the true cluster earns credit.** `c` is in its own cluster but in `b`'s equivalence set: `find = (0.5 − 0.1) / 0.9`.

```json scoring-example world=spec-w
{"ranking": ["c"], "chance": [0.1, 0.05],
 "expect": {"raw_recovery": 0.5, "find": 0.4444444444444445, "find_signed": 0.4444444444444445, "find_exact": 0.0}}
```

**Wrong features use depth; neutral features do not.** `d` is removed, so the top R = 2 representatives are `e` and `b`. `a` falls beyond the depth and earns nothing.

```json scoring-example world=spec-w
{"ranking": ["e", "d", "b", "a"], "chance": [0.1, 0.05],
 "expect": {"raw_recovery": 0.5, "find": 0.4444444444444445}}
```

**Any listed leak zeroes the world**, even at the end of an otherwise perfect list. The signed value is what the empty list would earn.

```json scoring-example world=spec-w
{"ranking": ["a", "b", "L"], "chance": [0.1, 0.05],
 "expect": {"leaked": true, "find": 0.0, "find_exact": 0.0, "find_signed": -0.11111111111111112}}
```

**Abstaining on a signal world** earns nothing and counts against Restraint, because it is abstention on a signal world.

```json scoring-example world=spec-w
{"ranking": [], "chance": [0.1, 0.05],
 "expect": {"abstained": true, "raw_recovery": 0.0, "find": 0.0, "find_signed": -0.11111111111111112}}
```

**Sequential efficiency.** `efficiency = min(1, reference cost / spend)`. A perfect list bought at four times the oracle reference cost carries efficiency 0.25 into Find.

```json scoring-example world=spec-w
{"ranking": ["a", "b"], "chance": [0.1, 0.05], "sequential": true, "spent": 400.0,
 "expect": {"find": 1.0, "efficiency": 0.25}}
```

### Null worlds

A null world has no recoverable part (R = 0). Its only question is whether the agent restrains. An empty list counts as restraint, and so does a list that contains only neutral features.

```json scoring-world name=spec-null
{
  "world_id": "spec-null",
  "difficulty_tier": 1,
  "groups": [
    {"group_id": "g3", "role": "decoy", "label": "neutral", "credit_rule": "single",
     "parts": [{"true_feature": "d", "equivalence_set": ["d"], "exact_recoverable": false}]}
  ],
  "reject_set": ["L"],
  "clusters": {"a": 0, "d": 1, "e": 2, "L": 3},
  "reference_cost": 100.0,
  "detection_threshold": 3.0,
  "oracle_version": "spec"
}
```

```json scoring-example world=spec-null
{"ranking": [], "expect": {"restrained": true, "find": 0.0}}
```

```json scoring-example world=spec-null
{"ranking": ["d"], "expect": {"restrained": true, "abstained": true}}
```

```json scoring-example world=spec-null
{"ranking": ["e"], "expect": {"restrained": false, "abstained": false}}
```

```json scoring-example world=spec-null
{"ranking": ["L"], "expect": {"restrained": false, "leaked": true}}
```

### Joint and weighted-coverage groups

An interaction counts only when both parts are inside the depth:

```json scoring-world name=spec-joint
{
  "world_id": "spec-joint",
  "difficulty_tier": 2,
  "groups": [
    {"group_id": "gx", "role": "interaction", "label": "recoverable", "credit_rule": "joint",
     "parts": [{"true_feature": "x", "equivalence_set": ["x"], "exact_recoverable": true},
               {"true_feature": "y", "equivalence_set": ["y"], "exact_recoverable": true}]}
  ],
  "clusters": {"x": 0, "y": 1, "z": 2},
  "reference_cost": 100.0,
  "detection_threshold": 3.0,
  "oracle_version": "spec"
}
```

```json scoring-example world=spec-joint
{"ranking": ["x", "z"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 0.0}}
```

```json scoring-example world=spec-joint
{"ranking": ["y", "x"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 1.0, "find_exact": 1.0}}
```

```json scoring-example world=spec-joint
{"ranking": ["y", "z", "x"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 0.0}}
```

A module earns the covered share of its absolute weights. With weights 3 and 1 and R = 2, listing only `m1` recovers 3/4 of the module:

```json scoring-world name=spec-module
{
  "world_id": "spec-module",
  "difficulty_tier": 2,
  "groups": [
    {"group_id": "gm", "role": "module", "label": "recoverable", "credit_rule": "weighted_coverage",
     "parts": [{"true_feature": "m1", "equivalence_set": ["m1"], "exact_recoverable": true, "weight": 3.0},
               {"true_feature": "m2", "equivalence_set": ["m2"], "exact_recoverable": true, "weight": 1.0}]}
  ],
  "clusters": {"m1": 0, "m2": 1, "z": 2},
  "reference_cost": 100.0,
  "detection_threshold": 3.0,
  "oracle_version": "spec"
}
```

```json scoring-example world=spec-module
{"ranking": ["m1"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 0.75}}
```

```json scoring-example world=spec-module
{"ranking": ["z", "m2"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 0.25}}
```

### Cases that look like shortcuts

These are the scorer's literal behaviour in situations an agent might try to exploit. Each was raised by adversarial testing.

**Listing only neutral features is abstaining** on a signal world, and is restraint on a null world. One non-neutral feature makes it a claim.

```json scoring-example world=spec-w
{"ranking": ["d"], "chance": [0.1, 0.05], "expect": {"abstained": true, "find": 0.0}}
```

```json scoring-example world=spec-null
{"ranking": ["d", "e"], "expect": {"restrained": false, "abstained": false}}
```

**Deduplication comes before neutral removal.** A neutral feature listed first represents its cluster, and the cluster then contributes nothing. Here `n` is neutral but shares a cluster with the true feature `a`. The cluster structure is truth-independent, so an agent cannot use this to hide a claim; it can only lose credit.

```json scoring-world name=spec-shadow
{
  "world_id": "spec-shadow",
  "difficulty_tier": 1,
  "groups": [
    {"group_id": "g1", "role": "direct", "label": "recoverable", "credit_rule": "single",
     "parts": [{"true_feature": "a", "equivalence_set": ["a"], "exact_recoverable": true}]},
    {"group_id": "g2", "role": "decoy", "label": "neutral", "credit_rule": "single",
     "parts": [{"true_feature": "n", "equivalence_set": ["n"], "exact_recoverable": false}]}
  ],
  "clusters": {"a": 0, "n": 0, "z": 1},
  "reference_cost": 100.0,
  "detection_threshold": 3.0,
  "oracle_version": "spec"
}
```

```json scoring-example world=spec-shadow
{"ranking": ["n", "a"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 0.0, "abstained": true}}
```

```json scoring-example world=spec-shadow
{"ranking": ["a", "n"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 1.0}}
```

**A leak behind a cluster-mate still zeroes the world.** The leak check reads the whole list before deduplication.

```json scoring-example world=spec-w
{"ranking": ["a", "a2", "b", "L"], "chance": [0.1, 0.05], "expect": {"leaked": true, "find": 0.0}}
```

**Credit is greedy in list order when equivalence sets overlap**. `v` is equivalent to the true feature `u` and is also a true feature itself. Listing `v` first spends it on `u`'s part, and `u` then has no part left to credit. Maximum matching would credit both orders fully. The choice between the two rules is under review.

```json scoring-world name=spec-overlap
{
  "world_id": "spec-overlap",
  "difficulty_tier": 2,
  "groups": [
    {"group_id": "gu", "role": "direct", "label": "recoverable", "credit_rule": "single",
     "parts": [{"true_feature": "u", "equivalence_set": ["u", "v"], "exact_recoverable": true}]},
    {"group_id": "gv", "role": "direct", "label": "recoverable", "credit_rule": "single",
     "parts": [{"true_feature": "v", "equivalence_set": ["v"], "exact_recoverable": true}]}
  ],
  "clusters": {"u": 0, "v": 1, "z": 2},
  "reference_cost": 100.0,
  "detection_threshold": 3.0,
  "oracle_version": "spec"
}
```

```json scoring-example world=spec-overlap
{"ranking": ["u", "v"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 1.0, "find_exact": 1.0}}
```

```json scoring-example world=spec-overlap
{"ranking": ["v", "u"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 0.5, "find_exact": 0.0}}
```

**Half an interaction earns nothing and still uses a slot**, unlike a module, where each member earns its share of the weight (compare `spec-module` above).

```json scoring-example world=spec-joint
{"ranking": ["x"], "chance": [0.0, 0.0], "expect": {"raw_recovery": 0.0, "abstained": false}}
```

**Malformed submissions are refused, never scored as empty.** A repeated or unknown feature raises a typed error. In process this is `ArenaError`; over HTTP it is a 400 or 422 response. Neither consumes the episode.

```json scoring-error world=spec-w
{"ranking": ["a", "a"], "code": "invalid_payload"}
```

```json scoring-error world=spec-w
{"ranking": ["a", "zz"], "code": "unknown_feature"}
```

**Sequential restraint is asymmetric by design.** Efficiency scales the credit for restraint on null worlds, but abstaining on a signal world counts in full. An agent that buys the whole pool and then always abstains therefore has negative Restraint (the third aggregate example in section 3). Its unfloored score is never positive.

## 2. Matched chance

Chance is not a property of the world alone. It belongs to *the list's own feature classes*: each listed feature is replaced by a random member of its stratum, without replacement, 200 times, and steps 3–5 are applied to each draw. A list built from data types, hubs, near-duplicate counts or synthetic-column tells therefore gains nothing over chance.

Suppose a world has four features in one stratum and a single true feature `a`. A one-feature list then has matched chance 1/4, whichever feature it names. The stated value is approximate (Monte Carlo, seeded by world and list), so it is checked to within ±0.07.

```json scoring-world name=spec-chance
{
  "world_id": "spec-chance",
  "difficulty_tier": 1,
  "groups": [
    {"group_id": "g1", "role": "direct", "label": "recoverable", "credit_rule": "single",
     "parts": [{"true_feature": "a", "equivalence_set": ["a"], "exact_recoverable": true}]}
  ],
  "clusters": {"a": 0, "p": 1, "q": 2, "r": 3},
  "strata": {"a": "s", "p": "s", "q": "s", "r": "s"},
  "reference_cost": 100.0,
  "detection_threshold": 3.0,
  "oracle_version": "spec"
}
```

```json scoring-chance world=spec-chance
{"ranking": ["p"], "expect_raw": 0.25, "tolerance": 0.07}
```

```json scoring-chance world=spec-chance
{"ranking": ["a"], "expect_raw": 0.25, "tolerance": 0.07}
```

**Chance cannot be re-rolled.** The Monte Carlo draws are fixed per world: draw *i* fixes one permutation of every stratum, and a list consumes those permutations in its own order. Two lists with the same sequence of strata therefore get the same chance. Permuting or padding the tail beyond the depth cannot fish for a luckier estimate.

```json scoring-invariant world=spec-chance
{"rankings": [["p", "q"], ["p", "r"], ["p", "a"]], "field": "chance_recovery"}
```

An empty list has chance 0 by definition:

```json scoring-chance world=spec-chance
{"ranking": [], "expect_raw": 0.0, "tolerance": 0.0}
```

## 3. Across worlds: the scorecard

Over a set of worlds:

- **Find** = mean over signal worlds of `find × efficiency`;
- **Restraint** = (mean over null worlds of `restrained × efficiency`) − (abstention rate on signal worlds). This is Youden's J: always-empty and always-claiming agents both score 0;
- **Discovery Score** = `Find × max(0, Restraint)`, which is the headline;
- **unfloored** = `find_signed × Restraint`, made negative whenever either factor is negative. It is the estimand for intervals and release gates;
- **Strict** = the same with exact credit.

Undefined values are `null`: Restraint without null worlds, and Find without signal worlds. The interval is a 2.5–97.5% percentile bootstrap of the unfloored score, resampling signal and null worlds separately. Per-tier rows put null worlds in the tier they inherited from their source world.

In this example, three signal worlds have Find 1, 0.5 and an abstention, and three null worlds have two restraints:

```json scoring-aggregate
{"worlds": [
   {"null": false, "find": 1.0, "find_signed": 1.0, "abstained": false},
   {"null": false, "find": 0.5, "find_signed": 0.5, "abstained": false},
   {"null": false, "find": 0.0, "find_signed": -0.1, "abstained": true},
   {"null": true, "restrained": true},
   {"null": true, "restrained": true},
   {"null": true, "restrained": false}
 ],
 "expect": {"find": 0.5, "restraint": 0.3333333333333333, "discovery_score": 0.16666666666666666,
            "unfloored": 0.15555555555555556, "abstention_on_signal": 0.3333333333333333,
            "restraint_on_null": 0.6666666666666666}}
```

Here Find = (1 + 0.5 + 0) / 3 = 0.5, Restraint = 2/3 − 1/3 = 1/3, and the unfloored value is `find_signed × Restraint = ((1 + 0.5 − 0.1) / 3) × (1/3) ≈ 0.156`.

An agent that always submits an empty list scores Restraint 1 − 1 = 0, and an agent that never submits one scores 0 − 0 = 0. Both have Discovery Score 0:

```json scoring-aggregate
{"worlds": [
   {"null": false, "find": 0.0, "find_signed": -0.2, "abstained": true},
   {"null": true, "restrained": true}
 ],
 "expect": {"restraint": 0.0, "discovery_score": 0.0, "unfloored": 0.0}}
```

When the components have opposite or both-negative signs, the unfloored value is never positive:

```json scoring-aggregate
{"worlds": [
   {"null": false, "find": 0.0, "find_signed": -0.1, "abstained": true},
   {"null": true, "restrained": true, "efficiency": 0.5}
 ],
 "expect": {"restraint": -0.5, "discovery_score": 0.0, "unfloored": -0.05}}
```

## 4. Agreement with PyPlasmode evaluators

For lists whose top R representatives each fall into at most one part's equivalence set, raw recovery equals PyPlasmode's `evaluate_groups` group recall at depth R. The answer key maps to a `MaterializedTruth(kind="custom")` with one `RecoveryGroup` per part. This package does not itself depend on PyPlasmode. The arena extends PyPlasmode's evaluators in four ways:

- partial lists, completed in a fixed, truth-blind order;
- one-to-one credit, so a single feature cannot credit two parts;
- the `joint` rule;
- matched chance normalisation.

Strict never exceeds PyPlasmode's `evaluate_ranking` exact recall over the same top R. It equals that recall unless a substitute is listed before its true feature: under one-to-one credit the substitute takes the slot, so the later true feature earns nothing. PyPlasmode's `evaluate_module` gives the weighted coverage of a module part set. The agreement is checked by a property test maintained alongside the world builder.

## 5. Latency envelope

Scoring one submission, chance included, must take at most 100 ms for p ≤ 2,000 features. The failure limit is 1 s. `tests/runtime/statistical/test_scoring_latency.py` measures it on a 2,000-feature world and reports the timing.
