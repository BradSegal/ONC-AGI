# Evaluation protocol

A score is interpretable only with its world set, software version and harness. This protocol defines reporting and comparison requirements. Server support for a tier does not announce a hosted service, and a development run is not an accepted leaderboard result.

## Tracks and exposure

| Track | Fixed | Comparison |
|---|---|---|
| Standard | Inspect prompt, task card, tools, sandbox and limits | Models using the same harness version |
| Open | Interface and scoring rules | Complete agents, including prompts, tools and acquisition policies |
| Reference | Built-in oracle, random and shortcut agents | Checks on the score's scale |

Public-train worlds include answer keys and permit per-world scores, recordings and explanations. Use them for development and disclose that exposure. Public-eval and private tiers return aggregate scorecards only after closure. Do not mix tiers or modes in a ranking.

For hidden-tier serving, the runtime draws fresh worlds, targets a 20% null share and spreads signal worlds across difficulty tiers. It caps public-eval scorecards at five per key per day and private scorecards at three per key in total. Issued keys are required for normal hidden-tier operation. Operators must publish the pool commitment before evaluation; the ability to attach a commitment is not evidence that publication occurred.

## Standard-track settings

| Setting | Value |
|---|---|
| Messages per world | 60 full access; 80 sequential |
| Safety timeout | 1,800 s per world |
| Tool-call timeout | 180 s |
| Sandbox | No network; 2 CPUs; 4 GB memory |
| Episodes | One per world per scorecard (`epochs = 1`) |
| Model parameters | Provider defaults unless declared; record reasoning effort, temperature and token limits |

Record the harness label (`inspect-standard-1.1+<digest>`), scorer, engine and oracle versions. Different code or harness labels define different comparison groups, even when headline version numbers match. Conformance checks the wire contract; it does not establish equivalence between harnesses.

## World selection, repetition and uncertainty

A leaderboard entry requires at least **120 worlds per scorecard** and **three scorecards per agent**, with the mean and between-run spread reported. These are protocol minima, not a guarantee of precision. The server's default minimum of 40 hidden-tier worlds is an exposure control, not the leaderboard sample-size requirement.

For public-train comparisons, publish the exact world IDs and store digest, use the same selection for every agent and check the cohort, mechanism and mode mix. Do not assume the first N ordered worlds are representative. Hidden-tier draws differ across scorecards; compare runs from the same committed pool and declared draw policy.

The scorecard's 95% stratified bootstrap interval concerns the **signed, unfloored statistic**, not the clipped headline Discovery Score. It resamples signal and null worlds separately and does not estimate a model's run-to-run variation. Report both the per-scorecard intervals and variation across repeated runs. The leaderboard convention gives overlapping intervals tied ranks; that convention is not a formal test of equivalence.

## Costs and provenance

Report tokens, cost, parameters, command, commit, world-store digest, selected IDs or pool commitment, and all version labels. Use provider-reported cost when available; otherwise state the prices used, including input, output and cache accounting. Mark unavailable cost as unknown.

A spend limit stops new worlds from starting; in-flight worlds finish. Unplayed worlds remain in the denominator and score as empty submissions. Never silently shorten the world set after a failure or budget stop.

## Integrity limits

Evaluation worlds must not be used for training, tuning, seed reconstruction, adaptive answer extraction or patient re-identification. Public source cohorts make record matching a material risk. Fake names, perturbation, fresh draws and a no-network sandbox are controls; they do not by themselves certify resistance to that risk. Sequential and hidden-variable results from unrestricted open-track harnesses carry unverified integrity assumptions.

The [scoring specification](scoring.md) defines chance correction and abstention. Its signed statistics are the basis for shortcut checks; clipping means a finite displayed score need not be exactly zero under chance. Report suspected integrity flaws privately to the maintainers before public disclosure.
