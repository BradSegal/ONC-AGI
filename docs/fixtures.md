# Toy fixture worlds

Release 1.0.0-rc1 ships 20 **toy contract fixtures**: ten mechanisms, each as a full-access world and a sequential world. The two are separate draws of the same mechanism, so they never share a pool. They live in `src/onc_agi/fixtures/store/public_train/` and are installed with the package. `onc_agi.adapters.cli.fixture_store()` returns their location.

These are **not benchmark tasks**:
- they are synthetic, so they contain no patient data;
- they are small;
- their effects are large;
- their answer keys are set by construction rather than certified by the Monte Carlo oracle.

They exist so that every scoring rule and episode path can be exercised offline. [`tools/build_toy.py`](../tools/build_toy.py) builds them deterministically and readably. It is separate from the private generator that builds benchmark worlds.

| World | Mechanism | Difficulty tier | R | Strata (sizes on the card) |
|---|---|---|---|---|
| `toy-null-a` | No signal | 0 | 0 | `all` |
| `toy-null-b` | No signal | 2 | 0 | `s-a`, `s-b` |
| `toy-driver` | One strong expression driver | 0 | 1 | `all` |
| `toy-stand-in` | A driver with a near-duplicate partner (either earns credit) | 1 | 1 | `all` |
| `toy-wrong-type` | A copy-number cause; its expression target is only a correlate | 1 | 1 | `all` |
| `toy-confounder` | An observed clinical confounder | 0 | 1 | `s-a`, `s-b` |
| `toy-leak` | A driver plus a post-outcome column that tracks the outcome | 0 | 1 | `all` |
| `toy-interaction` | Two features that matter only jointly | 2 | 2 | `all` |
| `toy-module` | A three-member module (weighted coverage) | 2 | 3 | `all` |
| `toy-neutral` | One driver and one unrecoverable planted effect | 1 | 1 | `all` |

Every toy world has:
- 240 patients and a 35% outcome rate;
- the same 23 columns by data type: 10 expression, 3 copy number, 2 protein, 2 clinical, 2 derived, 1 baseline lab and 3 post-outcome lab;
- fake lowercase names in a per-world random order;
- the same prices (recruit 40 USD per patient; assays 0.5–8 USD per patient), and a budget equal to the full pool.

All post-outcome columns are in every world's reject set. The `world_id` suffix `-full` or `-seq` gives the mode.

What the standard baselines do here is a quick check that an agent behaves sensibly:
- univariate tests, lasso and stability selection recover the single-driver, stand-in, wrong-type, confounder, leak and neutral worlds;
- they all abstain or fail on the interaction;
- they recover part of the module.

The oracle scores 1. Every cheater is held at the floor: its displayed score is 0, and its unfloored score is within 0.02 of 0.

## Scoring rules on the toy worlds

Each block below runs on its own, so it can be pasted into a session. The [scoring specification](scoring.md) defines the rules.

```python
from onc_agi.adapters.cli import fixture_store
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services import scoring

store = FileWorldStore(fixture_store())

# Stand-in: the planted feature and its near-duplicate are equivalent answers.
key = store.answer_key("toy-stand-in-full")
part = key.groups[0].parts[0]
partner = next(f for f in part.equivalence_set if f != part.true_feature)
assert key.depth == 1 and not part.exact_recoverable
assert scoring.score_world([partner], key).find == 1.0

# Leak: a post-outcome feature anywhere in the list zeroes the world.
leaked = scoring.score_world([part.true_feature, key.reject_set[0]], key)
assert leaked.leaked and leaked.find == 0.0

# No signal: the empty list is restrained.
null = store.answer_key("toy-null-a-full")
assert null.is_null and scoring.score_world([], null).restrained
```

```python
from onc_agi.adapters.cli import fixture_store
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services import scoring

store = FileWorldStore(fixture_store())

# Interaction: both members are needed; one alone earns nothing.
inter = store.answer_key("toy-interaction-full")
a, b = (p.true_feature for p in inter.groups[0].parts)
assert scoring.score_world([a, b], inter).find == 1.0
assert scoring.score_world([a], inter).find == 0.0

# Module: credit is the covered share of the module's weights.
module = store.answer_key("toy-module-full")
heaviest = max(module.groups[0].parts, key=lambda p: p.weight)
partial = scoring.score_world([heaviest.true_feature], module)
assert 0.0 < partial.find < 1.0
```

```python
from onc_agi.adapters.agents import make_agent
from onc_agi.adapters.cli import fixture_store
from onc_agi.core.schema import Tier
from onc_agi.infra.bundles import FileWorldStore
from onc_agi.services.kit import evaluate

store = FileWorldStore(fixture_store())
worlds = [w for w in store.world_ids(Tier.PUBLIC_TRAIN) if w.endswith("-full")]
oracle, _ = evaluate(make_agent("oracle", store), store, Tier.PUBLIC_TRAIN, world_ids=worlds)
empty, _ = evaluate(make_agent("always_empty"), store, Tier.PUBLIC_TRAIN, world_ids=worlds)
assert oracle.discovery_score == 1.0
assert empty.discovery_score == 0.0  # abstaining everywhere earns nothing
```
