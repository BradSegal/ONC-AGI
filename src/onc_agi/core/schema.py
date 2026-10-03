"""Interface contract v1 for ONC-AGI.

This module is the single owner of every payload that crosses a process, file or
network boundary: world cards, actions, observations, submissions, answer keys,
per-world scores, scorecards and trace events. Harnesses (the Agent kit, the HTTP
server and the Inspect standard harness) and the private world builder all
exchange these models, so a change here is a versioned interface change.

Change policy
-------------
``INTERFACE_VERSION`` follows ``MAJOR.MINOR``. A MINOR bump may add optional
fields only; anything that removes, renames or re-types a field, or changes a
scoring semantic, is a MAJOR bump and must be recorded in the decision log.

Survival outcomes are additive optional fields that are *omitted from
serialisation when absent*, so every binary payload is byte-identical to v1.0 and a strict
v1.0 client still parses it; ``INTERFACE_VERSION`` therefore stays "1.0".
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

INTERFACE_VERSION: Literal["1.0"] = "1.0"

FeatureId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")]
WorldId = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[a-z0-9][a-z0-9_.\-]*$")]
RequestId = Annotated[str, Field(min_length=1, max_length=96)]


class Frozen(BaseModel):
    """Strict, immutable base: unknown fields are rejected at every boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- world card


class Mode(StrEnum):
    FULL_ACCESS = "full_access"
    SEQUENTIAL = "sequential"


class Tier(StrEnum):
    PUBLIC_TRAIN = "public_train"
    PUBLIC_EVAL = "public_eval"
    PRIVATE = "private"


class NameVisibility(StrEnum):
    REAL = "real"
    FAKE = "fake"
    ANONYMOUS = "anonymous"


class Timing(StrEnum):
    """When a measurement is taken relative to the outcome.

    ``post_outcome`` features are recorded after the outcome occurred; listing one
    as a driver is a leak and zeroes the world.
    """

    BASELINE = "baseline"
    POST_OUTCOME = "post_outcome"


class FeatureMeta(Frozen):
    feature_id: FeatureId
    data_type: Literal["expression", "copy_number", "protein", "clinical", "mutation", "lab", "derived"]
    timing: Timing = Timing.BASELINE
    assay_price: float = Field(
        ge=0.0, description="Price per patient to measure this feature (sequential mode)."
    )


class PriceList(Frozen):
    recruit_per_patient: float = Field(ge=0.0)
    currency: Literal["USD"] = "USD"


class WorldCard(Frozen):
    """Everything an agent may know about a world before acting."""

    interface_version: Literal["1.0"] = INTERFACE_VERSION
    world_id: WorldId
    tier: Tier
    mode: Mode
    outcome_type: Literal["binary", "survival"] = Field(
        default="binary",
        description="binary: outcome is 0/1. survival: outcome is the event indicator and data carry follow-up time.",
    )
    horizon_days: float | None = Field(
        default=None,
        gt=0.0,
        exclude_if=lambda v: v is None,
        description="Survival only: administrative follow-up horizon in days (later events are censored).",
    )
    n_pool: int = Field(gt=0, description="Revealable patients in this world.")
    features: tuple[FeatureMeta, ...] = Field(min_length=1)
    strata: tuple[str, ...] = ("all",)
    stratum_sizes: dict[str, int] = Field(
        default_factory=dict,
        description="Patients per stratum (truth-blind; recruiting beyond a stratum's size is refused).",
    )
    prices: PriceList
    budget: float = Field(
        ge=0.0, description="Truth-blind budget: the full revealable pool at published prices."
    )
    name_visibility: NameVisibility = NameVisibility.FAKE
    premise: str = (
        "Here is a cohort with an outcome. Which measurements drive it? Return an ordered list, "
        "most likely first, or an empty list if nothing can be found."
    )

    @field_validator("features")
    @classmethod
    def _unique_features(cls, value: tuple[FeatureMeta, ...]) -> tuple[FeatureMeta, ...]:
        ids = [item.feature_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("feature identifiers must be unique")
        return value

    @model_validator(mode="after")
    def _horizon_matches_outcome(self) -> WorldCard:
        if (self.horizon_days is not None) != (self.outcome_type == "survival"):
            raise ValueError("survival cards (and only they) carry a follow-up horizon")
        return self

    def feature_ids(self) -> tuple[str, ...]:
        return tuple(item.feature_id for item in self.features)


# --------------------------------------------------------------------------- actions


class Reset(Frozen):
    kind: Literal["reset"] = "reset"
    request_id: RequestId
    world_id: WorldId


class Recruit(Frozen):
    kind: Literal["recruit"] = "recruit"
    request_id: RequestId
    count: int = Field(gt=0, le=100_000)
    stratum: str = "all"


class Assay(Frozen):
    kind: Literal["assay"] = "assay"
    request_id: RequestId
    feature_ids: tuple[FeatureId, ...] = Field(min_length=1)


class Submit(Frozen):
    kind: Literal["submit"] = "submit"
    request_id: RequestId
    ranking: tuple[FeatureId, ...] = Field(max_length=100_000)

    @field_validator("ranking")
    @classmethod
    def _no_repeats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("a ranking may not repeat a feature")
        return value


Action = Annotated[Reset | Recruit | Assay | Submit, Field(discriminator="kind")]


class ActionEnvelope(Frozen):
    """Wire wrapper so a single endpoint can accept any action."""

    action: Action


# --------------------------------------------------------------------------- observations


class EpisodeStatus(StrEnum):
    ACTIVE = "active"
    SUBMITTED = "submitted"


class RevealedData(Frozen):
    """Columnar revealed data. Missing cells are ``None``.

    Survival worlds: ``outcome`` is the event indicator (1 = event observed) and ``time`` the
    follow-up in days (to the event or to censoring). ``time`` is omitted for binary worlds.
    """

    patient_ids: tuple[str, ...]
    outcome: tuple[int, ...]
    stratum: tuple[str, ...]
    columns: dict[str, tuple[float | None, ...]]
    time: tuple[float, ...] | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def _aligned(self) -> RevealedData:
        n = len(self.patient_ids)
        if (
            len(self.outcome) != n
            or len(self.stratum) != n
            or (self.time is not None and len(self.time) != n)
        ):
            raise ValueError("revealed rows are misaligned")
        for name, values in self.columns.items():
            if len(values) != n:
                raise ValueError(f"column {name} is misaligned")
        return self


class Observation(Frozen):
    interface_version: Literal["1.0"] = INTERFACE_VERSION
    world_id: WorldId
    mode: Mode
    step: int = Field(ge=0)
    status: EpisodeStatus
    budget: float
    spent: float = Field(ge=0.0)
    available_actions: tuple[Literal["recruit", "assay", "submit"], ...]
    revealed: RevealedData


class ErrorCode(StrEnum):
    UNKNOWN_WORLD = "unknown_world"
    UNKNOWN_FEATURE = "unknown_feature"
    UNKNOWN_STRATUM = "unknown_stratum"
    OVER_BUDGET = "over_budget"
    ACTION_NOT_AVAILABLE = "action_not_available"
    EPISODE_CLOSED = "episode_closed"
    REQUEST_CONFLICT = "request_conflict"
    CAP_EXCEEDED = "cap_exceeded"
    SCORECARD_CLOSED = "scorecard_closed"
    INVALID_PAYLOAD = "invalid_payload"


class ArenaErrorPayload(Frozen):
    code: ErrorCode
    message: str


# --------------------------------------------------------------------------- answer keys (private data)


class GroupLabel(StrEnum):
    RECOVERABLE = "recoverable"
    NEUTRAL = "neutral"


class CreditRule(StrEnum):
    """How a true group's parts earn credit."""

    SINGLE = "single"  # one part, credited by any member of its equivalence set
    JOINT = "joint"  # interaction-like: every part must be credited, else nothing
    WEIGHTED_COVERAGE = "weighted_coverage"  # module: credit = covered share of absolute weights


class TruthPart(Frozen):
    """One scored slot of a true group: its representative feature and credit set."""

    true_feature: str = Field(
        description="The generating feature or, for a hidden cause, its best observed proxy."
    )
    equivalence_set: tuple[str, ...] = Field(min_length=1)
    exact_recoverable: bool
    weight: float = Field(
        default=1.0, gt=0.0, description="Absolute generating weight (used by weighted coverage)."
    )

    @model_validator(mode="after")
    def _contains_truth(self) -> TruthPart:
        if self.true_feature not in self.equivalence_set:
            raise ValueError("the equivalence set must contain the true feature")
        return self


class TrueGroup(Frozen):
    group_id: str
    role: str
    label: GroupLabel
    credit_rule: CreditRule = CreditRule.SINGLE
    underdetermined: bool = Field(
        default=False,
        description="Certified as not recoverable from the data (scored as neutral, only restraint counts).",
    )
    parts: tuple[TruthPart, ...] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def _rule_matches_parts(self) -> TrueGroup:
        if self.credit_rule is CreditRule.SINGLE and len(self.parts) != 1:
            raise ValueError("a single-credit group has exactly one part")
        if self.credit_rule is not CreditRule.SINGLE and len(self.parts) < 2:
            raise ValueError("joint and weighted-coverage groups have at least two parts")
        if self.underdetermined and self.label is not GroupLabel.NEUTRAL:
            raise ValueError("an underdetermined group cannot be recoverable")
        return self


class AnswerKey(Frozen):
    """Scorer-side truth for one world. Never revealed outside public-train."""

    interface_version: Literal["1.0"] = INTERFACE_VERSION
    world_id: WorldId
    difficulty_tier: int = Field(ge=0)
    groups: tuple[TrueGroup, ...]
    reject_set: tuple[str, ...] = ()
    clusters: dict[str, int]
    strata: dict[str, str] = Field(
        default_factory=dict,
        description="Truth-independent feature strata (data type x correlation bin) for matched chance.",
    )
    reference_cost: float = Field(ge=0.0)
    detection_threshold: float = Field(
        gt=0.0, description="Per-world Monte Carlo null max-statistic |z| threshold."
    )
    oracle_version: str

    @property
    def recoverable(self) -> tuple[TrueGroup, ...]:
        return tuple(g for g in self.groups if g.label is GroupLabel.RECOVERABLE)

    @property
    def depth(self) -> int:
        """R: the number of scored slots (an interaction group has two)."""
        return sum(len(g.parts) for g in self.recoverable)

    @property
    def is_null(self) -> bool:
        return self.depth == 0


# --------------------------------------------------------------------------- scores and scorecards


class WorldScore(Frozen):
    world_id: WorldId
    difficulty_tier: int
    is_null: bool
    find: float = Field(
        ge=0.0, le=1.0, description="Chance-normalised recovery q_w, clipped to [0, 1] (0 for null worlds)."
    )
    find_signed: float = Field(
        le=1.0,
        description="Unclipped chance-normalised recovery; zero in expectation for any outcome-blind list (gates, intervals).",
    )
    find_exact: float = Field(ge=0.0, le=1.0)
    raw_recovery: float = Field(ge=0.0, le=1.0)
    chance_recovery: float = Field(ge=0.0, le=1.0)
    restrained: bool = Field(description="Empty after neutral removal on a null world.")
    abstained: bool = Field(description="Submitted an empty list (after neutral removal).")
    leaked: bool
    spent: float = Field(ge=0.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    listed: int = Field(ge=0)
    tokens: int | None = Field(
        default=None, ge=0, description="LLM tokens spent on this world (standard harness)."
    )
    cost_usd: float | None = Field(default=None, ge=0.0)


class TierSummary(Frozen):
    """Per-tier summary; ``None`` where a metric is undefined (a tier lacking signal or null worlds)."""

    difficulty_tier: int
    n_signal: int
    n_null: int
    find: float | None
    restraint: float | None
    discovery_score: float | None


class Interval(Frozen):
    low: float | None
    high: float | None


class AlignmentDiagnostics(Frozen):
    analysis_regret: float
    acquisition_gap: float


class Scorecard(Frozen):
    interface_version: Literal["1.0"] = INTERFACE_VERSION
    scorecard_id: str
    track: Literal["standard", "open", "reference"]
    agent: str
    model: str | None = Field(default=None, description="Model identifier for LLM agents (standard track).")
    harness: str | None = Field(default=None, description="Harness name and version that produced the run.")
    versions: dict[str, str] = Field(default_factory=dict, description="Engine, scorer and oracle versions.")
    pool_commitment: str | None = Field(
        default=None, description="Committed manifest hash of the pool drawn from."
    )
    tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    tier: Tier
    n_worlds: int
    discovery_score: float | None = Field(
        description="Displayed headline: Find x max(0, Restraint); None when undefined."
    )
    discovery_score_unfloored: float | None
    interval: Interval
    find: float | None
    find_signed: float | None = Field(
        description="Mean unclipped Find over signal worlds (statistics and gates)."
    )
    restraint: float | None = Field(description="Chance-corrected (Youden's J), unfloored.")
    strict_discovery_score: float | None
    leak_rate: float
    abstention_on_signal: float | None
    restraint_on_null: float | None
    mean_data_cost: float
    per_tier: tuple[TierSummary, ...]
    alignment: AlignmentDiagnostics | None = None
    worlds: tuple[WorldScore, ...] = Field(
        default=(), description="Per-world results, populated only for the public-train tier."
    )
    tags: tuple[str, ...] = ()


# --------------------------------------------------------------------------- traces


class TraceEvent(Frozen):
    interface_version: Literal["1.0"] = INTERFACE_VERSION
    scorecard_id: str | None = Field(
        default=None, description="Scorecard (session scope) the event belongs to."
    )
    world_id: WorldId
    step: int = Field(ge=0)
    action_kind: Literal["reset", "recruit", "assay", "submit"]
    request_id: RequestId
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spent: float = Field(ge=0.0)
    action_json: str | None = Field(
        default=None, description="Exact action payload, so the episode can be replayed."
    )
    tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
