"""HTTP client for custom harnesses: runs any :class:`Agent` against a remote server."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import numpy as np

from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import (
    Action,
    ActionEnvelope,
    ArenaErrorPayload,
    Mode,
    Observation,
    Reset,
    Scorecard,
    Tier,
    WorldCard,
)
from onc_agi.services.engine import EpisodeView
from onc_agi.services.kit import MAX_STEPS, Agent


def view_from_observation(card: WorldCard, obs: Observation) -> EpisodeView:
    fids = card.feature_ids()
    n = len(obs.revealed.patient_ids)
    x = np.full((n, len(fids)), np.nan)
    measured = []
    for j, fid in enumerate(fids):
        column = obs.revealed.columns.get(fid)
        measured.append(column is not None)
        if column is not None:
            x[:, j] = [np.nan if v is None else v for v in column]
    return EpisodeView(
        world_id=obs.world_id,
        mode=obs.mode,
        step=obs.step,
        status=obs.status,
        budget=obs.budget,
        spent=obs.spent,
        available=tuple(obs.available_actions),
        rows=tuple(range(n)),
        outcome=np.asarray(obs.revealed.outcome, dtype=np.int64),
        stratum=obs.revealed.stratum,
        feature_ids=fids,
        x=x,
        measured=tuple(measured),
        time=None if obs.revealed.time is None else np.asarray(obs.revealed.time, dtype=np.float64),
    )


class ArenaClient:
    def __init__(self, base_url: str, api_key: str, *, client: httpx.Client | None = None) -> None:
        self.http = client or httpx.Client(base_url=base_url, timeout=120.0)
        self.headers = {"X-Arena-Key": api_key}

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code >= 400:
            payload = ArenaErrorPayload.model_validate(response.json())
            raise ArenaError(payload.code, payload.message)
        return response

    def open(
        self,
        agent: str,
        tier: Tier,
        n_worlds: int | None = None,
        *,
        track: str = "open",
        mode: Mode | None = None,
        world_ids: Sequence[str] | None = None,
        tags: Sequence[str] = (),
    ) -> tuple[str, list[WorldCard]]:
        """Open a scorecard of ``n_worlds`` drawn worlds, or of named public-train ``world_ids``."""
        body: dict[str, object] = {"agent": agent, "tier": tier.value, "track": track, "tags": list(tags)}
        if n_worlds is not None:
            body["n_worlds"] = n_worlds
        if world_ids is not None:
            body["world_ids"] = list(world_ids)
        if mode is not None:
            body["mode"] = mode.value
        data = self._check(self.http.post("/v1/scorecards", json=body, headers=self.headers)).json()
        return data["scorecard_id"], [WorldCard.model_validate(c) for c in data["cards"]]

    def worlds(self, tier: Tier = Tier.PUBLIC_TRAIN) -> list[WorldCard]:
        """Cards of every public-train world (eval tiers are never listed)."""
        r = self.http.get("/v1/worlds", params={"tier": tier.value}, headers=self.headers)
        return [WorldCard.model_validate(c) for c in self._check(r).json()["cards"]]

    def act(self, sid: str, wid: str, action: Action) -> Observation:
        body = ActionEnvelope(action=action).model_dump(mode="json")
        r = self.http.post(f"/v1/scorecards/{sid}/worlds/{wid}/actions", json=body, headers=self.headers)
        return Observation.model_validate(self._check(r).json())

    def state(self, sid: str, wid: str) -> Observation:
        r = self.http.get(f"/v1/scorecards/{sid}/worlds/{wid}", headers=self.headers)
        return Observation.model_validate(self._check(r).json())

    def close(self, sid: str) -> Scorecard:
        return Scorecard.model_validate(
            self._check(self.http.post(f"/v1/scorecards/{sid}/close", headers=self.headers)).json()
        )

    def scorecard(self, sid: str) -> Scorecard:
        """The server's record of a closed scorecard (only its owner may read it)."""
        r = self.http.get(f"/v1/scorecards/{sid}", headers=self.headers)
        return Scorecard.model_validate(self._check(r).json())

    def play(self, agent: Agent, sid: str, card: WorldCard, *, max_steps: int = MAX_STEPS) -> Observation:
        obs = self.act(sid, card.world_id, Reset(request_id=f"{agent.name}-reset", world_id=card.world_id))
        view = view_from_observation(card, obs)
        for _ in range(max_steps):
            if agent.is_done(view):
                return obs
            obs = self.act(sid, card.world_id, agent.choose_action(card, view))
            view = view_from_observation(card, obs)
        raise RuntimeError(f"agent {agent.name} did not submit within {max_steps} steps")
