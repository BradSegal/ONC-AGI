"""Versioned HTTP interface: scorecards, actions and state over JSON.

Endpoints (``X-Arena-Key`` header identifies the caller for caps)::

    GET  /v1/health
    POST /v1/scorecards                                   open (fresh draw)
    POST /v1/scorecards/{sid}/worlds/{wid}/actions        apply one action
    GET  /v1/scorecards/{sid}/worlds/{wid}                current state (resume)
    POST /v1/scorecards/{sid}/close                       aggregate scorecard

Answer keys never leave the server; eval scorecards expose aggregates only.
"""

from __future__ import annotations

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import (
    INTERFACE_VERSION,
    ActionEnvelope,
    ArenaErrorPayload,
    ErrorCode,
    Observation,
    Scorecard,
    Tier,
    WorldCard,
)
from onc_agi.services.scorecards import ScorecardService

_STATUS = {
    ErrorCode.UNKNOWN_WORLD: 404,
    ErrorCode.UNKNOWN_FEATURE: 422,
    ErrorCode.UNKNOWN_STRATUM: 422,
    ErrorCode.OVER_BUDGET: 402,
    ErrorCode.ACTION_NOT_AVAILABLE: 409,
    ErrorCode.EPISODE_CLOSED: 409,
    ErrorCode.REQUEST_CONFLICT: 409,
    ErrorCode.CAP_EXCEEDED: 429,
    ErrorCode.SCORECARD_CLOSED: 409,
    ErrorCode.INVALID_PAYLOAD: 400,
}


class OpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str = Field(min_length=1, max_length=96)
    track: str = "open"
    tier: Tier
    n_worlds: int = Field(gt=0, le=10_000)
    tags: tuple[str, ...] = ()


class OpenResponse(BaseModel):
    interface_version: str = INTERFACE_VERSION
    scorecard_id: str
    cards: tuple[WorldCard, ...]


def create_app(service: ScorecardService) -> FastAPI:
    app = FastAPI(title="ONC-AGI", version=INTERFACE_VERSION)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {"loc": (), "msg": "invalid request"}
        where = ".".join(str(part) for part in first.get("loc", ()))
        payload = ArenaErrorPayload(
            code=ErrorCode.INVALID_PAYLOAD, message=f"{where}: {first.get('msg', 'invalid')}"
        )
        return JSONResponse(
            status_code=_STATUS[ErrorCode.INVALID_PAYLOAD], content=payload.model_dump(mode="json")
        )

    @app.exception_handler(ArenaError)
    async def _arena_error(_: Request, exc: ArenaError) -> JSONResponse:
        return JSONResponse(status_code=_STATUS[exc.code], content=exc.payload().model_dump(mode="json"))

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "interface_version": INTERFACE_VERSION}

    @app.post("/v1/scorecards")
    def open_scorecard(body: OpenRequest, x_arena_key: str = Header(min_length=8)) -> OpenResponse:
        sid, cards = service.open(
            x_arena_key,
            agent=body.agent,
            track=body.track,
            tier=body.tier,
            n_worlds=body.n_worlds,
            tags=body.tags,
        )
        return OpenResponse(scorecard_id=sid, cards=cards)

    @app.post("/v1/scorecards/{sid}/worlds/{wid}/actions")
    def act(sid: str, wid: str, body: ActionEnvelope) -> Observation:
        view = service.act(sid, wid, body.action)
        return view.to_observation(service.episode(sid, wid).world.patient_ids)

    @app.get("/v1/scorecards/{sid}/worlds/{wid}")
    def state(sid: str, wid: str) -> Observation:
        episode = service.episode(sid, wid)
        return episode.view().to_observation(episode.world.patient_ids)

    @app.post("/v1/scorecards/{sid}/close")
    def close(sid: str) -> Scorecard:
        return service.close(sid)

    return app
