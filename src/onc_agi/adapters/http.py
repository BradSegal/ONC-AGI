"""Versioned HTTP interface: scorecards, actions and state over JSON.

Endpoints (every scorecard endpoint needs the ``X-Arena-Key`` header)::

    GET  /v1/health
    POST /v1/scorecards                                   open (fresh draw)
    POST /v1/scorecards/{sid}/worlds/{wid}/actions        apply one action
    GET  /v1/scorecards/{sid}/worlds/{wid}                current state (resume)
    POST /v1/scorecards/{sid}/close                       aggregate scorecard
    GET  /v1/scorecards/{sid}                             the closed scorecard on record

This adapter is the trust boundary: the key identifies the caller for caps and
owns the scorecard it opened. Any other key gets the same error as an unknown id.
Answer keys never leave the server; eval scorecards expose aggregates only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from onc_agi.core.errors import ArenaError
from onc_agi.core.schema import (
    INTERFACE_VERSION,
    ActionEnvelope,
    ArenaErrorPayload,
    ErrorCode,
    Mode,
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
    n_worlds: int | None = Field(default=None, gt=0, le=10_000)
    world_ids: tuple[str, ...] | None = Field(
        default=None, min_length=1, max_length=10_000, description="Named worlds (public train only)."
    )
    mode: Mode | None = Field(default=None, description="Restrict the worlds to one mode.")
    tags: tuple[str, ...] = ()


class WorldList(BaseModel):
    interface_version: str = INTERFACE_VERSION
    tier: Tier
    cards: tuple[WorldCard, ...]


class OpenResponse(BaseModel):
    interface_version: str = INTERFACE_VERSION
    scorecard_id: str
    cards: tuple[WorldCard, ...]


def create_app(service: ScorecardService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="ONC-AGI", version=INTERFACE_VERSION, lifespan=lifespan)
    # observations carry every revealed value (hundreds of KiB); they compress about tenfold
    app.add_middleware(GZipMiddleware, minimum_size=2048)

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
            mode=body.mode,
            world_ids=body.world_ids,
        )
        return OpenResponse(scorecard_id=sid, cards=cards)

    @app.get("/v1/worlds")
    def list_worlds(tier: Tier = Tier.PUBLIC_TRAIN, x_arena_key: str = Header(min_length=8)) -> WorldList:
        return WorldList(tier=tier, cards=service.list_worlds(tier))

    @app.post("/v1/scorecards/{sid}/worlds/{wid}/actions")
    def act(sid: str, wid: str, body: ActionEnvelope, x_arena_key: str = Header(min_length=8)) -> Observation:
        view = service.act(sid, wid, body.action, api_key=x_arena_key)
        return view.to_observation(service.store.world(wid).patient_ids)

    @app.get("/v1/scorecards/{sid}/worlds/{wid}")
    def state(sid: str, wid: str, x_arena_key: str = Header(min_length=8)) -> Observation:
        view = service.view(sid, wid, api_key=x_arena_key)
        return view.to_observation(service.store.world(wid).patient_ids)

    @app.post("/v1/scorecards/{sid}/close")
    def close(sid: str, x_arena_key: str = Header(min_length=8)) -> Scorecard:
        return service.close(sid, api_key=x_arena_key)

    @app.get("/v1/scorecards/{sid}")
    def scorecard(sid: str, x_arena_key: str = Header(min_length=8)) -> Scorecard:
        return service.scorecard(sid, api_key=x_arena_key)

    return app
