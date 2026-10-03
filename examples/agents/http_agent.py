"""Play any Agent through the versioned HTTP interface - the path a custom harness uses.

By default this starts the server in-process over the toy worlds. To use a running
server instead, start one and pass its URL::

    uv run onc-agi serve --store src/onc_agi/fixtures/store --ledger /tmp/onc-agi-ledger.json
    uv run python examples/agents/http_agent.py http://127.0.0.1:8787
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from onc_agi.adapters.cli import fixture_store
from onc_agi.adapters.client import ArenaClient
from onc_agi.core.schema import Scorecard, Tier
from onc_agi.services.kit import PipelineAgent
from pipeline_agent import bonferroni_t  # type: ignore[import-not-found]

KEY = "my-issued-key-0001"  # public-eval and private servers issue keys; the local server accepts any


def local_client() -> ArenaClient:
    from fastapi.testclient import TestClient
    from onc_agi.adapters.http import create_app
    from onc_agi.infra.bundles import FileWorldStore
    from onc_agi.infra.ledger import JsonLedger
    from onc_agi.services.scorecards import ScorecardService

    ledger = Path(tempfile.mkdtemp()) / "ledger.json"
    app = create_app(ScorecardService(FileWorldStore(fixture_store()), JsonLedger(ledger)))
    return ArenaClient("http://testserver", KEY, client=TestClient(app))


def main(url: str | None = None) -> Scorecard:
    client = ArenaClient(url, KEY) if url else local_client()
    agent = PipelineAgent("bonferroni-t-http", bonferroni_t)
    scorecard_id, cards = client.open(agent.name, Tier.PUBLIC_TRAIN, n_worlds=20)
    for card in cards:
        client.play(agent, scorecard_id, card)  # resumable: client.state(scorecard_id, world_id)
    scorecard = client.close(scorecard_id)
    print(scorecard.model_dump_json(indent=1, exclude={"worlds"}))
    return scorecard


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
