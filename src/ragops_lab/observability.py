"""Lightweight structured observability helpers."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list[JsonScalar] | dict[str, JsonScalar]

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
LOGGER = logging.getLogger("rag_quality_engine.observability")
LOGGER.setLevel(logging.INFO)


def resolve_request_id(candidate: str | None) -> str:
    """Return a validated caller request id or generate a new one."""
    if candidate is not None:
        normalized = candidate.strip()
        if REQUEST_ID_PATTERN.fullmatch(normalized):
            return normalized
    return f"req-{uuid.uuid4().hex}"


def log_event(event: str, **fields: JsonValue) -> None:
    """Emit one JSON structured log event."""
    payload: dict[str, JsonValue] = {"event": event, **fields}
    LOGGER.info(json.dumps(payload, sort_keys=True, default=str))

