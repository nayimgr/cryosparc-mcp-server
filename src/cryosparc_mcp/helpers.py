"""Shared helper utilities."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "item"):  # numpy scalar
            return o.item()
        if hasattr(o, "tolist"):  # numpy array
            return o.tolist()
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def to_json(obj: Any, indent: int = 2) -> str:
    """Serialise to JSON, handling numpy types and datetimes."""
    return json.dumps(obj, cls=_Encoder, indent=indent, ensure_ascii=False)


def fmt_error(exc: Exception) -> str:
    """Return a user-friendly error string."""
    msg = str(exc)
    cls = type(exc).__name__
    if "404" in msg or "not found" in msg.lower():
        return f"Error ({cls}): Resource not found — double-check the UID. {msg}"
    if "403" in msg or "permission" in msg.lower():
        return f"Error ({cls}): Permission denied — verify your credentials. {msg}"
    if "timeout" in msg.lower():
        return f"Error ({cls}): Request timed out — CryoSPARC may be overloaded. {msg}"
    return f"Error ({cls}): {msg}"
