"""Shared helper utilities for the CryoSPARC MCP server (cryosparc-tools v5)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from fastmcp.exceptions import ToolError

# --------------------------------------------------------------------------
# MCP Tasks extension (io.modelcontextprotocol/tasks, spec 2026-07-28)
# --------------------------------------------------------------------------
# Waiting on a CryoSPARC job can take hours. With the tasks extension the client
# gets a durable task handle and polls `tasks/get` instead of holding one request
# open. Requires the `fastmcp[tasks]` extra; without it the same tools still work
# as ordinary blocking calls.
try:  # pragma: no cover - depends on optional extra
    import fastmcp_tasks  # noqa: F401

    TASKS_AVAILABLE = True
except Exception:  # noqa: BLE001
    TASKS_AVAILABLE = False

TASK_MODE: Any = TASKS_AVAILABLE


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "model_dump"):  # pydantic model (cryosparc-tools v5)
            try:
                return o.model_dump(mode="json")
            except Exception:  # noqa: BLE001
                pass
        if hasattr(o, "item"):  # numpy scalar
            return o.item()
        if hasattr(o, "tolist"):  # numpy array
            return o.tolist()
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def to_json(obj: Any, indent: int = 2) -> str:
    """Serialise to JSON, handling pydantic models, numpy types and datetimes."""
    return json.dumps(obj, cls=_Encoder, indent=indent, ensure_ascii=False)


def jsonable(obj: Any) -> Any:
    """Return a JSON-safe copy of `obj` (plain dict/list/scalars).

    Tools return this instead of a JSON *string* so the MCP layer can emit
    `structuredContent` alongside an `outputSchema`, which an agent can read
    field-by-field rather than re-parsing prose. Pydantic models, numpy scalars
    and datetimes are normalised on the way through.
    """
    return json.loads(to_json(obj, indent=None))


def fmt_error(exc: Exception) -> str:
    """Build a user-friendly, actionable error message."""
    msg = str(exc)
    cls = type(exc).__name__
    low = msg.lower()
    if isinstance(exc, AttributeError):
        return (f"{cls}: {msg}. This usually means the installed cryosparc-tools version "
                f"does not match this server (it requires cryosparc-tools >= 5.0). "
                f"Run: pip install -U 'cryosparc-tools>=5.0'")
    if "404" in msg or "not found" in low:
        return f"{cls}: Resource not found — double-check the project/workspace/job UID. {msg}"
    if "403" in msg or "401" in msg or "permission" in low or "unauthor" in low:
        return f"{cls}: Permission denied — verify CRYOSPARC_EMAIL / CRYOSPARC_PASSWORD / CRYOSPARC_LICENSE. {msg}"
    if "timeout" in low or "timed out" in low:
        return f"{cls}: Request timed out — CryoSPARC may be busy. Raise CRYOSPARC_TIMEOUT. {msg}"
    if "connection" in low or "refused" in low:
        return (f"{cls}: Cannot reach CryoSPARC — check CRYOSPARC_HOST / CRYOSPARC_BASE_PORT "
                f"(or CRYOSPARC_BASE_URL) and that the instance is running. {msg}")
    return f"{cls}: {msg}"


def tool_error(exc: Exception) -> ToolError:
    """Wrap an exception in a ToolError.

    Raising this makes the MCP response a real error (isError=True) instead of a
    success payload that merely contains the word "Error" — which an agent
    cannot reliably distinguish from a valid result.
    """
    return ToolError(fmt_error(exc))


# --------------------------------------------------------------------------
# cryosparc-tools v5 adapters
# --------------------------------------------------------------------------

def dataset_fields(dset: Any) -> List[str]:
    """Return a dataset's column names.

    cryosparc-tools exposes `fields()`; there is no `columns()` in either the
    4.x or 5.x API, so this is the single place that knows the right call.
    """
    for attr in ("fields", "columns"):
        fn = getattr(dset, attr, None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:  # noqa: BLE001
                continue
    return []


def params_to_dict(params: Any) -> Dict[str, Any]:
    """Convert a v5 `job.params` object (pydantic) into a plain dict."""
    if params is None:
        return {}
    for attr in ("model_dump", "dict"):
        fn = getattr(params, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:  # noqa: BLE001
                continue
    if isinstance(params, dict):
        return dict(params)
    try:
        return dict(params)  # RootModel / mapping-like
    except Exception:  # noqa: BLE001
        return {k: v for k, v in vars(params).items() if not k.startswith("_")}


def col_stats(dset: Any, col: str) -> Optional[Dict[str, float]]:
    """count/min/mean/max for a numeric dataset column (None if unusable)."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        arr = np.asarray(dset[col], dtype="float64")
    except Exception:  # noqa: BLE001
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {"n": int(arr.size), "min": float(arr.min()),
            "mean": float(arr.mean()), "max": float(arr.max())}


def paginate(rows: Iterable[Any], offset: int, limit: int) -> List[Any]:
    from itertools import islice
    return list(islice(rows, offset, offset + limit))
