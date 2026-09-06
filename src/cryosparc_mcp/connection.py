"""CryoSPARC connection manager — cryosparc-tools v5.

The v5 ``CryoSPARC`` constructor is::

    CryoSPARC(base_url=None, *, host=None, base_port=None,
              email=None, password=None, license=None, timeout=300)

Both connection styles are supported:

* host + base_port (+ email/password/license)  — the classic CryoSPARC form
* base_url (+ email/password)                  — e.g. http://localhost:39000

The license ID is passed through whenever it is set, since instances that
require it will reject the login without it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_client: Optional[Any] = None


def _load_env() -> None:
    """Load a .env file from the cwd or the project root, if present."""
    candidates = [Path.cwd() / ".env",
                  Path(__file__).resolve().parent.parent.parent / ".env"]
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)
            break


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_client() -> Any:
    """Return the shared CryoSPARC client, creating it on first use.

    Environment variables
    ---------------------
    CRYOSPARC_HOST       hostname, e.g. ``cryosparc.mylab.org``
    CRYOSPARC_BASE_PORT  base port, e.g. ``39000``
    CRYOSPARC_BASE_URL   alternative to host+port, e.g. ``http://localhost:39000``
    CRYOSPARC_EMAIL      login email
    CRYOSPARC_PASSWORD   login password
    CRYOSPARC_LICENSE    license ID (required by most instances)
    CRYOSPARC_TIMEOUT    request timeout in seconds (default 300)
    """
    global _client
    if _client is not None:
        return _client

    _load_env()

    try:
        from cryosparc.tools import CryoSPARC
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "cryosparc-tools is not installed. Install it with:\n"
            "  pip install 'cryosparc-tools>=5.0'"
        ) from exc

    host = _env("CRYOSPARC_HOST")
    base_port = _env("CRYOSPARC_BASE_PORT")
    base_url = _env("CRYOSPARC_BASE_URL")
    email = _env("CRYOSPARC_EMAIL")
    password = _env("CRYOSPARC_PASSWORD")
    license_id = _env("CRYOSPARC_LICENSE")
    try:
        timeout = int(_env("CRYOSPARC_TIMEOUT", "300") or 300)
    except ValueError:
        timeout = 300

    if not (host and base_port) and not base_url:
        raise RuntimeError(
            "CryoSPARC connection is not configured. Set either\n"
            "  CRYOSPARC_HOST + CRYOSPARC_BASE_PORT   (e.g. myhost + 39000)\n"
            "or\n"
            "  CRYOSPARC_BASE_URL                     (e.g. http://localhost:39000)\n"
            "plus CRYOSPARC_EMAIL, CRYOSPARC_PASSWORD and CRYOSPARC_LICENSE."
        )

    kwargs: dict = {"timeout": timeout}
    if email:
        kwargs["email"] = email
    if password:
        kwargs["password"] = password
    if license_id:
        kwargs["license"] = license_id

    if host and base_port:
        try:
            kwargs["base_port"] = int(base_port)
        except ValueError as exc:
            raise RuntimeError(
                f"CRYOSPARC_BASE_PORT must be an integer, got {base_port!r}"
            ) from exc
        kwargs["host"] = host
        described = f"{host}:{base_port}"
    else:
        described = base_url

    try:
        _client = CryoSPARC(base_url, **kwargs) if not (host and base_port) else CryoSPARC(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to CryoSPARC at {described}: {exc}\n"
            "Check CRYOSPARC_HOST / CRYOSPARC_BASE_PORT (or CRYOSPARC_BASE_URL), "
            "CRYOSPARC_EMAIL, CRYOSPARC_PASSWORD and CRYOSPARC_LICENSE."
        ) from exc

    return _client


def describe_client(cs: Any) -> str:
    """Human-readable description of where a client is pointed."""
    url = getattr(cs, "base_url", None)
    if url:
        return str(url)
    host = getattr(cs, "host", None)
    port = getattr(cs, "base_port", None)
    if host:
        return f"{host}:{port}" if port else str(host)
    return "unknown"


def reset_client() -> None:
    """Drop the cached client (for reconnect scenarios)."""
    global _client
    _client = None
