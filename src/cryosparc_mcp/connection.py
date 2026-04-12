"""CryoSPARC connection manager — optimised for cryosparc-tools v4.7.1.

Authentication uses host + base_port + email + password + license (v4.7 format).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# Module-level singleton
_client = None


def _load_env() -> None:
    """Load .env file if it exists."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent.parent / ".env"]
    for p in candidates:
        if p.is_file():
            load_dotenv(p, override=False)
            break


def get_client():
    """Return the shared CryoSPARC client, creating it on first call.

    Supports both v4.7.1 (host/base_port) and v5.0+ (base_url) APIs.

    v4.7.1 environment variables (recommended):
      CRYOSPARC_HOST        — hostname, e.g. maws06.mpie.de
      CRYOSPARC_BASE_PORT   — base port, e.g. 61000
      CRYOSPARC_EMAIL       — login email
      CRYOSPARC_PASSWORD    — login password
      CRYOSPARC_LICENSE     — license ID
      CRYOSPARC_TIMEOUT     — optional timeout in seconds (default 300)

    v5.0+ environment variables (fallback):
      CRYOSPARC_BASE_URL    — full URL, e.g. http://localhost:39000
      CRYOSPARC_EMAIL       — login email
      CRYOSPARC_PASSWORD    — login password
    """
    global _client
    if _client is not None:
        return _client

    _load_env()

    from cryosparc.tools import CryoSPARC

    host      = os.getenv("CRYOSPARC_HOST", "").strip()
    base_port = os.getenv("CRYOSPARC_BASE_PORT", "").strip()
    base_url  = os.getenv("CRYOSPARC_BASE_URL", "http://localhost:39000")
    email     = os.getenv("CRYOSPARC_EMAIL", "").strip()
    password  = os.getenv("CRYOSPARC_PASSWORD", "").strip()
    license_id = os.getenv("CRYOSPARC_LICENSE", "").strip()
    timeout   = int(os.getenv("CRYOSPARC_TIMEOUT", "300"))

    try:
        # v4.7.1 format — host + base_port + email + password + license
        if host and base_port and email and password:
            _client = CryoSPARC(
                host=host,
                base_port=int(base_port),
                email=email,
                password=password,
                license=license_id,
                timeout=timeout,
            )
        # v5.0+ format — base_url + email + password
        elif email and password:
            _client = CryoSPARC(
                base_url=base_url,
                email=email,
                password=password,
                timeout=timeout,
            )
        # v5.0 token-based (no credentials needed)
        else:
            _client = CryoSPARC(
                base_url=base_url,
                timeout=timeout,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to CryoSPARC: {exc}\n"
            "Check your .env file — for v4.7.1 you need:\n"
            "  CRYOSPARC_HOST, CRYOSPARC_BASE_PORT, CRYOSPARC_EMAIL,\n"
            "  CRYOSPARC_PASSWORD, CRYOSPARC_LICENSE"
        ) from exc

    return _client


def reset_client() -> None:
    """Drop the cached client (for reconnect scenarios)."""
    global _client
    _client = None
