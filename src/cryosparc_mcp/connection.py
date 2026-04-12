"""CryoSPARC connection manager.

v5 authentication flow (from tools.cryosparc.com/intro.html):
  1. User runs: python -m cryosparc.tools login --url <URL>
     This saves a token locally that expires in two weeks.
  2. Scripts then use: CryoSPARC("<URL>")  — no email/password needed.

Legacy fallback:
  If the token is not available, email/password are passed directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

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

    Authentication priority:
      1. Email/password from environment variables (works with both versions)
      2. Token-based (from `python -m cryosparc.tools login`) — v5 only

    Returns:
        CryoSPARC: connected client instance

    Raises:
        RuntimeError: if connection fails
    """
    global _client
    if _client is not None:
        return _client

    _load_env()

    from cryosparc.tools import CryoSPARC

    # Support both v4.7.1 (host/base_port) and v5.0+ (base_url) formats
    base_url = os.getenv("CRYOSPARC_BASE_URL", "http://localhost:39000")
    host = os.getenv("CRYOSPARC_HOST", "").strip()
    base_port = os.getenv("CRYOSPARC_BASE_PORT", "").strip()

    email = os.getenv("CRYOSPARC_EMAIL", "").strip()
    password = os.getenv("CRYOSPARC_PASSWORD", "").strip()
    timeout = int(os.getenv("CRYOSPARC_TIMEOUT", "300"))
    license_id = os.getenv("CRYOSPARC_LICENSE", "15986a3e-b020-11f0-9649-436da5d3213b")

    try:
        # Try v4.7.1 format first (host + base_port + email/password)
        if host and base_port and email and password:
            _client = CryoSPARC(
                host=host,
                base_port=int(base_port),
                email=email,
                password=password,
                license=license_id,
                timeout=timeout,
            )
        # Fallback to v5.0+ format (base_url)
        elif email and password:
            _client = CryoSPARC(
                base_url=base_url,
                email=email,
                password=password,
                timeout=timeout,
            )
        # v5.0 token-based (no email/password needed)
        else:
            _client = CryoSPARC(
                base_url=base_url,
                timeout=timeout,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to CryoSPARC: {exc}\n"
            "Please verify environment variables:\n"
            "  For v4.7.1: CRYOSPARC_HOST, CRYOSPARC_BASE_PORT, CRYOSPARC_EMAIL, CRYOSPARC_PASSWORD, CRYOSPARC_LICENSE\n"
            "  For v5.0+: CRYOSPARC_BASE_URL and either token (login) or email/password"
        ) from exc

    return _client


def reset_client() -> None:
    """Drop the cached client (for reconnect scenarios)."""
    global _client
    _client = None
