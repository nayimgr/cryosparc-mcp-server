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

    Authentication priority:
      1. Token-based (from `python -m cryosparc.tools login`) — v5 recommended
      2. Email/password from environment variables — legacy fallback

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

    base_url = os.getenv("CRYOSPARC_BASE_URL", "http://localhost:39000")
    email = os.getenv("CRYOSPARC_EMAIL", "").strip()
    password = os.getenv("CRYOSPARC_PASSWORD", "").strip()
    timeout = int(os.getenv("CRYOSPARC_TIMEOUT", "300"))

    try:
        if email and password:
            # Legacy: email/password auth
            _client = CryoSPARC(
                base_url=base_url,
                email=email,
                password=password,
                timeout=timeout,
            )
        else:
            # v5 recommended: token-based auth (from `cryosparc.tools login`)
            _client = CryoSPARC(base_url, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to CryoSPARC at {base_url}: {exc}\n"
            "Please verify:\n"
            "  1. CryoSPARC is running and accessible at that URL\n"
            "  2. You have logged in with: python -m cryosparc.tools login --url <URL>\n"
            "     OR set CRYOSPARC_EMAIL and CRYOSPARC_PASSWORD environment variables"
        ) from exc

    return _client


def reset_client() -> None:
    """Drop the cached client (for reconnect scenarios)."""
    global _client
    _client = None
