"""Tests for the CryoSPARC MCP server.

These run entirely offline against a mocked CryoSPARC client, and cover the
areas that were previously broken:

* the cryosparc-tools v5 API surface actually used (fields(), params, outputs)
* tool failures surfacing as real MCP errors (isError) rather than success text
* results interpretation (class distributions, CTF/motion stats)

Run:  pip install -e '.[dev,report]' && pytest -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cryosparc_mcp import helpers  # noqa: E402
from cryosparc_mcp.helpers import (  # noqa: E402
    col_stats,
    dataset_fields,
    fmt_error,
    params_to_dict,
    tool_error,
)


# --------------------------------------------------------------------------
# Fakes mimicking the cryosparc-tools v5 surface
# --------------------------------------------------------------------------

class FakeDataset:
    """Mimics cryosparc.dataset.Dataset: fields(), rows(), len(), __getitem__."""

    def __init__(self, columns: dict):
        self._cols = columns
        self._n = len(next(iter(columns.values()))) if columns else 0

    def fields(self):
        return list(self._cols)

    def rows(self):
        return [{k: v[i] for k, v in self._cols.items()} for i in range(self._n)]

    def __len__(self):
        return self._n

    def __getitem__(self, key):
        return self._cols[key]


class FakeParams:
    """Mimics a pydantic params model (attribute access + model_dump)."""

    def __init__(self, values: dict):
        self._values = values
        for k, v in values.items():
            setattr(self, k, v)

    def model_dump(self):
        return dict(self._values)


class FakeJob:
    def __init__(self, uid="J42", type="class_2D", status="completed",
                 title="Class 2D", outputs=None, datasets=None, params=None):
        self.uid = uid
        self.type = type
        self.status = status
        self.title = title
        self.desc = ""
        self._outputs = outputs or {}
        self._datasets = datasets or {}
        self.params = FakeParams(params or {})

    @property
    def outputs(self):
        return self._outputs

    def load_output(self, name, slots=None):
        if name not in self._datasets:
            raise KeyError(f"no output {name}")
        return self._datasets[name]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_dataset_fields_uses_fields_not_columns():
    """`Dataset.columns()` does not exist in cryosparc-tools 4.x or 5.x."""
    ds = FakeDataset({"a": [1, 2], "b": [3, 4]})
    assert dataset_fields(ds) == ["a", "b"]
    assert not hasattr(ds, "columns")


def test_params_to_dict_handles_pydantic_params():
    p = FakeParams({"class2D_K": 50, "box_size_pix": 256})
    assert params_to_dict(p) == {"class2D_K": 50, "box_size_pix": 256}
    assert params_to_dict(None) == {}
    assert params_to_dict({"a": 1}) == {"a": 1}


def test_col_stats():
    pytest.importorskip("numpy")
    ds = FakeDataset({"x": [1.0, 2.0, 3.0, float("nan")]})
    st = col_stats(ds, "x")
    assert st["n"] == 3 and st["min"] == 1.0 and st["max"] == 3.0
    assert col_stats(ds, "missing") is None


def test_fmt_error_gives_version_hint_for_attributeerror():
    msg = fmt_error(AttributeError("'CryoSPARC' object has no attribute 'find_projects'"))
    assert "cryosparc-tools" in msg and ">= 5.0" in msg


def test_tool_error_is_a_tool_error():
    from mcp.server.fastmcp.exceptions import ToolError
    assert isinstance(tool_error(ValueError("boom")), ToolError)


# --------------------------------------------------------------------------
# results interpretation
# --------------------------------------------------------------------------

def test_summarise_2d_classification_class_distribution():
    pytest.importorskip("numpy")
    from cryosparc_mcp.tools_results import _summarise_job

    particles = FakeDataset({
        "alignments2D/class": [0, 0, 0, 1, 1, 2],
        "pick_stats/ncc_score": [0.5, 0.6, 0.7, 0.8, 0.9, 0.4],
    })
    job = FakeJob(outputs={"particles": {}, "class_averages": {}},
                  datasets={"particles": particles},
                  params={"class2D_K": 3})
    out = _summarise_job(job, max_classes=100)

    assert out["metrics"]["num_particles"] == 6
    dist = out["metrics"]["class_distribution_2D"]
    assert dist["total_particles"] == 6 and dist["num_classes"] == 3
    # ranked by particle count, descending
    assert [c["class"] for c in dist["classes"]] == [0, 1, 2]
    assert dist["classes"][0]["particles"] == 3
    assert dist["classes"][0]["fraction"] == pytest.approx(0.5)
    assert out["metrics"]["pick_ncc_score"]["n"] == 6
    assert out["key_params"]["class2D_K"] == 3


def test_summarise_ctf_job_stats():
    pytest.importorskip("numpy")
    from cryosparc_mcp.tools_results import _summarise_job

    exposures = FakeDataset({
        "ctf/df1_A": [10000.0, 12000.0],
        "ctf/df2_A": [10500.0, 12200.0],
        "ctf/ctf_fit_to_A": [3.5, 4.0],
    })
    job = FakeJob(uid="J3", type="patch_ctf_estimation_multi", title="CTF",
                  outputs={"exposures": {}}, datasets={"exposures": exposures})
    out = _summarise_job(job, max_classes=100)
    m = out["metrics"]
    assert m["num_exposures"] == 2
    assert m["ctf_fit_resolution_A"]["min"] == 3.5
    assert m["astigmatism_A"]["n"] == 2
    assert m["astigmatism_A"]["min"] == pytest.approx(200.0)


def test_summarise_reports_no_metrics_gracefully():
    from cryosparc_mcp.tools_results import _summarise_job
    job = FakeJob(uid="J9", type="import_movies", outputs={}, datasets={})
    out = _summarise_job(job, max_classes=10)
    assert out["metrics"] == {}
    assert "note" in out


# --------------------------------------------------------------------------
# MCP error semantics — failures must be isError, not success text
# --------------------------------------------------------------------------

@pytest.mark.anyio
async def test_tool_failure_is_reported_as_mcp_error():
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session

    srv = FastMCP("t")

    @srv.tool(name="boom")
    async def boom() -> str:
        raise tool_error(AttributeError("'CryoSPARC' object has no attribute 'find_projects'"))

    async with create_connected_server_and_client_session(srv._mcp_server) as client:
        result = await client.call_tool("boom", {})
        assert result.isError is True, "tool failure must set isError"
        assert "cryosparc-tools" in result.content[0].text


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --------------------------------------------------------------------------
# connection configuration
# --------------------------------------------------------------------------

def test_connection_requires_configuration(monkeypatch):
    from cryosparc_mcp import connection

    connection.reset_client()
    monkeypatch.setattr(connection, "_load_env", lambda: None)
    for var in ("CRYOSPARC_HOST", "CRYOSPARC_BASE_PORT", "CRYOSPARC_BASE_URL",
                "CRYOSPARC_EMAIL", "CRYOSPARC_PASSWORD", "CRYOSPARC_LICENSE"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(RuntimeError) as exc:
        connection.get_client()
    assert "CRYOSPARC_HOST" in str(exc.value)
    connection.reset_client()


def test_connection_passes_license_and_host(monkeypatch):
    """v5 signature: CryoSPARC(base_url=None, *, host, base_port, email, password, license, timeout)."""
    from cryosparc_mcp import connection

    connection.reset_client()
    monkeypatch.setattr(connection, "_load_env", lambda: None)
    monkeypatch.setenv("CRYOSPARC_HOST", "myhost")
    monkeypatch.setenv("CRYOSPARC_BASE_PORT", "39000")
    monkeypatch.setenv("CRYOSPARC_EMAIL", "me@lab.org")
    monkeypatch.setenv("CRYOSPARC_PASSWORD", "pw")
    monkeypatch.setenv("CRYOSPARC_LICENSE", "LIC-123")

    captured = {}

    class FakeCS:
        def __init__(self, base_url=None, **kwargs):
            captured["base_url"] = base_url
            captured.update(kwargs)

    import cryosparc.tools as ct
    monkeypatch.setattr(ct, "CryoSPARC", FakeCS)

    connection.get_client()
    assert captured["host"] == "myhost"
    assert captured["base_port"] == 39000
    assert captured["license"] == "LIC-123"
    assert captured["email"] == "me@lab.org"
    connection.reset_client()


# --------------------------------------------------------------------------
# server wiring
# --------------------------------------------------------------------------

def test_server_builds_and_registers_tools():
    import asyncio

    from cryosparc_mcp.server import create_server

    srv = create_server()
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    for expected in ("cryosparc_list_projects", "cryosparc_create_job",
                     "cryosparc_job_results", "cryosparc_render_class_averages"):
        assert expected in names
    assert len(names) >= 40


def test_http_host_port_go_through_settings():
    """FastMCP.run() accepts only (transport, mount_path) — host/port must be set
    on .settings, otherwise HTTP mode raises TypeError."""
    import inspect

    from mcp.server.fastmcp import FastMCP

    assert list(inspect.signature(FastMCP.run).parameters) == ["self", "transport", "mount_path"]
    srv = FastMCP("t")
    srv.settings.host = "127.0.0.1"
    srv.settings.port = 8299
    assert srv.settings.host == "127.0.0.1" and srv.settings.port == 8299
