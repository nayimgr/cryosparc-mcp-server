"""CryoSPARC MCP tools — all tools registered on the FastMCP server.

Every API call in this file follows patterns documented at:
  - https://tools.cryosparc.com/intro.html  (Getting Started)
  - https://tools.cryosparc.com/guides/jobs.html  (Working with Jobs)
  - https://tools.cryosparc.com/examples/custom-workflow.html  (Custom Workflow)
  - https://tools.cryosparc.com/api/cryosparc.tools.html  (CryoSPARC class)
  - https://tools.cryosparc.com/api/cryosparc.controllers.project.html  (ProjectController)
  - https://tools.cryosparc.com/api/cryosparc.controllers.job.html  (JobController / ExternalJobController)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

from .connection import get_client
from .helpers import fmt_error, to_json


# ====================================================================
# Pydantic input models
# ====================================================================

class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# --- Navigation ---

class TestConnectionIn(_Base):
    pass

class ListProjectsIn(_Base):
    pass

class FindProjectIn(_Base):
    project_uid: str = Field(..., description="Project UID, e.g. 'P3'", min_length=1)

class ListWorkspacesIn(_Base):
    project_uid: str = Field(..., min_length=1)

class ListJobsIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: Optional[str] = Field(default=None, description="Filter by workspace UID, e.g. 'W1'")
    type: Optional[str] = Field(default=None, description="Filter by job type, e.g. 'homo_refine_new'")

class FindJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., description="Job UID, e.g. 'J42'", min_length=1)

class ListJobTypesIn(_Base):
    category: Optional[str] = Field(default=None, description="Optional category filter, e.g. 'refinement'")

class GetLanesIn(_Base):
    pass

class GetTargetsIn(_Base):
    lane: Optional[str] = Field(default=None, description="Optional lane name filter")

# --- Job management ---

class CreateWorkspaceIn(_Base):
    project_uid: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=200)
    desc: Optional[str] = Field(default=None)

class CreateJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    job_type: str = Field(..., description="Job type, e.g. 'import_movies'. Use cryosparc_list_job_types to see all.", min_length=1)
    title: Optional[str] = Field(default="")
    desc: Optional[str] = Field(default="")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Parameter values, e.g. {'blob_paths': '/data/*.tif', 'psize_A': 1.0}")
    connections: Optional[Dict[str, Any]] = Field(default=None, description="Input connections: {input_name: (job_uid, output_name)} or {input_name: [(j1, o1), (j2, o2)]}")

class SetParamIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    param_name: str = Field(..., min_length=1)
    value: Any = Field(...)

class SetParamsBatchIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(...)

class ConnectIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., description="Target job UID", min_length=1)
    target_input: str = Field(..., description="Input slot name on target job, e.g. 'micrographs'", min_length=1)
    source_job_uid: str = Field(..., min_length=1)
    source_output: str = Field(..., description="Source output name, e.g. 'micrographs'", min_length=1)

class QueueJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    lane: Optional[str] = Field(default=None, description="Scheduler lane name. Omit to run on master.")

class JobRefIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)

class WaitJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    timeout: int = Field(default=600, ge=1, le=86400)
    error_on_incomplete: bool = Field(default=False, description="Raise error if job fails or is killed")

class WaitStatusIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    target_status: str = Field(..., description="Status to wait for, e.g. 'waiting', 'running', 'completed'", min_length=1)
    timeout: int = Field(default=300, ge=1, le=86400)

class InteractIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    action: str = Field(..., description="Interactive action name, e.g. 'get_fields_and_thresholds', 'set_thresholds', 'shutdown_interactive', 'finish'", min_length=1)
    data: Optional[Any] = Field(default=None, description="Data to pass to the interactive action (depends on action type)")

# --- Data ---

class DatasetSummaryIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., description="Output name, e.g. 'particles', 'micrographs', 'volume'", min_length=1)

class LoadOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None, description="Filter by result slots, e.g. ['location', 'ctf', 'blob']. Omit to load all.")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)

class ListFilesIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: Optional[str] = Field(default=None, description="Job UID to list files from. If omitted, lists project directory.")
    prefix: Optional[str] = Field(default=None, description="Subdirectory prefix")
    recursive: bool = Field(default=False)

class ListAssetsIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)

class GetJobSpecIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    spec_type: str = Field(default="params", description="'params', 'inputs', or 'outputs'")

# --- External jobs ---

class CreateExternalJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    title: Optional[str] = Field(default="")
    desc: Optional[str] = Field(default="")

class ExtConnectIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    target_input: str = Field(..., min_length=1)
    source_job_uid: str = Field(..., min_length=1)
    source_output: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None, description="Specific slots, e.g. ['micrograph_blob', 'ctf']")

class ExtAddOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_type: str = Field(..., description="'particle', 'exposure', 'volume', etc.", min_length=1)
    output_name: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None, description="Slots, e.g. ['location', 'pick_stats']")
    passthrough: Optional[str] = Field(default=None, description="Input name to passthrough from")
    alloc: Optional[int] = Field(default=None, ge=1, description="Pre-allocate dataset rows")

class ExtSaveOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., min_length=1)

class ExtJobRefIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)

class ExtLoadInputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    input_name: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None, description="Filter by slots, e.g. ['micrograph_blob']")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)

# --- Workflow ---

class ImportMoviesIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    blob_paths: str = Field(..., description="Glob path to movie files, e.g. '/data/movies/*.tif'", min_length=1)
    psize_A: float = Field(..., gt=0, description="Pixel size in Angstroms")
    accel_kv: float = Field(..., gt=0, description="Accelerating voltage in kV, e.g. 300")
    cs_mm: float = Field(..., ge=0, description="Spherical aberration in mm, e.g. 2.7")
    total_dose_e_per_A2: float = Field(..., gt=0, description="Total dose in e/Å²")
    gainref_path: Optional[str] = Field(default=None)
    lane: Optional[str] = Field(default=None)
    wait: bool = Field(default=False, description="Wait for completion")

class PreprocessIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    import_job_uid: str = Field(..., description="UID of completed Import Movies job", min_length=1)
    lane: Optional[str] = Field(default=None)


# ====================================================================
# Tool registration
# ====================================================================

def register_all_tools(mcp: FastMCP) -> None:
    """Register all CryoSPARC tools on the MCP server."""

    # ==============================================================
    # NAVIGATION (read-only)
    # ==============================================================

    @mcp.tool(name="cryosparc_test_connection", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_test_connection(params: TestConnectionIn) -> str:
        """Test connection to the CryoSPARC instance. Returns base URL and user info."""
        try:
            cs = get_client()
            cs.test_connection()
            return to_json({"connected": True, "base_url": cs.base_url, "user": str(cs.user)})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_projects", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_projects(params: ListProjectsIn) -> str:
        """List all CryoSPARC projects accessible to the current user."""
        try:
            cs = get_client()
            # cs.find_projects() returns an iterator of ProjectController
            projects = list(cs.find_projects())
            items = [{"uid": p.uid, "title": p.title, "desc": p.desc} for p in projects]
            return to_json({"count": len(items), "projects": items})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_find_project", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_find_project(params: FindProjectIn) -> str:
        """Get project details: UID, title, description, directory path."""
        try:
            cs = get_client()
            p = cs.find_project(params.project_uid)
            return to_json({"uid": p.uid, "title": p.title, "desc": p.desc, "dir": str(p.dir)})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_workspaces", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_workspaces(params: ListWorkspacesIn) -> str:
        """List workspaces in a project."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            # project.find_workspaces() returns iterator of WorkspaceController
            workspaces = list(project.find_workspaces())
            items = [{"uid": w.uid, "title": w.title, "desc": w.desc} for w in workspaces]
            return to_json({"count": len(items), "workspaces": items})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_jobs", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_jobs(params: ListJobsIn) -> str:
        """List jobs in a project, optionally filtered by workspace and/or type.

        Returns UID, type, title, and status for each job.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            # project.find_jobs(workspace_uid=..., type=...) — documented signature
            kwargs: dict = {}
            if params.workspace_uid:
                kwargs["workspace_uid"] = params.workspace_uid
            if params.type:
                kwargs["type"] = params.type
            jobs = list(project.find_jobs(**kwargs))
            items = [{"uid": j.uid, "type": j.type, "title": j.title, "status": j.status} for j in jobs]
            return to_json({"count": len(items), "jobs": items})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_find_job", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_find_job(params: FindJobIn) -> str:
        """Get detailed information about a job: UID, type, title, status, parameters, input/output names."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            j = project.find_job(params.job_uid)
            info = {
                "uid": j.uid, "type": j.type, "title": j.title,
                "desc": j.desc, "status": j.status,
                "params": dict(j.params) if j.params else {},
                "inputs": list(j.inputs.keys()) if j.inputs else [],
                "outputs": list(j.outputs.keys()) if j.outputs else [],
            }
            return to_json(info)
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_get_job_spec", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_get_job_spec(params: GetJobSpecIn) -> str:
        """Get parameter, input, or output specification for a job.

        Use spec_type='params' for parameters (name, title, type, default).
        Use spec_type='inputs' for input slots.
        Use spec_type='outputs' for output slots.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            j = project.find_job(params.job_uid)
            spec = j.full_spec
            if params.spec_type == "params" and spec and hasattr(spec, "params"):
                param_list = []
                for name, p in spec.params.items():
                    param_list.append({
                        "name": name,
                        "title": getattr(p, "title", name),
                        "type": getattr(p, "type", "unknown"),
                        "value": getattr(p, "value", None),
                    })
                return to_json({"job_uid": params.job_uid, "spec_type": "params", "params": param_list})
            elif params.spec_type == "inputs":
                return to_json({"job_uid": params.job_uid, "spec_type": "inputs", "inputs": list(j.inputs.keys()) if j.inputs else []})
            elif params.spec_type == "outputs":
                return to_json({"job_uid": params.job_uid, "spec_type": "outputs", "outputs": list(j.outputs.keys()) if j.outputs else []})
            return to_json({"error": f"Unknown spec_type: {params.spec_type}"})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_job_types", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_job_types(params: ListJobTypesIn) -> str:
        """List all job types available on the CryoSPARC instance.

        Optionally filter by category (e.g. 'refinement', 'extraction', 'import').
        """
        try:
            cs = get_client()
            register = cs.job_register
            items = []
            for section in register:
                sec_name = section.get("name", "")
                if params.category and sec_name != params.category:
                    continue
                for jt in section.get("contains", []):
                    items.append({
                        "type": jt.get("name", jt.get("type", "unknown")),
                        "title": jt.get("title", ""),
                        "category": sec_name,
                    })
            return to_json({"count": len(items), "job_types": items})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_get_lanes", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_get_lanes(params: GetLanesIn) -> str:
        """List available scheduler lanes (compute resource groups)."""
        try:
            cs = get_client()
            lanes = cs.get_lanes()
            return to_json({"lanes": lanes})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_get_targets", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_get_targets(params: GetTargetsIn) -> str:
        """List available scheduler targets (GPU workers)."""
        try:
            cs = get_client()
            targets = cs.get_targets(lane=params.lane) if params.lane else cs.get_targets()
            return to_json({"targets": targets})
        except Exception as e:
            return fmt_error(e)

    # ==============================================================
    # JOB MANAGEMENT
    # ==============================================================

    @mcp.tool(name="cryosparc_create_workspace", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_workspace(params: CreateWorkspaceIn) -> str:
        """Create a new workspace in a project."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            ws = project.create_workspace(params.title, desc=params.desc)
            return to_json({"workspace_uid": ws.uid, "title": ws.title})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_create_job", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_job(params: CreateJobIn) -> str:
        """Create a new CryoSPARC job.

        Uses the project-centric API: project.create_job(workspace_uid, type, connections, params).
        Connections format: {"input_name": ("source_job_uid", "source_output")}
        or {"input_name": [("j1", "o1"), ("j2", "o2")]} for multiple sources.

        The job is created in 'building' status. Set params, connect inputs, then queue.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)

            # Normalise connections to tuples as expected by cryosparc-tools
            connections = {}
            if params.connections:
                for input_name, conn in params.connections.items():
                    if not conn:
                        continue
                    if isinstance(conn, (list, tuple)) and len(conn) == 2 and isinstance(conn[0], str):
                        connections[input_name] = (conn[0], conn[1])
                    elif isinstance(conn, list) and all(isinstance(c, (list, tuple)) for c in conn):
                        connections[input_name] = [(c[0], c[1]) for c in conn]
                    else:
                        connections[input_name] = conn

            job = project.create_job(
                params.workspace_uid,
                params.job_type,
                connections=connections,
                params=params.params or {},
                title=params.title or "",
                desc=params.desc or "",
            )
            return to_json({"job_uid": job.uid, "type": job.type, "status": job.status})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_set_param", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
    async def cryosparc_set_param(params: SetParamIn) -> str:
        """Set a single parameter on a job. Job must be in 'building' status.

        Returns True on success. Use cryosparc_get_job_spec to see available parameters.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            result = job.set_param(params.param_name, params.value)
            return to_json({"success": result, "param": params.param_name, "value": params.value})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_set_params_batch", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
    async def cryosparc_set_params_batch(params: SetParamsBatchIn) -> str:
        """Set multiple parameters at once on a job in 'building' status."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            results = {}
            for name, value in params.params.items():
                results[name] = job.set_param(name, value)
            return to_json({"results": results})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_connect", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
    async def cryosparc_connect(params: ConnectIn) -> str:
        """Connect an output from a source job to an input on the target job.

        Uses: job.connect(target_input, source_job_uid, source_output)
        as documented at tools.cryosparc.com/guides/jobs.html
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            result = job.connect(params.target_input, params.source_job_uid, params.source_output)
            return to_json({"success": result, "connection": f"{params.source_job_uid}.{params.source_output} → {params.job_uid}.{params.target_input}"})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_queue_job", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_queue_job(params: QueueJobIn) -> str:
        """Queue a job for execution. Optionally specify a lane.

        Uses: job.queue(lane) or job.queue() as documented.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            if params.lane:
                job.queue(params.lane)
            else:
                job.queue()
            job.refresh()
            return to_json({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_wait_for_done", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_wait_for_done(params: WaitJobIn) -> str:
        """Wait for a job to reach a terminal status (completed/failed/killed).

        Uses: job.wait_for_done(timeout, error_on_incomplete) as documented.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            status = job.wait_for_done(timeout=params.timeout, error_on_incomplete=params.error_on_incomplete)
            return to_json({"job_uid": job.uid, "status": status})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_wait_for_status", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_wait_for_status(params: WaitStatusIn) -> str:
        """Wait for a job to reach a specific status (e.g. 'waiting' for interactive jobs).

        Uses: job.wait_for_status(target_status, timeout) as documented.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            status = job.wait_for_status(params.target_status, timeout=params.timeout)
            return to_json({"job_uid": job.uid, "status": status})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_interact", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_interact(params: InteractIn) -> str:
        """Invoke an interactive action on a waiting job.

        Used for interactive jobs like Curate Exposures, Inspect Picks, Select 2D.
        Common actions:
          - 'get_fields_and_thresholds' (Curate Exposures) → returns curation data
          - 'set_thresholds' (Curate Exposures) → pass curation data with updated thresholds
          - 'shutdown_interactive' (Curate Exposures, Inspect Picks) → finalize
          - 'finish' (Select 2D) → finalize

        Uses: job.interact(action, data) as documented.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            if params.data is not None:
                result = job.interact(params.action, params.data)
            else:
                result = job.interact(params.action)
            return to_json({"job_uid": job.uid, "action": params.action, "result": result})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_kill_job", annotations={"readOnlyHint": False, "destructiveHint": True})
    async def cryosparc_kill_job(params: JobRefIn) -> str:
        """Kill a running or queued job."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            job.kill()
            job.refresh()
            return to_json({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_clear_job", annotations={"readOnlyHint": False, "destructiveHint": True})
    async def cryosparc_clear_job(params: JobRefIn) -> str:
        """Clear a job — resets it to 'building' status. Existing outputs are removed."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            job.clear()
            job.refresh()
            return to_json({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            return fmt_error(e)

    # ==============================================================
    # DATA INSPECTION
    # ==============================================================

    @mcp.tool(name="cryosparc_dataset_summary", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_dataset_summary(params: DatasetSummaryIn) -> str:
        """Get dataset summary: row count, column names and dtypes.

        Use this before loading data to understand the schema.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            dset = job.load_output(params.output_name)
            cols = [{"name": c, "dtype": str(dset[c].dtype)} for c in dset.columns()]
            return to_json({"job_uid": params.job_uid, "output": params.output_name, "num_rows": len(dset), "columns": cols})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_load_output", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_load_output(params: LoadOutputIn) -> str:
        """Load data from a job output with optional slot filtering and pagination.

        Uses: job.load_output(output_name, slots=[...]) as documented.
        Returns paginated rows with selected columns.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)

            # load_output supports slots= parameter for filtering
            if params.slots:
                dset = job.load_output(params.output_name, slots=params.slots)
            else:
                dset = job.load_output(params.output_name)

            total = len(dset)
            col_names = list(dset.columns())

            if total > 10000 and not params.slots:
                return to_json({
                    "warning": f"Dataset has {total} rows. Specify 'slots' and use pagination.",
                    "num_rows": total, "columns": col_names,
                })

            start = min(params.offset, total)
            end = min(start + params.limit, total)
            rows = []
            for row in list(dset.rows())[start:end]:
                rows.append(dict(row))

            return to_json({
                "total_rows": total, "offset": start, "count": len(rows),
                "has_more": end < total,
                "next_offset": end if end < total else None,
                "columns": col_names, "rows": rows,
            })
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_files", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_files(params: ListFilesIn) -> str:
        """List files in a project or job directory."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            if params.job_uid:
                job = project.find_job(params.job_uid)
                files = job.list_files(params.prefix or "")
            else:
                files = project.list_files(prefix=params.prefix or "", recursive=params.recursive)
            return to_json({"files": list(files)})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_assets", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_assets(params: ListAssetsIn) -> str:
        """List image/plot assets for a job (previews, diagnostic plots)."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            assets = job.list_assets()
            items = [{"id": str(a.get("_id", a.get("id", ""))), "filename": a.get("filename", "")} for a in assets]
            return to_json({"assets": items})
        except Exception as e:
            return fmt_error(e)

    # ==============================================================
    # EXTERNAL JOBS
    # ==============================================================

    @mcp.tool(name="cryosparc_create_external_job", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_external_job(params: CreateExternalJobIn) -> str:
        """Create an External Job for importing results from third-party tools.

        Uses: project.create_external_job(workspace_uid, title, desc) as documented.
        Lifecycle: create → connect inputs → add outputs → start → populate → save → stop.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.create_external_job(params.workspace_uid, title=params.title or "", desc=params.desc or "")
            return to_json({"job_uid": job.uid, "type": "external", "status": job.status})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_external_connect", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_connect(params: ExtConnectIn) -> str:
        """Connect a source to an External Job input, with optional slot filtering.

        Uses: job.connect(target_input, source_job_uid, source_output, slots=[...])
        as documented in the crYOLO example.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            kwargs: dict = {}
            if params.slots:
                kwargs["slots"] = params.slots
            job.connect(params.target_input, params.source_job_uid, params.source_output, **kwargs)
            return to_json({"success": True, "connection": f"{params.source_job_uid}.{params.source_output} → {params.job_uid}.{params.target_input}"})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_external_add_output", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_add_output(params: ExtAddOutputIn) -> str:
        """Add an output slot to an External Job.

        Uses: job.add_output(type, name, slots=[...], passthrough=..., alloc=N)
        as documented.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            kwargs: dict = {"type": params.output_type, "name": params.output_name}
            if params.slots:
                kwargs["slots"] = params.slots
            if params.passthrough:
                kwargs["passthrough"] = params.passthrough
            if params.alloc is not None:
                kwargs["alloc"] = params.alloc
            result = job.add_output(**kwargs)
            return to_json({"success": True, "output_name": params.output_name})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_external_start", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_start(params: ExtJobRefIn) -> str:
        """Start an External Job — transitions to 'started' / 'waiting' status."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            job.start()
            return to_json({"job_uid": job.uid, "status": "started"})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_external_save_output", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_save_output(params: ExtSaveOutputIn) -> str:
        """Save (persist) an output of a started External Job."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            job.save_output(params.output_name)
            return to_json({"success": True, "output_name": params.output_name})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_external_stop", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_stop(params: ExtJobRefIn) -> str:
        """Stop an External Job — marks it as completed."""
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            job.stop()
            return to_json({"job_uid": job.uid, "status": "completed"})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_external_load_input", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_external_load_input(params: ExtLoadInputIn) -> str:
        """Load data from a connected input of an External Job.

        Uses: job.load_input(input_name, slots=[...]) as documented in crYOLO example.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            if params.slots:
                dset = job.load_input(params.input_name, params.slots)
            else:
                dset = job.load_input(params.input_name)

            total = len(dset)
            col_names = list(dset.columns())
            start = min(params.offset, total)
            end = min(start + params.limit, total)
            rows = [dict(r) for r in list(dset.rows())[start:end]]
            return to_json({"total_rows": total, "offset": start, "count": len(rows), "columns": col_names, "rows": rows})
        except Exception as e:
            return fmt_error(e)

    # ==============================================================
    # WORKFLOW HELPERS
    # ==============================================================

    @mcp.tool(name="cryosparc_workflow_import_movies", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_workflow_import_movies(params: ImportMoviesIn) -> str:
        """Import movies in one step: create job, set params, queue.

        Follows the exact pattern from the Custom Workflow tutorial.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            workspace = project.find_workspace(params.workspace_uid)

            job_params = {
                "blob_paths": params.blob_paths,
                "psize_A": params.psize_A,
                "accel_kv": params.accel_kv,
                "cs_mm": params.cs_mm,
                "total_dose_e_per_A2": params.total_dose_e_per_A2,
            }
            if params.gainref_path:
                job_params["gainref_path"] = params.gainref_path

            # workspace.create_job(type, params=...) — as in tutorial
            job = workspace.create_job("import_movies", params=job_params)

            if params.lane:
                job.queue(params.lane)
            else:
                job.queue()

            if params.wait:
                status = job.wait_for_done()
                return to_json({"job_uid": job.uid, "status": status})

            job.refresh()
            return to_json({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_workflow_preprocess", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_workflow_preprocess(params: PreprocessIn) -> str:
        """Run Motion Correction + CTF Estimation chained together.

        Follows the exact pattern from the Custom Workflow tutorial:
        both jobs queued at the same time, CryoSPARC scheduler handles ordering.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            workspace = project.find_workspace(params.workspace_uid)

            # Motion correction connected to import job
            mc_job = workspace.create_job(
                "patch_motion_correction_multi",
                connections={"movies": (params.import_job_uid, "imported_movies")},
            )
            # CTF connected to motion correction
            ctf_job = workspace.create_job(
                "patch_ctf_estimation_multi",
                connections={"exposures": (mc_job.uid, "micrographs")},
            )

            # Queue both — scheduler handles dependency
            if params.lane:
                mc_job.queue(params.lane)
                ctf_job.queue(params.lane)
            else:
                mc_job.queue()
                ctf_job.queue()

            mc_job.refresh()
            ctf_job.refresh()
            return to_json({
                "motion_correction": {"job_uid": mc_job.uid, "status": mc_job.status},
                "ctf_estimation": {"job_uid": ctf_job.uid, "status": ctf_job.status},
            })
        except Exception as e:
            return fmt_error(e)
