"""CryoSPARC MCP tools – universal interface for single-particle cryo-EM.

This module exposes the full CryoSPARC job creation, connection, and
data inspection API to an AI agent. The agent can build any processing
pipeline, either from scratch using the documented job types and
connection patterns, or by importing a workflow JSON file.

KEY CONCEPTS (read before using any tool):
- A *project* (UID like 'P3') contains workspaces.
- A *workspace* (UID like 'W1') contains jobs.
- A *job* has a type, parameters, and connections to outputs of previous jobs.
- Jobs are chained by connecting an output slot of an upstream job to an
  input slot of a downstream job (see cryosparc_connect).

COMMON PIPELINE STAGES (not an exhaustive recipe, but a mental map):
  1) Import movies
  2) Motion correction → CTF estimation
  3) Exposure curation (can be automatic with auto-thresholds)
  4) Denoising training (optional but common)
  5) Particle picking (blob, template, or deep-learning based)
  6) Junk detection to filter bad micrographs/picks
  7) Inspect picks (automatic clustering available)
  8) Particle extraction
  9) 2D classification → 2D selection (manual or automatic by reference)
  10) Ab-initio reconstruction
  11) Heterogeneous refinement (optionally using junk volumes to sort particles)
  12) Homogeneous / non-uniform refinement (often multiple parallel runs)
  13) Late-stage motion correction, local refinement, and volume selection.

ADVANCED PATTERNS documented for the agent:
- Two-round picking: blob picker → 2D class → auto-select 2D templates → template picker.
- Junk volumes: import several fake/junk maps and use them in hetero_refine
  to separate true particles from contaminants.
- Parallel non-uniform refinements with different CTF options (3rd, 4th,
  aniso mag) followed by select_volume to pick the best map.
- Automatic exposure curation using auto_threshold_* parameters.
- Automatic pick inspection using do_auto_cluster=True.

When the agent is given a workflow JSON, it should use
cryosparc_parse_workflow to extract jobs and connections, then recreate
them step by step.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

from .connection import get_client
from .helpers import TASK_MODE, dataset_fields, jsonable, params_to_dict, tool_error


# ====================================================================
# Pydantic input models
# ====================================================================

class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

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
    type: Optional[str] = Field(default=None, description="Filter by job type, e.g. 'homo_refine_new', 'class_2D', 'import_movies'")

class FindJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., description="Job UID, e.g. 'J42'", min_length=1)

class ListJobTypesIn(_Base):
    category: Optional[str] = Field(default=None, description="Category filter: 'import', 'motion_correction', 'ctf_estimation', 'particle_picking', 'extraction', 'particle_curation', 'reconstruction', 'refinement', 'postprocessing', 'live', etc.")

class GetLanesIn(_Base):
    pass

class GetTargetsIn(_Base):
    lane: Optional[str] = Field(default=None, description="Lane name filter")

class CreateWorkspaceIn(_Base):
    project_uid: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=200)
    desc: Optional[str] = Field(default=None)

class CreateJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    job_type: str = Field(..., description=(
        "CryoSPARC job type string. The list below is a common-case cheat sheet and may\n"
        "drift between CryoSPARC versions — call cryosparc_list_job_types for the\n"
        "authoritative list on THIS instance, and cryosparc_get_job_spec for parameters.\n"
        "  IMPORT: 'import_movies', 'import_micrographs', 'import_volumes', 'import_particles'\n"
        "  MOTION CORRECTION: 'patch_motion_correction_multi'\n"
        "  CTF: 'patch_ctf_estimation_multi'\n"
        "  CURATION: 'curate_exposures_v2', 'curate_micrographs', 'curate_particles'\n"
        "  DENOISE: 'denoise_train'\n"
        "  GROUPING: 'exposure_groups'\n"
        "  JUNK: 'junk_detector_v1'\n"
        "  PICKING: 'blob_picker_gpu', 'template_picker_gpu', 'deep_picker'\n"
        "  INSPECT: 'inspect_picks_v2'\n"
        "  EXTRACTION: 'extract_micrographs_cpu_parallel', 'extract_micrographs_multi'\n"
        "  2D CLASS: 'class_2D', 'class_2D_new'\n"
        "  2D SELECT: 'reference_select_2D', 'select_2D'\n"
        "  AB-INITIO: 'homo_abinit'\n"
        "  REFINEMENT: 'homo_refine_new', 'nonuniform_refine_new', 'hetero_refine'\n"
        "  DUPLICATE REMOVAL: 'remove_duplicate_particles', 'remove_duplicate_micrographs'\n"
        "  SUBSET: 'subset_particles'\n"
        "  ALIGNMENT: 'align_3D_new'\n"
        "  RECONSTRUCTION: 'homo_reconstruct'\n"
        "  VOLUME SELECT: 'select_volume', 'reference_select_3D'\n"
        "  LOCAL REFINE: 'new_local_refine', 'local_refine'\n"
        "  MOTION CORRECTION (LATE): 'reference_motion_correction'\n"
        "  POSTPROCESSING: 'sharpening', 'fsc_curve'\n"
        "Use cryosparc_list_job_types to see the complete list."
    ), min_length=1)
    title: Optional[str] = Field(default="")
    desc: Optional[str] = Field(default="")
    params: Optional[Dict[str, Any]] = Field(default=None, description=(
        "Job parameter values as key-value pairs.\n"
        "Examples:\n"
        "  import_movies: {'blob_paths': '/data/*.tif', 'psize_A': 1.06, 'accel_kv': 300, 'cs_mm': 2.7, 'total_dose_e_per_A2': 50}\n"
        "  blob_picker_gpu: {'diameter': 150, 'diameter_max': 200}\n"
        "  extract_micrographs_cpu_parallel: {'box_size_pix': 256}\n"
        "  class_2D: {'class2D_K': 50}\n"
        "  homo_abinit: {'abinit_K': 3}\n"
        "  homo_refine_new: {'refine_symmetry': 'C1'}\n"
        "Use cryosparc_get_job_spec to see all available parameters."
    ))
    connections: Optional[Dict[str, Any]] = Field(default=None, description=(
        "Input connections: {input_name: (source_job_uid, source_output_name)}.\n"
        "Examples:\n"
        "  motion_correction: {'movies': ('J1', 'imported_movies')}\n"
        "  ctf_estimation: {'exposures': ('J2', 'micrographs')}\n"
        "  curate_exposures: {'exposures': ('J3', 'exposures')}\n"
        "  blob_picker: {'micrographs': ('J3', 'exposures')}\n"
        "  extraction: {'micrographs': ('J3', 'exposures'), 'particles': ('J5', 'particles')}\n"
        "  class_2D: {'particles': ('J6', 'particles')}\n"
        "  select_2D: {'particles': ('J7', 'particles_class_0')}\n"
        "  abinit: {'particles': ('J8', 'particles_selected')}\n"
        "  refine: {'particles': ('J8', 'particles_selected'), 'volume': ('J9', 'volume_class_0')}"
    ))

class SetParamIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    param_name: str = Field(..., description="Parameter name", min_length=1)
    value: Any = Field(..., description="Parameter value")

class SetParamsBatchIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(..., description="Dictionary of {parameter_name: value} pairs")

class ConnectIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., description="Target job UID", min_length=1)
    target_input: str = Field(..., description="Input slot name on the target job", min_length=1)
    source_job_uid: str = Field(..., description="Source job UID", min_length=1)
    source_output: str = Field(..., description="Source output name", min_length=1)

class QueueJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    lane: Optional[str] = Field(default=None, description="Scheduler lane name")

class JobRefIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)

class WaitJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    timeout: int = Field(default=600, ge=1, le=86400)
    error_on_incomplete: bool = Field(default=False)

class WaitStatusIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    target_status: str = Field(..., description="Status to wait for, e.g. 'waiting'", min_length=1)
    timeout: int = Field(default=300, ge=1, le=86400)

class InteractIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    action: str = Field(..., description=(
        "Interactive action. Examples:\n"
        "  curate_exposures_v2: 'get_fields_and_thresholds', 'set_thresholds', 'shutdown_interactive'\n"
        "  inspect_picks_v2: 'shutdown_interactive'\n"
        "  select_2D: 'finish'"
    ), min_length=1)
    data: Optional[Any] = Field(default=None, description="Data payload (required for 'set_thresholds')")

class DatasetSummaryIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., min_length=1)

class LoadOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None, description="Optional column filters")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)

class ListFilesIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: Optional[str] = Field(default=None)
    prefix: Optional[str] = Field(default=None)
    recursive: bool = Field(default=False)

class ListAssetsIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)

class GetJobSpecIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    spec_type: str = Field(default="params", description="'params', 'inputs', or 'outputs'")

# ------------------ nouveaux inputs ------------------
class DownloadMrcIn(_Base):
    project_uid: str = Field(..., min_length=1)
    path: str = Field(..., description="Relative path to .mrc file, e.g. 'J42/volume.mrc'")

class DownloadFileIn(_Base):
    project_uid: str = Field(..., min_length=1)
    path: str = Field(..., description="Relative path to file")
    target: str = Field(default="", description="Local target path (default: current dir)")

class ParseWorkflowIn(_Base):
    workflow_json_path: str = Field(..., description="Path to workflow JSON file")

class SuggestParamsIn(_Base):
    target_type: str = Field(..., description="Target type: 'gpcr', 'soluble', 'membrane', 'complex'")
    pixel_size_A: Optional[float] = Field(default=None)
    num_particles_expected: Optional[int] = Field(default=None)


# ====================================================================
# Tool registration
# ====================================================================

def register_all_tools(mcp: FastMCP) -> None:
    """Register all CryoSPARC tools on the MCP server."""

    # ==============================================================
    # NAVIGATION (read-only)
    # ==============================================================

    @mcp.tool(name="cryosparc_test_connection", title="Test Connection", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_test_connection(params: TestConnectionIn) -> Dict[str, Any]:
        """Test connection to the CryoSPARC instance.

        Verifies that the API is reachable and working. Call this first
        before any other tool. Returns the base URL and user info.
        If this fails, check that CryoSPARC is running and credentials
        are set correctly.
        """
        try:
            cs = get_client()
            cs.test_connection()
            url = getattr(cs, "base_url", None) or f"{getattr(cs, 'host', 'unknown')}:{getattr(cs, 'base_port', 'unknown')}"
            return jsonable({"connected": True, "base_url": url, "user": str(getattr(cs, "user", "unknown"))})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_list_projects", title="List Projects", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_projects(params: ListProjectsIn) -> Dict[str, Any]:
        """List all accessible CryoSPARC projects.

        A project is the top-level container for cryo-EM processing.
        Returns each project's UID (e.g., 'P3'), title, and description.
        Use this to find the project_uid needed for all other tools.
        """
        try:
            cs = get_client()
            projects = list(cs.find_projects())
            items = [{"uid": p.uid, "title": p.title, "desc": p.desc} for p in projects]
            return jsonable({"count": len(items), "projects": items})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_find_project", title="Project Details", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_find_project(params: FindProjectIn) -> Dict[str, Any]:
        """Get detailed info about one project.

        Returns UID, title, description, and the filesystem directory
        where data is stored. The directory is useful for locating
        raw movie files or gain references.
        """
        try:
            cs = get_client()
            p = cs.find_project(params.project_uid)
            return jsonable({"uid": p.uid, "title": p.title, "desc": p.desc, "dir": str(p.dir)})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_list_workspaces", title="List Workspaces", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_workspaces(params: ListWorkspacesIn) -> Dict[str, Any]:
        """List workspaces in a project.

        Workspaces group jobs. Each has a UID (e.g., 'W1').
        You need a workspace UID to create jobs.
        If none exists, create one with cryosparc_create_workspace.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            workspaces = list(project.find_workspaces())
            items = [{"uid": w.uid, "title": w.title, "desc": w.desc} for w in workspaces]
            return jsonable({"count": len(items), "workspaces": items})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_list_jobs", title="List Jobs", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_jobs(params: ListJobsIn) -> Dict[str, Any]:
        """List jobs in a project, optionally filtered by workspace and/or type.

        Returns UID, type, title, and status for each job.
        Statuses: 'building', 'queued', 'running', 'waiting',
        'completed', 'failed', 'killed'.
        Use this to find job UIDs for connections.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            kwargs: dict = {}
            if params.workspace_uid:
                kwargs["workspace_uid"] = params.workspace_uid
            if params.type:
                kwargs["type"] = params.type
            jobs = list(project.find_jobs(**kwargs))
            items = [{"uid": j.uid, "type": j.type, "title": j.title, "status": j.status} for j in jobs]
            return jsonable({"count": len(items), "jobs": items})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_find_job", title="Job Details", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_find_job(params: FindJobIn) -> Dict[str, Any]:
        """Get detailed info about a specific job.

        Returns status, all parameter values, and lists of input/output
        slot names. Use this to check a job's progress, inspect its
        outputs, and learn which inputs/outputs it has before connecting.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            j = project.find_job(params.job_uid)
            info = {
                "uid": j.uid, "type": j.type, "title": j.title,
                "desc": j.desc, "status": j.status,
                "params": params_to_dict(j.params),
                "inputs": list(j.inputs.keys()) if j.inputs else [],
                "outputs": list(j.outputs.keys()) if j.outputs else [],
            }
            return jsonable(info)
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_get_job_spec", title="Job Specification", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_get_job_spec(params: GetJobSpecIn) -> Dict[str, Any]:
        """Get the parameter/input/output specification of a job.

        Use spec_type='params' to see all parameters with their types
        and default values. Essential before setting any parameter.

        Use spec_type='inputs' to see expected input slot names.

        Use spec_type='outputs' to see what outputs a completed job produces.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            j = project.find_job(params.job_uid)
            spec = j.full_spec
            if params.spec_type == "params" and spec and hasattr(spec, "params"):
                current = params_to_dict(j.params)
                param_list = []
                for name, p in spec.params.items():
                    if getattr(p, "hidden", False):
                        continue
                    entry = {
                        "name": name,
                        "title": getattr(p, "title", name),
                        "type": getattr(p, "type", "unknown"),
                        # v5 JobRegisterParamSpec exposes `default` (there is no `value`)
                        "default": getattr(p, "default", None),
                        "current": current.get(name),
                    }
                    desc = getattr(p, "description", None)
                    if desc:
                        entry["description"] = desc
                    enum = getattr(p, "enum", None)
                    if enum:
                        entry["allowed"] = enum
                    if getattr(p, "advanced", False):
                        entry["advanced"] = True
                    param_list.append(entry)
                return jsonable({"job_uid": params.job_uid, "spec_type": "params",
                                "count": len(param_list), "params": param_list})
            elif params.spec_type == "inputs":
                return jsonable({"job_uid": params.job_uid, "spec_type": "inputs", "inputs": list(j.inputs.keys()) if j.inputs else []})
            elif params.spec_type == "outputs":
                return jsonable({"job_uid": params.job_uid, "spec_type": "outputs", "outputs": list(j.outputs.keys()) if j.outputs else []})
            return jsonable({"error": f"Unknown spec_type: {params.spec_type}"})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_list_job_types", title="List Job Types", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_job_types(params: ListJobTypesIn) -> Dict[str, Any]:
        """List all available job types on this CryoSPARC instance.

        Returns the machine-readable type string (use in cryosparc_create_job),
        a human title, and category for each.
        """
        try:
            cs = get_client()
            register = cs.job_register  # v5: JobRegister(specs=[JobRegisterJobSpec], categories=[...])
            items = []
            for spec in register.specs:
                if getattr(spec, "hidden", False):
                    continue
                category = getattr(spec, "category", "") or ""
                if params.category and category != params.category:
                    continue
                items.append({
                    "type": spec.type,
                    "title": getattr(spec, "title", "") or "",
                    "category": category,
                    "interactive": bool(getattr(spec, "interactive", False)),
                })
            items.sort(key=lambda i: (i["category"], i["type"]))
            return jsonable({
                "count": len(items),
                "categories": sorted({i["category"] for i in items if i["category"]}),
                "job_types": items,
            })
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_get_lanes", title="List Scheduler Lanes", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_get_lanes(params: GetLanesIn) -> Dict[str, Any]:
        """List scheduler lanes (compute resource groups).

        A lane targets specific GPU workers. Omit to use default.
        """
        try:
            cs = get_client()
            lanes = cs.get_lanes()
            return jsonable({"lanes": lanes})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_get_targets", title="List GPU Targets", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_get_targets(params: GetTargetsIn) -> Dict[str, Any]:
        """List available scheduler targets (GPU workers/machines)."""
        try:
            cs = get_client()
            targets = cs.get_targets(lane=params.lane) if params.lane else cs.get_targets()
            return jsonable({"targets": targets})
        except Exception as e:
            raise tool_error(e) from e

    # ==============================================================
    # JOB MANAGEMENT
    # ==============================================================

    @mcp.tool(name="cryosparc_create_workspace", title="Create Workspace", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_workspace(params: CreateWorkspaceIn) -> Dict[str, Any]:
        """Create a new workspace in a project.

        Workspaces group jobs. You need one before creating any job.
        If you already have a workspace UID, you can skip this.
        Returns the workspace UID.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            ws = project.create_workspace(params.title, desc=params.desc)
            return jsonable({"workspace_uid": ws.uid, "title": ws.title})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_create_job", title="Create Job", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_job(params: CreateJobIn) -> Dict[str, Any]:
        """Create a new CryoSPARC job.

        This is the central tool for building processing pipelines.
        Specify job type, optional parameters, and input connections
        (outputs of previous jobs). The job starts in 'building' status.
        Afterwards, you can adjust parameters, add connections,
        and finally queue it with cryosparc_queue_job.

        Returns the job UID, type, and initial status.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            connections = {}
            if params.connections:
                for input_name, conn in params.connections.items():
                    if isinstance(conn, (list, tuple)) and len(conn) == 2 and isinstance(conn[0], str):
                        connections[input_name] = (conn[0], conn[1])
                    elif isinstance(conn, list):
                        connections[input_name] = [(c[0], c[1]) for c in conn if isinstance(c, (list, tuple))]
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
            return jsonable({"job_uid": job.uid, "type": job.type, "status": job.status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_set_param", title="Set Parameter", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
    async def cryosparc_set_param(params: SetParamIn) -> Dict[str, Any]:
        """Set a single parameter on a job in 'building' status.

        Use cryosparc_get_job_spec first to see available parameters.
        Common examples: box_size_pix, class2D_K, refine_symmetry.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            job.set_param(params.param_name, params.value)
            return jsonable({"success": True, "param": params.param_name, "value": params.value})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_set_params_batch", title="Set Parameters (Batch)", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
    async def cryosparc_set_params_batch(params: SetParamsBatchIn) -> Dict[str, Any]:
        """Set multiple parameters at once on a 'building' job.

        More efficient than calling cryosparc_set_param repeatedly.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            results = {}
            errors = {}
            for name, value in params.params.items():
                try:
                    results[name] = job.set_param(name, value)
                except Exception as pe:
                    errors[name] = str(pe)
            out = {"results": results}
            if errors:
                out["errors"] = errors
            return jsonable(out)
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_connect", title="Connect Job Inputs", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
    async def cryosparc_connect(params: ConnectIn) -> Dict[str, Any]:
        """Connect an output of one job to an input of another.

        This wires the pipeline: data flows from source_output to target_input.
        The target job must be 'building'.
        Example: connect CTF results to a blob picker:
          target_input='micrographs', source_job_uid='J3', source_output='exposures'
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            job.connect(params.target_input, params.source_job_uid, params.source_output)
            return jsonable({"success": True, "connection": f"{params.source_job_uid}.{params.source_output} -> {params.job_uid}.{params.target_input}"})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_queue_job", title="Queue Job", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_queue_job(params: QueueJobIn) -> Dict[str, Any]:
        """Submit a job for execution on GPU workers.

        The job must be in 'building' status with all required inputs
        connected. The scheduler handles dependencies automatically,
        so you can queue multiple chained jobs at once.
        For interactive jobs (curate_exposures, inspect_picks, select_2D),
        after queueing, use cryosparc_wait_for_status to wait for
        'waiting', then cryosparc_interact.
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
            return jsonable({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(task=TASK_MODE, name="cryosparc_wait_for_done", title="Wait for Job Completion", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_wait_for_done(params: WaitJobIn) -> Dict[str, Any]:
        """Wait until a job finishes (completed, failed, or killed).

        Use after queueing non-interactive jobs to block until they finish.
        For a full pipeline, you don't need to wait between every job;
        the scheduler manages dependencies.
        Only wait when you need outputs before the next step.
        For interactive jobs, use cryosparc_wait_for_status instead.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            status = job.wait_for_done(timeout=params.timeout, error_on_incomplete=params.error_on_incomplete)
            return jsonable({"job_uid": job.uid, "status": status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(task=TASK_MODE, name="cryosparc_wait_for_status", title="Wait for Job Status", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_wait_for_status(params: WaitStatusIn) -> Dict[str, Any]:
        """Wait for a job to reach a specific status (e.g., 'waiting').

        Essential for interactive jobs. After queueing an interactive job,
        wait for 'waiting', then call cryosparc_interact.

        Typical flow:
          1. create_job (curate_exposures_v2)
          2. queue_job
          3. wait_for_status (target_status='waiting')
          4. interact (get_fields_and_thresholds, set_thresholds, shutdown_interactive)
          5. wait_for_done
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            status = job.wait_for_status(params.target_status, timeout=params.timeout)
            return jsonable({"job_uid": job.uid, "status": status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_interact", title="Interactive Job Action", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_interact(params: InteractIn) -> Dict[str, Any]:
        """Perform an interactive action on a job in 'waiting' status.

        Used for jobs that require user-like decisions:

        CURATE EXPOSURES (curate_exposures_v2):
          1. action='get_fields_and_thresholds' → returns curation data
          2. action='set_thresholds', data=<modified> → apply thresholds
          3. action='shutdown_interactive' → finalize

        INSPECT PICKS (inspect_picks_v2):
          action='shutdown_interactive' (often with auto_cluster)

        SELECT 2D (select_2D):
          action='finish' → finalize selection

        The job must be in 'waiting' status; use cryosparc_wait_for_status first.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            if params.data is not None:
                result = job.interact(params.action, params.data)
            else:
                result = job.interact(params.action)
            return jsonable({"job_uid": job.uid, "action": params.action, "result": result})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_kill_job", title="Kill Job", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": True})
    async def cryosparc_kill_job(params: JobRefIn) -> Dict[str, Any]:
        """Kill a running or queued job.

        The job transitions to 'killed' status.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            job.kill()
            job.refresh()
            return jsonable({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_clear_job", title="Clear Job", annotations={"openWorldHint": True, "readOnlyHint": False, "destructiveHint": True})
    async def cryosparc_clear_job(params: JobRefIn) -> Dict[str, Any]:
        """Reset a completed/failed/killed job to 'building' status.

        This removes existing outputs so you can change parameters and re-run.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            job.clear()
            job.refresh()
            return jsonable({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            raise tool_error(e) from e

    # ==============================================================
    # DATA INSPECTION & DOWNLOAD
    # ==============================================================

    @mcp.tool(name="cryosparc_dataset_summary", title="Dataset Schema", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_dataset_summary(params: DatasetSummaryIn) -> Dict[str, Any]:
        """Get schema of a job output: row count and column names/types.

        Always call this before cryosparc_load_output to understand the
        data structure. Helps avoid loading too much data.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            dset = job.load_output(params.output_name)
            cols = [{"name": c, "dtype": str(dset[c].dtype)} for c in dataset_fields(dset)]
            return jsonable({"job_uid": params.job_uid, "output": params.output_name, "num_rows": len(dset), "columns": cols})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_load_output", title="Load Job Output", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_load_output(params: LoadOutputIn) -> Dict[str, Any]:
        """Load paginated data rows from a job output.

        Use 'slots' to filter columns and reduce transfer volume.
        For large datasets (>10k rows), you must specify slots and
        use pagination (offset/limit).
        Call cryosparc_dataset_summary first to see available columns.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            dset = job.load_output(params.output_name, slots=params.slots) if params.slots else job.load_output(params.output_name)
            total = len(dset)
            col_names = dataset_fields(dset)
            if total > 10000 and not params.slots:
                return jsonable({"warning": "Dataset is large; specify 'slots'.", "num_rows": total, "columns": col_names})
            start = min(params.offset, total)
            end = min(start + params.limit, total)
            from itertools import islice
            rows = [dict(r) for r in islice(dset.rows(), start, end)]
            return jsonable({
                "total_rows": total, "offset": start, "count": len(rows),
                "has_more": end < total, "next_offset": end if end < total else None,
                "columns": col_names, "rows": rows,
            })
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_list_files", title="List Files", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_files(params: ListFilesIn) -> Dict[str, Any]:
        """List files in a project or job directory.

        Useful for finding raw movies, MRC volumes, .cs dataset files,
        or any outputs produced by jobs.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            if params.job_uid:
                job = project.find_job(params.job_uid)
                files = job.list_files(params.prefix or "")
            else:
                files = project.list_files(prefix=params.prefix or "", recursive=params.recursive)
            return jsonable({"files": list(files)})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_list_assets", title="List Job Assets", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_assets(params: ListAssetsIn) -> Dict[str, Any]:
        """List image/plot assets stored for a job.

        Assets include 2D class previews, FSC curves, diagnostic plots.
        They are stored in CryoSPARC's database, not as files.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            assets = job.list_assets()
            items = [{"id": str(a.get("_id", a.get("id", ""))), "filename": a.get("filename", "")} for a in assets]
            return jsonable({"assets": items})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_download_mrc", title="Download MRC", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_download_mrc(params: DownloadMrcIn) -> Dict[str, Any]:
        """Download a .mrc file and return its header and data shape.

        The MRC is loaded as a NumPy array. This is useful for inspecting
        reconstructed volumes, masks, and reference maps.
        For large arrays, consider using cryosparc_download_file instead
        to save directly to disk.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            header, data = project.download_mrc(params.path)
            return jsonable({
                "path": params.path,
                "header": {
                    "nx": header.nx, "ny": header.ny, "nz": header.nz,
                    "mx": header.mx, "my": header.my, "mz": header.mz,
                    "xlen": header.xlen, "ylen": header.ylen, "zlen": header.zlen,
                },
                "data_shape": data.shape, "data_dtype": str(data.dtype),
            })
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_download_file", title="Download File", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_download_file(params: DownloadFileIn) -> Dict[str, Any]:
        """Download a file from the project directory to local disk.

        Use this to export MRC volumes, .cs files, STAR files, etc.
        The file is saved to target path (or current dir if target empty).
        Call cryosparc_list_files first to discover available files.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            result = project.download_file(params.path, target=params.target)
            return jsonable({"path": params.path, "target": str(result)})
        except Exception as e:
            raise tool_error(e) from e

    # ==============================================================
    # WORKFLOW & PARAMETER HELPERS
    # ==============================================================

    @mcp.tool(name="cryosparc_parse_workflow", title="Parse Workflow JSON", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_parse_workflow(params: ParseWorkflowIn) -> Dict[str, Any]:
        """Parse a CryoSPARC workflow JSON file and extract job structure.

        Returns a list of jobs with their type, connections, and
        flagged parameters. The agent can then recreate the pipeline
        step-by-step using cryosparc_create_job and cryosparc_connect.
        This is the programmatic equivalent of importing a workflow
        in the GUI.
        """
        import json
        try:
            with open(params.workflow_json_path) as f:
                wf = json.load(f)
            jobs = []
            for job_id, job in wf.get("jobs", {}).items():
                connections = {}
                for group in job.get("groups", []):
                    if len(group) >= 2:
                        src_ref, input_name = group[0]
                        if isinstance(src_ref, str) and "." in src_ref:
                            src_job, src_output = src_ref.split(".", 1)
                            connections[input_name] = (src_job, src_output)
                flagged_params = {k: v.get("value") for k, v in job.get("parameters", {}).items() if v.get("flagged")}
                jobs.append({
                    "job_id": job_id,
                    "type": job.get("jobType"),
                    "description": job.get("description", ""),
                    "connections": connections,
                    "flagged_params": flagged_params,
                })
            return jsonable({
                "title": wf.get("title"), "description": wf.get("description", ""),
                "num_jobs": len(jobs), "jobs": jobs,
            })
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_suggest_params", title="Suggest Parameters", annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_suggest_params(params: SuggestParamsIn) -> Dict[str, Any]:
        """Suggest starting parameters for common target types.

        This is not a prediction; it retrieves validated parameters
        from known reference workflows (e.g., GPCR workflow).
        Use these as a starting point and adapt based on actual
        pixel size, particle count, and sample characteristics.

        Currently supports: 'gpcr'.
        """
        reference = {
            "gpcr": {
                "source": "biorxiv 2025 GPCR workflow",
                "curate_exposures": {"auto_threshold_ctf_fit_to_A": "1.5,4.0", "auto_threshold_motion_total_pix": "0,200"},
                "blob_picker": {"diameter": 110, "diameter_max": 140, "min_distance": 0.7, "max_num_hits": 1250},
                "extract_round1": {"box_size_pix": 300, "bin_size_pix": 100},
                "class_2D": {"class2D_K": 100, "class2D_window_inner_A": 150},
                "abinit": {"abinit_K": 5, "abinit_max_res": 10},
                "nonuniform_refine": {"refine_ctf_global_refine": True, "refine_scale_min": True},
            }
        }
        if params.target_type in reference:
            return jsonable({"target_type": params.target_type, **reference[params.target_type]})
        return jsonable({"warning": f"No reference for '{params.target_type}'. Available: {list(reference.keys())}"})