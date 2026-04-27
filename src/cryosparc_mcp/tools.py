"""CryoSPARC MCP tools — all tools registered on the FastMCP server.

CRYO-EM PROCESSING PIPELINE REFERENCE (for the AI agent):

The standard single-particle cryo-EM workflow in CryoSPARC is:

  1. IMPORT MOVIES         → job type: "import_movies"
     Input: raw movie files (.tif, .mrc, .eer)
     Output: "imported_movies"
     Key params: blob_paths, psize_A, accel_kv, cs_mm, total_dose_e_per_A2

  2. MOTION CORRECTION     → job type: "patch_motion_correction_multi"
     Input: "movies" ← connected to import job's "imported_movies"
     Output: "micrographs" (aligned, dose-weighted micrographs)

  3. CTF ESTIMATION        → job type: "patch_ctf_estimation_multi"
     Input: "exposures" ← connected to motion correction's "micrographs"
     Output: "exposures" (micrographs with CTF parameters: defocus, astigmatism)

  4. EXPOSURE CURATION     → job type: "curate_exposures_v2" (interactive)
     Input: "exposures" ← connected to CTF estimation's "exposures"
     Output: "exposures_accepted" (curated micrographs)
     Interactive: queue → wait_for_status("waiting") → interact("get_fields_and_thresholds")
                  → interact("set_thresholds", data) → interact("shutdown_interactive")

  5. PARTICLE PICKING      → job type: "blob_picker_gpu" or "template_picker_gpu"
     Input: "micrographs" ← connected to CTF estimation's "exposures"
     Output: "particles" (picked particle locations)
     Key params: diameter (Å) for blob picker

  6. INSPECT PICKS         → job type: "inspect_picks_v2" (interactive)
     Input: "micrographs" ← CTF's "exposures", "particles" ← picker's "particles"
     Output: "particles" (filtered picks)

  7. EXTRACTION            → job type: "extract_micrographs_cpu_parallel"
     Input: "micrographs" ← CTF's "exposures", "particles" ← picker's "particles"
     Output: "particles" (extracted particle images)
     Key params: box_size_pix

  8. 2D CLASSIFICATION     → job type: "class_2D" or "class_2D_new"
     Input: "particles" ← extraction's "particles"
     Output: "class_averages", "particles_class_0", ..., "particles_class_N"
     Key params: class2D_K (number of classes, default 50)

  9. SELECT 2D             → job type: "select_2D" (interactive)
     Input: "particles" ← 2D classification's output
     Output: "particles_selected" (good particles), "templates_selected" (good classes)
     Interactive: queue → wait_for_status("waiting") → interact("finish")

  10. AB-INITIO            → job type: "homo_abinit"
      Input: "particles" ← select 2D's "particles_selected"
      Output: "volume_class_0", ..., "volume_class_N"
      Key params: abinit_K (number of initial volumes)

  11. REFINEMENT           → job type: "homo_refine_new" or "nonuniform_refine_new"
      Input: "particles" ← select 2D's "particles_selected",
             "volume" ← ab-initio's "volume_class_0"
      Output: "volume", "particles"
      Key params: refine_symmetry (e.g. "C1", "D7", "O")

CONNECTION PATTERN: {"input_name": ("source_job_uid", "source_output_name")}
Example: {"exposures": ("J5", "micrographs")} connects J5's micrographs output
to the new job's exposures input.
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

class TestConnectionIn(_Base):
    pass

class ListProjectsIn(_Base):
    pass

class FindProjectIn(_Base):
    project_uid: str = Field(..., description="Project UID, e.g. 'P3'", min_length=1)

class ListWorkspacesIn(_Base):
    project_uid: str = Field(..., description="Project UID", min_length=1)

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
        "CryoSPARC job type string. Common types in pipeline order:\n"
        "  'import_movies' — import raw movies\n"
        "  'patch_motion_correction_multi' — motion correction\n"
        "  'patch_ctf_estimation_multi' — CTF estimation\n"
        "  'curate_exposures_v2' — interactive exposure curation\n"
        "  'blob_picker_gpu' — blob particle picker\n"
        "  'template_picker_gpu' — template-based picker\n"
        "  'inspect_picks_v2' — interactive pick inspection\n"
        "  'extract_micrographs_cpu_parallel' — particle extraction\n"
        "  'class_2D' or 'class_2D_new' — 2D classification\n"
        "  'select_2D' — interactive 2D class selection\n"
        "  'homo_abinit' — ab-initio reconstruction\n"
        "  'homo_refine_new' — homogeneous refinement\n"
        "  'nonuniform_refine_new' — non-uniform refinement\n"
        "  'hetero_refine' — heterogeneous refinement\n"
        "Use cryosparc_list_job_types to see ALL available types."
    ), min_length=1)
    title: Optional[str] = Field(default="")
    desc: Optional[str] = Field(default="")
    params: Optional[Dict[str, Any]] = Field(default=None, description=(
        "Job parameter values as key-value pairs. Examples by job type:\n"
        "  import_movies: {'blob_paths': '/data/*.tif', 'psize_A': 1.06, 'accel_kv': 300, 'cs_mm': 2.7, 'total_dose_e_per_A2': 50}\n"
        "  blob_picker_gpu: {'diameter': 150, 'diameter_max': 200}\n"
        "  extract_micrographs_cpu_parallel: {'box_size_pix': 256}\n"
        "  class_2D: {'class2D_K': 50}\n"
        "  homo_abinit: {'abinit_K': 3}\n"
        "  homo_refine_new: {'refine_symmetry': 'C1'}\n"
        "Use cryosparc_get_job_spec to see all parameters for a specific job."
    ))
    connections: Optional[Dict[str, Any]] = Field(default=None, description=(
        "Input connections: {input_name: (source_job_uid, source_output_name)}.\n"
        "Standard connections by job type:\n"
        "  motion_correction: {'movies': ('J1', 'imported_movies')}\n"
        "  ctf_estimation: {'exposures': ('J2', 'micrographs')}\n"
        "  curate_exposures: {'exposures': ('J3', 'exposures')}\n"
        "  blob_picker: {'micrographs': ('J3', 'exposures')}\n"
        "  extraction: {'micrographs': ('J3', 'exposures'), 'particles': ('J5', 'particles')}\n"
        "  class_2D: {'particles': ('J6', 'particles')}\n"
        "  select_2D: {'particles': ('J7', 'particles_class_0')}\n"
        "  homo_abinit: {'particles': ('J8', 'particles_selected')}\n"
        "  homo_refine: {'particles': ('J8', 'particles_selected'), 'volume': ('J9', 'volume_class_0')}\n"
        "Multiple sources: {'particles': [('J5', 'particles'), ('J6', 'particles')]}"
    ))

class SetParamIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    param_name: str = Field(..., description="Parameter name, e.g. 'box_size_pix', 'class2D_K', 'refine_symmetry', 'diameter'", min_length=1)
    value: Any = Field(..., description="Parameter value (string, number, boolean, or list)")

class SetParamsBatchIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    params: Dict[str, Any] = Field(..., description="Dictionary of {parameter_name: value} pairs")

class ConnectIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., description="Target job UID (the job receiving the input)", min_length=1)
    target_input: str = Field(..., description=(
        "Input slot name on the target job. Common input names:\n"
        "  'movies' — for motion correction (from import)\n"
        "  'exposures' — for CTF, curation, picking (from motion correction or CTF)\n"
        "  'micrographs' — for picking, extraction (from CTF estimation)\n"
        "  'particles' — for extraction, 2D class, ab-initio, refinement\n"
        "  'volume' — for refinement (from ab-initio)"
    ), min_length=1)
    source_job_uid: str = Field(..., description="Source job UID (the job providing the output)", min_length=1)
    source_output: str = Field(..., description=(
        "Source output name. Common output names:\n"
        "  'imported_movies' — from import_movies\n"
        "  'micrographs' — from motion correction\n"
        "  'exposures' — from CTF estimation\n"
        "  'exposures_accepted' — from curate_exposures\n"
        "  'particles' — from picking, extraction\n"
        "  'particles_selected' — from select_2D\n"
        "  'class_averages' — from 2D classification\n"
        "  'templates_selected' — from select_2D\n"
        "  'volume_class_0' — from ab-initio\n"
        "  'volume' — from refinement"
    ), min_length=1)

class QueueJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    lane: Optional[str] = Field(default=None, description="Scheduler lane name. Omit to use default. Use cryosparc_get_lanes to see available lanes.")

class JobRefIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)

class WaitJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    timeout: int = Field(default=600, ge=1, le=86400, description="Max seconds to wait. Motion correction/CTF: 600-1800. 2D class: 1800-3600. Refinement: 3600-7200.")
    error_on_incomplete: bool = Field(default=False, description="If True, raises error when job fails or is killed instead of returning status")

class WaitStatusIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    target_status: str = Field(..., description="Status to wait for. Use 'waiting' for interactive jobs (curate_exposures, inspect_picks, select_2D) before calling cryosparc_interact.", min_length=1)
    timeout: int = Field(default=300, ge=1, le=86400)

class InteractIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    action: str = Field(..., description=(
        "Interactive action name. Depends on job type:\n"
        "  curate_exposures_v2:\n"
        "    'get_fields_and_thresholds' — get curation data (call first)\n"
        "    'set_thresholds' — apply curation thresholds (pass data from get_fields_and_thresholds with modified thresholds)\n"
        "    'shutdown_interactive' — finalize curation and produce outputs\n"
        "  inspect_picks_v2:\n"
        "    'shutdown_interactive' — finalize pick inspection\n"
        "  select_2D:\n"
        "    'finish' — finalize 2D selection and produce particles_selected output"
    ), min_length=1)
    data: Optional[Any] = Field(default=None, description="Data payload for the action. Required for 'set_thresholds' (pass modified data from 'get_fields_and_thresholds'). Optional/None for shutdown and finish actions.")

class DatasetSummaryIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., description=(
        "Output group name to inspect. Common names:\n"
        "  'imported_movies', 'micrographs', 'exposures', 'exposures_accepted',\n"
        "  'particles', 'particles_selected', 'particles_class_0',\n"
        "  'class_averages', 'templates_selected',\n"
        "  'volume', 'volume_class_0', 'mask'"
    ), min_length=1)

class LoadOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None, description=(
        "Filter by data slots to reduce data transfer. Common slots:\n"
        "  'blob' — file paths to image data\n"
        "  'micrograph_blob' — micrograph image path\n"
        "  'ctf' — CTF parameters (df1_A, df2_A, ctf_fit_to_A, etc.)\n"
        "  'location' — particle position (center_x_frac, center_y_frac, micrograph_uid)\n"
        "  'pick_stats' — picking scores (ncc_score, power)\n"
        "  'alignments3D' — 3D alignment parameters\n"
        "  'mscope_params' — microscope settings"
    ))
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)

class ListFilesIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: Optional[str] = Field(default=None, description="Job UID. If omitted, lists project-level directory.")
    prefix: Optional[str] = Field(default=None)
    recursive: bool = Field(default=False)

class ListAssetsIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)

class GetJobSpecIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    spec_type: str = Field(default="params", description="'params' (parameter specification), 'inputs' (input slot names), or 'outputs' (output slot names)")

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
    slots: Optional[List[str]] = Field(default=None, description="Specific slots to connect, e.g. ['micrograph_blob', 'ctf', 'location']. Omit for all slots.")

class ExtAddOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_type: str = Field(..., description="'particle', 'exposure', 'volume', 'mask', 'template'", min_length=1)
    output_name: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None)
    passthrough: Optional[str] = Field(default=None, description="Input name to passthrough unmodified fields from")
    alloc: Optional[int] = Field(default=None, ge=1, description="Pre-allocate dataset with this many rows")

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
    slots: Optional[List[str]] = Field(default=None)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)

class ImportMoviesIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    blob_paths: str = Field(..., description="Glob path to movie files, e.g. '/data/movies/*.tif' or '/data/**/*.eer'", min_length=1)
    psize_A: float = Field(..., gt=0, description="Pixel size in Angstroms, e.g. 1.06")
    accel_kv: float = Field(..., gt=0, description="Acceleration voltage in kV, e.g. 300 or 200")
    cs_mm: float = Field(..., ge=0, description="Spherical aberration in mm, e.g. 2.7")
    total_dose_e_per_A2: float = Field(..., gt=0, description="Total electron dose in e⁻/Å², e.g. 50")
    gainref_path: Optional[str] = Field(default=None, description="Path to gain reference file, e.g. '/data/gain.mrc'")
    lane: Optional[str] = Field(default=None)
    wait: bool = Field(default=False, description="Block until import completes")

class PreprocessIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    import_job_uid: str = Field(..., description="UID of the completed Import Movies job, e.g. 'J1'", min_length=1)
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
        """Test connection to the CryoSPARC instance.

        ALWAYS CALL THIS FIRST before any other tool to verify the server is reachable.
        Returns the base URL, connected user, and connection status.
        If this fails, check that CryoSPARC is running and credentials are correct.
        """
        try:
            cs = get_client()
            cs.test_connection()
            return to_json({"connected": True, "base_url": cs.base_url, "user": str(cs.user)})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_projects", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_projects(params: ListProjectsIn) -> str:
        """List all CryoSPARC projects accessible to the current user.

        A project is the top-level container for all cryo-EM processing.
        Each project has a UID (e.g. 'P3'), title, description, and data directory.
        Use this to find the project_uid needed by all other tools.
        """
        try:
            cs = get_client()
            projects = list(cs.find_projects())
            items = [{"uid": p.uid, "title": p.title, "desc": p.desc} for p in projects]
            return to_json({"count": len(items), "projects": items})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_find_project", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_find_project(params: FindProjectIn) -> str:
        """Get detailed information about a single project.

        Returns UID, title, description, and the filesystem directory where data is stored.
        The directory path is useful for locating raw data files (movies, gain references).
        """
        try:
            cs = get_client()
            p = cs.find_project(params.project_uid)
            return to_json({"uid": p.uid, "title": p.title, "desc": p.desc, "dir": str(p.dir)})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_workspaces", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_workspaces(params: ListWorkspacesIn) -> str:
        """List workspaces in a project.

        A workspace organizes jobs into logical groups within a project.
        Each workspace has a UID (e.g. 'W1'). You need a workspace_uid to create jobs.
        If no workspace exists, use cryosparc_create_workspace to create one.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            workspaces = list(project.find_workspaces())
            items = [{"uid": w.uid, "title": w.title, "desc": w.desc} for w in workspaces]
            return to_json({"count": len(items), "workspaces": items})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_list_jobs", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_list_jobs(params: ListJobsIn) -> str:
        """List jobs in a project, optionally filtered by workspace and/or type.

        Returns UID, type, title, and status for each job.
        Job statuses: 'building' (not yet queued), 'queued', 'running',
        'waiting' (interactive, awaiting user action), 'completed', 'failed', 'killed'.

        Use this to find job UIDs for connecting outputs to new jobs.
        For example, before creating a 2D classification job, list jobs to find
        the completed extraction job UID whose particles you want to classify.
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
            return to_json({"count": len(items), "jobs": items})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_find_job", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_find_job(params: FindJobIn) -> str:
        """Get detailed information about a specific job.

        Returns UID, type, title, description, status, all parameter values,
        and the names of input/output slots.

        Use this to:
        - Check a job's status before deciding what to do next
        - See what outputs are available (e.g. 'particles', 'micrographs', 'volume')
        - Review parameter values of a completed job
        - Find output names to connect to the next job in the pipeline
        """
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
        """Get the full parameter, input, or output specification for a job.

        Use spec_type='params' to see ALL available parameters with their names,
        types, and default values. Essential before setting parameters on a new job.

        Use spec_type='inputs' to see what input slots the job expects
        (helps you know what to connect).

        Use spec_type='outputs' to see what outputs the job produces
        (helps you know what to connect to the next job).
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
        """List all job types available on this CryoSPARC instance.

        Returns the machine-readable type string (used in cryosparc_create_job),
        human title, and category for each job type.

        Categories: 'import', 'motion_correction', 'ctf_estimation',
        'exposure_curation', 'particle_picking', 'extraction', 'deep_picker',
        'particle_curation', 'reconstruction', 'refinement', 'ctf_refinement',
        'variability', 'flexibility', 'postprocessing', 'local_refinement',
        'helix', 'utilities', 'simulations', 'live', 'workflows'.
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
        """List available scheduler lanes (compute resource groups).

        A lane groups GPU workers. When queueing a job, you can specify a lane
        to control which machines run it. Omitting the lane uses the default.
        """
        try:
            cs = get_client()
            lanes = cs.get_lanes()
            return to_json({"lanes": lanes})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_get_targets", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_get_targets(params: GetTargetsIn) -> str:
        """List available scheduler targets (GPU workers/machines)."""
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
        """Create a new workspace in a project.

        A workspace is required to create jobs. If the user hasn't specified
        a workspace, either list existing ones with cryosparc_list_workspaces
        or create a new one here.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            ws = project.create_workspace(params.title, desc=params.desc)
            return to_json({"workspace_uid": ws.uid, "title": ws.title})
        except Exception as e:
            return fmt_error(e)

    @mcp.tool(name="cryosparc_create_job", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_job(params: CreateJobIn) -> str:
        """Create a new CryoSPARC processing job.

        This is the core tool for building processing pipelines. Each job has:
        - A type (what processing to do)
        - Parameters (how to do it)
        - Input connections (where the data comes from)

        The job is created in 'building' status. You can then:
        1. Set additional parameters with cryosparc_set_param
        2. Add more connections with cryosparc_connect
        3. Queue it for execution with cryosparc_queue_job

        You can also provide params and connections directly at creation time.
        The CryoSPARC scheduler resolves dependencies automatically — you can
        queue multiple chained jobs at once and they run in the correct order.

        STANDARD PIPELINE CONNECTIONS (see module docstring for full pipeline):
          Motion Correction:  connections={"movies": ("J_import", "imported_movies")}
          CTF Estimation:     connections={"exposures": ("J_mc", "micrographs")}
          Blob Picker:        connections={"micrographs": ("J_ctf", "exposures")}
          Extraction:         connections={"micrographs": ("J_ctf", "exposures"), "particles": ("J_pick", "particles")}
          2D Classification:  connections={"particles": ("J_extract", "particles")}
          Ab-Initio:          connections={"particles": ("J_select2D", "particles_selected")}
          Refinement:         connections={"particles": ("J_select2D", "particles_selected"), "volume": ("J_abinit", "volume_class_0")}
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)

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
        """Set a single parameter on a job in 'building' status.

        Use cryosparc_get_job_spec first to see available parameters for the job.
        Common parameters by job type:
          import_movies: blob_paths, psize_A, accel_kv, cs_mm, total_dose_e_per_A2
          blob_picker_gpu: diameter, diameter_max
          extract: box_size_pix, bin_size_pix
          class_2D: class2D_K (number of classes)
          homo_abinit: abinit_K (number of volumes)
          homo_refine: refine_symmetry (e.g. 'C1', 'D7', 'O')
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
        """Set multiple parameters at once on a job in 'building' status.

        More efficient than calling cryosparc_set_param repeatedly.
        Example: {"class2D_K": 50, "class2D_max_res": 6}
        """
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
        """Connect an output from a source job to an input on a target job.

        This wires the processing pipeline: data flows from source_output → target_input.
        The target job must be in 'building' status.

        Example: To connect CTF estimation results to a blob picker:
          target_input='micrographs', source_job_uid='J3', source_output='exposures'
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
        """Queue a job for execution on GPU workers.

        The job must be in 'building' status with all required inputs connected.
        CryoSPARC's scheduler automatically handles dependencies — you can queue
        multiple chained jobs simultaneously and they will run in order.

        For interactive jobs (curate_exposures, inspect_picks, select_2D),
        after queueing, use cryosparc_wait_for_status with target_status='waiting',
        then use cryosparc_interact to perform the interactive actions.
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
        """Wait for a job to finish (reach 'completed', 'failed', or 'killed' status).

        Use after queueing a job when you need its outputs before creating the next job.
        For a full pipeline, you DON'T need to wait between every job — CryoSPARC handles
        dependencies. Only wait when you need to inspect results or make decisions.

        Do NOT use this for interactive jobs — use cryosparc_wait_for_status instead.
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
        """Wait for a job to reach a specific status.

        ESSENTIAL for interactive jobs. After queueing an interactive job
        (curate_exposures_v2, inspect_picks_v2, select_2D), it transitions to
        'waiting' status when ready for interaction. Use this tool with
        target_status='waiting', then call cryosparc_interact.

        Typical interactive job workflow:
          1. cryosparc_create_job (type='curate_exposures_v2')
          2. cryosparc_queue_job
          3. cryosparc_wait_for_status (target_status='waiting')
          4. cryosparc_interact (action='get_fields_and_thresholds')
          5. cryosparc_interact (action='set_thresholds', data=...)
          6. cryosparc_interact (action='shutdown_interactive')
          7. cryosparc_wait_for_done
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
        """Perform an interactive action on a job in 'waiting' status.

        Used for interactive jobs that require user input:

        CURATE EXPOSURES (curate_exposures_v2):
          1. action='get_fields_and_thresholds' → returns curation data with fields and thresholds
          2. action='set_thresholds', data=<modified data> → apply new thresholds
          3. action='shutdown_interactive' → finalize, produces 'exposures_accepted' output

        INSPECT PICKS (inspect_picks_v2):
          1. action='shutdown_interactive' → finalize pick inspection

        SELECT 2D (select_2D):
          1. action='finish' → finalize selection, produces 'particles_selected' and 'templates_selected'

        The job must be in 'waiting' status. Use cryosparc_wait_for_status first.
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
        """Kill a running or queued job. The job transitions to 'killed' status."""
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
        """Clear a completed/failed/killed job, resetting it to 'building' status.

        This lets you modify parameters and re-run the job.
        WARNING: existing outputs are removed.
        """
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
        """Get the schema of a job output dataset: row count, column names, and data types.

        ALWAYS call this before cryosparc_load_output to understand the data structure.
        This avoids loading too much data. Check available columns, then request
        only the columns you need using the 'slots' parameter in cryosparc_load_output.
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
        """Load paginated data rows from a job output.

        Use 'slots' to filter columns and reduce data transfer:
          slots=['ctf'] → only CTF columns (df1_A, df2_A, ctf_fit_to_A, ...)
          slots=['location'] → only particle positions (center_x_frac, center_y_frac, ...)
          slots=['blob'] → only file paths to image data

        For large datasets (>10,000 rows), you MUST specify slots and use pagination.
        Call cryosparc_dataset_summary first to see what's available.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)

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
            rows = [dict(r) for r in list(dset.rows())[start:end]]

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
        """List files in a project or job directory on the filesystem.

        Useful for finding raw data files, MRC volumes, .cs dataset files,
        or any output files produced by jobs.
        """
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
        """List image/plot assets stored for a job (previews, diagnostic plots, FSC curves).

        Assets are stored in CryoSPARC's database, not on the filesystem.
        They include diagnostic plots, 2D class previews, FSC curves, etc.
        """
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
    # EXTERNAL JOBS (third-party tool integration)
    # ==============================================================

    @mcp.tool(name="cryosparc_create_external_job", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_external_job(params: CreateExternalJobIn) -> str:
        """Create an External Job for importing results from third-party tools (RELION, crYOLO, etc.).

        External Jobs let you inject data from outside CryoSPARC back into the pipeline.
        Lifecycle:
          1. Create external job (this tool)
          2. Connect inputs with cryosparc_external_connect (with optional slot filtering)
          3. Add outputs with cryosparc_external_add_output (allocate dataset)
          4. Start with cryosparc_external_start
          5. Populate the allocated dataset (programmatically)
          6. Save with cryosparc_external_save_output
          7. Stop with cryosparc_external_stop
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

        The 'slots' parameter lets you select specific data fields:
          slots=['micrograph_blob'] — only micrograph image paths
          slots=['micrograph_blob', 'ctf'] — images + CTF data
          slots=['location'] — only particle coordinates
        Unspecified slots are passed through from the source.
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
        """Add an output slot to an External Job, optionally pre-allocating a dataset.

        Use alloc=N to create an empty dataset with N rows that you can populate.
        Use passthrough='input_name' to inherit fields from a connected input.
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
        """Start an External Job — transitions to 'started' status. Call after adding all outputs."""
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
        """Save (persist) a populated output dataset. The job must be started."""
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
        """Stop an External Job — marks as completed. Outputs become available for downstream jobs."""
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
        """Load data from a connected input of an External Job. Useful for inspecting incoming data."""
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
    # WORKFLOW HELPERS (multi-step shortcuts)
    # ==============================================================

    @mcp.tool(name="cryosparc_workflow_import_movies", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_workflow_import_movies(params: ImportMoviesIn) -> str:
        """Import raw movie files into CryoSPARC in one step.

        This is STEP 1 of any cryo-EM pipeline. Creates an Import Movies job
        with all acquisition parameters set and immediately queues it.

        After this job completes, its output 'imported_movies' is used as input
        for the next step: Motion Correction.

        The user MUST provide: pixel size, voltage, spherical aberration, total dose,
        and the path to the movie files. Ask for these if not specified.
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
        """Run STEPS 2-3: Patch Motion Correction + Patch CTF Estimation.

        Creates two chained jobs connected to the import job:
          1. Patch Motion Correction ← import's 'imported_movies'
          2. Patch CTF Estimation ← motion correction's 'micrographs'

        Both jobs are queued simultaneously — CryoSPARC's scheduler runs them in order.
        After CTF estimation completes, its output 'exposures' is the input for:
          - Particle picking (connects to 'micrographs' input)
          - Exposure curation (connects to 'exposures' input)
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            workspace = project.find_workspace(params.workspace_uid)

            mc_job = workspace.create_job(
                "patch_motion_correction_multi",
                connections={"movies": (params.import_job_uid, "imported_movies")},
            )
            ctf_job = workspace.create_job(
                "patch_ctf_estimation_multi",
                connections={"exposures": (mc_job.uid, "micrographs")},
            )

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
