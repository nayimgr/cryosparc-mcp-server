"""CryoSPARC MCP tools — External Jobs and high-level workflow helpers.

This module completes the tool set with:

EXTERNAL JOBS (save externally-computed results back into CryoSPARC):
    cryosparc_create_external_job   — create an External Job in a workspace
    cryosparc_external_connect      — add an input slot and wire it to an upstream output
    cryosparc_external_add_output   — declare an output slot, optionally pre-allocating rows
    cryosparc_external_start        — set the job to 'running' (or 'waiting')
    cryosparc_external_save_output  — populate and persist an output dataset
    cryosparc_external_stop         — mark the job 'completed' (or 'failed' with an error)
    cryosparc_external_load_input   — load rows from a connected input

WORKFLOW HELPERS (multi-step pipeline shortcuts):
    cryosparc_workflow_import_movies — create (and optionally queue) an import_movies job
    cryosparc_workflow_preprocess    — chain patch motion-correction + patch CTF estimation

TYPICAL EXTERNAL-JOB FLOW (each step is a separate tool call):
    1. cryosparc_create_external_job              -> returns job_uid (status 'building')
    2. cryosparc_external_connect (optional)      -> add input + connect to an upstream output
    3. cryosparc_external_add_output              -> declare the output slots (alloc optional)
    4. cryosparc_external_start                   -> status 'running'
    5. cryosparc_external_save_output             -> write the rows
    6. cryosparc_external_stop                    -> status 'completed'

Because each MCP call is independent, the ExternalJobController is re-fetched
with project.find_external_job() on every call. Any dataset pre-allocated by
cryosparc_external_add_output is kept in a small in-process registry so it can
be reused by cryosparc_external_save_output.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

from .connection import get_client
from .helpers import dataset_fields, params_to_dict, to_json, tool_error


# ====================================================================
# In-process registry of pre-allocated external-job output datasets.
# Keyed by (project_uid, job_uid, output_name). Lives for the lifetime
# of the server process, which is enough to bridge add_output -> save_output.
# ====================================================================
_alloc_registry: Dict[Tuple[str, str, str], Any] = {}


# ====================================================================
# Pydantic input models
# ====================================================================

class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


_DATATYPES_DOC = (
    "CryoSPARC data type. One of: 'exposure', 'particle', 'template', 'volume', "
    "'volume_multi', 'mask', 'live', 'ml_model', 'symmetry_candidate', 'flex_mesh', "
    "'flex_model', 'hyperparameter', 'denoise_model', 'annotation_model'."
)


class CreateExternalJobIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    title: str = Field(default="External Job", max_length=200)
    desc: Optional[str] = Field(default="")


class ExternalConnectIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., description="External job UID, e.g. 'J42'", min_length=1)
    target_input: str = Field(..., description="Name of the input slot on this external job", min_length=1)
    source_job_uid: str = Field(..., description="Upstream job UID providing the data", min_length=1)
    source_output: str = Field(..., description="Output name on the upstream job", min_length=1)
    input_type: Optional[str] = Field(
        default=None,
        description=(
            "If provided, the input slot is created with add_input() before connecting. "
            + _DATATYPES_DOC + " Omit if the input slot already exists."
        ),
    )
    slots: Optional[List[str]] = Field(
        default=None,
        description="Slot filter, e.g. ['micrograph_blob', 'ctf']. Restricts which fields are required/loaded.",
    )


class ExternalAddOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    type: str = Field(..., description=_DATATYPES_DOC)
    name: str = Field(..., description="Output name key, e.g. 'picked_particles'", min_length=1)
    slots: Optional[List[str]] = Field(
        default=None,
        description="Slots created for this output, e.g. ['location', 'pick_stats']. Omit passthrough slots.",
    )
    passthrough: Optional[str] = Field(
        default=None,
        description="Name of a previously added input from which to inherit (passthrough) slots.",
    )
    title: Optional[str] = Field(default=None)
    alloc: Optional[int] = Field(
        default=None,
        ge=0,
        description="Pre-allocate this many empty rows now. The allocated dataset is kept until cryosparc_external_save_output is called.",
    )


class ExternalStartIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    status: str = Field(default="running", description="'running' or 'waiting'")


class ExternalSaveOutputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    output_name: str = Field(..., min_length=1)
    data: Dict[str, List[Any]] = Field(
        ...,
        description=(
            "Column data as {column_name: [values...]}. All columns must have the same length. "
            "Column names match the output slots, e.g. 'location/center_x_frac'. "
            "The job must be 'running' (call cryosparc_external_start first)."
        ),
    )
    num_rows: Optional[int] = Field(
        default=None,
        ge=0,
        description="Row count to allocate if the output was not pre-allocated. Defaults to the length of the data columns.",
    )


class ExternalStopIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    error: Optional[str] = Field(
        default=None,
        description="If provided, the job is marked 'failed' with this error message. Otherwise marked 'completed'.",
    )


class ExternalLoadInputIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., min_length=1)
    input_name: str = Field(..., min_length=1)
    slots: Optional[List[str]] = Field(default=None, description="Optional column/slot filter")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)


class WorkflowImportMoviesIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    movies_path: str = Field(..., description="Glob path to movies, e.g. '/data/*.tif'", min_length=1)
    psize_A: float = Field(..., description="Pixel size in Angstroms, e.g. 1.06", gt=0)
    accel_kv: float = Field(..., description="Accelerating voltage in kV, e.g. 300", gt=0)
    cs_mm: float = Field(..., description="Spherical aberration in mm, e.g. 2.7", gt=0)
    total_dose_e_per_A2: float = Field(..., description="Total exposure dose in e/A^2, e.g. 50", gt=0)
    gain_path: Optional[str] = Field(default=None, description="Optional gain reference path")
    lane: Optional[str] = Field(default=None, description="Scheduler lane. If set, the job is queued.")
    title: Optional[str] = Field(default="Import Movies")


class WorkflowPreprocessIn(_Base):
    project_uid: str = Field(..., min_length=1)
    workspace_uid: str = Field(..., min_length=1)
    import_job_uid: str = Field(..., description="UID of the upstream import_movies job", min_length=1)
    import_output: str = Field(default="imported_movies", description="Output name on the import job to connect from")
    motion_params: Optional[Dict[str, Any]] = Field(default=None, description="Extra params for patch_motion_correction_multi")
    ctf_params: Optional[Dict[str, Any]] = Field(default=None, description="Extra params for patch_ctf_estimation_multi")
    lane: Optional[str] = Field(default=None, description="Scheduler lane. If set, both jobs are queued.")


# ====================================================================
# Registration
# ====================================================================

def register_extra_tools(mcp: FastMCP) -> None:
    """Register External Job tools and workflow helpers on the MCP server."""

    # ==============================================================
    # EXTERNAL JOBS
    # ==============================================================

    @mcp.tool(name="cryosparc_create_external_job", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_create_external_job(params: CreateExternalJobIn) -> str:
        """Create an External Job in a workspace.

        External Jobs let you inject results produced outside CryoSPARC
        (custom pickers, external refinements, etc.) back into a project so
        downstream CryoSPARC jobs can consume them.

        The job starts in 'building' status. Next steps:
        cryosparc_external_connect (optional inputs), cryosparc_external_add_output,
        cryosparc_external_start, cryosparc_external_save_output, cryosparc_external_stop.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.create_external_job(params.workspace_uid, title=params.title, desc=params.desc or "")
            return to_json({"job_uid": job.uid, "type": job.type, "status": job.status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_external_connect", annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
    async def cryosparc_external_connect(params: ExternalConnectIn) -> str:
        """Add an input slot to an external job and connect it to an upstream output.

        If 'input_type' is given, the input slot is created with add_input()
        first (with optional 'slots' filter), then connected. If the input
        already exists, omit 'input_type' and just connect.

        Example: feed micrographs from a CTF job into a custom picker:
          target_input='input_micrographs', input_type='exposure',
          slots=['micrograph_blob','ctf'], source_job_uid='J3', source_output='exposures'
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)

            created_input = None
            if params.input_type:
                existing = list(job.inputs.keys()) if job.inputs else []
                if params.target_input not in existing:
                    created_input = job.add_input(
                        type=params.input_type,
                        name=params.target_input,
                        slots=params.slots or [],
                    )

            if params.slots:
                job.connect(params.target_input, params.source_job_uid, params.source_output, slots=params.slots)
            else:
                job.connect(params.target_input, params.source_job_uid, params.source_output)

            return to_json({
                "success": True,
                "input_created": created_input,
                "connection": f"{params.source_job_uid}.{params.source_output} -> {params.job_uid}.{params.target_input}",
            })
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_external_add_output", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_add_output(params: ExternalAddOutputIn) -> str:
        """Declare an output slot on an external job, optionally pre-allocating rows.

        Outputs must be declared before data can be saved. Use 'slots' to list
        the fields this output will contain (e.g. ['location','pick_stats']).
        Use 'passthrough' to inherit fields from a previously added input.

        If 'alloc' (a row count) is given, an empty dataset is allocated now and
        held in memory so cryosparc_external_save_output can populate it later.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)

            kwargs: Dict[str, Any] = {"type": params.type, "name": params.name}
            if params.slots:
                kwargs["slots"] = params.slots
            if params.passthrough:
                kwargs["passthrough"] = params.passthrough
            if params.title:
                kwargs["title"] = params.title

            allocated = None
            if params.alloc is not None:
                dset = job.add_output(alloc=params.alloc, **kwargs)
                _alloc_registry[(params.project_uid, params.job_uid, params.name)] = dset
                allocated = params.alloc
            else:
                job.add_output(**kwargs)

            return to_json({"success": True, "output": params.name, "allocated_rows": allocated})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_external_start", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_start(params: ExternalStartIn) -> str:
        """Start an external job: set status to 'running' (or 'waiting').

        Outputs can only be saved while the job is 'running'. Call this after
        declaring outputs and before cryosparc_external_save_output.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            job.start(status=params.status)
            job.refresh()
            return to_json({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_external_save_output", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_save_output(params: ExternalSaveOutputIn) -> str:
        """Populate and persist an output dataset on a running external job.

        Provide 'data' as {column_name: [values...]} with equal-length columns.
        If the output was pre-allocated by cryosparc_external_add_output, that
        dataset is reused; otherwise a new one is allocated with 'num_rows'
        (defaulting to the length of the provided columns).

        The job must be 'running' (call cryosparc_external_start first).
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)

            # Determine row count
            lengths = {len(v) for v in params.data.values()} if params.data else {0}
            if len(lengths) > 1:
                return to_json({"error": f"All columns must have equal length; got {sorted(lengths)}."})
            data_len = next(iter(lengths)) if lengths else 0
            n_rows = params.num_rows if params.num_rows is not None else data_len

            key = (params.project_uid, params.job_uid, params.output_name)
            dset = _alloc_registry.pop(key, None)
            if dset is None or len(dset) != n_rows:
                dset = job.alloc_output(params.output_name, n_rows)

            # Populate columns
            written = []
            for col, values in params.data.items():
                if len(values) != len(dset):
                    return to_json({"error": f"Column '{col}' has {len(values)} values but dataset has {len(dset)} rows."})
                try:
                    dset[col] = values
                    written.append(col)
                except Exception as col_err:
                    return to_json({"error": f"Failed to set column '{col}': {col_err}"})

            job.save_output(params.output_name, dset)
            return to_json({"success": True, "output": params.output_name, "rows": len(dset), "columns_written": written})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_external_stop", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_external_stop(params: ExternalStopIn) -> str:
        """Finish an external job: mark 'completed', or 'failed' if an error is given.

        Call after all outputs are saved. Once completed, the job's outputs are
        available to downstream CryoSPARC jobs.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            if params.error:
                job.stop(params.error)
            else:
                job.stop()
            job.refresh()
            # Drop any leftover allocations for this job
            for k in [k for k in _alloc_registry if k[0] == params.project_uid and k[1] == params.job_uid]:
                _alloc_registry.pop(k, None)
            return to_json({"job_uid": job.uid, "status": job.status})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_external_load_input", annotations={"readOnlyHint": True, "destructiveHint": False})
    async def cryosparc_external_load_input(params: ExternalLoadInputIn) -> str:
        """Load paginated rows from a connected input of an external job.

        Use this to read data that was wired in with cryosparc_external_connect
        (e.g. micrographs or particles) so it can be processed externally.
        Use 'slots' to limit columns and 'offset'/'limit' to paginate.
        """
        from itertools import islice
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_external_job(params.job_uid)
            dset = job.load_input(params.input_name, slots=params.slots) if params.slots else job.load_input(params.input_name)
            total = len(dset)
            col_names = dataset_fields(dset)
            start = min(params.offset, total)
            end = min(start + params.limit, total)
            rows = [dict(r) for r in islice(dset.rows(), start, end)]
            return to_json({
                "input": params.input_name,
                "total_rows": total, "offset": start, "count": len(rows),
                "has_more": end < total, "next_offset": end if end < total else None,
                "columns": col_names, "rows": rows,
            })
        except Exception as e:
            raise tool_error(e) from e

    # ==============================================================
    # WORKFLOW HELPERS
    # ==============================================================

    @mcp.tool(name="cryosparc_workflow_import_movies", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_workflow_import_movies(params: WorkflowImportMoviesIn) -> str:
        """Create an import_movies job with the essential acquisition parameters.

        One-step shortcut over create_job + set_params. If 'lane' is provided,
        the job is queued immediately; otherwise it is left in 'building' status
        so you can adjust parameters before queueing.

        Returns the job UID and final status.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job_params: Dict[str, Any] = {
                "blob_paths": params.movies_path,
                "psize_A": params.psize_A,
                "accel_kv": params.accel_kv,
                "cs_mm": params.cs_mm,
                "total_dose_e_per_A2": params.total_dose_e_per_A2,
            }
            if params.gain_path:
                job_params["gainref_path"] = params.gain_path

            job = project.create_job(
                params.workspace_uid, "import_movies",
                params=job_params, title=params.title or "Import Movies",
            )
            queued = False
            if params.lane:
                job.queue(params.lane)
                job.refresh()
                queued = True
            return to_json({"job_uid": job.uid, "type": job.type, "status": job.status, "queued": queued})
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(name="cryosparc_workflow_preprocess", annotations={"readOnlyHint": False, "destructiveHint": False})
    async def cryosparc_workflow_preprocess(params: WorkflowPreprocessIn) -> str:
        """Chain Patch Motion Correction then Patch CTF Estimation off an import job.

        Creates patch_motion_correction_multi connected to the import job's
        movies output, then patch_ctf_estimation_multi connected to the motion
        job's micrographs output. If 'lane' is provided, both jobs are queued;
        the scheduler resolves the dependency automatically.

        Returns both new job UIDs.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)

            motion = project.create_job(
                params.workspace_uid, "patch_motion_correction_multi",
                connections={"movies": (params.import_job_uid, params.import_output)},
                params=params.motion_params or {}, title="Patch Motion Correction",
            )

            ctf = project.create_job(
                params.workspace_uid, "patch_ctf_estimation_multi",
                connections={"exposures": (motion.uid, "micrographs")},
                params=params.ctf_params or {}, title="Patch CTF Estimation",
            )

            queued = False
            if params.lane:
                motion.queue(params.lane)
                ctf.queue(params.lane)
                motion.refresh()
                ctf.refresh()
                queued = True

            return to_json({
                "motion_job_uid": motion.uid, "motion_status": motion.status,
                "ctf_job_uid": ctf.uid, "ctf_status": ctf.status,
                "queued": queued,
            })
        except Exception as e:
            raise tool_error(e) from e
