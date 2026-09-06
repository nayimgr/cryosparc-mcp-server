"""CryoSPARC MCP tools — results interpretation and visual inspection.

These are the tools that let an agent *decide* rather than merely *act*.

``cryosparc_job_results`` turns a finished job into the handful of numbers that
actually drive the next step (per-class particle distribution, CTF fit
resolution, motion, defocus, pick scores) instead of forcing the agent to load
and parse raw datasets.

``cryosparc_render_class_averages`` renders 2D class averages as an inline
image so a vision-capable agent can judge which classes are good.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from .connection import get_client
from .helpers import col_stats, dataset_fields, params_to_dict, to_json, tool_error


class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class JobResultsIn(_Base):
    project_uid: str = Field(..., description="Project UID, e.g. 'P3'", min_length=1)
    job_uid: str = Field(..., description="Job UID, e.g. 'J42'", min_length=1)
    max_classes: int = Field(default=100, ge=1, le=1000,
                             description="Maximum number of classes to list")


class RenderClassAveragesIn(_Base):
    project_uid: str = Field(..., min_length=1)
    job_uid: str = Field(..., description="Class2D / Select2D job UID", min_length=1)
    output_name: str = Field(default="class_averages",
                             description="Output holding the averages ('class_averages', "
                                         "or 'templates_selected' for a Select 2D job)")
    max_classes: int = Field(default=100, ge=1, le=400)
    columns: int = Field(default=10, ge=1, le=20, description="Montage columns")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _first_output(job: Any, *keywords: str) -> Optional[str]:
    """First output name containing one of the keywords (in keyword order)."""
    try:
        outs = list(job.outputs.keys()) if job.outputs else []
    except Exception:  # noqa: BLE001
        return None
    for kw in keywords:
        for o in outs:
            if kw in o:
                return o
    return None


def _try_load(job: Any, name: str, slots: Optional[List[str]] = None) -> Any:
    for attempt in ((slots,) if slots else ()) + (None,):
        try:
            return job.load_output(name, slots=attempt) if attempt else job.load_output(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def _class_distribution(dset: Any, field: str, max_classes: int) -> Optional[Dict[str, Any]]:
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        arr = np.asarray(dset[field])
        values, counts = np.unique(arr, return_counts=True)
    except Exception:  # noqa: BLE001
        return None
    total = int(counts.sum())
    if total == 0:
        return None
    rows = [{"class": int(v), "particles": int(c), "fraction": round(float(c) / total, 5)}
            for v, c in zip(values, counts)]
    rows.sort(key=lambda r: r["particles"], reverse=True)
    return {"total_particles": total, "num_classes": len(rows),
            "classes": rows[:max_classes]}


def _summarise_job(job: Any, max_classes: int) -> Dict[str, Any]:
    """Derive decision-relevant metrics from a job's output datasets."""
    out: Dict[str, Any] = {
        "job_uid": job.uid, "type": job.type,
        "status": job.status, "title": job.title,
    }
    params = params_to_dict(job.params)
    try:
        outputs = list(job.outputs.keys()) if job.outputs else []
    except Exception:  # noqa: BLE001
        outputs = []
    out["outputs"] = outputs

    metrics: Dict[str, Any] = {}
    next_steps: List[str] = []

    # ---- particles ----
    pname = _first_output(job, "particle")
    if pname:
        dset = _try_load(job, pname)
        if dset is not None:
            fields = dataset_fields(dset)
            metrics["num_particles"] = len(dset)
            if "alignments2D/class" in fields:
                dist = _class_distribution(dset, "alignments2D/class", max_classes)
                if dist:
                    metrics["class_distribution_2D"] = dist
                    next_steps.append(
                        "Inspect the classes visually with cryosparc_render_class_averages, "
                        "then keep the good ones with a select_2D job."
                    )
            elif "alignments3D/class" in fields:
                dist = _class_distribution(dset, "alignments3D/class", max_classes)
                if dist:
                    metrics["class_distribution_3D"] = dist
            for col, key in (("pick_stats/ncc_score", "pick_ncc_score"),
                             ("pick_stats/power", "pick_power")):
                st = col_stats(dset, col)
                if st:
                    metrics[key] = st

    # ---- exposures / micrographs ----
    ename = _first_output(job, "exposure", "micrograph")
    if ename:
        dset = _try_load(job, ename)
        if dset is not None:
            fields = dataset_fields(dset)
            metrics["num_exposures"] = len(dset)
            for col, key in (("ctf/df1_A", "defocus_df1_A"),
                             ("ctf/df2_A", "defocus_df2_A"),
                             ("ctf/ctf_fit_to_A", "ctf_fit_resolution_A"),
                             ("ctf/ice_thickness_rel", "ice_thickness_rel"),
                             ("rigid_motion/total_motion", "total_motion"),
                             ("spline_motion/total_motion", "total_motion")):
                if key in metrics:
                    continue
                st = col_stats(dset, col)
                if st:
                    metrics[key] = st
            if "ctf/df1_A" in fields and "ctf/df2_A" in fields:
                try:
                    import numpy as np
                    df1 = np.asarray(dset["ctf/df1_A"], dtype="float64")
                    df2 = np.asarray(dset["ctf/df2_A"], dtype="float64")
                    ast = np.abs(df1 - df2)
                    ast = ast[np.isfinite(ast)]
                    if ast.size:
                        metrics["astigmatism_A"] = {
                            "n": int(ast.size), "min": float(ast.min()),
                            "mean": float(ast.mean()), "max": float(ast.max()),
                        }
                except Exception:  # noqa: BLE001
                    pass

    # ---- volumes ----
    vol_outputs = [o for o in outputs if "volume" in o]
    if vol_outputs:
        metrics["volume_outputs"] = vol_outputs

    # ---- key parameters ----
    interesting = ("box_size_pix", "bin_size_pix", "psize_A", "class2D_K", "abinit_K",
                   "refine_symmetry", "abinit_symmetry", "hetero_num_classes",
                   "refine_res_init", "diameter", "diameter_max")
    key_params = {k: params[k] for k in interesting if k in params}
    if key_params:
        out["key_params"] = key_params

    if any(tok in (job.type or "") for tok in ("refine", "abinit", "reconstruct", "sharpen")):
        next_steps.append(
            "Final resolution / FSC is published as a job asset (plot), not as a dataset "
            "column — use cryosparc_list_assets on this job to locate the FSC curve."
        )

    out["metrics"] = metrics
    if next_steps:
        out["next_steps"] = next_steps
    if not metrics:
        out["note"] = ("No metrics could be derived — the job may still be running, or its "
                       "outputs may not be loadable yet. Check status with cryosparc_find_job.")
    return out


def _montage_png(images: List[Any], out_png: str, cols: int) -> None:
    import math

    import numpy as np
    from PIL import Image as PILImage

    n = len(images)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    canvas = np.zeros((rows * h, cols * w), dtype="uint8")
    for i, img in enumerate(images):
        arr = np.asarray(img, dtype="float32")
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        norm = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        r, c = divmod(i, cols)
        canvas[r * h:r * h + arr.shape[0], c * w:c * w + arr.shape[1]] = norm.astype("uint8")
    PILImage.fromarray(canvas, mode="L").save(out_png)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def register_results_tools(mcp: FastMCP) -> None:
    """Register results-interpretation and rendering tools."""

    @mcp.tool(
        name="cryosparc_job_results",
        annotations={"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": True},
    )
    async def cryosparc_job_results(params: JobResultsIn) -> str:
        """Extract the KEY METRICS of a job so you can decide what to do next.

        Auto-detects what the job produced and returns only decision-relevant numbers:
          - 2D/3D classification -> per-class particle counts and fractions (ranked)
          - picking              -> particle count and NCC/power score statistics
          - CTF estimation       -> defocus, astigmatism and CTF-fit resolution stats
          - motion correction    -> total-motion statistics
          - extraction           -> particle count and box size
          - refinement           -> particle count, symmetry, volume outputs

        Prefer this over cryosparc_load_output when you only need to judge a job's
        quality — it is far cheaper in tokens and needs no column knowledge.
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)
            return to_json(_summarise_job(job, params.max_classes))
        except ToolError:
            raise
        except Exception as e:
            raise tool_error(e) from e

    @mcp.tool(
        name="cryosparc_render_class_averages",
        annotations={"readOnlyHint": True, "destructiveHint": False,
                     "idempotentHint": True, "openWorldHint": True},
    )
    async def cryosparc_render_class_averages(params: RenderClassAveragesIn) -> Any:
        """Render 2D class averages as an inline PNG montage you can actually look at.

        Use after a Class 2D (output 'class_averages') or Select 2D job
        (output 'templates_selected') to visually judge which classes are good
        before selecting particles.

        Requires numpy and pillow (pip install 'cryosparc-mcp-server[report]').
        """
        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            job = project.find_job(params.job_uid)

            dset = _try_load(job, params.output_name, slots=["blob"])
            if dset is None:
                available = list(job.outputs.keys()) if job.outputs else []
                raise ToolError(
                    f"Could not load output '{params.output_name}' from {params.job_uid}. "
                    f"Available outputs: {available}"
                )
            fields = dataset_fields(dset)
            if "blob/path" not in fields:
                raise ToolError(
                    f"Output '{params.output_name}' has no 'blob/path' column "
                    f"(columns: {fields[:10]}), so it does not hold renderable images."
                )

            rows = list(dset.rows())[: params.max_classes]
            if not rows:
                raise ToolError(f"Output '{params.output_name}' is empty.")

            has_idx = "blob/idx" in fields
            blobs: Dict[str, Any] = {}
            for path in {r["blob/path"] for r in rows}:
                try:
                    _, data = project.download_mrc(path)
                    blobs[path] = data
                except Exception:  # noqa: BLE001
                    continue
            if not blobs:
                raise ToolError("Could not download any class-average MRC file.")

            images = []
            for r in rows:
                data = blobs.get(r["blob/path"])
                if data is None:
                    continue
                if getattr(data, "ndim", 2) == 3:
                    idx = int(r["blob/idx"]) if has_idx else 0
                    images.append(data[idx] if idx < data.shape[0] else data[0])
                else:
                    images.append(data)
            if not images:
                raise ToolError("No renderable images found in the output.")

            out_png = os.path.join(
                tempfile.gettempdir(),
                f"cryosparc_{params.project_uid}_{params.job_uid}_{params.output_name}.png",
            )
            try:
                _montage_png(images, out_png, params.columns)
            except ImportError as exc:
                raise ToolError(
                    "numpy and pillow are required to render class averages. "
                    "Install with: pip install 'cryosparc-mcp-server[report]'"
                ) from exc

            caption = (f"{len(images)} class average(s) from {params.job_uid}"
                       f".{params.output_name} — montage saved to {out_png}")
            return [caption, Image(path=out_png)]
        except ToolError:
            raise
        except Exception as e:
            raise tool_error(e) from e
