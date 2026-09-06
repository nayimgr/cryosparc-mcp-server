
"""CryoSPARC session report generator — comprehensive, adaptive.

Scans ALL completed jobs in a workspace. For each job that has data,
generates relevant statistics and plots. Gracefully skips jobs with
no accessible data.

Extra dependencies: matplotlib, numpy
"""
from __future__ import annotations

import io
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

from .connection import get_client
from .helpers import TASK_MODE, dataset_fields, jsonable, tool_error


class GenerateReportIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    project_uid: str = Field(..., description="Project UID, e.g. 'P3'", min_length=1)
    workspace_uid: str = Field(..., description="Workspace/session UID, e.g. 'W1' or 'S1'", min_length=1)
    output_path: Optional[str] = Field(
        default=None,
        description="Where to save the PDF. Default: /tmp/cryosparc_report_<project>_<workspace>.pdf",
    )
    num_sample_micrographs: int = Field(default=10, ge=1, le=50)
    include_running: bool = Field(
        default=True,
        description="Include jobs that are still running/waiting (useful for Live sessions). If False, only completed jobs are included."
    )


def register_report_tools(mcp: FastMCP) -> None:

    @mcp.tool(
        task=TASK_MODE, name="cryosparc_generate_live_report", title="Generate Session Report",
        annotations={"openWorldHint": True, "readOnlyHint": True, "destructiveHint": False},
    )
    async def cryosparc_generate_live_report(params: GenerateReportIn) -> Dict[str, Any]:
        """Generate a comprehensive PDF report from a CryoSPARC workspace or Live session.

        Scans ALL completed (and optionally running/waiting) jobs and reports on every job that has data:
        - Import: acquisition parameters
        - Motion Correction: drift statistics
        - CTF Estimation: defocus, astigmatism, resolution, and ice thickness (if available)
        - Exposure Curation: accepted/rejected counts
        - Particle Picking: pick counts, score distributions, and detailed picking settings
        - Random micrographs with picks overlaid (number controlled by num_sample_micrographs)
        - For each sampled micrograph: drift curve (motion) and CTF fit curve
        - Extraction: particle counts & box sizes
        - 2D Classification: class average gallery
        - Select 2D: selected class gallery
        - Ab-Initio: volume count & class info
        - Refinement (Homo/Hetero/NU): resolution, particle count
        - CTF Refinement: refined defocus stats
        - 3D Variability / Classification: component/class info
        - Postprocessing: resolution stats
        - Any other job: basic parameter summary

        Jobs with no accessible data are silently skipped.
        Returns the path to the generated PDF.

        Requires: pip install matplotlib numpy
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import numpy as np
        except ImportError:
            return jsonable({"error": "matplotlib and numpy required. Install: pip install matplotlib numpy"})

        try:
            cs = get_client()
            project = cs.find_project(params.project_uid)
            workspace = project.find_workspace(params.workspace_uid)

            pdf_path = params.output_path or f"/tmp/cryosparc_report_{params.project_uid}_{params.workspace_uid}.pdf"

            all_jobs = list(project.find_jobs(workspace_uid=params.workspace_uid))

            # Filter jobs based on include_running flag
            if params.include_running:
                allowed_statuses = ["completed", "running", "waiting"]
                target_jobs = [j for j in all_jobs if j.status in allowed_statuses]
                status_label = "completed+running+waiting"
            else:
                target_jobs = [j for j in all_jobs if j.status == "completed"]
                status_label = "completed"

            pages_created = []

            with PdfPages(pdf_path) as pdf:

                # ========================================================
                # PAGE 1: Title + overview
                # ========================================================
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(0.5, 0.88, "CryoSPARC Session Report", transform=ax.transAxes,
                        fontsize=24, fontweight="bold", ha="center")
                ax.text(0.5, 0.82, f"Project {params.project_uid} — Workspace {params.workspace_uid}",
                        transform=ax.transAxes, fontsize=14, ha="center", color="gray")
                ax.text(0.5, 0.78, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        transform=ax.transAxes, fontsize=10, ha="center", color="gray")

                # Job summary
                type_counts: Dict[str, int] = {}
                for j in target_jobs:
                    type_counts[j.type] = type_counts.get(j.type, 0) + 1

                y = 0.70
                ax.text(0.1, y, f"Total jobs in workspace: {len(all_jobs)}", fontsize=11,
                        transform=ax.transAxes, fontfamily="monospace")
                y -= 0.035
                ax.text(0.1, y, f"Jobs included in report ({status_label}): {len(target_jobs)}", fontsize=11,
                        transform=ax.transAxes, fontfamily="monospace")
                y -= 0.035
                ax.text(0.1, y, f"Excluded (failed/killed/queued): {len(all_jobs) - len(target_jobs)}", fontsize=11,
                        transform=ax.transAxes, fontfamily="monospace")
                y -= 0.05
                ax.text(0.1, y, "Jobs by type (included):", fontsize=11, fontweight="bold",
                        transform=ax.transAxes)
                y -= 0.035
                for jtype, count in sorted(type_counts.items()):
                    ax.text(0.12, y, f"  {jtype}: {count}", fontsize=9,
                            transform=ax.transAxes, fontfamily="monospace")
                    y -= 0.025
                    if y < 0.05:
                        break

                pdf.savefig(fig)
                plt.close(fig)
                pages_created.append("Overview")

                # ========================================================
                # Process each selected job in order
                # ========================================================
                # Keep references to motion and CTF jobs for later use
                motion_jobs = []
                ctf_jobs = []

                for job in target_jobs:
                    jtype = job.type
                    if jtype in ("patch_motion_correction_multi", "rigid_motion_correction_multi",
                                 "local_motion_correction", "local_motion_correction_multi"):
                        motion_jobs.append(job)
                    if jtype in ("patch_ctf_estimation_multi", "ctf_estimation", "patch_ctf_extract"):
                        ctf_jobs.append(job)

                skipped: List[Dict[str, str]] = []
                for job in target_jobs:
                    try:
                        made_pages = _report_job(pdf, plt, np, project, job, params, motion_jobs, ctf_jobs)
                        if made_pages:
                            pages_created.extend(made_pages)
                        else:
                            skipped.append({"job_uid": job.uid, "type": job.type,
                                            "reason": "no reportable data"})
                    except Exception as job_exc:  # noqa: BLE001
                        # Report the reason instead of hiding it: a silent skip made
                        # a fully-empty report look like a success.
                        skipped.append({"job_uid": job.uid, "type": job.type,
                                        "reason": f"{type(job_exc).__name__}: {job_exc}"})

            content_pages = [p for p in pages_created if p != "Overview"]
            return jsonable({
                "success": bool(content_pages),
                "pdf_path": pdf_path,
                "total_jobs_scanned": len(target_jobs),
                "jobs_reported": len(target_jobs) - len(skipped),
                "pages": pages_created,
                "skipped": skipped,
                "message": (
                    f"Report saved to {pdf_path}" if content_pages else
                    f"No job produced reportable data — {pdf_path} contains only the "
                    f"overview page. See 'skipped' for the per-job reasons."
                ),
            })

        except Exception as e:
            raise tool_error(e) from e


# ====================================================================
# Per-job report dispatcher (returns list of descriptions, one per page)
# ====================================================================

def _report_job(pdf, plt, np, project, job, params, motion_jobs, ctf_jobs) -> Optional[List[str]]:
    """Generate page(s) for a single job. Returns list of descriptions or None."""
    jtype = job.type
    descriptions = []

    # --- Import ---
    if jtype in ("import_movies", "import_micrographs", "import_particles", "import_volumes"):
        desc = _page_import(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- Motion Correction ---
    if jtype in ("patch_motion_correction_multi", "rigid_motion_correction_multi",
                 "local_motion_correction", "local_motion_correction_multi"):
        desc = _page_motion(pdf, plt, np, job)
        if desc:
            descriptions.append(desc)

    # --- CTF Estimation ---
    if jtype in ("patch_ctf_estimation_multi", "ctf_estimation", "patch_ctf_extract"):
        desc = _page_ctf(pdf, plt, np, job)
        if desc:
            descriptions.append(desc)
        # Add ice thickness page if available
        ice_desc = _page_ice_thickness(pdf, plt, np, job)
        if ice_desc:
            descriptions.append(ice_desc)

    # --- Exposure Curation ---
    if jtype in ("curate_exposures_v2",):
        desc = _page_curation(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- Particle Picking ---
    if jtype in ("blob_picker_gpu", "template_picker_gpu", "filament_tracer_gpu",
                 "auto_blob_picker_gpu", "topaz_extract"):
        # 1. Picking settings page
        settings_desc = _page_picking_settings(pdf, plt, job)
        if settings_desc:
            descriptions.append(settings_desc)
        # 2. Score distributions page
        score_desc = _page_picking_scores(pdf, plt, np, job)
        if score_desc:
            descriptions.append(score_desc)
        # 3. Sample micrographs with picks overlay + motion curves + CTF fit curves
        micro_desc = _page_sample_micrographs(pdf, plt, np, project, job, params, motion_jobs, ctf_jobs)
        if micro_desc:
            descriptions.append(micro_desc)

    # --- Inspect Picks ---
    if jtype in ("inspect_picks_v2",):
        desc = _page_inspect_picks(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- Extraction ---
    if jtype in ("extract_micrographs_multi", "extract_micrographs_cpu_parallel",
                 "downsample_particles", "restack_particles"):
        desc = _page_extraction(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- 2D Classification ---
    if jtype in ("class_2D", "class_2D_new"):
        desc = _page_2d_class(pdf, plt, np, project, job)
        if desc:
            descriptions.append(desc)

    # --- Select 2D ---
    if jtype in ("select_2D",):
        desc = _page_select_2d(pdf, plt, np, project, job)
        if desc:
            descriptions.append(desc)

    # --- Ab-Initio ---
    if jtype in ("homo_abinit",):
        desc = _page_abinit(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- Refinement ---
    if jtype in ("homo_refine_new", "hetero_refine", "nonuniform_refine_new",
                 "homo_reconstruct", "hetero_reconstruct_new"):
        desc = _page_refinement(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- CTF Refinement ---
    if jtype in ("ctf_refine_global", "ctf_refine_local"):
        desc = _page_ctf_refine(pdf, plt, np, job)
        if desc:
            descriptions.append(desc)

    # --- 3D Variability ---
    if jtype in ("var_3D", "var_3D_disp"):
        desc = _page_variability(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- 3D Classification ---
    if jtype in ("class_3D", "regroup_3D_new"):
        desc = _page_3d_class(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- Postprocessing ---
    if jtype in ("sharpen", "local_resolution", "local_filter", "validation", "reslog"):
        desc = _page_postprocessing(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- Local Refinement ---
    if jtype in ("new_local_refine", "particle_subtract"):
        desc = _page_local_refine(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    # --- Anything else: generic summary ---
    if not descriptions:
        desc = _page_generic(pdf, plt, job)
        if desc:
            descriptions.append(desc)

    return descriptions if descriptions else None


# ====================================================================
# Page generators — each returns a string description or None
# ====================================================================

def _make_page_header(fig, ax, title: str, subtitle: str = ""):
    ax.text(0.5, 0.95, title, transform=ax.transAxes,
            fontsize=16, fontweight="bold", ha="center", va="top")
    if subtitle:
        ax.text(0.5, 0.90, subtitle, transform=ax.transAxes,
                fontsize=10, ha="center", va="top", color="gray")


def _params_text(job, max_lines=20) -> List[str]:
    """Extract readable parameter lines from a job."""
    lines = []
    p = job.params or {}
    for k, v in list(p.items())[:max_lines]:
        val_str = str(v)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        lines.append(f"  {k}: {val_str}")
    return lines


def _page_import(pdf, plt, job) -> Optional[str]:
    p = job.params or {}
    if not p:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    _make_page_header(fig, ax, f"Import — {job.uid}", job.title)

    lines = [
        f"Job type: {job.type}",
        f"Status: {job.status}",
        "",
        "Parameters:",
    ] + _params_text(job, max_lines=18)

    y = 0.82
    for line in lines:
        ax.text(0.08, y, line, transform=ax.transAxes, fontsize=9, fontfamily="monospace", va="top")
        y -= 0.028
        if y < 0.05:
            break

    pdf.savefig(fig)
    plt.close(fig)
    return f"Import {job.uid}: acquisition parameters"


def _page_motion(pdf, plt, np, job) -> Optional[str]:
    try:
        mics = job.load_output("micrographs")
    except Exception:
        return None
    if len(mics) == 0:
        return None

    cols = dataset_fields(mics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Motion Correction — {job.uid} ({len(mics)} micrographs)", fontsize=14, fontweight="bold")

    plotted = False
    for motion_col in ["rigid_motion/total_motion", "spline_motion/total_motion"]:
        if motion_col in cols:
            data = np.array(list(mics[motion_col]))
            data = data[np.isfinite(data)]
            if len(data) > 0:
                axes[0].hist(data, bins=min(50, len(data)), color="coral", edgecolor="white")
                axes[0].set_xlabel("Total Motion (Å)")
                axes[0].set_ylabel("Count")
                axes[0].set_title("Motion Distribution")

                axes[1].scatter(range(len(data)), data, s=2, alpha=0.5, color="coral")
                axes[1].set_xlabel("Micrograph Index")
                axes[1].set_ylabel("Total Motion (Å)")
                axes[1].set_title("Motion Over Time")
                plotted = True
                break

    if not plotted:
        axes[0].text(0.5, 0.5, f"{len(mics)} micrographs processed", transform=axes[0].transAxes, ha="center", fontsize=14)
        axes[0].set_title("Motion Correction")
        axes[1].text(0.5, 0.5, f"Available columns:\n{', '.join(cols[:8])}", transform=axes[1].transAxes, ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)
    return f"Motion {job.uid}: {len(mics)} micrographs"


def _page_ctf(pdf, plt, np, job) -> Optional[str]:
    try:
        exposures = job.load_output("exposures", slots=["ctf"])
    except Exception:
        return None
    if len(exposures) == 0:
        return None

    cols = dataset_fields(exposures)
    has_df = "ctf/df1_A" in cols and "ctf/df2_A" in cols

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"CTF Estimation — {job.uid} ({len(exposures)} micrographs)", fontsize=14, fontweight="bold")

    if has_df:
        df1 = np.array(list(exposures["ctf/df1_A"]))
        df2 = np.array(list(exposures["ctf/df2_A"]))
        defocus_mean = (df1 + df2) / 2.0 / 10000  # convert to μm

        axes[0, 0].hist(defocus_mean, bins=min(50, len(defocus_mean)), color="steelblue", edgecolor="white")
        axes[0, 0].set_xlabel("Mean Defocus (μm)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Defocus Distribution")

        axes[0, 1].scatter(range(len(defocus_mean)), defocus_mean, s=2, alpha=0.5, color="steelblue")
        axes[0, 1].set_xlabel("Micrograph Index")
        axes[0, 1].set_ylabel("Mean Defocus (μm)")
        axes[0, 1].set_title("Defocus Over Time")

        astigmatism = np.abs(df1 - df2)
        axes[1, 0].hist(astigmatism, bins=min(50, len(astigmatism)), color="teal", edgecolor="white")
        axes[1, 0].set_xlabel("Astigmatism |DF1-DF2| (Å)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Astigmatism Distribution")

    if "ctf/ctf_fit_to_A" in cols:
        fit_res = np.array(list(exposures["ctf/ctf_fit_to_A"]))
        fit_res = fit_res[np.isfinite(fit_res) & (fit_res > 0)]
        if len(fit_res) > 0:
            axes[1, 1].hist(fit_res, bins=min(50, len(fit_res)), color="goldenrod", edgecolor="white")
            axes[1, 1].set_xlabel("CTF Fit Resolution (Å)")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("CTF Resolution Distribution")
    else:
        axes[1, 1].text(0.5, 0.5, "CTF fit resolution\nnot available", transform=axes[1, 1].transAxes, ha="center")
        axes[1, 1].set_title("CTF Resolution")

    if not has_df:
        for ax in axes.flat:
            ax.text(0.5, 0.5, f"Columns: {', '.join(cols[:6])}", transform=ax.transAxes, ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)
    return f"CTF {job.uid}: {len(exposures)} exposures, defocus+astigmatism+resolution"


def _page_ice_thickness(pdf, plt, np, job) -> Optional[str]:
    """Ice thickness histogram if available."""
    try:
        exposures = job.load_output("exposures", slots=["ctf"])
    except Exception:
        return None
    if len(exposures) == 0:
        return None

    cols = dataset_fields(exposures)
    ice_col = None
    for candidate in ["ctf/ice_thickness_A", "ctf/ice_thickness"]:
        if candidate in cols:
            ice_col = candidate
            break
    if ice_col is None:
        return None

    ice = np.array(list(exposures[ice_col]))
    ice = ice[np.isfinite(ice)]
    if len(ice) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(ice, bins=min(50, len(ice)), color="cornflowerblue", edgecolor="white")
    ax.set_xlabel("Ice Thickness (Å)")
    ax.set_ylabel("Count")
    ax.set_title(f"Ice Thickness Distribution — {job.uid} (n={len(ice)})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return f"Ice thickness {job.uid}: {len(ice)} measurements"


def _page_curation(pdf, plt, job) -> Optional[str]:
    outputs = list(job.outputs.keys()) if job.outputs else []
    if not outputs:
        return None

    counts = {}
    for out_name in outputs:
        try:
            dset = job.load_output(out_name)
            counts[out_name] = len(dset)
        except Exception:
            counts[out_name] = "?"

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis("off")
    _make_page_header(fig, ax, f"Exposure Curation — {job.uid}", job.title)

    y = 0.75
    for name, count in counts.items():
        label = name.replace("_", " ").title()
        ax.text(0.15, y, f"{label}: {count} exposures", transform=ax.transAxes, fontsize=13, va="top")
        y -= 0.06

    pdf.savefig(fig)
    plt.close(fig)
    return f"Curation {job.uid}: {counts}"


def _page_picking_settings(pdf, plt, job) -> Optional[str]:
    """Dedicated page showing picking thresholds, model, and other parameters."""
    p = job.params or {}
    if not p:
        return None

    interesting_keys = []
    if job.type in ("blob_picker_gpu", "auto_blob_picker_gpu"):
        interesting_keys = ["low_threshold", "high_threshold", "min_particle_diameter", "max_particle_diameter",
                            "min_separation", "max_particles", "diameter", "blob_detector_method"]
    elif job.type == "template_picker_gpu":
        interesting_keys = ["threshold", "template", "max_peaks", "min_peak_distance", "template_diameter"]
    elif job.type == "topaz_extract":
        interesting_keys = ["model_path", "threshold", "pixel_size", "box_size", "use_gpu"]
    elif job.type == "filament_tracer_gpu":
        interesting_keys = ["threshold", "min_length", "max_length", "filament_width", "max_gap"]

    if not interesting_keys:
        interesting_keys = list(p.keys())[:15]

    lines = [f"Picker type: {job.type}", f"Job title: {job.title}", ""]
    lines.append("Picking parameters:")
    for k in interesting_keys:
        if k in p:
            val = p[k]
            if isinstance(val, float):
                val = f"{val:.4g}"
            lines.append(f"  {k}: {val}")

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    _make_page_header(fig, ax, f"Picking Settings — {job.uid}", job.title)

    y = 0.82
    for line in lines:
        ax.text(0.08, y, line, transform=ax.transAxes, fontsize=10, fontfamily="monospace", va="top")
        y -= 0.03
        if y < 0.05:
            break

    pdf.savefig(fig)
    plt.close(fig)
    return f"Picking settings {job.uid}: {len(interesting_keys)} parameters"


def _page_picking_scores(pdf, plt, np, job) -> Optional[str]:
    """Score distributions (NCC, power) for picks."""
    try:
        particles = job.load_output("particles", slots=["pick_stats"])
    except Exception:
        try:
            particles = job.load_output("particles")
        except Exception:
            return None
    if len(particles) == 0:
        return None

    cols = dataset_fields(particles)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Picking Scores — {job.uid} ({job.type}) — {len(particles)} particles", fontsize=14, fontweight="bold")

    if "pick_stats/ncc_score" in cols:
        scores = np.array(list(particles["pick_stats/ncc_score"]))
        scores = scores[np.isfinite(scores)]
        if len(scores) > 0:
            axes[0].hist(scores, bins=min(50, len(scores)), color="forestgreen", edgecolor="white")
            axes[0].set_xlabel("NCC Score")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Pick Score Distribution")
    else:
        axes[0].text(0.5, 0.5, f"{len(particles)} particles picked\nNCC score not available", transform=axes[0].transAxes, ha="center", fontsize=12)

    if "pick_stats/power" in cols:
        power = np.array(list(particles["pick_stats/power"]))
        power = power[np.isfinite(power)]
        if len(power) > 0:
            axes[1].hist(power, bins=min(50, len(power)), color="mediumpurple", edgecolor="white")
            axes[1].set_xlabel("Power")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Pick Power Distribution")
    else:
        axes[1].text(0.5, 0.5, "Power score not available", transform=axes[1].transAxes, ha="center", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)
    return f"Picking scores {job.uid}: {len(particles)} particles"


def _plot_motion_drift(ax, job, micrograph_uid):
    """Plot drift curve for a specific micrograph from motion correction job."""
    import numpy as np
    try:
        # Load micrographs data with motion details
        mics = job.load_output("micrographs", slots=["motion"])
        if mics is None or len(mics) == 0:
            return False
        # Find the row for this micrograph
        row = None
        for r in mics.rows():
            if r.get("uid") == micrograph_uid:
                row = r
                break
        if row is None:
            return False
        # Try to get per-frame shifts
        shift_x = row.get("rigid_motion/shift_x_per_frame")
        shift_y = row.get("rigid_motion/shift_y_per_frame")
        if shift_x is None or shift_y is None:
            # Try alternative names
            shift_x = row.get("motion/shift_x_per_frame")
            shift_y = row.get("motion/shift_y_per_frame")
        if shift_x is not None and shift_y is not None:
            # Convert to numpy arrays if they are lists
            if not isinstance(shift_x, (list, np.ndarray)):
                return False
            shift_x = np.array(shift_x)
            shift_y = np.array(shift_y)
            frames = np.arange(len(shift_x))
            # Cumulative drift
            cum_x = np.cumsum(shift_x)
            cum_y = np.cumsum(shift_y)
            ax.plot(frames, cum_x, label='X drift', color='blue')
            ax.plot(frames, cum_y, label='Y drift', color='red')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Cumulative drift (Å)')
            ax.legend()
            ax.set_title('Motion drift')
            return True
        # Fallback: total motion if available
        total_motion = row.get("rigid_motion/total_motion")
        if total_motion is not None:
            ax.text(0.5, 0.5, f"Total motion: {total_motion:.2f} Å", transform=ax.transAxes, ha='center')
            ax.set_title('Motion (total only)')
            return True
        return False
    except Exception:
        return False


def _plot_ctf_fit(ax, project, ctf_job, micrograph_uid):
    """Plot CTF fit curve for a specific micrograph from CTF job."""
    try:
        # List files in the job directory under ctfestimated/
        # We need to find the .npy file that corresponds to this micrograph
        # Typically named like <micrograph_uid>_ctf_diag_plt.npy or similar
        # Let's try to get the list of files
        files = ctf_job.list_files(prefix="ctfestimated/")
        target_file = None
        uid_str = str(micrograph_uid)
        for f in files:
            if uid_str in f and f.endswith('.npy'):
                target_file = f
                break
        if target_file is None:
            return False
        # Read the .npy file as a stream (download_file would write it to disk
        # and return a path, not bytes). job.download() yields a readable response.
        import numpy as np
        import io
        with ctf_job.download(target_file) as res:
            data = np.load(io.BytesIO(res.read()), allow_pickle=True).item()
        # Now data is a dict with keys: freqs_trim, EPA_trim, BGINT, ENVINT, CTF, CC
        freqs = data['freqs_trim']
        power_spectrum = np.cbrt(data['EPA_trim'] - data['BGINT']) + 0.5
        ctf_curve = np.cbrt(data['ENVINT'] * (2 * data['CTF']**2 - 1)) + 0.5
        cc_curve = data['CC']
        ax.plot(freqs, power_spectrum, color='black', label='Power Spectrum')
        ax.plot(freqs, ctf_curve, color='orange', label='Envelope x |CTF|²')
        ax.plot(freqs, cc_curve, color='blue', label='Cross-correlation')
        ax.legend(loc='upper right', fontsize=6)
        ax.set_xlabel('Frequency (1/Å)')
        ax.set_ylabel('Intensity')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, freqs.max())
        ax.set_title('CTF fit')
        return True
    except Exception:
        return False


def _page_sample_micrographs(pdf, plt, np, project, picking_job, params, motion_jobs, ctf_jobs) -> Optional[str]:
    """Render random micrographs with picks overlay, motion drift, and CTF fit curves."""
    # Load micrographs with blob
    try:
        mics = picking_job.load_output("micrographs", slots=["blob"])
    except Exception:
        return None
    if len(mics) == 0:
        return None

    # Load particles
    try:
        particles = picking_job.load_output("particles")
    except Exception:
        return None
    if len(particles) == 0:
        return None

    # Find coordinate columns
    coord_x_col = None
    coord_y_col = None
    for cx in ["location/center_x_pix", "location/center_x_frac"]:
        if cx in dataset_fields(particles):
            coord_x_col = cx
            break
    for cy in ["location/center_y_pix", "location/center_y_frac"]:
        if cy in dataset_fields(particles):
            coord_y_col = cy
            break
    if coord_x_col is None or coord_y_col is None:
        return None
    is_fractional = "_frac" in coord_x_col

    # --- Sample FIRST, then download only the sampled micrographs. ---
    # Previously every micrograph in the job was downloaded before sampling,
    # i.e. potentially tens of GB of transfer just to render ~10 images.
    if "micrograph_uid" not in dataset_fields(particles):
        return None
    mic_rows_by_uid = {row["uid"]: row for row in mics.rows()}
    pool = sorted(uid for uid in set(particles["micrograph_uid"]) if uid in mic_rows_by_uid)
    if not pool:
        return None

    n_sample = min(params.num_sample_micrographs, len(pool))
    selected_uids = random.sample(pool, n_sample)

    mic_uid_to_data = {}
    width = height = None
    for uid in selected_uids:
        row = mic_rows_by_uid[uid]
        try:
            _, data = project.download_mrc(row["blob/path"])
        except Exception:
            continue
        if data.ndim == 3:
            idx = row["blob/idx"]
            data = data[idx] if idx < data.shape[0] else data[0]
        mic_uid_to_data[uid] = data
        if width is None:
            height, width = data.shape[0], data.shape[1]
    selected_uids = [u for u in selected_uids if u in mic_uid_to_data]
    if not selected_uids:
        return None

    # For each selected micrograph, collect its particles in a SINGLE pass over
    # the particle set (previously this scanned all particles once per micrograph).
    selected_set = set(selected_uids)
    mic_particles = {uid: [] for uid in selected_uids}
    for p_row in particles.rows():
        uid = p_row["micrograph_uid"]
        if uid in selected_set:
            mic_particles[uid].append(p_row)

    # Prepare figure: for each micrograph, create a row with three subplots (micrograph, motion, CTF)
    # We'll create one figure per micrograph to keep layout manageable
    for uid in selected_uids:
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        fig.suptitle(f"Micrograph {uid} — {len(mic_particles[uid])} picks", fontsize=12, fontweight="bold")

        # Left: micrograph with picks
        ax_img = axes[0]
        img = mic_uid_to_data[uid]
        ax_img.imshow(img, cmap="gray", origin="lower")
        coords = []
        for p in mic_particles[uid]:
            x = p[coord_x_col]
            y = p[coord_y_col]
            if is_fractional:
                x = x * width
                y = y * height
            coords.append((x, y))
        if coords:
            xs, ys = zip(*coords)
            ax_img.scatter(xs, ys, c='red', s=4, marker='x', alpha=0.7, linewidth=0.5)
        ax_img.set_title("Micrograph + picks")
        ax_img.axis("off")

        # Middle: motion drift (use the first motion job that has data for this micrograph)
        ax_motion = axes[1]
        motion_ok = False
        for mjob in motion_jobs:
            if _plot_motion_drift(ax_motion, mjob, uid):
                motion_ok = True
                break
        if not motion_ok:
            ax_motion.text(0.5, 0.5, "Motion data not available", transform=ax_motion.transAxes, ha='center')
            ax_motion.set_title("Motion drift")
            ax_motion.axis('off')

        # Right: CTF fit curve
        ax_ctf = axes[2]
        ctf_ok = False
        for cjob in ctf_jobs:
            if _plot_ctf_fit(ax_ctf, project, cjob, uid):
                ctf_ok = True
                break
        if not ctf_ok:
            ax_ctf.text(0.5, 0.5, "CTF fit not available", transform=ax_ctf.transAxes, ha='center')
            ax_ctf.set_title("CTF fit")
            ax_ctf.axis('off')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    return f"Sample micrographs {picking_job.uid}: {n_sample} micrographs with picks, motion drift, CTF fit"


def _page_inspect_picks(pdf, plt, job) -> Optional[str]:
    outputs = list(job.outputs.keys()) if job.outputs else []
    counts = {}
    for out_name in outputs:
        try:
            dset = job.load_output(out_name)
            counts[out_name] = len(dset)
        except Exception:
            pass
    if not counts:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis("off")
    _make_page_header(fig, ax, f"Inspect Picks — {job.uid}", job.title)
    y = 0.75
    for name, count in counts.items():
        ax.text(0.15, y, f"{name}: {count}", transform=ax.transAxes, fontsize=12, va="top")
        y -= 0.06
    pdf.savefig(fig)
    plt.close(fig)
    return f"Inspect picks {job.uid}: {counts}"


def _page_extraction(pdf, plt, job) -> Optional[str]:
    try:
        particles = job.load_output("particles")
    except Exception:
        return None
    if len(particles) == 0:
        return None

    p = job.params or {}
    box_size = p.get("box_size_pix", "N/A")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis("off")
    _make_page_header(fig, ax, f"Extraction — {job.uid}", job.title)
    info = [
        f"Particles extracted: {len(particles)}",
        f"Box size: {box_size} px",
        f"Fourier crop: {p.get('bin_size_pix', 'N/A')} px",
    ]
    y = 0.72
    for line in info:
        ax.text(0.15, y, line, transform=ax.transAxes, fontsize=13, va="top")
        y -= 0.06

    pdf.savefig(fig)
    plt.close(fig)
    return f"Extraction {job.uid}: {len(particles)} particles, box={box_size}"


def _page_2d_class(pdf, plt, np, project, job) -> Optional[str]:
    try:
        templates = job.load_output("class_averages", slots=["blob"])
    except Exception:
        return None
    if len(templates) == 0:
        return None

    return _render_class_gallery(pdf, plt, np, project, templates, job,
                                 f"2D Classification — {job.uid} — {len(templates)} classes")


def _page_select_2d(pdf, plt, np, project, job) -> Optional[str]:
    try:
        templates = job.load_output("templates_selected", slots=["blob"])
    except Exception:
        return None
    if len(templates) == 0:
        return None

    return _render_class_gallery(pdf, plt, np, project, templates, job,
                                 f"Selected 2D Classes — {job.uid} — {len(templates)} classes")


def _render_class_gallery(pdf, plt, np, project, templates, job, title) -> Optional[str]:
    """Render a grid of class averages from MRC blobs."""
    unique_paths = set(templates["blob/path"])
    all_blobs = {}
    for path in unique_paths:
        try:
            _, data = project.download_mrc(path)
            all_blobs[path] = data
        except Exception:
            continue

    if not all_blobs:
        return None

    num = len(templates)
    ncols = min(10, num)
    nrows = max(1, (num + ncols - 1) // ncols)
    fig_h = max(3, nrows * 1.0 + 1.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, min(fig_h, 16)))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    if nrows == 1 and ncols == 1:
        flat = [axes]
    elif nrows == 1:
        flat = list(axes)
    elif ncols == 1:
        flat = list(axes)
    else:
        flat = [ax for row in axes for ax in row]

    for i, row in enumerate(templates.rows()):
        if i >= len(flat):
            break
        path = row["blob/path"]
        idx = row["blob/idx"]
        ax = flat[i]
        if path in all_blobs:
            blob = all_blobs[path]
            if blob.ndim == 3 and idx < blob.shape[0]:
                ax.imshow(blob[idx], cmap="gray", origin="lower")
            elif blob.ndim == 2:
                ax.imshow(blob, cmap="gray", origin="lower")
        ax.axis("off")

    for i in range(num, len(flat)):
        flat[i].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)
    return f"2D classes {job.uid}: {num} class averages"


def _page_abinit(pdf, plt, job) -> Optional[str]:
    outputs = list(job.outputs.keys()) if job.outputs else []
    volume_outputs = [o for o in outputs if "volume" in o]
    if not volume_outputs:
        return None

    p = job.params or {}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis("off")
    _make_page_header(fig, ax, f"Ab-Initio Reconstruction — {job.uid}", job.title)

    info = [
        f"Number of classes (K): {p.get('abinit_K', len(volume_outputs))}",
        f"Volume outputs: {', '.join(volume_outputs)}",
        f"Symmetry: {p.get('abinit_symmetry', 'C1')}",
        f"Max resolution: {p.get('abinit_max_res', 'N/A')} Å",
    ]
    y = 0.72
    for line in info:
        ax.text(0.15, y, line, transform=ax.transAxes, fontsize=12, va="top")
        y -= 0.06

    pdf.savefig(fig)
    plt.close(fig)
    return f"Ab-initio {job.uid}: {len(volume_outputs)} volumes"


def _page_refinement(pdf, plt, job) -> Optional[str]:
    outputs = list(job.outputs.keys()) if job.outputs else []
    if not outputs:
        return None

    p = job.params or {}

    particle_count = "N/A"
    for out_name in outputs:
        if "particle" in out_name:
            try:
                dset = job.load_output(out_name)
                particle_count = len(dset)
            except Exception:
                pass
            break

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.axis("off")
    _make_page_header(fig, ax, f"Refinement — {job.uid} ({job.type})", job.title)

    info = [
        f"Particles: {particle_count}",
        f"Symmetry: {p.get('refine_symmetry', p.get('hetero_symmetry', 'N/A'))}",
        f"Outputs: {', '.join(outputs)}",
    ]

    refine_keys = ["refine_res_init", "refine_res_gsfsc_split", "refine_num_final_iterations",
                   "refine_defocus_refine", "hetero_num_classes"]
    for k in refine_keys:
        if k in p:
            info.append(f"{k}: {p[k]}")

    y = 0.72
    for line in info:
        ax.text(0.1, y, line, transform=ax.transAxes, fontsize=11, va="top", fontfamily="monospace")
        y -= 0.045

    pdf.savefig(fig)
    plt.close(fig)
    return f"Refinement {job.uid}: particles={particle_count}"


def _page_ctf_refine(pdf, plt, np, job) -> Optional[str]:
    try:
        exposures = job.load_output("exposures", slots=["ctf"])
    except Exception:
        return None
    if len(exposures) == 0:
        return None

    cols = dataset_fields(exposures)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.suptitle(f"CTF Refinement — {job.uid} ({job.type})", fontsize=14, fontweight="bold")

    if "ctf/df1_A" in cols:
        df1 = np.array(list(exposures["ctf/df1_A"]))
        df2 = np.array(list(exposures["ctf/df2_A"]))
        defocus = (df1 + df2) / 2.0 / 10000
        ax.hist(defocus, bins=min(50, len(defocus)), color="darkcyan", edgecolor="white")
        ax.set_xlabel("Refined Mean Defocus (μm)")
        ax.set_ylabel("Count")
        ax.set_title(f"Refined Defocus — {len(exposures)} exposures")
    else:
        ax.text(0.5, 0.5, f"{len(exposures)} exposures refined", transform=ax.transAxes, ha="center", fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)
    return f"CTF refine {job.uid}: {len(exposures)} exposures"


def _page_variability(pdf, plt, job) -> Optional[str]:
    return _page_generic(pdf, plt, job, title_prefix="3D Variability")


def _page_3d_class(pdf, plt, job) -> Optional[str]:
    return _page_generic(pdf, plt, job, title_prefix="3D Classification")


def _page_postprocessing(pdf, plt, job) -> Optional[str]:
    return _page_generic(pdf, plt, job, title_prefix="Postprocessing")


def _page_local_refine(pdf, plt, job) -> Optional[str]:
    return _page_generic(pdf, plt, job, title_prefix="Local Refinement")


def _page_generic(pdf, plt, job, title_prefix: str = "") -> Optional[str]:
    """Generic page for any job: shows UID, type, params, and output names."""
    outputs = list(job.outputs.keys()) if job.outputs else []
    p = job.params or {}
    if not outputs and not p:
        return None

    prefix = title_prefix or job.type.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    _make_page_header(fig, ax, f"{prefix} — {job.uid}", job.title)

    lines = [f"Job type: {job.type}", f"Status: {job.status}", ""]
    if outputs:
        lines.append("Outputs:")
        for o in outputs:
            try:
                dset = job.load_output(o)
                lines.append(f"  {o}: {len(dset)} items")
            except Exception:
                lines.append(f"  {o}: (not loadable)")
        lines.append("")

    if p:
        lines.append("Parameters:")
        lines.extend(_params_text(job, max_lines=25))

    y = 0.82
    for line in lines:
        ax.text(0.08, y, line, transform=ax.transAxes, fontsize=9, fontfamily="monospace", va="top")
        y -= 0.025
        if y < 0.03:
            break

    pdf.savefig(fig)
    plt.close(fig)
    return f"{prefix} {job.uid}: {len(outputs)} outputs"