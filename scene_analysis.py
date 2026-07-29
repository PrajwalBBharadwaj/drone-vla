"""
Per-object bounding-box + monocular-depth scene analysis.

Pipeline
--------
1. Detect objects          → facebook/detr-resnet-50
2. Estimate depth          → Intel/dpt-hybrid-midas
3. Per-object metrics      → heading and pseudo-metric distance via a generic
                             pinhole camera model.
4. Annotated image output  → saved alongside the input as <name>_analysed.png

Camera model
------------
Assumes a typical drone / action-camera field of view:
  Horizontal FOV  = 84°   (GoPro-style wide-angle approximation)
  Vertical FOV    = 60°
  Principal point = image centre
  Focal lengths   = derived from FOV + resolution

Distance estimation
-------------------
Monocular depth is *relative*, not metric. To get a pseudo-metric scale we
use an object-height heuristic for known categories (car, truck, person, …).
For every detected object:
  1. Sample the normalised inverse-depth map inside the bounding box.
  2. For categories with a known real-world height, additionally estimate
     distance via: d = f_y * H_real / h_pixels
  3. Both estimates are shown; labelled as "est." to be explicit.

Output columns per object
-------------------------
  ID | label | conf | distance (est.) | heading_H | heading_V
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

# ── model names ──────────────────────────────────────────────────────────────
DETECT_MODEL = "facebook/detr-resnet-50"
DEPTH_MODEL  = "Intel/dpt-hybrid-midas"

# ── generic camera intrinsics ─────────────────────────────────────────────────
HFOV_DEG = 84.0    # horizontal field of view (degrees)
VFOV_DEG = 60.0    # vertical field of view (degrees)

# ── detection threshold ───────────────────────────────────────────────────────
CONF_THRESHOLD = 0.60

# ── real-world height priors (metres) for common COCO labels ─────────────────
HEIGHT_PRIOR: dict[str, float] = {
    "person":      1.70,
    "bicycle":     1.10,
    "car":         1.55,
    "motorcycle":  1.15,
    "bus":         3.20,
    "truck":       3.00,
    "train":       4.00,
    "traffic light": 0.60,
    "fire hydrant": 0.60,
}

# ── colour palette (one per object) ──────────────────────────────────────────
PALETTE = plt.cm.tab20.colors  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────

def _focal_lengths(width: int, height: int) -> tuple[float, float]:
    fx = width  / (2 * np.tan(np.deg2rad(HFOV_DEG / 2)))
    fy = height / (2 * np.tan(np.deg2rad(VFOV_DEG / 2)))
    return fx, fy


def _normalise_depth(raw: np.ndarray) -> np.ndarray:
    """Normalise inverse-depth map to [0, 1] using 5-95 percentile clipping."""
    finite = raw[np.isfinite(raw)]
    lo, hi = np.percentile(finite, [5, 95])
    denom = max(hi - lo, 1e-6)
    return np.clip((raw - lo) / denom, 0.0, 1.0)


def _mean_depth_in_box(
    norm_inv_depth: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    """Return mean normalised inverse-depth inside (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    h, w = norm_inv_depth.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    region = norm_inv_depth[y0:y1, x0:x1]
    return float(np.mean(region)) if region.size > 0 else 0.5


def _height_based_distance(
    label: str,
    y0: int,
    y1: int,
    fy: float,
) -> float | None:
    """Return distance in metres using object-height heuristic, or None."""
    h_real = HEIGHT_PRIOR.get(label.lower())
    if h_real is None:
        return None
    h_pix = max(y1 - y0, 1)
    return fy * h_real / h_pix


def _heading(
    cx_pix: float,
    cy_pix: float,
    img_w: int,
    img_h: int,
    fx: float,
    fy: float,
) -> tuple[float, float]:
    """
    Return (horizontal_deg, vertical_deg) heading relative to camera bore.
    Positive H = right, positive V = up.
    """
    dx = cx_pix - img_w / 2
    dy = img_h / 2 - cy_pix          # flip y so up is positive
    h_deg = float(np.degrees(np.arctan2(dx, fx)))
    v_deg = float(np.degrees(np.arctan2(dy, fy)))
    return h_deg, v_deg


# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    image_path: str,
    output_path: str | None = None,
    verbose: bool = True,
    save_visualization: bool = True,
) -> tuple[str | None, list[dict], tuple[int, int]]:
    """
    Run the full pipeline on *image_path*.
    Returns the path of the saved annotated image.
    """
    src = Path(image_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {src}")

    out = Path(output_path) if output_path else src.with_name(src.stem + "_analysed.png")

    if verbose:
        print(f"Loading image: {src}")
    image = Image.open(src).convert("RGB")
    W, H = image.size

    fx, fy = _focal_lengths(W, H)
    if verbose:
        print(
            f"Image size: {W}x{H}  |  fx={fx:.1f}  fy={fy:.1f}  "
            f"(generic {HFOV_DEG}°/{VFOV_DEG}° FOV)"
        )

    # ── 1. Object detection ───────────────────────────────────────────────────
    if verbose:
        print(f"\nRunning detection ({DETECT_MODEL}) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detector = pipeline("object-detection", model=DETECT_MODEL)
    detections = detector(image)
    detections = [d for d in detections if d["score"] >= CONF_THRESHOLD]
    if verbose:
        print(f"  {len(detections)} objects above {CONF_THRESHOLD:.0%} confidence")

    # ── 2. Depth estimation ───────────────────────────────────────────────────
    if verbose:
        print(f"\nRunning depth estimation ({DEPTH_MODEL}) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL)
    raw_depth = depth_pipe(image)["predicted_depth"]
    raw_np    = raw_depth.squeeze().detach().cpu().numpy().astype(np.float32)
    norm_inv  = _normalise_depth(raw_np)

    # ── 3. Per-object metrics ─────────────────────────────────────────────────
    records: list[dict] = []
    for idx, det in enumerate(detections):
        label = det["label"]
        score = det["score"]
        bx    = det["box"]                                    # {xmin,ymin,xmax,ymax}
        box   = (bx["xmin"], bx["ymin"], bx["xmax"], bx["ymax"])
        cx    = (bx["xmin"] + bx["xmax"]) / 2
        cy    = (bx["ymin"] + bx["ymax"]) / 2

        inv_d      = _mean_depth_in_box(norm_inv, box)        # 0=far, 1=near
        h_based    = _height_based_distance(label, bx["ymin"], bx["ymax"], fy)
        h_deg, v_deg = _heading(cx, cy, W, H, fx, fy)

        records.append(
            dict(
                id=idx + 1,
                label=label,
                score=score,
                box=box,
                cx=cx, cy=cy,
                inv_depth=inv_d,
                dist_height_m=h_based,
                heading_h=h_deg,
                heading_v=v_deg,
            )
        )

    # sort closest-first (highest inverse depth = nearest)
    records.sort(key=lambda r: r["inv_depth"], reverse=True)
    for rank, rec in enumerate(records, start=1):
        rec["id"] = rank

    # ── 4. Print table ────────────────────────────────────────────────────────
    if verbose:
        print(
            f"\n{'ID':>3}  {'Label':<18} {'Conf':>5}  "
            f"{'Rel.Depth':>9}  {'Dist(est)':>9}  {'Head_H':>7}  {'Head_V':>7}"
        )
        print("-" * 70)
        for r in records:
            dist_str = f"{r['dist_height_m']:.1f}m" if r["dist_height_m"] else "  —"
            print(
                f"{r['id']:>3}  {r['label']:<18} {r['score']:>5.2f}  "
                f"{r['inv_depth']:>9.3f}  {dist_str:>9}  "
                f"{r['heading_h']:>+7.1f}°  {r['heading_v']:>+7.1f}°"
            )

    if not save_visualization:
        return None, records, (W, H)

    # ── 5. Annotated image ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#111")

    # left: RGB + bounding boxes
    ax_rgb = axes[0]
    ax_rgb.imshow(image)
    ax_rgb.set_title("Object Detection + Headings", color="white", fontsize=12)
    ax_rgb.axis("off")

    for r in records:
        color = PALETTE[r["id"] % len(PALETTE)]
        x0, y0, x1, y1 = r["box"]
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="square,pad=1",
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax_rgb.add_patch(rect)

        dist_str = f"~{r['dist_height_m']:.1f}m" if r["dist_height_m"] else f"rel={r['inv_depth']:.2f}"
        label_str = (
            f"#{r['id']} {r['label']} ({r['score']:.0%})\n"
            f"{dist_str}\n"
            f"H:{r['heading_h']:+.1f}° V:{r['heading_v']:+.1f}°"
        )
        ax_rgb.text(
            x0, max(y0 - 4, 0), label_str,
            color="white", fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=2", fc=color, alpha=0.75, lw=0),
        )

    # right: depth heatmap
    ax_dep = axes[1]
    dep_img = ax_dep.imshow(norm_inv, cmap="plasma", vmin=0, vmax=1)
    ax_dep.set_title("Monocular Depth Map (brighter = closer)", color="white", fontsize=12)
    ax_dep.axis("off")
    plt.colorbar(dep_img, ax=ax_dep, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color="white")

    for r in records:
        color = PALETTE[r["id"] % len(PALETTE)]
        x0, y0, x1, y1 = r["box"]
        rect2 = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="square,pad=1",
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        ax_dep.add_patch(rect2)
        ax_dep.text(
            (x0 + x1) / 2, (y0 + y1) / 2, f"#{r['id']}",
            color="white", fontsize=8, ha="center", va="center", fontweight="bold",
        )

    plt.tight_layout(pad=0.5)
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    if verbose:
        print(f"\nAnnotated image saved → {out}")
    return str(out), records, (W, H)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-object depth + heading analysis")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output", default=None, help="Output image path")
    args = parser.parse_args()
    out_path, _, _ = analyse(args.image, args.output)
    _ = out_path  # result already printed inside analyse()


if __name__ == "__main__":
    main()
