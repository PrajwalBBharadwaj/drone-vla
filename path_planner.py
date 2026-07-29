"""
path_planner.py
---------------
1. Calls scene_analysis.analyse() to get per-object bounding boxes + depth.
2. Merges overlapping / nearby boxes into obstacle convex-hull polygons.
3. Rasterises the polygons into a binary obstacle mask and dilates by the
   drone's approximate angular footprint.
4. Runs Dijkstra from bottom-centre to top, penalising lateral movement, to
   find the safest low-deviation forward path.
5. Saves a two-panel annotated image:
      LEFT  – RGB frame with merged polygons + safe path corridor overlay
      RIGHT – dilated obstacle mask with planned path
6. Prints the heading recommendation (degrees left/right of bore).

Usage
-----
    python path_planner.py test_image_6.png
    python path_planner.py test_image_6.png -o result.png
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull

from scene_analysis import PALETTE, _focal_lengths, analyse

# ── tuneable constants ────────────────────────────────────────────────────────
# Two boxes are merged when their expanded bounding rects overlap.
MERGE_MARGIN_PX: int = 8
# Half the drone's lateral footprint in pixels at a "typical" distance.
# Used to dilate the obstacle mask so the path stays clear of edges.
DRONE_HALF_WIDTH_PX: int = 45
# Extra cost per pixel of lateral movement in Dijkstra.
# Higher value = path hugs the centre lane more aggressively.
LATERAL_PENALTY: float = 4.0
# Smoothing window for the raw path (pixels).
SMOOTH_WINDOW: int = 50
# Cost multiplier for pixels inside the dilated obstacle zone.
# Path through obstacles is always possible but very expensive.
OBSTACLE_COST: float = 80.0


@dataclass
class PathRecommendation:
    heading_offset_deg: float
    quality: str
    note: str
    detection_count: int
    obstacle_group_count: int


# ── bounding-box merging ──────────────────────────────────────────────────────

def _boxes_near(b1: tuple, b2: tuple, margin: int) -> bool:
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2
    return (
        x0a - margin <= x1b
        and x1a + margin >= x0b
        and y0a - margin <= y1b
        and y1a + margin >= y0b
    )


def _merge_groups(records: list[dict], margin: int) -> list[list[dict]]:
    """Union-Find grouping of boxes that are close or overlapping."""
    n = len(records)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_near(records[i]["box"], records[j]["box"], margin):
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(records[i])
    return list(groups.values())


def _hull_polygon(group: list[dict]) -> np.ndarray:
    """Return (N,2) convex-hull vertices of all box corners in the group."""
    corners = []
    for r in group:
        x0, y0, x1, y1 = r["box"]
        corners += [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts = np.array(corners, dtype=float)
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        return pts


# ── obstacle mask ─────────────────────────────────────────────────────────────

def _build_cost_map(
    hulls: list[np.ndarray],
    W: int,
    H: int,
    half_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        mask      – bool (H, W) – True where dilated obstacle zone is
        cost_map  – float32 (H, W) – 1.0 in free space, OBSTACLE_COST in zone
    """
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    for poly in hulls:
        flat = [(int(x), int(y)) for x, y in poly]
        if len(flat) >= 3:
            draw.polygon(flat, fill=255)

    mask = np.array(img) > 0
    if half_width > 0:
        h_struct = np.ones((1, half_width * 2 + 1), dtype=bool)
        mask = binary_dilation(mask, structure=h_struct)
        v_half = max(1, half_width // 2)
        v_struct = np.ones((v_half * 2 + 1, 1), dtype=bool)
        mask = binary_dilation(mask, structure=v_struct)

    cost_map = np.where(mask, OBSTACLE_COST, 1.0).astype(np.float32)
    return mask, cost_map


# ── path planning ─────────────────────────────────────────────────────────────

def _dijkstra_path(
    cost_map: np.ndarray,
    mask: np.ndarray,
    W: int,
    H: int,
) -> tuple[Optional[list[tuple[int, int]]], bool]:
    """
    Dijkstra from bottom-centre to the top row using a cost map.
    Obstacles are expensive but not impassable — always returns a path.
    Returns (path, is_obstacle_free).
    """
    start_x = W // 2
    start_y = H - 1

    # prefer free starting column if centre is in obstacle zone
    if mask[start_y, start_x]:
        free = np.where(~mask[start_y])[0]
        if len(free) > 0:
            start_x = int(free[np.argmin(np.abs(free - W // 2))])

    INF = float("inf")
    dist = np.full((H, W), INF, dtype=np.float32)
    dist[start_y, start_x] = 0.0
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[float, int, int]] = [(0.0, start_x, start_y)]
    goal: Optional[tuple[int, int]] = None

    while pq:
        cost, x, y = heapq.heappop(pq)
        if cost > dist[y, x] + 1e-6:
            continue
        if y == 0:
            goal = (x, y)
            break
        ny = y - 1
        for dx in (-1, 0, 1):
            nx = x + dx
            if 0 <= nx < W:
                nc = cost + float(cost_map[ny, nx]) + abs(dx) * LATERAL_PENALTY
                if nc < dist[ny, nx]:
                    dist[ny, nx] = nc
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (nc, nx, ny))

    if goal is None:
        rows = np.where(np.any(dist < INF, axis=1))[0]
        if len(rows) == 0:
            return None, False
        top_row = int(rows.min())
        cols = np.where(dist[top_row] < INF)[0]
        best_col = int(cols[np.argmin(dist[top_row, cols])])
        goal = (best_col, top_row)

    path: list[tuple[int, int]] = []
    cur = goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append((start_x, start_y))
    path = path[::-1]

    # check if path is entirely obstacle-free
    obstacle_free = all(not mask[y, x] for x, y in path)
    return path, obstacle_free


def _smooth(
    path: list[tuple[int, int]],
    window: int,
) -> list[tuple[int, int]]:
    if not path or len(path) <= window:
        return path
    xs = np.array([p[0] for p in path], dtype=float)
    ys = np.array([p[1] for p in path], dtype=float)
    kernel = np.ones(window) / window
    xs_s = np.convolve(xs, kernel, mode="same")
    return [(int(x), int(y)) for x, y in zip(xs_s, ys)]


def _build_path_solution(
    records: list[dict],
    W: int,
    H: int,
) -> tuple[list[list[dict]], list[np.ndarray], np.ndarray, Optional[list[tuple[int, int]]], bool]:
    groups = _merge_groups(records, MERGE_MARGIN_PX)
    hulls = [_hull_polygon(g) for g in groups]
    mask, cost_map = _build_cost_map(hulls, W, H, DRONE_HALF_WIDTH_PX)
    raw_path, is_clear = _dijkstra_path(cost_map, mask, W, H)
    path = _smooth(raw_path, SMOOTH_WINDOW) if raw_path else None
    return groups, hulls, mask, path, is_clear


def _compute_recommendation(
    path: Optional[list[tuple[int, int]]],
    is_clear: bool,
    W: int,
    fx: float,
) -> PathRecommendation:
    if not path:
        return PathRecommendation(
            heading_offset_deg=0.0,
            quality="NO_PATH",
            note="No forward path found.",
            detection_count=0,
            obstacle_group_count=0,
        )

    top_slice = path[: max(1, len(path) // 4)]
    avg_x = float(np.mean([p[0] for p in top_slice]))
    delta_h = float(np.degrees(np.arctan2(avg_x - W / 2.0, fx)))
    quality = "CLEAR" if is_clear else "FORCED"
    if abs(delta_h) < 1.0:
        note = f"Path [{quality}]: straight ahead."
    else:
        direction = "right" if delta_h > 0 else "left"
        note = f"Path [{quality}]: adjust {abs(delta_h):.1f}° {direction}."

    return PathRecommendation(
        heading_offset_deg=round(delta_h, 2),
        quality=quality,
        note=note,
        detection_count=0,
        obstacle_group_count=0,
    )


def recommend_heading_offset(image_path: str) -> PathRecommendation:
    """Return heading recommendation from merged obstacle geometry for one image."""
    _, records, (W, H) = analyse(
        image_path,
        verbose=False,
        save_visualization=False,
    )
    fx, _ = _focal_lengths(W, H)
    groups, _, _, path, is_clear = _build_path_solution(records, W, H)
    rec = _compute_recommendation(path, is_clear, W, fx)
    rec.detection_count = len(records)
    rec.obstacle_group_count = len(groups)
    return rec


# ── main pipeline ─────────────────────────────────────────────────────────────

def plan_and_visualise(
    image_path: str,
    output_path: Optional[str] = None,
) -> str:
    # 1. run detection + depth
    _, records, (W, H) = analyse(image_path)
    image = Image.open(Path(image_path).expanduser().resolve()).convert("RGB")
    fx, _ = _focal_lengths(W, H)

    # 2. merge boxes and compute path solution
    groups, hulls, mask, path, is_clear = _build_path_solution(records, W, H)
    print(
        f"\n[path_planner] {len(records)} detections merged into "
        f"{len(groups)} obstacle groups."
    )

    # 3. heading recommendation
    rec = _compute_recommendation(path, is_clear, W, fx)
    heading_note = rec.note
    path_color = "red"
    if path:
        path_color = "lime" if is_clear else "orange"
    print(f"[path_planner] {heading_note}")

    # 5. visualise
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(groups))]

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        heading_note, color="lime", fontsize=14,
        fontweight="bold", y=0.015, va="bottom",
    )

    # ── left: RGB + merged polygons + path ───────────────────────────────────
    ax = axes[0]
    ax.imshow(image)
    ax.set_title("Merged Obstacle Polygons  +  Safe Path", color="white", fontsize=12)
    ax.axis("off")

    legend_handles = []
    for hull_pts, color, group in zip(hulls, colors, groups):
        if len(hull_pts) >= 3:
            patch = plt.Polygon(
                hull_pts, closed=True,
                facecolor=color, alpha=0.38,
                edgecolor=color, linewidth=2.5,
            )
            ax.add_patch(patch)

        cx_g = float(np.mean([r["cx"] for r in group]))
        cy_g = float(np.mean([r["cy"] for r in group]))
        ids_str = ", ".join(f"#{r['id']}" for r in group)
        labels_str = ", ".join(sorted({r["label"] for r in group}))
        ax.text(
            cx_g, cy_g, f"{ids_str}\n{labels_str}",
            color="white", fontsize=7, ha="center", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=1.5", fc=color, alpha=0.75, lw=0),
        )
        legend_handles.append(
            mpatches.Patch(color=color, label=f"Obstacle: {labels_str} ({ids_str})")
        )

    if path:
        px = np.array([p[0] for p in path])
        py = np.array([p[1] for p in path])
        ax.fill_betweenx(
            py,
            np.clip(px - DRONE_HALF_WIDTH_PX, 0, W),
            np.clip(px + DRONE_HALF_WIDTH_PX, 0, W),
            color=path_color, alpha=0.15, label="_nolegend_",
        )
        ax.plot(
            px,
            py,
            color=path_color,
            linewidth=3.0,
            alpha=0.95,
            label="Safe path" if is_clear else "Least-bad path (crosses obstacles)",
        )
        ax.plot(px[0], py[0], "o", color=path_color, markersize=10, zorder=5)
        ax.plot(px[-1], py[-1], "*", color=path_color, markersize=15, zorder=5)
        legend_handles.append(
            mpatches.Patch(color=path_color, alpha=0.6, label="Safe corridor" if is_clear else "Least-bad corridor")
        )

    ax.legend(
        handles=legend_handles, loc="lower right",
        facecolor="#1a1a1a", labelcolor="white", fontsize=7,
        framealpha=0.85,
    )

    # ── right: dilated mask + path ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.imshow(mask, cmap="inferno", origin="upper", aspect="auto")
    ax2.set_title(
        "Dilated Obstacle Mask  +  Dijkstra Path\n"
        "(bright = obstacle / exclusion zone, dark = free space)",
        color="white", fontsize=11,
    )
    ax2.axis("off")

    if path:
        ax2.fill_betweenx(
            py,
            np.clip(px - DRONE_HALF_WIDTH_PX, 0, W),
            np.clip(px + DRONE_HALF_WIDTH_PX, 0, W),
            color=path_color, alpha=0.25,
        )
        ax2.plot(px, py, color=path_color, linewidth=2.5, alpha=0.95)
        ax2.plot(px[0], py[0], "o", color=path_color, markersize=9)
        ax2.plot(px[-1], py[-1], "*", color=path_color, markersize=13)

    plt.tight_layout(pad=1.0)
    src = Path(image_path)
    out = Path(output_path) if output_path else src.with_name(src.stem + "_path.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[path_planner] Output saved → {out}")
    return str(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obstacle polygon merge + safe-path planner"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output", default=None, help="Output image path")
    args = parser.parse_args()
    plan_and_visualise(args.image, args.output)


if __name__ == "__main__":
    main()
