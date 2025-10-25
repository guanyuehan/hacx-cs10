"""
heatmap_renderer.py
-------------------
Render a haze/smoke intensity grid to a PNG heatmap for the dashboard.

This is called by:
- pipeline/infer_grid.py           (nowcast view: current conditions)
- pipeline/forecast_runner.py      (future view: T+30, T+60, ...)

Input to generate_heatmap():
    grid_matrix : 2D numpy array [n_rows, n_cols] of float severity scores
    out_path    : where to save the PNG
    title       : optional string for dev/debug overlays

Convention for grid_matrix:
    Higher values = worse air quality risk / thicker smoke.
    Typically built as:
        smoke  -> 3.0 * confidence
        haze   -> 2.0 * confidence
        normal -> 1.0 * confidence
    So values are usually in [0, 3] or so.

The renderer does NOT know about lat/lon yet. It just plots the grid
as an image and saves it.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server usage
import matplotlib.pyplot as plt
import config


def generate_heatmap(
    grid_matrix: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
    show_colorbar: bool = False,
) -> str:
    """
    Render the given 2D severity grid to a PNG heatmap and save it.

    Args:
        grid_matrix: np.ndarray of shape [n_rows, n_cols], float
                     higher = more severe (smoke > haze > normal)
        out_path: filesystem path (outputs/heatmaps/...png)
        title: optional string to display on the plot (for dev)
        show_colorbar: if True, draw a colorbar (useful for debugging,
                       probably False in production)

    Returns:
        The out_path string.
    """
    # Make sure directories exist
    config.ensure_output_dirs(verbose=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(grid_matrix, np.ndarray):
        grid_matrix = np.array(grid_matrix, dtype=float)

    # Avoid weird cases where the entire grid is the same value,
    # which can cause imshow to warn about singular colormap.
    vmin = float(np.nanmin(grid_matrix)) if grid_matrix.size else 0.0
    vmax = float(np.nanmax(grid_matrix)) if grid_matrix.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1e-6

    plt.figure(figsize=(6, 6), dpi=200)

    img = plt.imshow(
        grid_matrix,
        origin="upper",       # row 0 is top, row increases downward
        interpolation="nearest",
        cmap="inferno",       # high = bright/hot
        vmin=vmin,
        vmax=vmax,
    )

    if title:
        plt.title(title, fontsize=8)

    plt.xticks([])
    plt.yticks([])

    if show_colorbar:
        plt.colorbar(img, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    return out_path
