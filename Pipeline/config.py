"""
config.py
----------
Central configuration and runtime metadata for the AI Haze Sentinel pipeline.

This module is imported by:
- pipeline/infer_grid.py        (nowcasting / per-tile classification)
- pipeline/forecast_runner.py   (multi-step advection forecast)
- pipeline/alert_engine.py      (alert messaging)
- webapp routes                 (serving latest heatmaps / forecast)
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import pytz  # make sure pytz is in requirements

# -----------------------------------------------------------------------------
# CLASS MAPPINGS (GLOBAL / CANONICAL)
# -----------------------------------------------------------------------------
# This is the final, canonical ID order we expose to the dashboard and alerts.
# 0 = smoke
# 1 = haze
# 2 = normal
#
# Everything leaving the pipeline (CSV, heatmap, forecast, alert_engine)
# should be expressed in this space.
CLASS_ID_TO_NAME = {
    0: "smoke",
    1: "haze",
    2: "normal",
}

NAME_TO_CLASS_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}

# -----------------------------------------------------------------------------
# SPATIAL / TEMPORAL CONSTANTS
# -----------------------------------------------------------------------------
# Each tile is resized to this pixel size before inference.
TILE_SIZE_PX = 256

# Forecast timestep in minutes (dt for advection model).
FORECAST_DT_MIN = 30

# Default timezone for timestamps, naming, and ETA reporting.
DEFAULT_TIMEZONE = "Asia/Singapore"

# -----------------------------------------------------------------------------
# GRID / AOI METADATA
# -----------------------------------------------------------------------------
# GRID_META describes the current working prediction grid for a given run.
# n_rows / n_cols are dynamic: infer_grid.py will update this based on
# how many tiles were found for a given timestamp.
#
# tile_size_px is fixed by preprocessing and mainly here for reference.
#
GRID_META = {
    "n_rows": None,          # set at runtime by infer_grid or forecast_runner
    "n_cols": None,          # set at runtime by infer_grid or forecast_runner
    "tile_size_px": TILE_SIZE_PX,
}

# AOI_BBOX describes the area of interest in geospatial terms.
# We'll keep it empty for now, but downstream we might store:
# {
#     "lat_min": ...,
#     "lat_max": ...,
#     "lon_min": ...,
#     "lon_max": ...
# }
#
AOI_BBOX: Dict[str, float] = {}

# Human-readable region name. Used in alert text, dashboard labels, etc.
DEFAULT_REGION = "Singapore"

def set_grid_size(rows: int, cols: int) -> None:
    """
    Dynamically set the grid size for the current scene / timestamp batch.

    This should be called by infer_grid.py after it parses tile_rX_cY filenames.
    """
    GRID_META["n_rows"] = int(rows)
    GRID_META["n_cols"] = int(cols)

def set_region(name: str, bbox: Optional[Dict[str, float]] = None) -> None:
    """
    Dynamically update AOI region metadata.
    Use this if the user zooms out to Southeast Asia, etc.

    Args:
        name: human-readable region name ("Singapore", "Southeast Asia", ...)
        bbox: optional dict with lat/lon bounds. If provided, we merge it.
              Example:
              {
                  "lat_min": 0.5,
                  "lat_max": 6.0,
                  "lon_min": 100.0,
                  "lon_max": 106.0
              }
    """
    global DEFAULT_REGION
    DEFAULT_REGION = name
    if bbox:
        AOI_BBOX.update(bbox)

# -----------------------------------------------------------------------------
# OUTPUT PATHS / RUNTIME DIRECTORIES
# -----------------------------------------------------------------------------
# We never commit runtime artifacts to git. These are created at runtime.
#
# nowcast_csv/      per-timestamp classification grid (current observed haze)
# forecast_csv/     forecasted haze fields at +30m, +60m, etc
# heatmaps/         rendered PNGs for dashboard overlay (nowcast + forecast)
# alerts/           structured alert JSONs ("Haze incoming, ETA 6h")
# logs/             debug logs, errors, runtimes
#
OUTPUT_DIRS = {
    "nowcast_csv": "outputs/nowcast_csv",
    "forecast_csv": "outputs/forecast_csv",
    "heatmaps": "outputs/heatmaps",
    "alerts": "outputs/alerts",
    "logs": "outputs/logs",
}

def ensure_output_dirs(verbose: bool = False) -> None:
    """
    Ensure that all required runtime output directories exist.
    Call this before writing CSV/PNGs/etc.

    Args:
        verbose: if True, print created/missing dirs; if False, stay quiet.
    """
    for logical_name, path_str in OUTPUT_DIRS.items():
        p = Path(path_str)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"✅ Created {logical_name} dir at {p}")
        else:
            if verbose:
                # directory already exists
                print(f"ℹ {logical_name} dir already exists at {p}")

# -----------------------------------------------------------------------------
# TIMESTAMP HELPERS
# -----------------------------------------------------------------------------
def timestamp_now(tz_name: str = DEFAULT_TIMEZONE) -> str:
    """
    Return a human-readable timestamp string in the configured timezone.
    Format: YYYY-MM-DD_T_HH-MM-SS
    Example: '2025-10-25_T_18-30-00'

    This is used to name:
    - nowcast CSVs      nowcast_2025-10-25_T_18-30-00.csv
    - forecast CSVs     forecast_2025-10-25_T_18-30-00_T+30.csv
    - heatmaps          heatmap_2025-10-25_T_18-30-00.png
    """
    tz = pytz.timezone(tz_name)
    now_local = datetime.now(tz)
    return now_local.strftime("%Y-%m-%d_T_%H-%M-%S")

# -----------------------------------------------------------------------------
# SANITY SELF-TEST (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("---- AI Haze Sentinel config self-check ----")
    ensure_output_dirs(verbose=True)

    print("Current DEFAULT_REGION:", DEFAULT_REGION)
    print("Current GRID_META:", GRID_META)
    print("Current timestamp_now():", timestamp_now())

    # simulate user zooming out
    set_region("Southeast Asia", bbox={
        "lat_min": 0.5,
        "lat_max": 6.0,
        "lon_min": 100.0,
        "lon_max": 106.0,
    })
    set_grid_size(rows=32, cols=48)

    print("Updated DEFAULT_REGION:", DEFAULT_REGION)
    print("Updated AOI_BBOX:", AOI_BBOX)
    print("Updated GRID_META:", GRID_META)
