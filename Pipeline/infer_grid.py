"""
infer_grid.py
-------------
Takes a folder of tiles for a single snapshot timestamp, runs haze/smoke
classification on each tile, assembles the spatial grid, writes a nowcast CSV,
renders a heatmap PNG, and returns a summary dict.

This will be called by:
- the web API (/api/classify)
- internal batch runners

Input assumptions:
- tiles_dir looks like: tiles/2025-10-25_T_18-30-00/
- Inside are files named like: tile_r12_c8_2025-10-25_T_18-30-00.png
  where r{row}_c{col} encodes the zero-based tile position in the grid.

Outputs:
1. CSV:  outputs/nowcast_csv/nowcast_<timestamp>.csv
   columns:
   timestamp,row,col,class_id,class_name,confidence,lat,lon,tile_path

2. PNG heatmap:  outputs/heatmaps/heatmap_<timestamp>.png
   (severity-weighted, smoke > haze > normal)

3. Summary dict (returned to caller):
   {
     "timestamp": ts,
     "nowcast_csv": ".../nowcast_<ts>.csv",
     "heatmap_png": ".../heatmap_<ts>.png",
     "grid_rows": N,
     "grid_cols": M,
     "smoke_detected": True/False,
     "max_confidence_smoke": 0.xx
   }
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import config
import stitch_to_grid
import heatmap_renderer
from Models.model_utils import load_model, load_checkpoint, preprocess_image
from Models.inference import classify_tensor


# -----------------------------------------------------------------------------
# Helper: extract timestamp from folder name if possible
# -----------------------------------------------------------------------------
_TS_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}_T_\d{2}-\d{2}-\d{2})"
    # e.g. "2025-10-25_T_18-30-00"
)

def _infer_timestamp_from_dirname(tiles_dir: Path) -> str:
    """
    Try to pull a timestamp out of the tiles directory name using our
    standard '%Y-%m-%d_T_%H-%M-%S' format. If not found, fall back to
    config.timestamp_now().

    Example:
      tiles/2025-10-25_T_18-30-00/  -> "2025-10-25_T_18-30-00"
    """
    m = _TS_PATTERN.search(str(tiles_dir))
    if m:
        return m.group(1)
    # fallback
    return config.timestamp_now()


# -----------------------------------------------------------------------------
# Helper: build severity grid from classified rows
# -----------------------------------------------------------------------------
def _build_severity_grid(
    rows: List[Dict[str, Any]],
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """
    Build a 2D numeric severity field with shape [n_rows, n_cols].

    Each cell = base_severity(class_name) * confidence
    where base_severity is ranked smoke > haze > normal.

    If multiple tiles claim the same (row,col), last one wins. (This
    shouldn't normally happen if filenames are unique per cell.)
    """
    severity_lookup = {
        "smoke":  3.0,
        "haze":   2.0,
        "normal": 1.0,
    }

    grid = np.zeros((n_rows, n_cols), dtype=float)

    for r in rows:
        rr = r["row"]
        cc = r["col"]
        cname = r["class_name"]
        conf = r["confidence"]

        base_val = severity_lookup.get(cname, 0.0)
        grid[rr, cc] = base_val * conf

    return grid


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
def run_nowcast_for_folder(
    tiles_dir: str,
    arch_name: str,
    weights_path: str,
    device: str = None,
    use_heads: bool = False,
) -> Dict[str, Any]:
    """
    Process all tiles in a timestamped folder, generate nowcast CSV + heatmap PNG.

    Args:
        tiles_dir: directory holding tiles for ONE timestamp snapshot.
                   e.g. "tiles/2025-10-25_T_18-30-00"
        arch_name: model architecture string ("basiccnn", "vgg16", "resnet50", ...)
        weights_path: path to .pth weights
        device: "cpu" or "cuda" (if None, auto-pick)
        use_heads: whether to load the +heads variant of the model

    Returns:
        summary dict:
        {
          "timestamp": ts,
          "nowcast_csv": ".../nowcast_<ts>.csv",
          "heatmap_png": ".../heatmap_<ts>.png",
          "grid_rows": N,
          "grid_cols": M,
          "smoke_detected": bool,
          "max_confidence_smoke": float
        }
    """
    tiles_dir = Path(tiles_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. determine timestamp
    ts = _infer_timestamp_from_dirname(tiles_dir)

    # 2. prepare model once
    model = load_model(
        arch_name=arch_name,
        num_classes=3,
        use_heads=use_heads,
        pretrained=True,
        freeze_features=True,
    )
    model = load_checkpoint(model, weights_path, device=device)
    model.eval()

    # 3. classify each tile
    rows: List[Dict[str, Any]] = []
    max_r = -1
    max_c = -1

    tile_files = sorted(
        [p for p in tiles_dir.iterdir() if p.is_file()],
        key=lambda p: p.name
    )

    for tile_path in tqdm(tile_files, desc=f"Nowcast {ts}"):
        # parse grid position
        rc: Optional[Tuple[int, int]] = stitch_to_grid.parse_tile_position(tile_path.name)
        if rc is None:
            # skip tiles that don't match r{row}_c{col}
            continue

        row_idx, col_idx = rc
        max_r = max(max_r, row_idx)
        max_c = max(max_c, col_idx)

        # preprocess + infer
        tensor = preprocess_image(str(tile_path), arch_name=arch_name, device=device)
        pred = classify_tensor(
            model=model,
            tensor=tensor,
            device=device,
            return_global=True,  # map to global 0=smoke,1=haze,2=normal
        )
        # lat/lon stub (future georeferencing)
        latlon = stitch_to_grid.tile_to_latlon(row_idx, col_idx)

        rows.append({
            "timestamp": ts,
            "row": row_idx,
            "col": col_idx,
            "class_id": pred["class_id"],          # global class id
            "class_name": pred["class_name"],      # "smoke"/"haze"/"normal"
            "confidence": pred["confidence"],
            "lat": latlon[0] if latlon else None,
            "lon": latlon[1] if latlon else None,
            "tile_path": str(tile_path),
        })

    if max_r < 0 or max_c < 0:
        # No valid tiles -> return empty result
        print(f"[infer_grid] WARN: no valid tiles found in {tiles_dir}")
        return {
            "timestamp": ts,
            "nowcast_csv": None,
            "heatmap_png": None,
            "grid_rows": 0,
            "grid_cols": 0,
            "smoke_detected": False,
            "max_confidence_smoke": 0.0,
        }

    # 4. update GRID_META with final grid size
    grid_rows = max_r + 1
    grid_cols = max_c + 1
    config.set_grid_size(grid_rows, grid_cols)

    # 5. write nowcast CSV
    config.ensure_output_dirs(verbose=False)
    nowcast_csv_path = Path(config.OUTPUT_DIRS["nowcast_csv"]) / f"nowcast_{ts}.csv"
    df = pd.DataFrame(rows, columns=[
        "timestamp",
        "row",
        "col",
        "class_id",
        "class_name",
        "confidence",
        "lat",
        "lon",
        "tile_path",
    ])
    nowcast_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(nowcast_csv_path, index=False)

    # 6. build severity grid for heatmap
    severity_grid = _build_severity_grid(
        rows=rows,
        n_rows=grid_rows,
        n_cols=grid_cols,
    )

    # 7. render heatmap
    heatmap_path = Path(config.OUTPUT_DIRS["heatmaps"]) / f"heatmap_{ts}.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_renderer.generate_heatmap(
        grid_matrix=severity_grid,
        out_path=str(heatmap_path),
        title=f"Nowcast {ts} ({config.DEFAULT_REGION})",
        show_colorbar=False,
    )

    # 8. summary stats for alert banner / dashboard
    smoke_conf_list = [r["confidence"] for r in rows if r["class_name"] == "smoke"]
    max_conf_smoke = max(smoke_conf_list) if smoke_conf_list else 0.0
    smoke_detected = len(smoke_conf_list) > 0

    # 9. return summary
    summary = {
        "timestamp": ts,
        "nowcast_csv": str(nowcast_csv_path),
        "heatmap_png": str(heatmap_path),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "smoke_detected": smoke_detected,
        "max_confidence_smoke": float(max_conf_smoke),
    }

    return summary


# -----------------------------------------------------------------------------
# Optional: quick manual test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Example manual test (replace paths with real ones before running):

    python -m pipeline.infer_grid

    Assumes:
    - tiles/2025-10-25_T_18-30-00/ exists with tile_rX_cY_*.png tiles
    - saved_weights/resnet50_best.pth exists
    """

    test_tiles_dir = "tiles/2025-10-25_T_18-30-00"
    arch = "resnet50"
    weights = "saved_weights/resnet50_best.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary = run_nowcast_for_folder(
        tiles_dir=test_tiles_dir,
        arch_name=arch,
        weights_path=weights,
        device=device,
        use_heads=False,
    )

    print("Nowcast summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
