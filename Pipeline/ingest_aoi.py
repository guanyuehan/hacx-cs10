"""
ingest_aoi.py
-------------
Stage -1 of the AI Haze Sentinel pipeline.

This module is responsible for:
1. Taking a user-defined AOI (region + bounding box).
2. Fetching the latest satellite snapshot for that AOI (stubbed for now).
3. Resampling it so each pixel ~ 1 km (TODO: future).
4. Cutting it into 256x256 tiles (padding at edges so the grid is rectangular).
5. Saving those tiles into a timestamped folder under ./tiles/<timestamp>/.
6. Cleaning up older snapshots to control storage.

After this runs, Stage 0 (`infer_grid.run_nowcast_for_folder`) can take over.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import shutil
import numpy as np
from PIL import Image
import re
import torch
import config


# -----------------------------------------------------------------------------
# 1. Region / AOI setup
# -----------------------------------------------------------------------------

def set_active_region(region_name: str, bbox: Dict[str, float]) -> None:
    """
    Update global config with the AOI metadata.

    Args:
        region_name: Human-readable name (e.g. "Singapore", "Southeast Asia")
        bbox: Dict with keys like:
              {
                "lat_min": ...,
                "lat_max": ...,
                "lon_min": ...,
                "lon_max": ...
              }
    """
    config.set_region(region_name, bbox)


# -----------------------------------------------------------------------------
# 2. Fetch AOI imagery (stubbed for now)
# -----------------------------------------------------------------------------

def fetch_satellite_aoi(bbox: Dict[str, float]) -> np.ndarray:
    """
    Fetch the latest satellite RGB image for the AOI bbox.

    For MVP / offline dev:
    - We don't yet call a real satellite provider.
    - Instead, we synthesize a deterministic "fake AOI" image so that
      the rest of the pipeline can run end-to-end.

    We'll generate a smooth gradient so tiles visibly differ spatially,
    which helps us debug tiling and stitching.

    Returns:
        img_np: uint8 array of shape [H, W, 3]

    Current default: 1024 px tall x 1536 px wide
        - 1024 / 256 = 4 tile rows
        - 1536 / 256 = 6 tile cols
    """
    H = 1024  # pixels vertically -> 4 tiles of 256px
    W = 1536  # pixels horizontally -> 6 tiles of 256px

    # Create coordinate grids
    y = np.linspace(0, 1, H, endpoint=True).reshape(H, 1)
    x = np.linspace(0, 1, W, endpoint=True).reshape(1, W)

    # Build simple gradient:
    # R channel: horizontal gradient
    # G channel: vertical gradient
    # B channel: mild mix
    R = (x * 255).astype(np.uint8)
    G = (y * 255).astype(np.uint8)
    B = (((x + y) / 2) * 255).astype(np.uint8)

    img_np = np.stack([R, G, B], axis=2)  # [H, W, 3]
    return img_np


# -----------------------------------------------------------------------------
# 3. Resample to km-scale grid
# -----------------------------------------------------------------------------

def resample_to_km_grid(
    aoi_img: np.ndarray,
    bbox: Dict[str, float],
    km_per_px: float = 1.0,
) -> np.ndarray:
    """
    Enforce ~1 km per pixel resolution.

    For now (MVP):
    - We don't know the actual geospatial scale of the dummy image.
    - We return the input as-is (no-op).
    - We keep this function so we can later insert real reprojection /
      resampling logic using bbox and km_per_px.

    Args:
        aoi_img: np.ndarray [H, W, 3] uint8
        bbox: AOI bounding box dict from config.set_region()
        km_per_px: desired km per pixel (default 1.0)

    Returns:
        np.ndarray [H, W, 3] uint8, same for now.
    """
    # TODO: implement real resampling using bbox + km_per_px
    return aoi_img


# -----------------------------------------------------------------------------
# 4. Slice AOI into 256x256 tiles (with padding on edges)
# -----------------------------------------------------------------------------

def slice_into_tiles(
    img: np.ndarray,
    tile_size_px: int = config.TILE_SIZE_PX,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Cut the AOI image into fixed-size tiles, padding as needed so that
    the final grid is strictly rectangular (no ragged edges).

    Args:
        img: np.ndarray [H, W, 3] uint8
        tile_size_px: int, default 256

    Returns:
        tiles: list of tuples (tile_img_np, row_idx, col_idx)
            - tile_img_np: [tile_size_px, tile_size_px, 3] uint8
            - row_idx: zero-based tile row
            - col_idx: zero-based tile col

    Behavior:
        - If img dims (H,W) are not multiples of tile_size_px,
          we pad the image with zeros (black) on the bottom/right edges.
    """
    H, W, C = img.shape
    assert C == 3, "Expected RGB image"

    tile = tile_size_px

    # Compute how many tiles we need in each direction
    n_rows = int(np.ceil(H / tile))
    n_cols = int(np.ceil(W / tile))

    # Pad the image to (n_rows*tile, n_cols*tile)
    pad_H = n_rows * tile - H
    pad_W = n_cols * tile - W

    if pad_H > 0 or pad_W > 0:
        padded = np.zeros(
            (H + pad_H, W + pad_W, 3),
            dtype=np.uint8
        )
        padded[:H, :W, :] = img
        img = padded
        H, W, _ = img.shape  # update after padding

    tiles_out: List[Tuple[np.ndarray, int, int]] = []

    # Now slice cleanly by tile_size
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * tile
            y1 = y0 + tile
            x0 = c * tile
            x1 = x0 + tile

            tile_img = img[y0:y1, x0:x1, :]
            tiles_out.append((tile_img, r, c))

    return tiles_out


# -----------------------------------------------------------------------------
# 5. Save tiles into ./tiles/<timestamp>/tile_r{row}_c{col}_{timestamp}.png
# -----------------------------------------------------------------------------

def save_tiles_for_timestamp(
    tiles: List[Tuple[np.ndarray, int, int]],
    timestamp_str: str,
    tiles_root: str = "tiles",
) -> Path:
    """
    Save each tile to disk inside a timestamped directory.

    Output layout:
        tiles/<timestamp_str>/
            tile_r0_c0_<timestamp_str>.png
            tile_r0_c1_<timestamp_str>.png
            ...

    Args:
        tiles: list of (tile_img_np, row_idx, col_idx)
        timestamp_str: timestamp token like "2025-10-25_T_18-30-00"
        tiles_root: base directory for tile snapshots

    Returns:
        Path to the created directory (tiles/<timestamp_str>)
    """
    ts_dir = Path(tiles_root) / timestamp_str
    ts_dir.mkdir(parents=True, exist_ok=True)

    for tile_img, r, c in tiles:
        out_name = f"tile_r{r}_c{c}_{timestamp_str}.png"
        out_path = ts_dir / out_name

        # convert np.ndarray -> PIL Image for saving
        pil_img = Image.fromarray(tile_img.astype(np.uint8), mode="RGB")
        pil_img.save(out_path, format="PNG")

    return ts_dir


# -----------------------------------------------------------------------------
# 6. Cleanup old snapshots
# -----------------------------------------------------------------------------

_TS_NAME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}_T_\d{2}-\d{2}-\d{2}$"
    # matches "2025-10-25_T_18-30-00"
)

def _list_timestamp_dirs(tiles_root: str = "tiles") -> List[str]:
    """
    Return a sorted list of timestamp folder names under tiles_root that
    match our timestamp naming convention.
    Sorted newest-last (lexicographic works because format is sortable).
    """
    root = Path(tiles_root)
    if not root.exists():
        return []

    ts_folders = []
    for p in root.iterdir():
        if p.is_dir() and _TS_NAME_PATTERN.match(p.name):
            ts_folders.append(p.name)

    # Sort ascending by timestamp string (YYYY-mm-dd_T_HH-MM-SS sorts naturally)
    ts_folders.sort()
    return ts_folders


def cleanup_old_snapshots(
    keep_last: int = 12,
    tiles_root: str = "tiles",
) -> None:
    """
    Keep only the most recent `keep_last` timestamp folders in ./tiles/.
    For older snapshots, also remove their paired nowcast CSV + heatmap PNG.

    Specifically removes:
    - tiles/<old_ts>/
    - outputs/nowcast_csv/nowcast_<old_ts>.csv
    - outputs/heatmaps/heatmap_<old_ts>.png

    NOTE: In the future we should also remove:
    - outputs/forecast_csv/forecast_<old_ts>_T+XX.csv
    - outputs/alerts/<old_ts>_alert.json
    """
    all_ts = _list_timestamp_dirs(tiles_root=tiles_root)
    if len(all_ts) <= keep_last:
        return  # nothing to clean

    to_delete = all_ts[0 : len(all_ts) - keep_last]

    for ts in to_delete:
        # 1. delete tiles folder
        ts_path = Path(tiles_root) / ts
        if ts_path.exists():
            print(f"[ingest_aoi] cleanup: removing tiles snapshot {ts_path}")
            shutil.rmtree(ts_path, ignore_errors=True)

        # 2. delete nowcast CSV
        nowcast_csv = Path(config.OUTPUT_DIRS["nowcast_csv"]) / f"nowcast_{ts}.csv"
        if nowcast_csv.exists():
            print(f"[ingest_aoi] cleanup: removing nowcast CSV {nowcast_csv}")
            nowcast_csv.unlink(missing_ok=True)

        # 3. delete heatmap PNG
        heatmap_png = Path(config.OUTPUT_DIRS["heatmaps"]) / f"heatmap_{ts}.png"
        if heatmap_png.exists():
            print(f"[ingest_aoi] cleanup: removing heatmap PNG {heatmap_png}")
            heatmap_png.unlink(missing_ok=True)

        # TODO later: also clear forecast_csv + alerts for ts


# -----------------------------------------------------------------------------
# 7. High-level entrypoint for Stage -1
# -----------------------------------------------------------------------------

def run_aoi_ingest(
    region_name: str,
    bbox: Dict[str, float],
    km_per_px: float = 1.0,
    keep_last: int = 12,
) -> Dict[str, str]:
    """
    Full AOI ingest pipeline for a single snapshot.

    Steps:
    1. Record AOI config (region, bbox)
    2. Generate timestamp
    3. Fetch AOI satellite image (stubbed gradient for now)
    4. Resample to ~1 km/pixel (no-op placeholder)
    5. Slice into padded 256x256 tiles
    6. Save tiles into tiles/<timestamp>/tile_r{row}_c{col}_<timestamp>.png
    7. Cleanup older snapshots (tiles + stale outputs)
    8. Return metadata so Stage 0 (nowcast) can run

    Returns:
        {
          "timestamp": <ts>,
          "tiles_dir": "tiles/<ts>",
          "region": <region_name>,
          "bbox_used": bbox
        }
    """
    # 1. region/AOI metadata
    set_active_region(region_name, bbox)

    # 2. timestamp for this snapshot
    ts = config.timestamp_now()

    # 3. fetch AOI imagery (dummy gradient for now)
    raw_img = fetch_satellite_aoi(bbox)

    # 4. resample to ~1 km/px (future geospatial scaling hook)
    km_img = resample_to_km_grid(raw_img, bbox, km_per_px=km_per_px)

    # 5. slice AOI into tiles (with padding to full 256x256 blocks)
    tiles = slice_into_tiles(km_img, tile_size_px=config.TILE_SIZE_PX)

    # 6. save tiles under tiles/<timestamp>/
    tiles_dir_path = save_tiles_for_timestamp(tiles, timestamp_str=ts)

    # 7. cleanup retention across historical snapshots
    cleanup_old_snapshots(keep_last=keep_last, tiles_root="tiles")

    # 8. return summary for chaining into Stage 0
    return {
        "timestamp": ts,
        "tiles_dir": str(tiles_dir_path),
        "region": region_name,
        "bbox_used": bbox,
    }


# -----------------------------------------------------------------------------
# optional quick manual self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Example manual test (dummy AOI, using a fake bbox):

    python -m pipeline.ingest_aoi

    This will:
    - create tiles/<timestamp>/tile_r*_c*_<timestamp>.png
    - prune old snapshots if > keep_last
    - print the returned summary
    """

    fake_bbox = {
        "lat_min": 0.5,
        "lat_max": 6.0,
        "lon_min": 100.0,
        "lon_max": 106.0,
    }

    summary = run_aoi_ingest(
        region_name="Southeast Asia",
        bbox=fake_bbox,
        km_per_px=1.0,
        keep_last=12,
    )

    print("Ingest summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
