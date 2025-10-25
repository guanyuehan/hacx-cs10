"""
stitch_to_grid.py
-----------------
Helpers for mapping tile image filenames to grid positions
(row, col) and (eventually) to geographic coordinates.

This is used by:
- pipeline/infer_grid.py to rebuild the spatial grid
- forecast code / alerting to know where smoke will advect

Conventions
-----------
We assume tiles are named with an embedded row/col index:
    tile_r{row}_c{col}_... .png/.tif/etc

Examples:
    tile_r0_c0_20251025.png
    tile_r03_c15_sceneA.tif
    TILE_R12_C8_SG.png        (case-insensitive is OK)

Indices are expected to be zero-based (r0 == top row, c0 == leftmost col).

If a filename doesn't match this pattern, we return None and warn,
instead of crashing the pipeline.
"""

import re
from typing import Optional, Tuple

# case-insensitive regex for r{row}_c{col}
# - row: digits after r or R
# - col: digits after c or C
_TILE_REGEX = re.compile(
    r".*?_r(?P<row>\d+)_c(?P<col>\d+).*",
    re.IGNORECASE
)


def parse_tile_position(filename: str) -> Optional[Tuple[int, int]]:
    """
    Extract (row, col) from a tile filename.

    Args:
        filename: just the basename, e.g. "tile_r03_c15_20251025.png"
                  (not required to be lowercase, not required to be zero-padded)

    Returns:
        (row:int, col:int) if match is found,
        None if not found (and prints a warning).
    """
    m = _TILE_REGEX.match(filename)
    if not m:
        print(f"[stitch_to_grid] WARN: could not parse row/col from filename '{filename}'")
        return None

    row = int(m.group("row"))
    col = int(m.group("col"))
    return row, col


def tile_to_latlon(row: int, col: int) -> Optional[Tuple[float, float]]:
    """
    Placeholder for mapping grid indices -> geographic coordinates.

    Args:
        row, col: zero-based grid indices

    Returns:
        (lat, lon) in degrees if we have a mapping,
        or None for now.

    Notes:
        Eventually, this will use:
        - AOI_BBOX from pipeline.config
        - GRID_META["n_rows"], GRID_META["n_cols"]

        For now we don't have calibrated georeference, so we leave this
        as a stub and return None to avoid lying.
    """
    # TODO: implement once AOI_BBOX + grid georeferencing is defined
    return None
