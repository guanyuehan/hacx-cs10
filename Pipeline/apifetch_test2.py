#!/usr/bin/env python3
"""
gibs_fetch.py
Fetch near-real-time RGB true-colour satellite imagery (VIIRS) from NASA GIBS,
covering a wide (~1000+ km) Southeast Asia-scale region around a given lat/lon,
at ~1 km per pixel, and save it as output.png.

Usage example:
    python gibs_fetch.py --lat 1.3521 --lon 103.8198 --km 1000 --date 2025-10-24
If --date is not provided, we auto-use (today - 1 day).

What you get:
- output.png in the current working directory
- printed absolute path
"""

import math
import os
import io
import sys
import argparse
import datetime
import requests
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------------------------------------

# Default layer: VIIRS SNPP True Color (good for haze/smoke)
# This is one of the standard near-real-time browse layers in GIBS.
DEFAULT_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"

# We target ~1 km per pixel. For GIBS WMTS, that's typically served under EPSG4326_1km
# using a 512x512 tile size.
TILEMATRIXSET = "EPSG4326_1km"
TILE_SIZE = 512  # pixels

# GIBS WMTS template for EPSG:4326 grid
# {layer}, {date}, {tilematrixset}, {zoom}, {row}, {col}
# Typically JPEG or PNG depending on the layer; VIIRS TrueColor is usually JPEG.
WMTS_URL_TEMPLATE = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
    "{layer}/default/{date}/{tilematrixset}/{zoom}/{row}/{col}.jpg"
)

# We'll use this zoom level logic:
# EPSG4326_1km in GIBS defines discrete zoom levels that map the globe into tiles.
# For our "about 1 km/pixel" ask, we assume zoom level 0 is coarse, larger zoom = finer.
# We'll choose a zoom that yields ~1 km/pixel. For EPSG4326_1km, zoom 0 covers the whole
# globe with a small number of tiles; higher zoom subdivides. Empirically, zoom=6 or 7
# is a good practical regional scale for ~1km-ish.
DEFAULT_ZOOM = 6

# Coverage threshold. If *all* tiles fail we error out.
MIN_SUCCESSFUL_TILES = 1


# ---------------------------------------------------------------------------------
# GEO UTILS
# ---------------------------------------------------------------------------------

def km_to_deg_lat(km: float) -> float:
    # ~111.32 km per deg latitude
    return km / 111.32

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    # longitude degrees per km shrinks with cos(latitude)
    return km / (111.32 * max(1e-6, math.cos(math.radians(lat_deg))))


def make_bbox_from_center(lat_deg: float, lon_deg: float, size_km: float):
    """
    Build geographic bbox around a centre point, roughly size_km x size_km.
    Returns (min_lat, min_lon, max_lat, max_lon)
    """
    half_km = size_km / 2.0
    dlat = km_to_deg_lat(half_km)
    dlon = km_to_deg_lon(half_km, lat_deg)

    min_lat = lat_deg - dlat
    max_lat = lat_deg + dlat
    min_lon = lon_deg - dlon
    max_lon = lon_deg + dlon
    return (min_lat, min_lon, max_lat, max_lon)


# ---------------------------------------------------------------------------------
# TILE GRID MATH (EPSG4326_1km assumptions)
# ---------------------------------------------------------------------------------
#
# For EPSG:4326 WMTS in GIBS, the tiling scheme is geographic (lat/lon),
# not WebMercator. We approximate it like this:
#
# - Latitude spans from +90 (north) down to -90 (south) → 180 degrees total.
# - Longitude spans from -180 to +180 → 360 degrees total.
#
# At a given zoom Z:
#   num_tiles_x = 2^Z * 2        (heuristic for EPSG4326_1km horizontal tiling)
#   num_tiles_y = 2^Z            (heuristic for EPSG4326_1km vertical tiling)
#
# That arrangement makes each tile width in degrees_lon:
#   tile_lon_deg = 360 / num_tiles_x
#
# and each tile height in degrees_lat:
#   tile_lat_deg = 180 / num_tiles_y
#
# IMPORTANT:
# GIBS actually encodes this in its TileMatrixSet metadata (GetCapabilities). We're
# baking a pragmatic model here for direct use. This heuristic matches common usage
# for EPSG4326_*km sets where tiles are 512px and pixel resolution ~1km.
#
# We'll clamp indices to valid ranges just in case.


def grid_geometry(zoom: int):
    """
    Return (#tiles_x, #tiles_y, tile_lon_deg, tile_lat_deg)
    for our assumed EPSG4326_1km grid at a given zoom.
    """
    tiles_y = 2 ** zoom
    tiles_x = tiles_y * 2  # 2:1 aspect (360° lon vs 180° lat)
    tile_lat_deg = 180.0 / tiles_y
    tile_lon_deg = 360.0 / tiles_x
    return tiles_x, tiles_y, tile_lon_deg, tile_lat_deg


def latlon_to_tile_indices(lat_deg: float, lon_deg: float, zoom: int):
    """
    Convert a lat/lon to (row, col) tile indices at a given zoom level
    in our assumed EPSG4326_1km tiling scheme.
    - row 0 is northmost tiles, increasing southward
    - col 0 is westmost tiles, increasing eastward
    """
    tiles_x, tiles_y, tile_lon_deg, tile_lat_deg = grid_geometry(zoom)

    # normalise lon to [-180,180), lat to [-90,90]
    lon_norm = (lon_deg + 180.0)
    lat_norm = (90.0 - lat_deg)  # invert latitude so row 0 = north

    col = int(lon_norm // tile_lon_deg)
    row = int(lat_norm // tile_lat_deg)

    # clamp
    col = max(0, min(tiles_x - 1, col))
    row = max(0, min(tiles_y - 1, row))

    return row, col


def bbox_to_tile_range(min_lat, min_lon, max_lat, max_lon, zoom):
    """
    Given a bbox in lat/lon, compute the inclusive range of tile rows and cols needed.
    We'll sample all tiles that intersect this bbox.
    """
    # note: bbox could cross negative/positive lon but for SEA (95E-120E) it's fine
    r1, c1 = latlon_to_tile_indices(max_lat, min_lon, zoom)  # top-left-ish
    r2, c2 = latlon_to_tile_indices(min_lat, max_lon, zoom)  # bottom-right-ish

    row_min = min(r1, r2)
    row_max = max(r1, r2)
    col_min = min(c1, c2)
    col_max = max(c1, c2)

    return row_min, row_max, col_min, col_max


def tile_bounds(row: int, col: int, zoom: int):
    """
    Get the geographic bounding box of a specific tile.
    Return (min_lat, min_lon, max_lat, max_lon) for that tile.
    """
    tiles_x, tiles_y, tile_lon_deg, tile_lat_deg = grid_geometry(zoom)

    # lat
    lat_north = 90.0 - (row * tile_lat_deg)
    lat_south = lat_north - tile_lat_deg

    # lon
    lon_west = -180.0 + (col * tile_lon_deg)
    lon_east = lon_west + tile_lon_deg

    # return in (min_lat, min_lon, max_lat, max_lon)
    return (lat_south, lon_west, lat_north, lon_east)


# ---------------------------------------------------------------------------------
# FETCH TILE IMAGE
# ---------------------------------------------------------------------------------

def fetch_tile(layer, date_str, zoom, row, col):
    """
    Download one tile (row,col) at given zoom as a PIL Image.
    Returns Image or None if fail.
    """
    url = WMTS_URL_TEMPLATE.format(
        layer=layer,
        date=date_str,
        tilematrixset=TILEMATRIXSET,
        zoom=zoom,
        row=row,
        col=col,
    )
    resp = requests.get(url, timeout=20)
    ctype = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or ("image" not in ctype.lower()):
        print(f"[WARN] tile r{row} c{col} zoom{zoom} not available ({resp.status_code}, {ctype})")
        return None
    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] tile r{row} c{col} zoom{zoom} could not decode: {e}")
        return None


# ---------------------------------------------------------------------------------
# STITCH
# ---------------------------------------------------------------------------------

def stitch_tiles(layer, date_str, zoom, bbox, out_km):
    """
    1. Figure out which tiles we need for the bbox.
    2. Download each tile.
    3. Paste into a large canvas.
    4. Crop down to the exact bbox.
    5. Return final RGB PIL.Image.
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    row_min, row_max, col_min, col_max = bbox_to_tile_range(
        min_lat, min_lon, max_lat, max_lon, zoom
    )

    n_rows = (row_max - row_min + 1)
    n_cols = (col_max - col_min + 1)

    print(f"[INFO] Need tiles rows {row_min}..{row_max}, cols {col_min}..{col_max}")
    print(f"[INFO] Total tiles to fetch: {n_rows * n_cols}")

    # We'll paste into a big canvas in tile grid coordinates.
    canvas_w = n_cols * TILE_SIZE
    canvas_h = n_rows * TILE_SIZE
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

    success_count = 0

    # Keep track of geographic extents per tile so we can crop accurately later
    # We'll build arrays of lon/lat per pixel row/col for the final crop math.
    # Simpler approach:
    #   - We'll compute the lat/lon bounds of the entire stitched canvas from row_min..row_max, col_min..col_max
    #   - Then convert your desired bbox to pixel coords in that stitched canvas.
    tiles_geo_bounds = {}  # (row_i, col_j) -> (min_lat, min_lon, max_lat, max_lon)

    for tile_row in range(row_min, row_max + 1):
        for tile_col in range(col_min, col_max + 1):

            img = fetch_tile(layer, date_str, zoom, tile_row, tile_col)
            y_off = (tile_row - row_min) * TILE_SIZE
            x_off = (tile_col - col_min) * TILE_SIZE

            if img is not None:
                canvas.paste(img, (x_off, y_off))
                success_count += 1
            else:
                # leave black where missing
                pass

            tb_min_lat, tb_min_lon, tb_max_lat, tb_max_lon = tile_bounds(tile_row, tile_col, zoom)
            tiles_geo_bounds[(tile_row, tile_col)] = (tb_min_lat, tb_min_lon, tb_max_lat, tb_max_lon)

    if success_count < MIN_SUCCESSFUL_TILES:
        raise RuntimeError("No tiles available for this request/date/layer.")

    # Now we crop the stitched canvas down to exactly the requested bbox.
    # We need a function to map lat/lon -> pixel in the stitched canvas.
    # We'll assume lat/lon vary linearly across each tile (true for geographic tiling).
    # For the stitched canvas:
    #   top-left corresponds to tile_row=row_min, tile_col=col_min, tile pixel (0,0)
    #   bottom-right corresponds to tile_row=row_max, tile_col=col_max, pixel (TILE_SIZE-1, TILE_SIZE-1)

    # Compute full stitched geo-extent:
    full_min_lat, full_min_lon, full_max_lat, full_max_lon = stitched_bounds(row_min, row_max, col_min, col_max, zoom)

    # Convert desired bbox edges to pixel coords in stitched space
    crop_left   = lon_to_px(min_lon,  full_min_lon, full_max_lon, canvas_w)
    crop_right  = lon_to_px(max_lon,  full_min_lon, full_max_lon, canvas_w)
    crop_top    = lat_to_py(max_lat,  full_min_lat, full_max_lat, canvas_h)
    crop_bottom = lat_to_py(min_lat,  full_min_lat, full_max_lat, canvas_h)

    # Clamp + round
    crop_left   = max(0, min(canvas_w, int(math.floor(crop_left))))
    crop_right  = max(0, min(canvas_w, int(math.ceil(crop_right))))
    crop_top    = max(0, min(canvas_h, int(math.floor(crop_top))))
    crop_bottom = max(0, min(canvas_h, int(math.ceil(crop_bottom))))

    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise RuntimeError("Crop region collapsed; check bbox / zoom.")

    cropped = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
    return cropped


def stitched_bounds(row_min, row_max, col_min, col_max, zoom):
    """
    Get geographic bounding box (min_lat, min_lon, max_lat, max_lon)
    for the ENTIRE stitched canvas covering tile rows row_min..row_max and
    tile cols col_min..col_max.
    """
    # top-left tile:
    tl_min_lat, tl_min_lon, tl_max_lat, tl_max_lon = tile_bounds(row_min, col_min, zoom)
    # bottom-right tile:
    br_min_lat, br_min_lon, br_max_lat, br_max_lon = tile_bounds(row_max, col_max, zoom)

    stitched_min_lat = min(tl_min_lat, tl_max_lat, br_min_lat, br_max_lat)
    stitched_max_lat = max(tl_min_lat, tl_max_lat, br_min_lat, br_max_lat)
    stitched_min_lon = min(tl_min_lon, tl_max_lon, br_min_lon, br_max_lon)
    stitched_max_lon = max(tl_min_lon, tl_max_lon, br_min_lon, br_max_lon)

    return (stitched_min_lat, stitched_min_lon, stitched_max_lat, stitched_max_lon)


def lon_to_px(lon, min_lon, max_lon, width_px):
    # linear map lon [-180..180] to pixel [0..width_px]
    return ( (lon - min_lon) / (max_lon - min_lon) ) * width_px

def lat_to_py(lat, min_lat, max_lat, height_px):
    # NOTE: y grows downward in images, but latitude grows upward (north).
    # top pixel (y=0) corresponds to max_lat.
    return ( (max_lat - lat) / (max_lat - min_lat) ) * height_px


# ---------------------------------------------------------------------------------
# DATE HANDLING
# ---------------------------------------------------------------------------------

def resolve_date_str(user_date_str: str | None) -> str:
    """
    If user gave --date, use it.
    Else use yesterday (today - 1 day) so imagery is likely available.
    """
    if user_date_str:
        return user_date_str
    today = datetime.date.today()
    yday = today - datetime.timedelta(days=1)
    return yday.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch ~1km true-colour VIIRS RGB patch from NASA GIBS and save output.png")
    parser.add_argument("--lat", type=float, required=True, help="centre latitude (deg)")
    parser.add_argument("--lon", type=float, required=True, help="centre longitude (deg)")
    parser.add_argument("--km", type=float, default=1000.0, help="size of square region in km (default 1000km)")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD date to fetch (default: yesterday)")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM, help=f"zoom level for EPSG4326_1km (default {DEFAULT_ZOOM})")
    parser.add_argument("--layer", type=str, default=DEFAULT_LAYER, help=f"GIBS layer ID (default {DEFAULT_LAYER})")

    args = parser.parse_args()

    date_str = resolve_date_str(args.date)

    # 1. Make target bbox around (lat,lon)
    bbox = make_bbox_from_center(args.lat, args.lon, args.km)
    min_lat, min_lon, max_lat, max_lon = bbox
    print("[INFO] BBOX deg:", bbox)
    print(f"[INFO] DATE: {date_str}")
    print(f"[INFO] LAYER: {args.layer}")
    print(f"[INFO] ZOOM: {args.zoom}")

    # 2. Stitch tiles for that bbox/date
    final_img = stitch_tiles(
        layer=args.layer,
        date_str=date_str,
        zoom=args.zoom,
        bbox=bbox,
        out_km=args.km,
    )

    # 3. Save output
    out_path = os.path.abspath("output.png")
    final_img.save(out_path)
    print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()
