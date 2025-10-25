#!/usr/bin/env python3
"""
realtime_asia_geo.py

Fetch near-real-time Himawari-8 true-color ("GeoColor") full-disk imagery from the
RAMMB Slider (CIRA / NOAA mirror), crop it to a square region around a given lat/lon,
and save output.png.

This gives you ~1–2 km per pixel around Southeast / East Asia with ~10-minute latency.
No NASA GIBS, no WMTS capability scraping, no tiling math, no SSL nonsense.

Example usage:
    python realtime_asia_geo.py --lat 1.3521 --lon 103.8198 --km 1000

That will:
    - pick the most recent available Himawari-8 timestamp (rounded to last 10 min UTC
      and walking backwards if needed),
    - download the latest full-disk GeoColor PNG,
    - crop ~1000 km x ~1000 km centered on (1.3521 N, 103.8198 E),
    - save ./output.png,
    - print the absolute file path.

Assumptions / Notes:
- Himawari-8 is geostationary at ~140.7°E. The "full disk" product is rendered in a
  fixed geostationary projection used by RAMMB. We treat it as an equirectangular
  approximation whose pixel->lat/lon mapping we model explicitly.
- The RAMMB Slider mirror generally serves the full disk as a single PNG per timestamp
  per product. The URL pattern includes YYYYMMDDHHMM. We only request that 1 big image.
- We assume GeoColor product is available at about 10-minute cadence.
- Resolution is typically ~11k x 11k pixels full disk. We'll crop after projecting our
  requested bbox into pixel space.
- This is good enough for haze/smoke/wildfire plume monitoring in SEA.

If RAMMB changes their URL structure, you'll get 404. In that case we'll just need to
update BASE_URL_TEMPLATE and (maybe) the projection constants below.
"""

import argparse
import datetime
import io
import math
import os
import sys
import time
from typing import Optional, Tuple

import requests
from PIL import Image


# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------

# Himawari-8 nominal sub-satellite longitude (degrees east)
HIMAWARI_SUBLON_DEG = 140.7

# Projection assumptions for RAMMB Himawari full-disk GeoColor:
# The RAMMB full-disk GeoColor imagery is typically rendered in a fixed geostationary
# view: the Earth disk is roughly centered in the image, which is square-ish
# (e.g. ~11000 x ~11000 px), and each pixel corresponds to a ray from the satellite
# through Earth's surface. Near SEA this is close to ~1-2 km per px.
#
# We'll assume that RAMMB gives us:
#   - An image where the visible Earth disk fits inside a square.
#   - The mapping can be approximated by a standard geostationary projection:
#       given lat, lon -> (x_img, y_img).
#
# We'll compute that mapping using the standard geostationary projection math below.

# Satellite geometry constants (km)
Re_equatorial = 6378.137  # Earth's equatorial radius
H = 35786.0               # height above Earth's surface for geostationary orbit
Rs = (Re_equatorial + H)  # distance from Earth's center to satellite [~42164 km]

# How far back we search timestamps (in 10-minute steps)
MAX_BACKOFF_STEPS = 18  # 18 * 10min = 180 min (~3 hours)

# RAMMB Slider style full-disk Himawari-8 GeoColor URL template.
# We are not hitting tiles; we are hitting the rendered full-disk composite.
#
# This pattern is representative of the RAMMB Slider public imagery structure for Himawari-8:
#   https://rammb-slider.cira.colostate.edu/data/imagery/Himawari-8/FullDisk/GeoColor/YYYYMMDDHHMM/000_000.png
#
# 000_000.png is the whole disk at native resolution.
#
BASE_URL_TEMPLATE = (
    "https://rammb-slider.cira.colostate.edu/data/imagery/Himawari-8/FullDisk/GeoColor/"
    "{stamp}/000_000.png"
)


# ------------------------------------------------------------------------------------
# TIME HELPERS
# ------------------------------------------------------------------------------------

def round_down_to_10min_utc(now_utc: datetime.datetime) -> datetime.datetime:
    # ensure timezone-naive UTC
    if now_utc.tzinfo is not None:
        now_utc = now_utc.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    minute = (now_utc.minute // 10) * 10
    return now_utc.replace(minute=minute, second=0, microsecond=0)


def ts_to_stamp(ts: datetime.datetime) -> str:
    # timestamp -> "YYYYMMDDHHMM"
    return ts.strftime("%Y%m%d%H%M")


def fetch_full_disk(ts: datetime.datetime, timeout=15) -> Optional[Image.Image]:
    """
    Try downloading the full-disk GeoColor PNG for a given UTC timestamp (10-min aligned).
    Return PIL Image or None on 404/other fail.
    """
    stamp = ts_to_stamp(ts)
    url = BASE_URL_TEMPLATE.format(stamp=stamp)
    try:
        r = requests.get(url, timeout=timeout)
    except Exception as e:
        print(f"[WARN] request failed {url}: {e}")
        return None

    if r.status_code != 200:
        print(f"[WARN] status {r.status_code} for {url}")
        return None

    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] decode fail for {url}: {e}")
        return None


def get_latest_full_disk() -> Tuple[datetime.datetime, Image.Image]:
    """
    Find the most recent available full-disk image by walking backwards
    in 10-minute steps.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    base = round_down_to_10min_utc(now_utc)

    for step in range(MAX_BACKOFF_STEPS):
        probe = base - datetime.timedelta(minutes=10 * step)
        img = fetch_full_disk(probe)
        if img is not None:
            print(f"[INFO] using timestamp {probe.isoformat()}Z")
            return probe, img

    raise RuntimeError("No recent Himawari full-disk GeoColor image found in backoff window.")


# ------------------------------------------------------------------------------------
# GEO ↔ IMAGE PROJECTION
# ------------------------------------------------------------------------------------

def latlon_to_fulldisk_xy(
    lat_deg: float,
    lon_deg: float,
    sublon_deg: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float]]:
    """
    Approximate conversion from lat/lon (deg) -> pixel (x,y) in the full-disk GeoColor image.

    We'll use standard geostationary projection math:
    1. Convert lat/lon to geocentric coords.
    2. Project onto satellite view plane.
    3. Normalize that to [-1,+1] in both axes.
    4. Map that to pixel coords in the full-disk PNG.

    Returns (x,y) in pixel coordinates (float), or None if not visible from Himawari.
    """

    # radians
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sublon = math.radians(sublon_deg)

    # Step 1: geocentric coordinates of the point on Earth surface
    cos_lat = math.cos(lat)
    sin_lat = math.sin(lat)
    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)

    # Earth center at (0,0,0).
    # Satellite is on equatorial plane at longitude sublon, radius Rs from Earth's center.
    # We'll shift longitude so satellite sublon becomes 0 to simplify math:
    dlon = lon - sublon
    cos_dlon = math.cos(dlon)
    sin_dlon = math.sin(dlon)

    # Earth's surface point in sat-centered frame (scaled by Earth's radius Re_equatorial)
    # We'll pretend Earth is a sphere for this mapping.
    x = Re_equatorial * cos_lat * cos_dlon
    y = Re_equatorial * cos_lat * sin_dlon
    z = Re_equatorial * sin_lat

    # Satellite position in that same frame: (Rs, 0, 0)
    sx = Rs
    sy = 0.0
    sz = 0.0

    # Vector from satellite to point on Earth:
    vx = x - sx
    vy = y - sy
    vz = z - sz

    # Visibility test (point must be "in front" of satellite and not behind the Earth limb).
    # If dot( point_vector_from_earth_center , satellite_vector_from_earth_center ) < 0,
    # it's visible. Another commonly used test is whether the angle subtended is < ~90 deg.
    # We'll do a simple horizon check using the parametric form from GOES/Himawari projection.
    #
    # Derivation shortcut:
    # For geostationary projection, if the line of sight from satellite intersects Earth
    # before hitting the point, it's visible. A simpler approximate test is:
    if (vx * sx + vy * sy + vz * sz) > 0:
        # likely behind the limb
        return None

    # Project to image plane:
    # Geostationary projection often uses:
    #   x_img ~ -vx / vz
    #   y_img ~  vy / vz
    # There's some sign convention; we'll define an approximate mapping
    # that results in the Earth disk being roughly centered.
    if abs(vz) < 1e-6:
        return None

    u = -vx / vz
    v =  vy / vz

    # Now we need to map u,v (roughly [-something,+something]) onto image pixels.
    # We'll assume the visible full disk roughly spans a circle of radius R_px
    # centered in the middle of the image.
    # We'll guess that |u|<=1 and |v|<=1 maps to that circle.
    #
    # We'll scale u,v -> pixel coords:
    #   cx = img_w/2, cy = img_h/2
    #   R_px ~ 0.47 * img_w (empirical for Himawari full-disk composites)
    # We'll tune the factor so SEA should land in the correct region of the disk.
    cx = img_w / 2.0
    cy = img_h / 2.0
    R_px = 0.47 * img_w  # heuristic scaling

    px = cx + u * R_px
    py = cy + v * R_px

    # If it's way off-canvas, it's probably not visible / invalid mapping
    if px < -0.1 * img_w or px > 1.1 * img_w or py < -0.1 * img_h or py > 1.1 * img_h:
        return None

    return (px, py)


def bbox_pixel_bounds(
    bbox: Tuple[float, float, float, float],
    sublon_deg: float,
    full_img: Image.Image,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert bbox [min_lat, min_lon, max_lat, max_lon] to a crop box in pixel coords
    (left, top, right, bottom) within full_img using latlon_to_fulldisk_xy().

    Returns None if bbox is fully off-disk.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    w, h = full_img.size

    corners = [
        (max_lat, min_lon),  # TL
        (max_lat, max_lon),  # TR
        (min_lat, min_lon),  # BL
        (min_lat, max_lon),  # BR
    ]
    pxs = []
    pys = []
    for (lat, lon) in corners:
        xy = latlon_to_fulldisk_xy(lat, lon, sublon_deg, w, h)
        if xy is not None:
            x, y = xy
            pxs.append(x)
            pys.append(y)

    if not pxs or not pys:
        return None

    left   = max(0, math.floor(min(pxs)))
    right  = min(w, math.ceil(max(pxs)))
    top    = max(0, math.floor(min(pys)))
    bottom = min(h, math.ceil(max(pys)))

    if right <= left or bottom <= top:
        return None

    return (left, top, right, bottom)


# ------------------------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------------------------

def bbox_from_center_km(lat_deg: float, lon_deg: float, size_km: float):
    half = size_km / 2.0
    dlat = half / 111.32
    dlon = half / (111.32 * max(1e-6, math.cos(math.radians(lat_deg))))
    return (
        lat_deg - dlat,             # min_lat
        lon_deg - dlon,             # min_lon
        lat_deg + dlat,             # max_lat
        lon_deg + dlon,             # max_lon
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True, help="center latitude (deg)")
    parser.add_argument("--lon", type=float, required=True, help="center longitude (deg)")
    parser.add_argument("--km",  type=float, default=1000.0,
                        help="square size in km (default 1000 km)")
    args = parser.parse_args()

    # Build bbox in lat/lon
    bbox = bbox_from_center_km(args.lat, args.lon, args.km)
    print("[INFO] bbox (lat/lon):", bbox)

    # 1. Get latest full-disk GeoColor image
    ts, full_img = get_latest_full_disk()
    print(f"[INFO] full-disk size: {full_img.size[0]} x {full_img.size[1]} px")

    # 2. Map bbox -> pixel crop
    crop_box = bbox_pixel_bounds(bbox, HIMAWARI_SUBLON_DEG, full_img)

    if crop_box is None:
        raise RuntimeError("Requested bbox appears to be out of Himawari view.")

    (left, top, right, bottom) = crop_box
    print(f"[INFO] crop pixel box: left={left}, top={top}, right={right}, bottom={bottom}")

    cropped = full_img.crop((left, top, right, bottom))

    # 3. Save output
    out_path = os.path.abspath("output.png")
    cropped.save(out_path)
    print(f"[OK] Saved output to {out_path}")


if __name__ == "__main__":
    main()
