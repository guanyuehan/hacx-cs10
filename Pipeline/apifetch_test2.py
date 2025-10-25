import math
import requests
from PIL import Image
from io import BytesIO
import numpy as np

SNAPSHOT_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"

# ===================== CONFIG =====================
CENTER_LAT = 1.3521      # Singapore approx
CENTER_LON = 103.8198
PATCH_SIZE_KM = 256      # ~256km x 256km box
PX = 256                 # output tile size 256x256 -> ~1km/pixel

# Try multiple days, newest first. You can extend this list.
DATE_CANDIDATES = [
    "2023-10-01",
    "2023-09-30",
    "2023-09-29",
]

# NASA layer: combined Terra + Aqua corrected reflectance (true colour daily mosaic)
LAYER_COMBINED = "MODIS_Terra_Aqua_CorrectedReflectance_TrueColor"
# ==================================================


def make_bbox(center_lat, center_lon, size_km):
    """
    Build a bbox [minLon,minLat,maxLon,maxLat] that spans ~size_km x size_km
    around a center point, accounting for lon scale at that latitude.
    """
    half_km = size_km / 2.0

    km_per_deg_lat = 111.32
    km_per_deg_lon = max(1e-6, 111.32 * math.cos(math.radians(center_lat)))

    half_deg_lat = half_km / km_per_deg_lat
    half_deg_lon = half_km / km_per_deg_lon

    min_lat = center_lat - half_deg_lat
    max_lat = center_lat + half_deg_lat
    min_lon = center_lon - half_deg_lon
    max_lon = center_lon + half_deg_lon

    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    return bbox_str


def fetch_snapshot(layer_name, bbox, width_px, height_px, date_str):
    """
    Fetch one layer as RGBA.
    Returns (rgb_arr uint8[H,W,3], alpha uint8[H,W]) or (None, None) on fail.
    """
    params = {
        "REQUEST": "GetSnapshot",
        "TIME": date_str,
        "BBOX": bbox,
        "CRS": "EPSG:4326",
        "LAYERS": layer_name,
        "FORMAT": "image/png",
        "WIDTH": width_px,
        "HEIGHT": height_px,
    }

    r = requests.get(SNAPSHOT_URL, params=params)
    print(f"[{layer_name}] URL:", r.url)
    print(f"[{layer_name}] Status:", r.status_code, r.headers.get("Content-Type"))

    if r.status_code != 200:
        print(f"[{layer_name}] request failed for {date_str}")
        return None, None

    img_rgba = Image.open(BytesIO(r.content)).convert("RGBA")
    arr_rgba = np.array(img_rgba).astype(np.uint8)  # (H,W,4)

    rgb_arr   = arr_rgba[:, :, :3]   # (H,W,3)
    alpha_arr = arr_rgba[:, :, 3]    # (H,W)

    opaque_fraction = (alpha_arr > 0).mean()
    print(f"[{layer_name}] {date_str} opaque_fraction={opaque_fraction:.4f}")

    return rgb_arr, alpha_arr


def merge_fill(base_rgb, base_a, new_rgb, new_a):
    """
    Fill holes in (base_rgb, base_a) using valid pixels from (new_rgb, new_a).

    base_* and new_* are uint8 arrays shaped:
      base_rgb/new_rgb: [H,W,3]
      base_a/new_a:     [H,W]

    Returns updated (base_rgb, base_a).
    """
    # Where base is empty but new is valid, copy new pixel over
    holes = base_a == 0
    fill_mask = (new_a > 0) & holes

    base_rgb[fill_mask] = new_rgb[fill_mask]
    base_a[fill_mask]   = new_a[fill_mask]

    return base_rgb, base_a


def fetch_best_recent(center_lat, center_lon, size_km, px, date_list, layer_name):
    """
    Try multiple dates (newest first). Keep filling missing pixels
    until we either (a) have good coverage or (b) run out of dates.

    Returns final_rgb uint8[H,W,3], final_a uint8[H,W],
    plus the list of dates actually used.
    """
    bbox = make_bbox(center_lat, center_lon, size_km)
    print("BBOX:", bbox)

    final_rgb = None
    final_a   = None
    used_dates = []

    for d in date_list:
        rgb, a = fetch_snapshot(layer_name, bbox, px, px, d)
        if rgb is None or a is None:
            continue

        if final_rgb is None:
            final_rgb = rgb.copy()
            final_a   = a.copy()
        else:
            final_rgb, final_a = merge_fill(final_rgb, final_a, rgb, a)

        used_dates.append(d)

        coverage = (final_a > 0).mean()
        print(f"[ACCUM] after {d}, coverage={coverage:.4f}")

        # if we're >90% filled, that's probably good enough
        if coverage > 0.9:
            break

    return final_rgb, final_a, used_dates


def main():
    final_rgb, final_a, used_dates = fetch_best_recent(
        CENTER_LAT,
        CENTER_LON,
        PATCH_SIZE_KM,
        PX,
        DATE_CANDIDATES,
        LAYER_COMBINED,
    )

    if final_rgb is None or final_a is None:
        raise RuntimeError("No data from any of the requested dates.")

    coverage = (final_a > 0).mean()
    print("Final coverage fraction:", coverage)
    print("Dates actually used (newest first):", used_dates)

    # Anything still uncovered becomes black [0,0,0] which we already have.
    # Save result
    out_png = f"singapore_patch_{used_dates[0]}_{PX}px.png"
    Image.fromarray(final_rgb, mode="RGB").save(out_png)
    print("Saved", out_png)

    # Quick preview (may block on some systems, but useful for dev)
    Image.fromarray(final_rgb, mode="RGB").show()


if __name__ == "__main__":
    main()
