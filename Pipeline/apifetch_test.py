import math
import requests
from PIL import Image
from io import BytesIO
import numpy as np

SNAPSHOT_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"

# ----- CONFIG -----
DATE = "2023-10-01"      # date to sample
CENTER_LAT = 1.35        # location of singapore SUPPOSEDLY BUT I CANT GET THIS SHIT TO WORK UNLESS I SET IT ELSEWHERE THIS IS PROBABLY CUZ OF THE STUPID SWASH PATTERN FROM SATELITE ORBITALS
CENTER_LON = 20.82
PATCH_SIZE_KM = 256      # want ~256 km x 256 km area
PX = 256                 # output tile size: 256x256 -> ~1 km/pixel

LAYER_TERRA = "MODIS_Terra_CorrectedReflectance_TrueColor"
LAYER_AQUA  = "MODIS_Aqua_CorrectedReflectance_TrueColor"

def make_bbox(center_lat, center_lon, size_km):
    """
    Build a bbox [minLon,minLat,maxLon,maxLat] that spans ~size_km x size_km
    around a center point, accounting for lon scale at that latitude.
    """
    half_km = size_km / 2.0

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))

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
    Fetch one layer (Terra or Aqua) as RGBA.
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
        return None, None

    img_rgba = Image.open(BytesIO(r.content)).convert("RGBA")
    arr_rgba = np.array(img_rgba).astype(np.uint8)  # (H,W,4)

    rgb_arr   = arr_rgba[:, :, :3]   # (H,W,3)
    alpha_arr = arr_rgba[:, :, 3]    # (H,W)

    opaque_fraction = (alpha_arr > 0).mean()
    print(f"[{layer_name}] opaque_fraction={opaque_fraction:.4f}")

    return rgb_arr, alpha_arr

def composite_terra_aqua(terra_rgb, terra_a, aqua_rgb, aqua_a):
    """
    Pixelwise merge to reduce swath seam:
    - If Terra alpha > 0, use Terra pixel.
    - Else if Aqua alpha > 0, use Aqua pixel.
    - Else fill [0,0,0].
    Returns final_rgb uint8[H,W,3].
    """
    h, w, _ = terra_rgb.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    terra_valid = terra_a > 0
    aqua_valid  = aqua_a > 0

    # Use Terra where valid
    out[terra_valid] = terra_rgb[terra_valid]

    # Where Terra not valid but Aqua is valid, fill with Aqua
    fill_from_aqua = (~terra_valid) & (aqua_valid)
    out[fill_from_aqua] = aqua_rgb[fill_from_aqua]

    # Else remains black (0,0,0)
    return out

# 1. Build bbox for ~256km x 256km, ~1 km/px at 256x256
bbox = make_bbox(CENTER_LAT, CENTER_LON, PATCH_SIZE_KM)
print("BBOX:", bbox)

# 2. Fetch Terra + Aqua
terra_rgb, terra_a = fetch_snapshot(LAYER_TERRA, bbox, PX, PX, DATE)
aqua_rgb,  aqua_a  = fetch_snapshot(LAYER_AQUA,  bbox, PX, PX, DATE)

if terra_rgb is None and aqua_rgb is None:
    raise RuntimeError("Both Terra and Aqua requests failed.")

# Handle missing one side gracefully
if terra_rgb is None:
    print("Terra failed, using Aqua only.")
    final_rgb = aqua_rgb if aqua_rgb is not None else None
elif aqua_rgb is None:
    print("Aqua failed, using Terra only.")
    final_rgb = terra_rgb
else:
    # 3. Composite to hide the swath/orbit seam
    final_rgb = composite_terra_aqua(terra_rgb, terra_a, aqua_rgb, aqua_a)

print("final_rgb shape:", final_rgb.shape, "dtype:", final_rgb.dtype)
print("final pixel sum:", int(final_rgb.sum()))

# 4. Save just the merged RGB as a normal PNG (no alpha channel)
final_img = Image.fromarray(final_rgb, mode="RGB")
out_png = f"singapore{DATE}_1km_256_merged.png"
# final_img.save(out_png)
# print("Saved", out_png)

# Optional quick look
final_img.show()
