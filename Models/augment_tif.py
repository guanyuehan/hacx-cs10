#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

AUG_PREFIXES = ("90_", "180_", "270_")
ROTATIONS = {
    "90_": 90,
    "180_": 180,
    "270_": 270,
}

def is_tif(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".tif", ".tiff"}

def is_already_augmented(p: Path) -> bool:
    # Check if the filename already starts with one of our augmentation prefixes
    return p.name.startswith(AUG_PREFIXES)

def augment_image(img_path: Path):
    try:
        with Image.open(img_path) as im:
            # For 90/180/270, expand=True keeps safe geometry if needed
            for prefix, angle in ROTATIONS.items():
                out_name = prefix + img_path.name
                out_path = img_path.parent / out_name
                if out_path.exists():
                    # Don’t overwrite if already created in a previous run
                    continue
                rotated = im.rotate(angle, expand=True)
                # Save as TIFF, preserving mode; Pillow will pick a sensible default
                rotated.save(out_path, format="TIFF")
    except Exception as e:
        print(f"[WARN] Skipped {img_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Rotate .tif images by 90/180/270 degrees and save with prefixed filenames."
    )
    parser.add_argument("folder", type=Path, help="Path to folder containing .tif images")
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    # Collect only original (non-augmented) .tif files
    all_tifs = [p for p in folder.iterdir() if is_tif(p)]
    originals = [p for p in all_tifs if not is_already_augmented(p)]

    if not originals:
        print("No original .tif images found to augment (or everything is already augmented).")
        return

    print(f"Found {len(originals)} original .tif images. Creating rotated copies...")

    for img_path in originals:
        augment_image(img_path)

    # Recount to summarise
    final_tifs = [p for p in folder.iterdir() if is_tif(p)]
    created = len(final_tifs) - len(all_tifs)
    print(f"Done. Created {created} new images.")
    print(f"Original images: {len(originals)} | Expected new images: {len(originals)*3}")
    print(f"Total .tif files now: {len(final_tifs)}")

    # Sanity hint: if you started with only originals, total should be originals * 4
    if len(final_tifs) == len(originals) * 4:
        print("Sanity check passed: total images are exactly 4× the original count.")
    else:
        print("Note: Total is not exactly 4× (likely because some augmented files already existed).")

if __name__ == "__main__":
    main()

# Usage example:
# python augment_tif.py /path/to/your/folder