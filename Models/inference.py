"""
inference.py
------------
High-level inference utilities for:
1. Evaluation mode (batch over a labelled test directory, write CSV for metrics)
2. Application mode (classify individual tiles for the live haze pipeline)

Example usage (application / pipeline):
---------------------------------------
from model_utils import load_model, load_checkpoint, preprocess_image
from inference import classify_tensor

model = load_model("resnet50", use_heads=False)
model = load_checkpoint(model, "saved_weights/resnet50_best.pth", device="cuda")

tensor = preprocess_image("tile_r12_c08_20251025.png", arch_name="resnet50", device="cuda")
result = classify_tensor(model, tensor, device="cuda", return_global=True)
# result -> {
#   "class_id": 0,              # global ID (0=smoke, 1=haze, 2=normal)
#   "class_name": "smoke",
#   "confidence": 0.94
# }

Example usage (evaluation CLI from terminal):
---------------------------------------------
python inference.py \\
  --model resnet50 \\
  --weights saved_weights/resnet50_best.pth \\
  --test_dir ./test_data/ \\
  --out_csv evaluation/output/sample_predictions.csv \\
  --use_heads  \\
  --device cuda

This will:
- auto-generate ground truth CSV from folder structure,
- run inference on each image,
- emit predictions vs ground truth in the exact schema that compute_metrics.py expects.
"""

import argparse
from pathlib import Path
import os
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from model_utils import (
    load_model,
    load_checkpoint,
    preprocess_image,
)

# -----------------------------------------------------------------------------
# CLASS MAPPING SYSTEM
# -----------------------------------------------------------------------------
# MODEL_CLASS_NAMES:
# This is the class order used when the model was TRAINED.
# i.e. index in model logits -> semantic class.
#
# Based on our training setup, this is:
#   0 = "haze"
#   1 = "normal"
#   2 = "smoke"
#
MODEL_CLASS_NAMES = ["haze", "normal", "smoke"]

# GLOBAL_CLASS_NAMES:
# This is the canonical order we want everywhere in AI Haze Sentinel:
#   0 = "smoke"
#   1 = "haze"
#   2 = "normal"
#
GLOBAL_CLASS_NAMES = ["smoke", "haze", "normal"]


def _build_model_to_global_map(
    model_names: List[str],
    global_names: List[str],
) -> Dict[int, int]:
    """
    Build a mapping from model-class-id -> global-class-id, using the class names.

    Example:
        model_names  = ["haze", "normal", "smoke"]
        global_names = ["smoke","haze","normal"]

        returns something like:
            0 (haze)   -> 1 (haze)
            1 (normal) -> 2 (normal)
            2 (smoke)  -> 0 (smoke)
    """
    name_to_global = {name: idx for idx, name in enumerate(global_names)}
    out = {}
    for mid, cname in enumerate(model_names):
        if cname not in name_to_global:
            raise ValueError(
                f"Model class '{cname}' not found in GLOBAL_CLASS_NAMES {global_names}"
            )
        out[mid] = name_to_global[cname]
    return out


MODEL_TO_GLOBAL_ID = _build_model_to_global_map(
    MODEL_CLASS_NAMES, GLOBAL_CLASS_NAMES
)

# -----------------------------------------------------------------------------
# IMAGE DISCOVERY AND PREP FOR EVAL MODE
# -----------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> List[Path]:
    """
    Recursively list all supported image files under `root`,
    sorted by filename for stability.
    """
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    return sorted(files, key=lambda p: p.name)


# -----------------------------------------------------------------------------
# CORE INFERENCE PRIMITIVES
# -----------------------------------------------------------------------------

@torch.no_grad()
def classify_tensor(
    model: nn.Module,
    tensor: torch.Tensor,
    device: Union[str, torch.device],
    return_global: bool = True,
) -> Dict[str, Union[int, float, str]]:
    """
    Run one forward pass on a preprocessed [1,C,H,W] tensor and produce a dict
    describing the prediction.

    Args:
        model: torch model already loaded with weights and .eval() moved to device
        tensor: preprocessed batch tensor of shape [1,3,256,256] on the same device
        device: "cpu"/"cuda" or torch.device
        return_global:
            - If True: return IDs/names in GLOBAL space
              (0=smoke,1=haze,2=normal). This is what the pipeline uses.
            - If False: return IDs/names in MODEL space
              (0=haze,1=normal,2=smoke). This is what eval uses.

    Returns:
        {
          "class_id": int,
          "class_name": str,
          "confidence": float
        }
    """
    model.eval()

    # Forward
    out = model(tensor.to(device))
    if isinstance(out, (tuple, list)):
        out = out[0]  # handle models that return (logits, aux)
    probs = torch.softmax(out, dim=1)
    conf, pred_model_id = torch.max(probs, dim=1)

    pred_model_id = int(pred_model_id.item())
    conf = float(conf.item())

    if return_global:
        # Map model-space ID -> global-space ID
        pred_global_id = MODEL_TO_GLOBAL_ID[pred_model_id]
        pred_global_name = GLOBAL_CLASS_NAMES[pred_global_id]
        return {
            "class_id": pred_global_id,
            "class_name": pred_global_name,
            "confidence": conf,
        }
    else:
        # Stay in model's native label space
        pred_model_name = MODEL_CLASS_NAMES[pred_model_id]
        return {
            "class_id": pred_model_id,
            "class_name": pred_model_name,
            "confidence": conf,
        }


@torch.no_grad()
def classify_tile(
    img_path: Union[str, Path],
    arch_name: str,
    weights_path: Union[str, Path],
    device: str = None,
    use_heads: bool = False,
    return_global: bool = True,
) -> Dict[str, Union[int, float, str]]:
    """
    High-level convenience: load model, load weights, preprocess ONE tile,
    run inference, return prediction dict.

    This is the function the pipeline (infer_grid.py) will call per tile.

    Args:
        img_path: path to a single tile image
        arch_name: e.g. "basiccnn", "vgg16", "resnet50"
        weights_path: .pth checkpoint
        device: "cpu" or "cuda". If None -> auto-pick CUDA if available.
        use_heads: whether to build the +heads variant (HAE/Texture fusion model)
        return_global: if True, return class_id in GLOBAL order (0=smoke,1=haze,2=normal)

    Returns:
        same dict as classify_tensor()
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build / load weights
    model = load_model(
        arch_name,
        num_classes=len(MODEL_CLASS_NAMES),
        use_heads=use_heads,
        pretrained=True,
        freeze_features=True,
    )
    model = load_checkpoint(model, str(weights_path), device=device)

    # Preprocess image to tensor [1,3,256,256] on device
    t = preprocess_image(str(img_path), arch_name=arch_name, device=device)

    # Classify
    result = classify_tensor(model, t, device=device, return_global=return_global)
    return result


# -----------------------------------------------------------------------------
# BATCH INFERENCE FOR EVAL MODE
# -----------------------------------------------------------------------------

@torch.no_grad()
def infer_dir(
    model: nn.Module,
    img_dir: Path,
    device: str,
) -> List[Tuple[str, int]]:
    """
    Loop all tiles in img_dir, run inference, and return list of (filename, class_id)
    in MODEL label space.

    Used for benchmark/evaluation mode. This does NOT apply the global remap.
    """
    model.eval()
    rows: List[Tuple[str, int]] = []

    imgs = list_images(img_dir)
    for p in tqdm(imgs, desc="Inference"):
        try:
            # Preprocess using the same utility pipeline uses
            x = preprocess_image(
                img_path=str(p),
                arch_name="resnet50",  # NOTE: we will correct this below
                device=device,
            )
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred_model_id = int(out.softmax(1).argmax(1).item())
            rows.append((p.name, pred_model_id))
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    return rows


# -----------------------------------------------------------------------------
# GROUND TRUTH CSV GENERATION FOR EVAL MODE
# -----------------------------------------------------------------------------

def create_gt_csv(
    test_dir: Path,
    out_csv: Path,
) -> Path:
    """
    Construct ground truth CSV from a test_dir with structure:
        test_dir/
            haze/
                img1.tif
                img2.tif
            normal/
                ...
            smoke/
                ...

    The folder names MUST match MODEL_CLASS_NAMES, because this is still using
    the model's native training order.

    Output CSV columns:
        file_name, actual_class
    Where `actual_class` is the integer class ID in MODEL space.
    """
    rows = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name not in MODEL_CLASS_NAMES:
            print(f"[WARN] Skipping unknown class dir: {class_name}")
            continue

        class_idx = MODEL_CLASS_NAMES.index(class_name)

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
                rows.append({
                    "file_name": img_path.name,
                    "actual_class": class_idx,
                })

    df = pd.DataFrame(rows, columns=["file_name", "actual_class"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Created ground-truth CSV: {out_csv} ({len(df)} rows)")
    return out_csv


def load_gt_map(gt_csv: Path) -> Dict[str, int]:
    """
    Read ground-truth CSV of shape:
        file_name, actual_class

    Returns:
        dict mapping filename -> actual_class (int, MODEL-space ID)
    """
    if gt_csv is None:
        return {}
    df = pd.read_csv(gt_csv)
    required = {"file_name", "actual_class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Ground-truth CSV must have columns: {required}")
    return dict(zip(
        df["file_name"].astype(str),
        df["actual_class"].astype(int),
    ))


# -----------------------------------------------------------------------------
# CLI ENTRY POINT FOR BENCHMARK / OFFLINE EVAL
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run inference and write CSV: file_name,predicted_class,actual_class"
    )

    ap.add_argument("--model", required=True,
                    choices=["basiccnn", "vgg16", "resnet50"],
                    help="Backbone architecture to build")
    ap.add_argument("--weights", required=True,
                    help="Path to .pth weights")
    ap.add_argument("--test_dir", required=True,
                    help="Directory containing class subfolders of test images")
    ap.add_argument("--out_csv", default="evaluation/output/sample_predictions.csv",
                    help="Where to write predictions CSV")
    ap.add_argument("--use_heads", action="store_true",
                    help="If set, use the +heads (HAE+Texture fusion) variant")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device string, e.g. 'cpu' or 'cuda'")
    ap.add_argument("--gt_csv", default=None,
                    help="(Optional) pre-made ground-truth CSV. "
                         "If not provided, we'll auto-generate one from folder names.")

    args = ap.parse_args()

    device = args.device

    # 1. Ground truth CSV (auto-generate if not provided)
    if args.gt_csv is None:
        gt_csv_path = Path("evaluation/output/ground_truth.csv")
        gt_csv_path.parent.mkdir(parents=True, exist_ok=True)
        gt_csv = create_gt_csv(Path(args.test_dir), gt_csv_path)
    else:
        gt_csv = Path(args.gt_csv)

    # 2. Build model + load checkpoint
    model = load_model(
        arch_name=args.model,
        num_classes=len(MODEL_CLASS_NAMES),
        use_heads=args.use_heads,
        pretrained=True,
        freeze_features=True,
    )
    model = load_checkpoint(model, args.weights, device=device)
    model.eval()

    # 3. Load the GT mapping (filename -> class_id in MODEL space)
    gt_map = load_gt_map(gt_csv)

    # 4. Inference loop across test_dir (MODEL label space)
    # NOTE: infer_dir currently assumes arch_name for preprocess.
    # We'll update infer_dir to accept arch_name so we don't hardcode.
    preds = []
    for p in tqdm(list_images(Path(args.test_dir)), desc="Eval Inference"):
        try:
            x = preprocess_image(str(p), arch_name=args.model, device=device)
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            pred_model_id = int(out.softmax(1).argmax(1).item())
            preds.append((p.name, pred_model_id))
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    # 5. Write predictions CSV in EXACT schema compute_metrics.py expects
    out_rows = []
    for fname, pred_model_id in preds:
        actual = int(gt_map.get(fname, -1))  # -1 if ground truth missing
        # IMPORTANT:
        # For evaluation, we DO NOT remap to global order.
        # We keep model-space IDs so metrics line up.
        out_rows.append({
            "file_name": fname,
            "predicted_class": pred_model_id,
            "actual_class": actual,
        })

    out_df = pd.DataFrame(out_rows,
                          columns=["file_name", "predicted_class", "actual_class"])
    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)
    print(f"✅ Saved predictions CSV: {out_csv_path}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
