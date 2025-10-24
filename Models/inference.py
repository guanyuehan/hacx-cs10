# ============================= inference.py =============================
import argparse
from pathlib import Path
import os
from typing import List, Tuple

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 🔁 EDIT THIS import to wherever your build_model() lives.
# If build_model is in your notebook, copy it into models.py and import from there.
from model_utils import build_model  # <-- change 'models' to your actual module if different

# Class id order (must match your training + metrics)
# 0=haze, 1=normal, 2=smoke (adjust if your project is different)
CLASS_NAMES = ["haze", "normal", "smoke"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def list_images(root: Path) -> List[Path]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    return sorted(files, key=lambda p: p.name)


@torch.no_grad()
def infer_dir(model: torch.nn.Module, img_dir: Path, device: str) -> List[Tuple[str, int]]:
    model.eval()
    rows: List[Tuple[str, int]] = []
    for p in tqdm(list_images(img_dir), desc="Inference"):
        try:
            img = Image.open(p).convert("RGB")
            x = TEST_TRANSFORM(img).unsqueeze(0).to(device)
            out = model(x)
            if isinstance(out, (tuple, list)):  # handle models that return (logits, aux)
                out = out[0]
            pred = int(out.softmax(1).argmax(1).item())
            rows.append((p.name, pred))
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    return rows

"""test dir 
------class1
        ----imgs...
------class2
        ----imgs...

"""
def create_gt_csv(test_dir: Path, out_csv: Path):
    rows = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in CLASS_NAMES:
            print(f"[WARN] Skipping unknown class dir: {class_name}")
            continue
        class_idx = CLASS_NAMES.index
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in IMG_EXTS and img_path.is_file():
                rows.append({"file_name": img_path.name, "actual_class": class_idx(class_name)})
    df = pd.DataFrame(rows, columns=["file_name", "actual_class"])
    df.to_csv(out_csv, index=False)
    print(f"✅ Created ground-truth CSV: {out_csv} ({len(df)} rows)")
    return out_csv


def load_gt_map(gt_csv: Path):
    """Optional ground-truth CSV with columns: file_name,actual_class (ints)."""
    if gt_csv is None:
        return {}
    df = pd.read_csv(gt_csv)
    required = {"file_name", "actual_class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Ground-truth CSV must have columns: {required}")
    return dict(zip(df["file_name"].astype(str), df["actual_class"].astype(int)))


def main():
    ap = argparse.ArgumentParser(
        description="Run inference and write CSV: file_name,predicted_class,actual_class"
    )
    ap.add_argument("--model", required=True, choices=["basiccnn", "vgg16", "resnet50"])
    ap.add_argument("--weights", required=True, help="Path to .pth weights")
    ap.add_argument("--test_dir", required=True, help="Directory containing test images")
    ap.add_argument("--out_csv", default="evaluation/output/sample_predictions.csv")
    ap.add_argument("--use_heads", action="store_true", help="Use the +heads variant")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gt_csv", default=None, help="Optional ground-truth CSV file")


    args = ap.parse_args()
    # produce ground truth csv from test dir using foldername and structure
    gt_csv = create_gt_csv(Path(args.test_dir), Path(args.gt_csv))

    device = args.device

    # Build model (you said you “just return base instead of backbone” for unmasked)
    model = build_model(model_name=args.model, use_heads=args.use_heads)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Optional GT
    gt_map = load_gt_map(gt_csv)

    # Inference
    preds = infer_dir(model, Path(args.test_dir), device=device)

    # Write EXACT schema required by compute_metrics.py
    out_rows = []
    for fname, pred in preds:
        actual = int(gt_map.get(fname, -1))  # use -1 if ground truth is unknown
        out_rows.append({"file_name": fname, "predicted_class": pred, "actual_class": actual})

    out_df = pd.DataFrame(out_rows, columns=["file_name", "predicted_class", "actual_class"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"✅ Saved: {args.out_csv}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
