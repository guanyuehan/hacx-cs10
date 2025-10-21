import argparse
import csv
import glob
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
import torchvision
import numpy as np

def detect_and_load_model(model_path):
    """
    Load a model from a file. Returns a torch.nn.Module on success.
    Strategy:
     - torch.load(model_path) -> if object has .parameters(), use as model
     - if dict/state_dict: try to detect VGG-like keys and load into vgg16
     - otherwise raise
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    # If it's already a model object
    if hasattr(checkpoint, "parameters"):
        return checkpoint

    # If dict: possibly a state_dict or checkpoint with state_dict inside
    if isinstance(checkpoint, dict):
        # unwrap common container keys
        for k in ("model_state_dict", "state_dict", "model"):
            if k in checkpoint:
                state_dict = checkpoint[k]
                break
        else:
            state_dict = checkpoint

        # If this state_dict looks like VGG (feature/classifier keys) -> load into vgg16
        keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
        key_str = " ".join(keys)[:300].lower()
        if "features.0.weight" in key_str or any(k.startswith("features.") for k in keys):
            m = models.vgg16(pretrained=False)
            m.load_state_dict(state_dict, strict=False)
            return m
        # If the dict is already a state_dict with linear/conv keys but not vgg, try generic load by shape:
        # As a last resort, try to construct a torchvision resnet50 and load (non-strict)
        try:
            m = models.resnet50(pretrained=False)
            m.load_state_dict(state_dict, strict=False)
            return m
        except Exception:
            pass

    raise RuntimeError(f"Could not load model from {model_path}. Saved object not recognized as a model or compatible state_dict.")

def find_weight_file(weights_dir):
    # find common weight extensions
    patterns = ["*.pth", "*.pt", "*.pkl"]
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(weights_dir, pat)))
        if files:
            return files[0]
    return None

def infer_label_from_filename(fname):
    n = fname.lower()
    if "haze" in n:
        return 1  # NOTE: mapping may vary; adjust as needed
    if "smoke" in n:
        return 0
    if "normal" in n:
        return 2
    return ""

def evaluate_model(model_path, weights_dir, test_data_dir, output_csv, gt_csv=None, device="cpu"):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # collect test files
    test_files = sorted(glob.glob(os.path.join(test_data_dir, "*.tif")))
    if not test_files:
        print(f"No .tif files found in {test_data_dir}")
        return

    # determine model file
    model_file = model_path
    if model_file is None:
        model_file = find_weight_file(weights_dir)
        if model_file is None:
            raise SystemExit(f"No model specified and no weight files found in {weights_dir}")

    print(f"Loading model from: {model_file}")
    try:
        model = detect_and_load_model(model_file)
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")

    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(device)
    model.eval()

    # load optional ground truth CSV
    gt_map = {}
    if gt_csv and os.path.exists(gt_csv):
        with open(gt_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gt_map[r["file_name"]] = r.get("actual_class", "")
        print(f"Loaded ground truth for {len(gt_map)} files from {gt_csv}")

    # simple transform: resize/crop to 256 and to tensor (match training in notebook)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    rows = []
    with torch.no_grad():
        for p in test_files:
            fname = os.path.basename(p)
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"Warning: could not open {p}: {e}")
                continue
            inp = transform(img).unsqueeze(0).to(device)
            out = model(inp)
            if isinstance(out, (list, tuple)):
                out = out[0]
            # ensure tensor
            if isinstance(out, torch.Tensor):
                pred_idx = int(torch.argmax(out, dim=1).cpu().item())
            else:
                # fallback if model returns numpy
                pred_idx = int(np.argmax(out, axis=1)[0])

            actual = gt_map.get(fname, infer_label_from_filename(fname))
            rows.append({"file_name": fname, "predicted_class": pred_idx, "actual_class": actual})

    # write CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "predicted_class", "actual_class"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} predictions to {out_path}")