"""
Haze/Smoke/Normal inference -> organiser-ready CSV/Excel

- Test dir split into: haze/, smoke/, normal/ with 256x256 RGB TIFFs.
- Normalise with ImageNet mean/std, no resize.
- Supports: BasicCNN, VGG16Tuned, ResNetTuned, and optional ModelWithHeads.
- Loads weights in modes: auto/full/head.
- Output columns: file_name,predicted_class,actual_class with mapping
  smoke->0, haze->1, normal->2.

example usage:
python inference.py \
  --model resnet50 \
  --use-heads \
  --weights ./weights/model_with_heads_state.pth \
  --load-mode full \
  --test-dir ./test_images \
  --out ./output/predictions.csv \
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import torchvision
import torchvision.models as tvm
from torchvision import transforms

import pandas as pd


# =================== Models ===================
class BasicCNNModule(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return self.relu(x)

class BasicCNN(nn.Module):
    def __init__(self, in_channels=3, hdim=64, num_classes=3, input_size=256):
        super().__init__()
        self.conv_modules = nn.ModuleList(
            [BasicCNNModule(in_channels=in_channels, out_channels=hdim, kernel_size=3, stride=1, padding=1)]
            + [BasicCNNModule(
                in_channels=hdim * (2**i),
                out_channels=hdim * (2**(i+1)),
                kernel_size=3, stride=1, padding=1) for i in range(4)
            ]
        )
        final_features = int((input_size * input_size) * hdim * (2 ** 4) / (2 ** 10))
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_features=final_features, out_features=num_classes)

    def forward(self, x):
        for module in self.conv_modules:
            x = module(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class VGG16Tuned(nn.Module):
    """VGG16 edited for 3-class haze classification."""
    def __init__(self, num_classes=3, pretrained=True, freeze_features=True):
        super().__init__()
        base = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        if freeze_features:
            for p in base.features.parameters():
                p.requires_grad = False
        in_f = base.classifier[-1].in_features
        base.classifier[-1] = SimpleMLP(input_size=in_f, hidden_size=4096, output_size=num_classes)
        self.trainable_model = base.classifier[-1]
        self.model = base

    def forward(self, x):
        return self.model(x)

    def save_weights(self, path):
        torch.save(self.model.classifier[-1].state_dict(), path)

    def load_weights(self, path, DEVICE="cpu"):
        self.model.classifier[-1].load_state_dict(torch.load(path, map_location=DEVICE))

class ResNetTuned(nn.Module):
    """ResNet50 fine-tuned for 3-class haze/smoke/normal classification."""
    def __init__(self, num_classes=3, pretrained=True, freeze_features=True):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        if freeze_features:
            for p in base.parameters():
                p.requires_grad = False
        in_f = base.fc.in_features
        base.fc = SimpleMLP(input_size=in_f, hidden_size=4096, output_size=num_classes)
        self.trainable_model = base.fc
        self.model = base

    def forward(self, x):
        return self.model(x)

    def save_weights(self, path):
        torch.save(self.model.fc.state_dict(), path)

    def load_weights(self, path, DEVICE="cpu"):
        self.model.fc.load_state_dict(torch.load(path, map_location=DEVICE))

# ----- Attention/heads -----
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class HAEHead(nn.Module):
    """Predicts a 1-channel haze transmission map and compresses to a 128-D vector."""
    def __init__(self, in_ch=3, width=32, out_dim=128):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(in_ch, width),
            ConvBNAct(width, width),
            nn.Conv2d(width, 1, 1)
        )
        self.proj = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, out_dim))
    def forward(self, x):
        t = torch.sigmoid(self.body(x))
        pooled = F.adaptive_avg_pool2d(t, 1).view(x.size(0), 1)
        vec = self.proj(pooled)
        return vec, t

class EdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32)
        sobel_y = sobel_x.t()
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32)
        self.register_buffer("kx", sobel_x[None,None])
        self.register_buffer("ky", sobel_y[None,None])
        self.register_buffer("kl", lap[None,None])
    @torch.no_grad()
    def forward(self, rgb):
        r,g,b = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
        gray = 0.2989*r + 0.587*g + 0.114*b
        ex = F.conv2d(gray, self.kx, padding=1)
        ey = F.conv2d(gray, self.ky, padding=1)
        el = F.conv2d(gray, self.kl, padding=1)
        e = torch.cat([ex,ey,el],1)
        e = e/(e.abs().amax((2,3),keepdim=True)+1e-6)
        return e

class TextureHead(nn.Module):
    """Extracts structural cues from Sobel/Laplacian edge maps -> 128-D vector."""
    def __init__(self, width=32, out_dim=128):
        super().__init__()
        self.edge = EdgeExtractor()
        self.fe = nn.Sequential(
            ConvBNAct(3,width),
            ConvBNAct(width,width*2),
            ConvBNAct(width*2,width*2)
        )
        self.proj = nn.Sequential(nn.Linear(width*2,out_dim), nn.ReLU(True))
    def forward(self, x):
        e = self.edge(x)
        f = self.fe(e)
        g = F.adaptive_avg_pool2d(f,1).flatten(1)
        vec = self.proj(g)
        return vec, e

class FusionClassifier(nn.Module):
    def __init__(self, dim_backbone, dim_hae=128, dim_tex=128, num_classes=3, p_drop=0.2):
        super().__init__()
        dim_in = dim_backbone + (dim_hae if dim_hae else 0) + (dim_tex if dim_tex else 0)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, 512), nn.ReLU(True),
            nn.Dropout(p_drop),
            nn.Linear(512, num_classes)
        )
    def forward(self, f_b, f_h=None, f_t=None):
        feats = [f_b]
        if f_h is not None: feats.append(f_h)
        if f_t is not None: feats.append(f_t)
        return self.mlp(torch.cat(feats, dim=1))

class ModelWithHeads(nn.Module):
    """Wrapper that adds HAE/Texture heads to any backbone producing a feature vector."""
    def __init__(self, backbone, feature_dim, num_classes=3, use_hae=True, use_tex=True):
        super().__init__()
        self.backbone = backbone
        self.use_hae = use_hae
        self.use_tex = use_tex
        self.hae = HAEHead(out_dim=128) if use_hae else None
        self.tex = TextureHead(out_dim=128) if use_tex else None
        self.cls = FusionClassifier(feature_dim,
                                    dim_hae=128 if use_hae else 0,
                                    dim_tex=128 if use_tex else 0,
                                    num_classes=num_classes)

    def forward(self, x):
        f_b = self.backbone(x)
        if f_b.ndim > 2:
            f_b = F.adaptive_avg_pool2d(f_b, 1).flatten(1)
        f_h = f_t = None
        if self.use_hae: f_h, _ = self.hae(x)
        if self.use_tex: f_t, _ = self.tex(x)
        logits = self.cls(f_b, f_h, f_t)
        return logits

def build_backbone(model_name: str, num_classes: int = 3,
                   pretrained: bool = True, freeze_features: bool = True):
    model_name = model_name.lower()
    if model_name == "basiccnn":
        base = BasicCNN(in_channels=3, hdim=64, num_classes=num_classes, input_size=256)
        feature_dim = num_classes  # fallback
        backbone = base
        return base, backbone, feature_dim
    elif model_name == "vgg16":
        base = VGG16Tuned(num_classes=num_classes, pretrained=pretrained, freeze_features=freeze_features)
        backbone = nn.Sequential(
            base.model.features,
            base.model.avgpool,
            nn.Flatten(),
            *list(base.model.classifier.children())[:-1]
        )
        feature_dim = 4096
        return base, backbone, feature_dim
    elif model_name == "resnet50":
        base = ResNetTuned(num_classes=num_classes, pretrained=pretrained, freeze_features=freeze_features)
        body = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        modules = list(body.children())[:-1]  # up to avgpool
        backbone = nn.Sequential(*modules, nn.Flatten())
        feature_dim = body.fc.in_features
        return base, backbone, feature_dim
    else:
        raise ValueError("model_name must be one of: basiccnn, vgg16, resnet50")


# =================== Data & transforms ===================
CLASS_TO_ID_OUT = {"smoke": 0, "haze": 1, "normal": 2}  # organiser mapping

class TestFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.items: List[Tuple[Path, int]] = []
        for cname in ["haze", "smoke", "normal"]:
            cdir = self.root / cname
            if not cdir.exists():
                warnings.warn(f"Missing folder: {cdir}")
                continue
            for p in sorted(cdir.glob("*.tif")):
                self.items.append((p, CLASS_TO_ID_OUT[cname]))
        if len(self.items) == 0:
            raise RuntimeError(f"No .tif files found in {self.root}/(haze|smoke|normal)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, actual = self.items[i]
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            if self.transform:
                im = self.transform(im)
        return im, actual, path.name

def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])


# =================== Loading helpers ===================
def load_weights_auto(model: nn.Module, weights_path: str, head_only: bool | None):
    ckpt = torch.load(weights_path, map_location="cpu")
    if head_only is False or head_only is None:
        try:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if head_only is False or (len(missing) + len(unexpected)) < len(list(model.state_dict().keys())):
                print(f"[load] Loaded as full state_dict (missing={len(missing)}, unexpected={len(unexpected)})")
                return
        except Exception as e:
            if head_only is False:
                raise
            print(f"[load] Full state_dict load failed ({e}); will try head-only...")

    if isinstance(model, VGG16Tuned):
        model.model.classifier[-1].load_state_dict(ckpt)
        print("[load] Loaded VGG16 classifier head only.")
        return
    if isinstance(model, ResNetTuned):
        model.model.fc.load_state_dict(ckpt)
        print("[load] Loaded ResNet50 classifier head only.")
        return
    raise RuntimeError("Head-only load requested, but model doesn't expose a single-classifier head.")


# =================== Inference ===================
@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval().to(device)
    out_rows = []
    for x, actual, names in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        actual = actual.tolist()
        out_rows.extend([(n, p, a) for n, p, a in zip(names, preds, actual)])
    return out_rows

def write_outputs(rows, out_csv: str):
    df = pd.DataFrame(rows, columns=["file_name","predicted_class","actual_class"]).sort_values("file_name")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    try:
        df.to_excel(out_csv.with_suffix(".xlsx"), index=False)
    except Exception as e:
        print(f"[warn] Excel save failed: {e}")
    print(f"Saved: {out_csv}")


# =================== Main ===================
def main():
    ap = argparse.ArgumentParser(description="Haze inference -> CSV/Excel")
    ap.add_argument("--model", choices=["basiccnn","vgg16","resnet50"], required=True)
    ap.add_argument("--use-heads", action="store_true",
                    help="Wrap backbone with HAE/Texture heads and FusionClassifier (requires compatible weights)." )
    ap.add_argument("--weights", required=True, help="Path to weights .pth/.pt")
    ap.add_argument("--load-mode", choices=["auto","full","head"], default="auto")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", default="./output/predictions.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    device = torch.device(args.device)

    base, backbone, feat_dim = build_backbone(args.model, num_classes=3, pretrained=True, freeze_features=False)
    model: nn.Module = base
    if args.use_heads:
        model = ModelWithHeads(backbone, feat_dim, num_classes=3, use_hae=True, use_tex=True)

    head_only = None if args.load_mode == "auto" else (args.load_mode == "head")
    load_weights_auto(model, args.weights, head_only=head_only)

    tfm = get_eval_transform()
    ds = TestFolder(args.test_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type=="cuda"))

    rows = infer(model, loader, device)
    write_outputs(rows, args.out)
    print("Preview:", rows[:5])

if __name__ == "__main__":
    main()
