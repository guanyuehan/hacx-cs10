import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from typing import Union



#------ Basic CNN ---------------------------------------------------------
class BasicCNNModule(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    

class BasicCNN(nn.Module):
    def __init__(self, in_channels=3, hdim=64, num_classes=3, input_size=256):
        super().__init__()
        # 5 layers of conv with downsampling each spatial dimension by factor of 1/2 while
        # doubling the channels
        self.conv_modules = nn.ModuleList(
            [BasicCNNModule(in_channels=in_channels, out_channels=hdim, kernel_size=3, stride=1, padding=1)]
            + [BasicCNNModule(
                in_channels=hdim * (2**i), 
                out_channels=hdim * (2**(i+1)), 
                kernel_size=3, 
                stride=1, 
                padding=1) for i in range(4)
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

# ----- VGG16Tuned ---------------------------------------------------------
class VGG16Tuned(nn.Module):
    """VGG16 edited for 3-class haze classification."""
    def __init__(self, num_classes=3, pretrained=True, freeze_features=True):
        super().__init__()
        base = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        if freeze_features:
            for p in base.features.parameters(): p.requires_grad = False
        # replace classifier output
        in_f = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_f, num_classes)
        self.trainable_model = base.classifier[-1]
        self.model = base
    def forward(self, x):
        return self.model(x)
    
    def save_weights(self, path):
        torch.save(self.model.classifier[-1].state_dict(), path)

    def load_weights(self, path, DEVICE="cpu"):
        self.model.classifier[-1].load_state_dict(torch.load(path, map_location=DEVICE))
    
# ----- ResNetTuned ---------------------------------------------------------
class ResNetTuned(nn.Module):
    """ResNet fine-tuned for 3-class haze/smoke/normal classification."""
    def __init__(self, num_classes=3, pretrained=True, freeze_features=True):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # optionally freeze feature extractor
        if freeze_features:
            for p in base.parameters():
                p.requires_grad = False

        # replace the final fully-connected layer
        in_f = base.fc.in_features
        base.fc = nn.Linear(in_f, num_classes)
        self.trainable_model = base.fc
        self.model = base

    def forward(self, x):
        return self.model(x)
    
    def save_weights(self, path):
        torch.save(self.model.fc.state_dict(), path)

    def load_weights(self, path, DEVICE="cpu"):
        self.model.fc.load_state_dict(torch.load(path, map_location=DEVICE))

"""Attention modules"""
#----- default settings -----------------------------------------------------
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

# ----- Haze-Aware Enhancement (HAE) ----------------------------------------
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
        t = torch.sigmoid(self.body(x))                # transmission map
        pooled = F.adaptive_avg_pool2d(t, 1).view(x.size(0), 1)
        vec = self.proj(pooled)
        return vec, t

# ----- Dual-View Texture ----------------------------------------------------
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
    """Extracts structural cues from Sobel/Laplacian edge maps → 128-D vector."""
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

"""Utils for model wrapping"""
# === Model Builder Wrapper ==================================================

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
    """Generic wrapper that adds HAE/Texture heads to any backbone producing a feature vector."""
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
        if f_b.ndim > 2:  # if feature map, pool and flatten
            f_b = F.adaptive_avg_pool2d(f_b, 1).flatten(1)
        f_h = f_t = None
        if self.use_hae: f_h, _ = self.hae(x)
        if self.use_tex: f_t, _ = self.tex(x)
        logits = self.cls(f_b, f_h, f_t)
        return logits
    
def build_model(model_name="basiccnn", use_heads=False, num_classes=3,
                pretrained=True, freeze_features=True):
    """
    Returns a model by name, optionally with HAE/Texture heads attached.
    model_name: 'basiccnn', 'vgg16'
    """
    model_name = model_name.lower()

    # --- baseline backbones ---
    if model_name == "basiccnn":
        base = BasicCNN(in_channels=3, hdim=64, num_classes=num_classes, input_size=256)
        backbone = BasicCNN(in_channels=3, hdim=64, num_classes=num_classes, input_size=256)
        feature_dim = num_classes  # BasicCNN outputs logits directly

    elif model_name == "vgg16":
        base = VGG16Tuned(num_classes=num_classes, pretrained=pretrained, freeze_features=freeze_features)
        # expose pre-classifier features instead of logits
        backbone = nn.Sequential(base.model.features, base.model.avgpool, nn.Flatten(), *list(base.model.classifier.children())[:-1])
        feature_dim = 4096

    elif model_name == "resnet50":
        base = ResNetTuned(num_classes=num_classes, pretrained=pretrained, freeze_features=freeze_features)
        # expose pre-classifier features instead of logits
        backbone = nn.Sequential(
            *list(base.model.children())[:-1],  # everything except fc
            nn.Flatten()  # flatten 2D features to vector
        )
        feature_dim = base.model.fc.in_features

    else:
        raise ValueError("model_name must be one of: basiccnn, vgg16, resnet50")

    # --- wrap or return plain ---
    if use_heads:
        return ModelWithHeads(backbone, feature_dim, num_classes=num_classes,
                              use_hae=True, use_tex=True)
    else:
        return base


# --- Added helper functions for pipeline and inference ---

ARCH_NORMALISATION = {
    "basiccnn": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    "vgg16":    {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    "resnet50": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
}

INPUT_SIZE = {"basiccnn": (256, 256), "vgg16": (256, 256), "resnet50": (256, 256)}


def _to_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(arch_name: str, num_classes: int = 3, use_heads: bool = False,
               pretrained: bool = True, freeze_features: bool = True):
    model = build_model(arch_name, use_heads, num_classes, pretrained, freeze_features)
    return model


def load_checkpoint(model, weights_path: str, device="cpu", strict: bool = False):
    dev = _to_device(device)
    ckpt = torch.load(weights_path, map_location=dev)

    if hasattr(model, "load_weights"):
        try:
            model.load_weights(weights_path, DEVICE=str(dev))
            model.to(dev).eval()
            return model
        except Exception:
            pass

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=strict)
    return model.to(dev).eval()


def preprocess_image(img_path: str,
                     arch_name: str,
                     device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    Load an image tile from disk, resize to 256x256, convert to float tensor.
    No mean/std normalization is applied here because the model was trained
    directly on raw pixel intensities (besides resizing).

    """
    from PIL import Image
    import numpy as np

    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256), resample=Image.BILINEAR)

    arr = np.array(img)            # [H,W,3], uint8 0-255
    x = torch.tensor(arr).permute(2, 0, 1).float()  # [3,H,W], float32 0-255

    # If training used ToTensor(), enable this:
    # x = x / 255.0

    x = x.unsqueeze(0)  # [1,3,H,W]
    if isinstance(device, str):
        device = torch.device(device)
    x = x.to(device)
    return x


if __name__ == "__main__":
    # plain backbones
    CNNunmasked = build_model("basiccnn", use_heads=False)
    VGG16unmasked = build_model("vgg16", use_heads=False)
    ResNet50unmasked = build_model("resnet50", use_heads=False)

    # augmented versions
    CNNmasked = build_model("basiccnn", use_heads=True)
    VGG16masked = build_model("vgg16", use_heads=True)
    ResNet50masked = build_model("resnet50", use_heads=True)

    # quick test
    for name, model in [("basiccnn+heads", CNNmasked), ("vgg16+heads", VGG16masked), ("resnet50+heads", ResNet50masked)]:
        x = torch.randn(1,3,256,256)
        y = model(x)
        print(f"{name:15s} ->", tuple(y.shape))

    for name, model in [("basiccnn", CNNunmasked), ("vgg16", VGG16unmasked), ("resnet50", ResNet50unmasked)]:
        x = torch.randn(1,3,256,256)
        y = model(x)
        print(f"{name:15s} ->", tuple(y.shape))

