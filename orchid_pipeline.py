#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORCHID Histopathology Pipeline
Author: Zeid Al Ruzouq, Allen High School – 2025

Complete pipeline:
- Dataset bootstrap from Zenodo (train/val/test zips)
- Augmentations (Albumentations preferred, torchvision fallback)
- ConvNeXt (default; extensible to ResNet, EfficientNet, DeiT, RegNetY)
- AdamW, ReduceLROnPlateau, early stopping
- Weighted CE loss (optional)
- Eval-only mode to produce logs/metrics + predictions CSV + Grad-CAMs
"""

import os, sys, json, random, csv
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict
from tqdm import tqdm
import cv2
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ------------------------
# Environment Info Logger
# ------------------------
def save_env_info(out_path="env_info.txt"):
    import platform, subprocess
    pkgs = {}
    try:
        import torch, timm, albumentations, numpy, sklearn, cv2
        pkgs["torch"] = torch.__version__
        pkgs["timm"] = timm.__version__
        pkgs["albumentations"] = albumentations.__version__
        pkgs["numpy"] = numpy.__version__
        import sklearn as sk
        pkgs["scikit-learn"] = sk.__version__
        import cv2 as _cv2
        pkgs["opencv"] = _cv2.__version__
    except Exception as e:
        pkgs["error"] = str(e)

    gpu_info = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        gpu_info = result.stdout.strip()
    except Exception:
        gpu_info = "nvidia-smi not available"

    env = {
        "python": platform.python_version(),
        "system": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_info": gpu_info,
        "packages": pkgs
    }

    with open(out_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"[INFO] Environment info saved to {out_path}")

# ------------------------
# Constants & config
# ------------------------
VALID_EXTS = {".png",".jpg",".jpeg",".tif",".tiff"}

ZENODO = {
    "train": "https://zenodo.org/records/12636426/files/train.zip?download=1",
    "val":   "https://zenodo.org/records/12646943/files/val.zip?download=1",
    "test":  "https://zenodo.org/records/12646943/files/test.zip?download=1",
}

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------
# Utils
# ------------------------
def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dst: Path, expected_size: int = None):
    import requests
    import tqdm

    # If file exists and is already complete, skip
    if dst.exists():
        if expected_size is None or dst.stat().st_size >= expected_size:
            print(f"[INFO] {dst.name} already exists and is complete ({dst.stat().st_size} bytes). Skipping download.")
            return
        else:
            print(f"[INFO] {dst.name} exists but is incomplete ({dst.stat().st_size} bytes). Resuming...")

    headers = {}
    mode = "wb"
    pos = 0
    if dst.exists():
        pos = dst.stat().st_size
        headers["Range"] = f"bytes={pos}-"
        mode = "ab"  # append

    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + pos
        with open(dst, mode) as f, tqdm.tqdm(
            total=total, initial=pos, unit="B", unit_scale=True, desc=dst.name
        ) as pbar:
            for chunk in r.iter_content(1 << 20):  # 1 MB
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if dst.stat().st_size == 0:
        raise RuntimeError(f"Download failed, {dst} is empty!")

def unzip_file(zip_path: Path, out_dir: Path):
    import zipfile
    print(f"[INFO] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting", unit="file"):
            zf.extract(member, out_dir)
    print(f"[INFO] Extraction complete for {zip_path.name}")

def discover_classes(root: Path) -> Dict[str, int]:
    # Look one level deeper if only a single folder is found
    folders = [f for f in sorted(root.iterdir()) if f.is_dir()]
    if len(folders) == 1 and folders[0].name.lower() in ("train", "val", "test"):
        folders = [f for f in sorted(folders[0].iterdir()) if f.is_dir()]
    class_to_idx = {f.name: i for i, f in enumerate(folders[:5])}
    idx_to_class = {i: f.name for i, f in enumerate(folders[:5])}
    print("[INFO] Class mapping:", class_to_idx)
    return class_to_idx, idx_to_class

def scan_split(split_dir: Path, class_to_idx: Dict[str, int]) -> List[Tuple[str,int]]:
    items = []
    for root, _, files in os.walk(split_dir):
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext not in VALID_EXTS:
                continue
            for cls_name, cls_idx in class_to_idx.items():
                if cls_name in Path(root).parts:
                    items.append((str(Path(root) / fn), cls_idx))
                    break
    return items

CLASS_TO_IDX = {}
IDX_TO_CLASS = {}

def bootstrap_dataset(root: Path):
    zips = root / "zips"
    safe_mkdir(zips)
    splits = {}

    # expected sizes in bytes (from Zenodo)
    expected_sizes = {
        "train": 39745604871,
        "val":   11269377883,
        "test":   6138851343
    }

    for split in ["train", "val", "test"]:
        url = ZENODO[split]
        dst = zips / f"{split}.zip"
        download_file(url, dst, expected_size=expected_sizes[split])

        out = root / split
        if not out.exists():
            unzip_file(dst, out)

        # Drill down if needed (some zips have an extra "val"/"test" folder inside)
        folders = [f for f in out.iterdir() if f.is_dir()]
        if len(folders) == 1 and folders[0].name.lower() in ("val", "test"):
            out = folders[0]

        global CLASS_TO_IDX, IDX_TO_CLASS
        if split == "train":
            CLASS_TO_IDX, IDX_TO_CLASS = discover_classes(out)

        splits[split] = scan_split(out, CLASS_TO_IDX)
        counts = Counter([IDX_TO_CLASS[y] for _, y in splits[split]])
        print(f"{split} counts:", dict(counts))

        if split in ("train", "val"):
            for cls in CLASS_TO_IDX:
                if counts.get(cls, 0) == 0:
                    raise RuntimeError(f"No {cls} images in {split}")

    # Save mapping for reproducibility
    with open(root / "class_index.json", "w") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)

    return splits

# ------------------------
# Dataset Preparation Helper
# ------------------------
def prepare_orchid_dataset(root: Path = Path("orchid_dataset")):
    """
    Download ORCHID train/val/test splits, unzip, verify sizes,
    discover class mappings (case-insensitive), and return dataset splits.
    """
    print("[INFO] Preparing ORCHID dataset...")
    root.mkdir(parents=True, exist_ok=True)

    expected_sizes = {
        "train": 39745604871,   # ~39.7 GB
        "val":   11269377883,   # ~11.2 GB
        "test":   6138851343    # ~6.1 GB
    }

    zips = root / "zips"
    safe_mkdir(zips)
    splits = {}

    for split in ["train", "val", "test"]:
        url = ZENODO[split]
        dst = zips / f"{split}.zip"

        # Download and check size
        download_file(url, dst, expected_size=expected_sizes[split])
        if dst.stat().st_size < expected_sizes[split] * 0.95 or dst.stat().st_size > expected_sizes[split] * 1.05:
            raise RuntimeError(
                f"[ERROR] {split}.zip size mismatch "
                f"({dst.stat().st_size} bytes, expected ~{expected_sizes[split]})"
            )

        # Unzip if missing
        out = root / split
        if not out.exists():
            unzip_file(dst, out)

        # Drill down if needed (some zips have an extra "val"/"test" folder inside)
        folders = [f for f in out.iterdir() if f.is_dir()]
        if len(folders) == 1 and folders[0].name.lower() in ("val", "test"):
            out = folders[0]

        # Discover classes from train set (auto-handle nested dirs)
        global CLASS_TO_IDX, IDX_TO_CLASS
        if split == "train":
            folders = [f for f in sorted(out.iterdir()) if f.is_dir()]
            while len(folders) == 1:
                inner = [f for f in sorted(folders[0].iterdir()) if f.is_dir()]
                if not inner:
                    break
                folders = inner
            CLASS_TO_IDX = {f.name.lower(): i for i, f in enumerate(folders[:5])}
            IDX_TO_CLASS = {i: f.name for i, f in enumerate(folders[:5])}
            print("[INFO] Class mapping:", CLASS_TO_IDX)

        # Scan files (case-insensitive match)
        items = []
        for root_dir, _, files in os.walk(out):
            root_parts = [p.lower() for p in Path(root_dir).parts]
            for fn in files:
                ext = Path(fn).suffix.lower()
                if ext not in VALID_EXTS:
                    continue
                for cls_name, cls_idx in CLASS_TO_IDX.items():
                    if cls_name in root_parts:
                        items.append((str(Path(root_dir) / fn), cls_idx))
                        break
        splits[split] = items

        counts = Counter([IDX_TO_CLASS[y] for _, y in splits[split]])
        print(f"[INFO] {split} counts:", dict(counts))

        # Sanity check: all classes present in train/val
        if split in ("train", "val"):
            for cls in CLASS_TO_IDX:
                if counts.get(IDX_TO_CLASS[CLASS_TO_IDX[cls]], 0) == 0:
                    raise RuntimeError(f"[ERROR] No {cls} images in {split} set!")

    # Save mapping for reproducibility
    with open(root / "class_index.json", "w") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)

    print("[INFO] ORCHID dataset ready.")
    return splits

# ------------------------
# Dataset
# ------------------------
class OrchidDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        import cv2
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, label, path

# ------------------------
# Transforms
# ------------------------
def build_transforms(size=224):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_tf = A.Compose([
        A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0)),
        A.Rotate(limit=360),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3, p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.5),
        A.ColorJitter(0.2, 0.2, 0.15, 0.05, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    eval_tf = A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return {
        "train": lambda im: train_tf(image=im)["image"],
        "val":   lambda im: eval_tf(image=im)["image"],
        "test":  lambda im: eval_tf(image=im)["image"]
    }

# ------------------------
# Model
# ------------------------
def build_model(name="convnext", dropout=0.0, num_classes=5):
    import timm
    reg = {
        "convnext": "convnext_tiny",
        "resnet": "resnet50",
        "efficientnet": "efficientnet_b0",
        "deit": "deit_small_patch16_224",
        "regnety": "regnety_008"
    }
    timm_name = reg.get(name.lower(), None)
    if timm_name is None:
        raise ValueError(f"Unsupported model '{name}'. Choose from {list(reg.keys())}")
    model = timm.create_model(
        timm_name, pretrained=True, num_classes=num_classes, drop_rate=dropout
    )
    return model, timm_name

# ------------------------
# Training + Evaluation helpers
# ------------------------
def semantic_label(correct, maxp):
    if correct:
        if maxp>=0.85: return "High-Confidence Correct"
        if maxp>=0.55: return "Low-Confidence Correct"
        return "Ambiguous Prediction"
    else:
        if maxp>=0.85: return "High-Confidence Misclassification"
        if maxp>=0.55: return "Low-Confidence Misclassification"
        return "Ambiguous Prediction"

def entropy(probs: np.ndarray) -> float:
    return -np.sum(probs * np.log(probs + 1e-12))

def mc_dropout_predictions(model, x, device, passes=20):
    """
    Run Monte Carlo dropout without breaking BatchNorm.
    Keeps model in eval mode, but enables dropout layers only.
    Returns array of shape [passes, batch, num_classes].
    """
    # Ensure eval mode
    model.eval()
    # Enable dropout layers only
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(passes):
            out = torch.softmax(model(x.to(device)), dim=1)
            preds.append(out.cpu().numpy())
    return np.stack(preds, axis=0)

def evaluate(model, loader, device):
    model.eval(); loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
    loss,acc,n=0,0,0
    with torch.no_grad():
        for x,y,_ in loader:
            x,y=x.to(device),y.to(device)
            out=model(x); loss+=loss_fn(out,y).item()
            acc+=(out.argmax(1)==y).sum().item(); n+=x.size(0)
    return loss/n, acc/n

def evaluate_and_write_csv(model, loader, device, out_csv, model_label):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build header dynamically from IDX_TO_CLASS
    header = ["Title","model","true_label","predicted_label","semantic_info",
              "entropy","variance"] + [f"confidence_{IDX_TO_CLASS[i]}" for i in range(len(IDX_TO_CLASS))]

    y_true_all, y_pred_all, y_prob_all = [], [], []

    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        rows=[]
        for x,y,paths in tqdm(loader, desc="Evaluating/Test CSV", unit="batch"):
            x=x.to(device)
            mc_probs = mc_dropout_predictions(model, x, device, passes=20)  # [P, B, C]
            mean_probs = mc_probs.mean(axis=0)                               # [B, C]
            var_map = mc_probs.var(axis=0)                                   # [B, C]
            var_per_sample = var_map.mean(axis=1)                            # [B]

            preds = mean_probs.argmax(1)
            for i,path in enumerate(paths):
                true_lbl=int(y[i])
                probs = mean_probs[i]

                # Normalize to sum=1
                s = probs.sum()
                probs = probs / s if s > 0 else np.ones_like(probs)/len(probs)

                pred_lbl = int(np.argmax(probs))
                maxp = float(probs.max())
                sem = semantic_label(true_lbl==pred_lbl, maxp)

                # Build row dynamically
                row=[Path(path).name, model_label,
                     IDX_TO_CLASS[true_lbl], IDX_TO_CLASS[pred_lbl], sem,
                     f"{entropy(probs):.5f}", f"{float(var_per_sample[i]):.5f}"] \
                    + [f"{float(probs[j]):.5f}" for j in range(len(IDX_TO_CLASS))]

                w.writerow(row)
                rows.append((path,true_lbl,pred_lbl,maxp,sem))

                # Accumulate metrics
                y_true_all.append(true_lbl)
                y_pred_all.append(pred_lbl)
                y_prob_all.append(probs.tolist())

    print(f"[INFO] Test predictions written to {out_csv}")
    return rows, np.array(y_true_all), np.array(y_pred_all), np.array(y_prob_all)


# ------------------------
# Metrics & Confusion Matrix
# ------------------------
def save_metrics_and_confusion(y_true, y_pred, y_prob, out_dir: Path):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=[IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))],
        output_dict=True,
        digits=3
    )
    acc = float((y_true == y_pred).mean())
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro"))
    except Exception:
        macro_auc = None
    try:
        ll = float(log_loss(y_true, y_prob))
    except Exception:
        ll = None

    metrics = {
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_auc": macro_auc,
        "log_loss": ll,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
        },
    }

    # Print results live
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.3f}")
    if macro_auc is not None:
        print(f"Macro AUC: {macro_auc:.3f}")
    if ll is not None:
        print(f"Log Loss: {ll:.3f}")
    print("\n--- Per-Class Report ---")
    for cls in metrics["per_class"]:
        p = metrics["per_class"][cls]
        print(f"{cls:10s} | "
              f"Prec: {p['precision']:.3f} "
              f"Rec: {p['recall']:.3f} "
              f"F1: {p['f1']:.3f} "
              f"(Support: {int(p['support'])})")

    # Save JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved {out_dir/'metrics.json'}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(IDX_TO_CLASS))))
    with open(out_dir / "confusion_matrix.json", "w") as f:
        json.dump(
            {"labels": [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))],
             "matrix": cm.tolist()},
            f,
            indent=2,
        )
    print("[INFO] Confusion matrix:")
    print(cm)

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))],
        yticklabels=[IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved {out_dir/'confusion_matrix.png'}")

# ------------------------
# Grad-CAM or Attention Rollout (CNN vs DeiT)
# ------------------------
def generate_gradcam_examples(model, device, rows, out_dir):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import torch, cv2, numpy as np
    from pathlib import Path

    safe_mkdir(out_dir)
    model.eval()

    # Pick 1 sample per semantic label
    categories = {}
    for path, true_idx, pred_idx, maxp, sem in rows:
        if sem not in categories:
            categories[sem] = (path, true_idx, pred_idx, maxp)
        if len(categories) == 5:
            break

    # ---- Helper: DeiT / ViT attention rollout ----
    def attention_rollout(x):
        attn_weights = []
        handles = []
        if not hasattr(model, "blocks"):
            return None
        for blk in model.blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "attn_drop"):
                handles.append(
                    blk.attn.attn_drop.register_forward_hook(
                        lambda m, i, o: attn_weights.append(o)
                    )
                )
        with torch.no_grad():
            _ = model(x.to(device))
        for h in handles:
            h.remove()

        if not attn_weights:
            print("[WARN] No attention weights captured. Skipping attention rollout.")
            return None

        attn = torch.stack(attn_weights)       # [L, B, H, T, T]
        attn = attn.mean(dim=2)                # avg heads → [L, B, T, T]
        attn = attn.squeeze(1)
        eye = torch.eye(attn.size(-1)).to(device)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)

        rollout = attn[0]
        for i in range(1, attn.size(0)):
            rollout = rollout @ attn[i]
        cls_attn = rollout[0, 1:]              # CLS → patches
        return cls_attn

    for sem, (path, true_idx, pred_idx, maxp) in tqdm(
        categories.items(), desc="Generating Explanations", unit="case"
    ):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        tf = build_transforms()["test"]
        x = tf(img).unsqueeze(0).to(device)
        rgb_float = img.astype(np.float32) / 255.0

        overlay, heatmap = None, None

        # Transformer attention rollout
        if any(k in model.__class__.__name__.lower() for k in ["deit", "vit", "swin"]):
            attn_map = attention_rollout(x)
            if attn_map is not None:
                side = int(attn_map.size(0) ** 0.5)
                attn_map = attn_map.reshape(side, side).cpu().numpy()
                attn_map = cv2.resize(attn_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)
                attn_map = cv2.GaussianBlur(attn_map, (7, 7), sigmaX=3)
                overlay = show_cam_on_image(rgb_float, attn_map, use_rgb=True, image_weight=0.6)
                heatmap = attn_map

        # Fallback / CNNs → Grad-CAM
        if overlay is None:
            target_layer = None
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    target_layer = m
            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=x)[0]
            grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() + 1e-8)
            overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=0.6)
            heatmap = grayscale_cam

        # ---- Save figure: Input | Heatmap (with colorbar) | Overlay ----
        fname = out_dir / f"explain_{sem.replace(' ', '_').lower()}.png"
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img)
        axes[0].set_title(f"Input\nTrue={IDX_TO_CLASS[true_idx]}")
        axes[0].axis("off")

        im = axes[1].imshow(heatmap, cmap="plasma", vmin=0, vmax=1)
        axes[1].set_title("Heatmap")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nPred={IDX_TO_CLASS[pred_idx]} Conf={maxp:.2f}")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved {fname}")

# ------------------------
# Training loop 
# ------------------------
def train_loop(cfg, splits, tuning=False, trial=None):
    tf = build_transforms(size=cfg["img_size"])
    train_ds, val_ds, test_ds = [OrchidDataset(splits[s], tf[s]) for s in ("train","val","test")]
    train_loader = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["bs"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg["bs"], shuffle=False, num_workers=2, pin_memory=True)

    model, _ = build_model(cfg["model"], dropout=cfg["dropout"])
    model = model.to(cfg["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # AMP scaler
    scaler = GradScaler(enabled=cfg.get("amp", False))
    if cfg.get("amp", False):
        print("[INFO] AMP enabled: mixed precision training")

    # ------------------------
    # Trial-specific ckpts/logs if Optuna
    # ------------------------
    if trial is not None:
        trial_id = trial.number
        ckpt_dir = Path(f"checkpoints/{cfg['model']}/trial_{trial_id}")
        log_path = Path(f"logs/{cfg['model']}_trial{trial_id}.csv")
    else:
        # 🔹 Normal training: put last.pt in workspace, logs in logs/
        ckpt_dir = Path(".")
        log_path = Path("logs/train_log.csv")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Resume support
    # ------------------------
    def try_resume():
        best_acc, start_epoch = -1, 0
        resume_path = ckpt_dir / "last.pt"
        if resume_path.exists():
            print(f"[INFO] Resuming from checkpoint {resume_path}")
            checkpoint = torch.load(resume_path, map_location=cfg["device"])
            model.load_state_dict(checkpoint["model_state"])
            opt.load_state_dict(checkpoint["optimizer_state"])
            sched.load_state_dict(checkpoint["scheduler_state"])
            best_acc = checkpoint.get("best_acc", -1)
            start_epoch = checkpoint.get("epoch", 0)

            # Sync with log
            if log_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(log_path)
                    if not df.empty:
                        last_logged_epoch = int(df["epoch"].iloc[-1])
                        start_epoch = max(start_epoch, last_logged_epoch)
                        print(f"[INFO] Log shows epoch {last_logged_epoch}, resuming from {start_epoch+1}")
                except Exception as e:
                    print(f"[WARN] Could not parse {log_path.name}: {e}")
        return best_acc, start_epoch

    best_acc, start_epoch = try_resume()

    # ---- Early stopping patience ----
    if tuning or trial is not None:
        patience = 5
    else:
        patience = 10
    no_improve = 0

    # CSV logger
    log_file = open(log_path, "a" if start_epoch > 0 else "w", newline="")
    log_writer = csv.writer(log_file)
    if start_epoch == 0:
        log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # ------------------------
    # Training loop
    # ------------------------
    epoch = start_epoch
    while epoch < cfg["epochs"]:
        try:
            model.train()
            run_loss, run_acc = 0, 0
            for x, y, _ in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{cfg['epochs']}] Training", leave=False):
                x, y = x.to(cfg["device"]), y.to(cfg["device"])
                opt.zero_grad()
                with autocast(enabled=cfg.get("amp", False)):
                    out = model(x)
                    loss = loss_fn(out, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item() * x.size(0)
                run_acc  += (out.argmax(1) == y).sum().item()

            train_loss = run_loss / len(train_ds)
            train_acc  = run_acc / len(train_ds)
            val_loss, val_acc = evaluate(model, val_loader, cfg["device"])
            sched.step(val_loss)

            print(f"[Epoch {epoch+1}/{cfg['epochs']}] "
                  f"train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, "
                  f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")

            # Log row
            log_writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])
            log_file.flush()

            # Optuna pruning
            if trial is not None:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    print(f"[INFO] Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()

            # Save checkpoints
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scheduler_state": sched.state_dict(),
                "best_acc": best_acc,
            }, ckpt_dir / "last.pt")

            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                print(f"[INFO] New best model saved with val_acc={val_acc:.3f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("[INFO] Early stopping triggered.")
                    break

            epoch += 1

        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side" in str(e):
                print(f"[WARN] GPU issue detected: {e}. Restarting from checkpoint...")
                best_acc, start_epoch = try_resume()
                epoch = start_epoch
                continue
            else:
                raise e

    log_file.close()

    if tuning:
        return best_acc

    # Final evaluation
    print("[INFO] Loading best model for evaluation...")
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=cfg["device"]))
    rows, y_true, y_pred, y_prob = evaluate_and_write_csv(
        model, test_loader, cfg["device"],
        Path("predictions/test_predictions.csv"), cfg["model"]
    )
    save_metrics_and_confusion(y_true, y_pred, y_prob, Path("logs"))
    generate_gradcam_examples(model, cfg["device"], rows, Path("gradcam_examples"))
    print("[INFO] Training complete. Results saved.")
    return best_acc

# ------------------------
# Eval-only runner
# ------------------------
def eval_only_runner(cfg):
    # dataset
    splits = bootstrap_dataset(Path("orchid_dataset"))
    tf = build_transforms(size=cfg["img_size"])
    split = cfg.get("split", "test")
    ds = OrchidDataset(splits[split], tf[split])
    loader = DataLoader(ds, batch_size=cfg["bs"], shuffle=False,
                        num_workers=2, pin_memory=True)

    # model
    model, _ = build_model(cfg["model"], dropout=cfg["dropout"])
    device = cfg["device"]
    model = model.to(device).eval()

    weights_path = cfg.get("weights", "best.pt")
    print(f"[INFO] Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # eval
    out_csv = Path(f"predictions/{split}_{cfg['model']}.csv")
    rows, y_true, y_pred, y_prob = evaluate_and_write_csv(
        model, loader, device, out_csv, cfg["model"]
    )
    save_metrics_and_confusion(y_true, y_pred, y_prob, Path("logs")/split)
    generate_gradcam_examples(model, device, rows, Path("gradcam_examples")/split)
    print(f"[INFO] Eval-only complete for {split}. Results in predictions/, logs/, gradcam_examples/")

# ------------------------
# CLI + Optuna
# ------------------------
if __name__ == "__main__":
    import argparse, optuna, pynvml, time, random
    from optuna.exceptions import TrialPruned

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convnext")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (last.pt) to resume training from")

    # Eval-only
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--weights", type=str, default="best.pt")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Which split to evaluate when using --eval-only")

    # Optuna
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--study-name", type=str, default="orchid")
    args = p.parse_args()

    # ------------------------
    # GPU memory helper
    # ------------------------
    def get_free_gpus(min_free_gb=10):
        pynvml.nvmlInit()
        free = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem.free / (1024**3)
            if free_gb >= min_free_gb:
                free.append(i)
        pynvml.nvmlShutdown()
        return free

    if args.eval_only:
        eval_only_runner(vars(args))

    elif args.tune:
        def objective(trial):
            # Wait until some GPU has enough free memory
            gpu_id = None
            while gpu_id is None:
                free_gpus = get_free_gpus(min_free_gb=10)  # adjust threshold for your model
                if free_gpus:
                    gpu_id = random.choice(free_gpus)
                else:
                    print(f"[Trial {trial.number}] No GPU free, sleeping 60s...")
                    time.sleep(60)

            device = f"cuda:{gpu_id}"
            print(f"[Trial {trial.number}] Assigned to GPU {gpu_id} ({device})")

            # Trial configuration
            cfg = vars(args).copy()
            cfg["device"] = device
            cfg["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            cfg["bs"] = trial.suggest_categorical("bs", [8, 16])
            cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
            cfg["resume"] = None

            splits = bootstrap_dataset(Path("orchid_dataset"))
            acc = train_loop(cfg, splits, tuning=True, trial=trial)
            return acc

        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=2,
            interval_steps=1
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            pruner=pruner
        )

        study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    else:
        if args.resume:
            print(f"[INFO] Resuming training from {args.resume} ...")
            splits = bootstrap_dataset(Path("orchid_dataset"))
            train_loop(vars(args), splits, tuning=False)
        else:
            print("[INFO] Running standard training from scratch...")
            for ckpt in ["last.pt", "best.pt"]:
                if os.path.exists(ckpt):
                    os.remove(ckpt)
                    print(f"[INFO] Removed old checkpoint: {ckpt}")
            splits = bootstrap_dataset(Path("orchid_dataset"))
            train_loop(vars(args), splits, tuning=False)

    save_env_info("env_info.txt")
