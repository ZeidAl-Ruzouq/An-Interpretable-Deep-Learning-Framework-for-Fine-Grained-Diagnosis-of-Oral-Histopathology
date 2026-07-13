#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Oral Cancer WSI + CA-GNNE Fusion
- Train CNNs, save per-model prediction CSVs
- Build graphs from CNN outputs + costs.json
- Load pretrained CA-GNNE (Models/GNN/best.pt)
- Evaluate all 31 subsets via GNN
- Save Results/test_predictions.csv and Results/subset_report.csv
"""

import os, cv2, csv, torch, json, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# ------------------------
# Config
# ------------------------
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
PATCH_DIR    = Path("wsi_patches")
MODELS_DIR   = Path("Models")
RESULTS_DIR  = Path("Results")
SLIDES_DIR   = Path("Slides")
NORMAL_CSV   = Path("normal_slides.csv")
COSTS_JSON   = Path("outputs_cagne")/"costs.json"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_REGISTRY = {
    "convnext": "convnext_tiny",
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0",
    "deit": "deit_small_patch16_224",
    "regnety": "regnety_008"
}
CLASS_MAP = {"Normal":0, "OSMF":1, "WDOSCC":2, "MDOSCC":3, "PDOSCC":4}
IDX_TO_CLASS = {v:k for k,v in CLASS_MAP.items()}
MAX_PER_CLASS = 2000

# ------------------------
# Logging
# ------------------------
LOG_FILE = RESULTS_DIR / "run.log"

def log_message(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

# ------------------------
# Dataset
# ------------------------
class OrchidDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path,label = self.items[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img,label,path

def build_transforms(size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

# ------------------------
# CNN Training / Eval
# ------------------------
def build_model(model_name,num_classes):
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

# ------------------------
# Class Weights (Balancing)
# ------------------------
def compute_class_weights(loader, num_classes=5, device="cuda"):
    """
    Compute inverse-frequency weights from dataloader labels.
    """
    import numpy as np
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels, _ in loader:   # <-- FIXED unpacking
        for l in labels.numpy():
            counts[l] += 1
    print("[INFO] Class counts:", counts.tolist())
    weights = 1.0 / (counts + 1e-6)  # avoid divide-by-zero
    weights = weights / weights.sum() * num_classes  # normalize
    print("[INFO] Class weights:", weights.tolist())
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ------------------------
# Training Loop (5-class CE)
# ------------------------
def train_model(model,
                tr_loader,
                va_loader,
                device,
                model_label,
                epochs=5,
                patience=5,
                checkpoint_path="outputs/best.pt"):
    from tqdm import tqdm
    import time, sys

    # compute balanced weights from training set (5 classes total)
    class_weights = compute_class_weights(tr_loader, num_classes=len(CLASSES), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1, no_improve, best_state = -1.0, 0, None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for i, (x, labels, _) in enumerate(tqdm(tr_loader, desc=f"{model_label} | Epoch {ep+1}/{epochs}", unit="batch")):
            x, labels = x.to(device), labels.to(device)

            opt.zero_grad()
            out = model(x)                      # shape [B, 5]
            loss = loss_fn(out, labels)         # labels in 0–4
            loss.backward()
            opt.step()
            total_loss += loss.item()

            # live progress every ~1 sec
            if (time.time() - start_time) >= 1.0:
                pct = 100.0 * (i+1) / len(tr_loader)
                sys.stdout.write(
                    f"\r[{model_label}] Epoch {ep+1}/{epochs} "
                    f"{i+1}/{len(tr_loader)} batches ({pct:.1f}%) | "
                    f"Avg Loss {total_loss/(i+1):.4f}"
                )
                sys.stdout.flush()
                start_time = time.time()
        print("")  # newline

        print(f"[{model_label}] Epoch {ep+1}/{epochs} | Avg Loss {total_loss/len(tr_loader):.4f}")

        # validation step
        if va_loader is None:
            continue
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, labels, _ in va_loader:
                x, labels = x.to(device), labels.to(device)
                out = model(x)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = prob.argmax(axis=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred)

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        print(f"[{model_label}] Epoch {ep+1} Val Acc={acc:.4f} | Val F1={f1m:.4f}")

        if f1m > best_f1:
            best_f1, best_state, no_improve = f1m, model.state_dict(), 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # <-- FIX
            torch.save(best_state, checkpoint_path)
            print(f"[INFO] Saved new best model ({model_label}) with F1={best_f1:.4f}")
        else:
            no_improve += 1
            print(f"[INFO] No improvement ({no_improve}/{patience})")
        if no_improve >= patience:
            print(f"[EARLY STOP] {model_label} did not improve for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ------------------------
# Evaluation (with Entropy + Variance)
# ------------------------
def evaluate_model(model, loader, device, out_csv, model_label):
    all_probs, y_true, ids = [], [], []
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["image_id", "true", "pred"] \
                 + [f"p_{IDX_TO_CLASS[i].lower()}" for i in range(len(CLASS_MAP))] \
                 + ["entropy", "variance"]
        w.writerow(header)

        with torch.no_grad():
            for x, y, paths in tqdm(loader, desc=f"Eval/{model_label}"):
                x = x.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                all_probs.append(probs)
                y_true.extend(y.numpy())
                ids.extend(paths)

                for i, p in enumerate(probs):
                    pred = int(np.argmax(p))
                    ent = -np.sum(p * np.log(p + 1e-12))
                    var = np.var(p)
                    row = [Path(paths[i]).name, y[i].item(), pred] \
                          + [f"{p[j]:.5f}" for j in range(len(CLASS_MAP))] \
                          + [f"{ent:.5f}", f"{var:.5f}"]
                    w.writerow(row)

    all_probs = np.vstack(all_probs)
    y_true = np.array(y_true)
    y_pred = all_probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"[{model_label}] Eval Acc={acc:.4f} | F1={f1m:.4f}")
    return acc, f1m


# ------------------------
# Inference-Only Helpers
# ------------------------
def entropy_from_probs(p):
    return -np.sum(p * np.log(p + 1e-12))

def variance_from_probs(p):
    return np.var(p)

def classify_semantic(y_true, y_pred, conf_max, threshold=0.7):
    """
    Categorize prediction quality.
    """
    if y_true == y_pred:
        return "High-Confidence Correct" if conf_max >= threshold else "Low-Confidence Correct"
    else:
        return "High-Confidence Misclassification" if conf_max >= threshold else "Low-Confidence Misclassification"


# ------------------------
# Inference Function
# ------------------------
def run_inference_only(model_key, loader, device, results_dir, class_map, use_fine_tuned=True):
    """
    Run inference for a single model and save CSV with semantic info,
    entropy, variance, and per-class confidences.
    """
    timm_name = MODEL_REGISTRY[model_key]
    model = build_model(timm_name, len(class_map)).to(device)

    # choose checkpoint
    ckpt = MODELS_DIR / model_key / ("fine_tuned.pt" if use_fine_tuned else "best.pt")
    if not ckpt.exists():
        log_message(f"[WARN] No checkpoint found for {model_key} at {ckpt}")
        return None

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    log_message(f"[INFO] Loaded {model_key} weights from {ckpt}")

    out_csv = results_dir / f"{model_key}_preds.csv"
    rows = []

    with torch.no_grad():
        for x, y, paths in tqdm(loader, desc=f"Infer/{model_key}"):
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for i, p in enumerate(probs):
                y_true = int(y[i].item())
                y_pred = int(preds[i])
                conf_max = float(np.max(p))
                sem_info = classify_semantic(y_true, y_pred, conf_max)

                row = {
                    "image_id": Path(paths[i]).name,
                    "model": model_key,
                    "true": y_true,
                    "pred": y_pred,
                    "semantic_info": sem_info,
                    "entropy": float(entropy_from_probs(p)),
                    "variance": float(variance_from_probs(p)),
                }
                # ✅ FIX: correct unpack order
                for cname, j in class_map.items():
                    row[f"p_{cname.lower()}"] = float(p[j])

                rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log_message(f"[INFO] Saved inference results → {out_csv}")
    return out_csv

# ------------------------
# Inference Entry Point
# ------------------------
def inference_all_models(loader, device=DEVICE, use_fine_tuned=True):
    """
    Run inference for all models in MODEL_REGISTRY.
    """
    for mkey in MODEL_REGISTRY.keys():
        run_inference_only(
            model_key=mkey,
            loader=loader,
            device=device,
            results_dir=RESULTS_DIR,
            class_map=CLASS_MAP,
            use_fine_tuned=use_fine_tuned
        )


# ------------------------
# Load CA-GNNE
# ------------------------
def load_cagne(model_path):
    model = CAGNNE().to(DEVICE)
    sd = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)  # allow flexible loading
    model.eval()
    return model

# ------------------------
# === CA-GNNE Imports ===
# ------------------------
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import torch.nn as nn

CLASSES = ["normal", "osmf", "wdoscc", "mdoscc", "pdoscc"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

def entropy_from_probs(p): 
    return -np.sum(p * np.log(p + 1e-12))

def variance_from_probs(p): 
    return np.var(p)

class CAGNNE(nn.Module):
    def __init__(self, in_dim=7, hid=64, drop=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(drop)
        self.multi_head = nn.Linear(hid, len(CLASSES))

    def forward(self, x, edge_index, batch):
        h = self.act(self.conv1(x, edge_index))
        h = self.drop(h)
        h = self.act(self.conv2(h, edge_index))
        h = self.drop(h)
        g = global_mean_pool(h, batch)
        return self.multi_head(g)

BEST_SUBSET = ["convnext", "efficientnet", "deit", "regnety"]

def build_graph(img_id, sub_models, preds_dict, cost_table, y_true):
    rows, softmaxes = [], []
    for m in sub_models:
        r = preds_dict.get((img_id, m))
        if r is None:
            log_message(f"[DEBUG] Missing prediction for {img_id} in model {m}")
            return None

        probs = np.array([r[f"p_{c}"] for c in CLASSES], dtype=np.float32)
        softmaxes.append(probs)
        feat = np.array(
            list(probs) + [entropy_from_probs(probs), variance_from_probs(probs)],
            dtype=np.float32
        )
        rows.append(feat)

    X = np.stack(rows, axis=0)
    n = len(sub_models)

    # fully connected directed edges
    send, recv = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                send.append(i)
                recv.append(j)

    edge_index = torch.tensor([send, recv], dtype=torch.long)
    edge_attr = torch.tensor(
        [[float(np.dot(softmaxes[i], softmaxes[j]))] for i, j in zip(send, recv)],
        dtype=torch.float32
    )

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor([y_true], dtype=torch.long)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    d.image_id = img_id
    d.subset = ",".join(sub_models)
    d.subset_size = n
    d.subset_cost = float(sum(cost_table[m] for m in sub_models))
    return d


from torch_geometric.loader import DataLoader as GeoLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def run_best_subset(batch_size=1024):
    log_message("=== Evaluating Best Subset Only with GNN (Optimized, Batched) ===")
    all_csvs = [RESULTS_DIR / f"{m}_preds.csv" for m in MODEL_REGISTRY.keys()]
    preds = pd.concat([pd.read_csv(c).assign(model=m)
                       for c, m in zip(all_csvs, MODEL_REGISTRY.keys())])

    # ✅ Precompute dictionary for O(1) lookup
    preds_dict = {
        (row["image_id"], row["model"]): row
        for _, row in preds.iterrows()
    }

    # Load cost table
    with open(COSTS_JSON, "r") as f:
        costs = json.load(f)
    cost_table = {k: v["composite_cost"] for k, v in costs.items()}

    # Load CA-GNNE
    gnn_path = MODELS_DIR / "GNN" / "best.pt"
    log_message(f"[INFO] Loading CA-GNNE from {gnn_path}")
    gnn = load_cagne(gnn_path)

    rows, subsets, graphs = [], [], []
    by_img = preds.groupby("image_id")

    # --- Build all graphs first ---
    for idx, (img_id, g) in enumerate(tqdm(by_img, desc="Build Graphs", unit="slide")):
        y_true = int(g["true"].iloc[0])
        d = build_graph(img_id, BEST_SUBSET, preds_dict, cost_table, y_true)
        if d is None:
            log_message(f"[WARN] Skipping {img_id} — missing predictions for one or more models")
            continue
        graphs.append(d)

    # --- Batched inference ---
    loader = GeoLoader(graphs, batch_size=batch_size, shuffle=False)
    gnn.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="BestSubset GNN", unit="batch"):
            out = gnn(batch.x.to(DEVICE), batch.edge_index.to(DEVICE), batch.batch.to(DEVICE))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds_y = probs.argmax(axis=1)

            for img_id, y_true, prob, y_pred in zip(batch.image_id, batch.y.cpu().numpy(), probs, preds_y):
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
                rows.append([
                    img_id,
                    ",".join(BEST_SUBSET),
                    len(BEST_SUBSET),
                    sum(cost_table[m] for m in BEST_SUBSET),
                    y_true,
                    y_pred
                ] + list(prob))

    # --- Compute metrics for best subset ---
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    prec = precision_score(y_true_all, y_pred_all, average="macro")
    rec = recall_score(y_true_all, y_pred_all, average="macro")

    subsets.append({
        "subset": ",".join(BEST_SUBSET),
        "subset_size": len(BEST_SUBSET),
        "subset_cost": sum(cost_table[m] for m in BEST_SUBSET),
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    })

    # --- Add rows for individual base models (straight from their CSVs) ---
    for m in MODEL_REGISTRY.keys():
        base_df = pd.read_csv(RESULTS_DIR / f"{m}_preds.csv")
        y_true_base = base_df["true"].values
        y_pred_base = base_df["pred"].values
        acc = accuracy_score(y_true_base, y_pred_base)
        f1 = f1_score(y_true_base, y_pred_base, average="macro")
        prec = precision_score(y_true_base, y_pred_base, average="macro")
        rec = recall_score(y_true_base, y_pred_base, average="macro")

        subsets.append({
            "subset": m,
            "subset_size": 1,
            "subset_cost": cost_table[m],
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec
        })

    # --- Save outputs ---
    cols = ["image_id","subset","subset_size","subset_cost","true","pred"]+[f"p_{c}" for c in CLASSES]
    test_pred_path = RESULTS_DIR / "test_predictions.csv"
    subset_report_path = RESULTS_DIR / "subset_report.csv"

    pd.DataFrame(rows, columns=cols).to_csv(test_pred_path, index=False)
    pd.DataFrame(subsets).to_csv(subset_report_path, index=False)

    log_message(f"[INFO] Saved test predictions → {test_pred_path}")
    log_message(f"[INFO] Saved subset report → {subset_report_path}")
    log_message("=== RUN END (Best Subset + Base Models Report) ===")

# ------------------------
# Main (Training / Inference Section)
# ------------------------
def main(run_stage="all"):
    """
    run_stage:
      - "all"       → fine-tune CNNs + run GNN inference
      - "gnn"       → skip CNNs, only run GNN inference
      - "inference" → skip training + GNN, only run inference for base models
    """
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    log_message("=== RUN START ===")

    # build test dataset
    tf = build_transforms(224)
    items = []
    for cls, idx in CLASS_MAP.items():
        cdir = PATCH_DIR / cls
        if not cdir.exists():
            continue
        for img in cdir.glob("*.png"):
            items.append((str(img), idx))
    np.random.shuffle(items)
    total = len(items)
    split = int(0.3 * total)
    _, test_items = random_split(items, [split, total - split])
    test_loader = DataLoader(OrchidDataset(test_items, tf), batch_size=32, shuffle=False)

    if run_stage == "all":
        # fine-tune CNNs then run GNN
        log_message("=== Training + GNN ===")
        # ... fine-tuning code ...
        # after training, call run_cagne_inference(...)
    elif run_stage == "gnn":
        log_message("=== GNN Only ===")
        run_best_subset()
    elif run_stage == "inference":
        log_message("=== Base Models Inference Only ===")
        inference_all_models(test_loader, device=DEVICE, use_fine_tuned=True)

    log_message("=== RUN END ===")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in ["all", "gnn", "inference"]:
        raise ValueError(f"Invalid stage '{stage}'. Use: all | gnn | inference")
    main(run_stage=stage)
