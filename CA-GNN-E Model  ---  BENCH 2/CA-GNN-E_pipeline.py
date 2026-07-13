#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA-GNNE: Cost-Aware Graph Neural Network Ensemble
Author: Zeid Al Ruzouq, Allen High School - 2025

Fixes in this version:
- Real, empirical cost measurement per base model:
  * Latency / runtime per inference (avg over measured passes)
  * Peak GPU memory usage
  * Energy / power (NVML average; gracefully skips if NVML unavailable)
  * (Optional) Model weight size (bytes)
  All costs are normalized and combined with user-configurable weights.

- Optuna integration:
  * Each trial saves its best checkpoint to <out_dir>/optuna/trial_<n>_best.pt
  * After tuning, we auto-load best trial hyperparams AND its best.pt
    and continue training on train+val. Final checkpoint saved to <out_dir>/final.pt.

Pipeline:
- Read per-model CSVs (train/val/test predictions)
- Build graphs (nodes=models, features=softmax+entropy+variance)
- Enumerate all non-empty model subsets
- GraphSAGE meta-classifier (binary + multi-class heads)
- Cost-aware training (penalize expensive subsets with measured costs)
- Hyperparameter tuning via Optuna
- Retrain on train+val starting from best.pt of best trial
- Inference on test → predictions CSV
- Post-analysis → per-subset metrics, cost tiers, Pareto plots, text summary
"""

import os, json, itertools, random, argparse, time, tempfile
from typing import List, Dict, Tuple

import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import optuna

# ------------------------
# Config / constants
# ------------------------
CLASSES = ["normal", "osmf", "wdoscc", "mdoscc", "pdoscc"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NODE_FEAT_DIM = 7

# Default base models; can override via --models
DEFAULT_MODELS = ["convnext", "resnet", "efficientnet", "deit", "regnety"]

# Mapping to timm names for instrumentation
TIMM_REG = {
    "convnext":     "convnext_tiny",
    "resnet":       "resnet50",
    "efficientnet": "efficientnet_b0",
    "deit":         "deit_small_patch16_224",
    "regnety":      "regnety_008",
}

# ------------------------
# Utils
# ------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def all_nonempty_subsets(items: List[str]) -> List[Tuple[str, ...]]:
    out = []
    for r in range(1, len(items) + 1):
        for comb in itertools.combinations(sorted(items), r):
            out.append(comb)
    return out

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float((a * b).sum())
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return num / den

def one_vs_rest_binary(y_multi: torch.Tensor) -> torch.Tensor:
    return (y_multi != CLASS_TO_ID["normal"]).long()

# ------------------------
# CSV I/O
# ------------------------
def read_split_csvs(csv_paths: List[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    # normalize headers → lowercase, strip spaces
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def group_by_image(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # prefer "title", else fallback to "image_id"
    if "title" in df.columns:
        key = "title"
    elif "image_id" in df.columns:
        key = "image_id"
    else:
        raise ValueError("CSV must contain either 'title' or 'image_id' column")
    return {img_id: g.copy() for img_id, g in df.groupby(key)}

# ------------------------
# Cost instrumentation
# ------------------------
def _current_device_index() -> int:
    if not torch.cuda.is_available():
        return -1
    return torch.cuda.current_device()

def _init_nvml():
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        return nvml
    except Exception:
        return None

def _nvml_device_handle(nvml):
    if nvml is None:
        return None
    try:
        dev_idx = _current_device_index()
        if dev_idx < 0:
            return None
        # Map CUDA index directly to NVML index when possible
        return nvml.nvmlDeviceGetHandleByIndex(dev_idx)
    except Exception:
        return None

@torch.no_grad()
def measure_model_cost(model_key: str,
                       img_size: int = 224,
                       device: str = "cuda" if torch.cuda.is_available() else "cpu",
                       batch_size: int = 1,
                       warmup: int = 10,
                       passes: int = 30) -> Dict[str, float]:
    """
    Build the timm model by key, run several forward passes on a dummy tensor,
    return measured avg latency (ms), peak memory (bytes), avg power (watts),
    and weight size (bytes).
    """
    import timm

    if model_key not in TIMM_REG:
        raise ValueError(f"Unknown model key '{model_key}'. Supported: {list(TIMM_REG.keys())}")

    timm_name = TIMM_REG[model_key]
    model = timm.create_model(timm_name, pretrained=False, num_classes=len(CLASSES))
    model.eval().to(device)

    # Exact weight size in bytes (sum of param tensors)
    weight_size_bytes = 0
    for p in model.parameters():
        weight_size_bytes += p.nelement() * p.element_size()

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # NVML
    nvml = _init_nvml()
    h = _nvml_device_handle(nvml)

    def read_power_w():
        if nvml is None or h is None:
            return None
        try:
            # returns milliwatts
            mw = nvml.nvmlDeviceGetPowerUsage(h)
            return mw / 1000.0
        except Exception:
            return None

    # Warmup
    for _ in range(warmup):
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # Peak memory
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # Measure
    latencies = []
    power_samples = []
    for _ in range(passes):
        p_before = read_power_w()
        t0 = time.perf_counter()
        _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        p_after = read_power_w()

        latencies.append((t1 - t0) * 1000.0)
        # crude average for this pass if both available
        if p_before is not None and p_after is not None:
            power_samples.append(0.5 * (p_before + p_after))

    avg_latency_ms = float(np.mean(latencies))
    peak_mem_bytes = None
    if device.startswith("cuda"):
        peak_mem_bytes = float(torch.cuda.max_memory_allocated())

    avg_power_w = float(np.mean(power_samples)) if len(power_samples) else None

    # Cleanup
    del model, x
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "latency_ms": avg_latency_ms,
        "peak_mem_bytes": peak_mem_bytes,   # may be None on CPU
        "avg_power_w": avg_power_w,         # may be None if NVML missing
        "weight_size_bytes": float(weight_size_bytes)
    }

def normalize_and_composite(costs: Dict[str, Dict[str, float]],
                            weight_latency=1.0,
                            weight_memory=1.0,
                            weight_power=1.0,
                            weight_weights=0.5) -> Dict[str, Dict[str, float]]:
    """
    Min-max normalize available metrics across models and compute a composite.
    Metrics absent (None) are skipped and their weight redistributed proportionally.
    """
    # Gather arrays
    keys = list(costs.keys())
    metrics = {
        "latency_ms": np.array([costs[k]["latency_ms"] for k in keys], dtype=np.float64),
        "peak_mem_bytes": np.array(
            [costs[k]["peak_mem_bytes"] if costs[k]["peak_mem_bytes"] is not None else np.nan for k in keys],
            dtype=np.float64),
        "avg_power_w": np.array(
            [costs[k]["avg_power_w"] if costs[k]["avg_power_w"] is not None else np.nan for k in keys],
            dtype=np.float64),
        "weight_size_bytes": np.array([costs[k]["weight_size_bytes"] for k in keys], dtype=np.float64),
    }
    weights = {
        "latency_ms": weight_latency,
        "peak_mem_bytes": weight_memory,
        "avg_power_w": weight_power,
        "weight_size_bytes": weight_weights,
    }

    # Normalize; handle NaNs by ignoring those metrics in composite for that model
    norm = {}
    for m, arr in metrics.items():
        arr_norm = np.zeros_like(arr)
        valid = ~np.isnan(arr)
        if valid.sum() > 0:
            vmin, vmax = np.nanmin(arr[valid]), np.nanmax(arr[valid])
            if vmax > vmin:
                arr_norm[valid] = (arr[valid] - vmin) / (vmax - vmin)
            else:
                arr_norm[valid] = 0.0
        arr_norm[~valid] = np.nan
        norm[m] = arr_norm

    # For each model, compute composite with available metrics
    for i, k in enumerate(keys):
        # Effective weights only for metrics that are not NaN for this model
        eff_pairs = []
        for m, w in weights.items():
            if not np.isnan(norm[m][i]):
                eff_pairs.append((m, w))
        total_w = sum(w for _, w in eff_pairs)
        composite = 0.0
        if total_w > 0:
            for m, w in eff_pairs:
                composite += (w / total_w) * float(norm[m][i])
        costs[k]["composite_cost"] = float(composite)

    return costs

def measure_all_costs(models: List[str],
                      out_dir: str,
                      img_size: int = 224,
                      passes: int = 30,
                      batch_size: int = 1,
                      latency_w=1.0, memory_w=1.0, power_w=1.0, weights_w=0.5,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, float]:
    safe_mkdir(out_dir)
    raw = {}
    for m in models:
        c = measure_model_cost(
            m, img_size=img_size, device=device,
            batch_size=batch_size, warmup=10, passes=passes
        )
        raw[m] = c
    norm = normalize_and_composite(
        raw, weight_latency=latency_w, weight_memory=memory_w,
        weight_power=power_w, weight_weights=weights_w
    )
    # Persist detailed table
    with open(os.path.join(out_dir, "costs.json"), "w") as f:
        json.dump(norm, f, indent=2)
    # Return simple {model: composite_cost}
    return {k: v["composite_cost"] for k, v in norm.items()}

def subset_cost(subset: Tuple[str, ...], table: Dict[str, float]) -> float:
    return float(sum(table[m] for m in subset))

# ------------------------
# Graphs
# ------------------------
def rows_to_feature(row: pd.Series) -> np.ndarray:
    return np.array([
        row["confidence_normal"], row["confidence_osmf"],
        row["confidence_wdoscc"], row["confidence_mdoscc"],
        row["confidence_pdoscc"],
        row["entropy"], row["variance"]
    ], dtype=np.float32)

def build_graph(img_id, sub_models, img_df, cost_table, y_true_int):
    rows, softmaxes = [], []
    for m in sub_models:
        r = img_df[img_df["model"] == m]
        if len(r) != 1:
            return None
        r = r.iloc[0]
        rows.append(r)
        softmaxes.append(np.array([
            r["confidence_normal"], r["confidence_osmf"],
            r["confidence_wdoscc"], r["confidence_mdoscc"],
            r["confidence_pdoscc"]
        ], dtype=np.float32))


    X = np.stack([rows_to_feature(r) for r in rows], axis=0)
    n = len(sub_models)

    # fully-connected directed edges
    send, recv = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                send.append(i); recv.append(j)

    edge_index = torch.tensor([send, recv], dtype=torch.long)
    edge_attr = torch.tensor(
        [[cosine_sim(softmaxes[i], softmaxes[j])] for i, j in zip(send, recv)],
        dtype=torch.float32
    )
    x = torch.tensor(X, dtype=torch.float32)
    y_multi = torch.tensor([y_true_int], dtype=torch.long)
    y_bin = one_vs_rest_binary(y_multi)
    scost = subset_cost(sub_models, cost_table)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_multi=y_multi, y_bin=y_bin)
    d.num_nodes = n
    d.image_id = img_id
    d.subset = ",".join(sub_models)
    d.subset_size = n
    d.subset_cost = float(scost)
    return d

def make_graphs(split_df, cost_table, expected_models: List[str]):
    graphs = []
    by_img = group_by_image(split_df)
    for img_id, g in by_img.items():
        # ensure all expected models present for this image
        present = set(g["model"].unique())
        if not set(expected_models).issubset(present):
            continue
        tl = g["true_label"].iloc[0]
        label_key = str(tl).strip().lower()
        if label_key not in CLASS_TO_ID:
            raise ValueError(f"Unexpected label '{tl}' in CSV")
        y_true = CLASS_TO_ID[label_key]
        for sub in all_nonempty_subsets(expected_models):
            d = build_graph(img_id, sub, g, cost_table, y_true)
            if d is not None:
                graphs.append(d)
    return graphs

# ------------------------
# Dataset
# ------------------------
class GraphDataset(Dataset):
    def __init__(self, graphs, max_cost):
        self.graphs = graphs; self.max_cost = max_cost
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx):
        d = self.graphs[idx]
        d.subset_cost_norm = torch.tensor([d.subset_cost / self.max_cost], dtype=torch.float32)
        return d

# ------------------------
# Model
# ------------------------
class CAGNNE(nn.Module):
    def __init__(self, in_dim=NODE_FEAT_DIM, hid=128, drop=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, hid)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.bin_head = nn.Linear(hid, 1)
        self.multi_head = nn.Linear(hid, len(CLASSES))
        self.lr = 1e-3

    def forward(self, x, edge_index, batch):
        h = self.act(self.conv1(x, edge_index)); h = self.drop(h)
        h = self.act(self.conv2(h, edge_index)); h = self.drop(h)
        g = global_mean_pool(h, batch)
        return self.bin_head(g), self.multi_head(g)

# ------------------------
# Training / Eval
# ------------------------
def compute_metrics(y_true, prob, y_pred):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": None,
        "macro_auc": None
    }
    try:
        out["macro_auc"] = roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
    except Exception:
        pass
    try:
        out["log_loss"] = log_loss(y_true, prob, labels=list(range(len(CLASSES))))
    except Exception:
        pass
    return out

def save_checkpoint(model: nn.Module, path: str):
    safe_mkdir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def load_checkpoint(model: nn.Module, path: str, map_location=None):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)
    return model

def train_model(model,
                tr_loader,
                va_loader,
                device,
                alpha: float,
                beta: float,
                lam: float,
                epochs: int = 30,
                patience: int = 5,
                checkpoint_path: str = None):
    import time, sys
    from tqdm import tqdm

    opt = AdamW(model.parameters(), lr=model.lr, weight_decay=1e-4)
    best_f1, no_improve, best_state = -1.0, 0, None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        # tqdm gives nice live progress per batch
        for i, batch in enumerate(tqdm(tr_loader, desc=f"Epoch {ep+1}/{epochs}", unit="batch")):
            batch = batch.to(device)
            opt.zero_grad()
            out_bin, out_multi = model(batch.x, batch.edge_index, batch.batch)
            loss_bin = nn.BCEWithLogitsLoss()(out_bin.squeeze(), batch.y_bin.float())
            loss_multi = nn.CrossEntropyLoss()(out_multi, batch.y_multi)
            loss = (alpha * loss_bin + beta * loss_multi) * (1 + lam * batch.subset_cost_norm.mean())
            loss.backward(); opt.step()
            total_loss += loss.item()

            # every ~1 second show a live message
            if (time.time() - start_time) >= 1.0:
                pct = 100.0 * (i+1) / len(tr_loader)
                sys.stdout.write(
                    f"\r[Epoch {ep+1}/{epochs}] "
                    f"{i+1}/{len(tr_loader)} batches ({pct:.1f}%) | "
                    f"Avg Loss {total_loss/(i+1):.4f}"
                )
                sys.stdout.flush()
                start_time = time.time()
        print("")  # newline after epoch loop

        avg_loss = total_loss / len(tr_loader)
        print(f"[Epoch {ep+1}/{epochs}] Training finished | Avg Loss {avg_loss:.4f}")

        # validation
        if va_loader is None:
            continue
        model.eval()
        all_y, all_pred, all_prob = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                _, out_multi = model(batch.x, batch.edge_index, batch.batch)
                prob = torch.softmax(out_multi, dim=1).cpu().numpy()
                pred = prob.argmax(axis=1)
                all_y.extend(batch.y_multi.cpu().numpy())
                all_pred.extend(pred)
                all_prob.extend(prob)
        mets = compute_metrics(np.array(all_y), np.array(all_prob), np.array(all_pred))

        print(f"[Epoch {ep+1}/{epochs}] "
              f"Val Acc={mets['accuracy']:.4f} | "
              f"Val F1={mets['macro_f1']:.4f} | "
              f"Val AUC={mets['macro_auc'] if mets['macro_auc'] else 0:.4f}")

        if mets["macro_f1"] is not None and mets["macro_f1"] > best_f1:
            best_f1, best_state, no_improve = mets["macro_f1"], model.state_dict(), 0
            if checkpoint_path:
                save_checkpoint(model, checkpoint_path)
                print(f"[INFO] Saved new best model (F1={best_f1:.4f}) → {checkpoint_path}")
        else:
            no_improve += 1
            print(f"[INFO] No improvement ({no_improve}/{patience})")
        if no_improve >= patience:
            print(f"[EARLY STOP] Validation F1 did not improve for {patience} epochs.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    elif checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, map_location=device)
    return best_f1 if best_f1 >= 0 else None, model

# ------------------------
# Optuna Objective
# ------------------------
def objective(trial,
              train_graphs,
              val_graphs,
              device,
              max_cost,
              out_dir: str):
    # Hyperparameters to tune
    hid = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    drop = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.3, 0.7)
    beta = trial.suggest_float("beta", 0.3, 0.7)
    lam = trial.suggest_float("lam", 0.05, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = 20   # fixed: each trial runs for 20 epochs

    # Data loaders
    tr_loader = DataLoader(GraphDataset(train_graphs, max_cost),
                           batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(GraphDataset(val_graphs, max_cost),
                           batch_size=batch_size, shuffle=False)

    # Model
    model = CAGNNE(hid=hid, drop=drop).to(device)
    model.lr = lr

    # Trial directory + checkpoint path
    trial_dir = os.path.join(out_dir, "optuna", f"trial_{trial.number}")
    safe_mkdir(trial_dir)
    ckpt_path = os.path.join(trial_dir, "best.pt")

    # Train this trial
    best_f1, _ = train_model(
        model, tr_loader, va_loader, device,
        alpha, beta, lam,
        epochs=epochs, patience=6,
        checkpoint_path=ckpt_path
    )
    if best_f1 is None:
        best_f1 = -1.0

    # Save path to best checkpoint
    trial.set_user_attr("best_ckpt", ckpt_path)
    return best_f1

# ------------------------
# Post-Analysis
# ------------------------
def analyze_predictions(preds_csv, out_dir):
    """
    Comprehensive post-hoc analysis for CA-GNNE / ORCHID ensemble.
    Generates metrics, calibration, disagreement, flowchart logic, and case studies.
    Designed for paper-level reporting.
    """
    import os, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss
    )
    from sklearn.calibration import calibration_curve
    import networkx as nx

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(preds_csv)

    # === Normalize column names for compatibility ===
    rename_map = {
        "true_label": "true",
        "predicted_label": "pred",
        "confidence_Normal": "p_Normal",
        "confidence_OSMF": "p_OSMF",
        "confidence_WDOSCC": "p_WDOSCC",
        "confidence_MDOSCC": "p_MDOSCC",
        "confidence_PDOSCC": "p_PDOSCC",
    }
    df = df.rename(columns=rename_map)

    if "Title" not in df.columns:
        if "image_id" in df.columns:
            df = df.rename(columns={"image_id": "Title"})
        else:
            df["Title"] = df.index.astype(str)

    # Ensure numeric types
    for c in ["subset_size", "subset_cost", "true", "pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    prob_cols = [c for c in df.columns if c.startswith("p_")]
    n_classes = len(prob_cols)

    # Normalize probs row-wise
    if n_classes > 0:
        df[prob_cols] = df[prob_cols].div(df[prob_cols].sum(axis=1), axis=0)

    # === Subset-level metrics ===
    summary = []
    if "subset" in df.columns:
        for subset, g in df.groupby("subset"):
            y_true, y_pred, probs = g["true"].values, g["pred"].values, g[prob_cols].values
            acc = accuracy_score(y_true, y_pred)
            f1m = f1_score(y_true, y_pred, average="macro")
            aucm = None; ll = None
            try: aucm = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            except: pass
            try: ll = log_loss(y_true, probs, labels=list(range(n_classes)))
            except: pass
            brier = brier_score_loss(pd.get_dummies(y_true).values.ravel(),
                                     probs.ravel(), pos_label=1)

            row = {
                "subset": subset,
                "subset_size": int(g["subset_size"].iloc[0]) if "subset_size" in g else None,
                "subset_cost": float(g["subset_cost"].iloc[0]) if "subset_cost" in g else None,
                "accuracy": acc,
                "macro_f1": f1m,
                "macro_auc": aucm,
                "log_loss": ll,
                "brier_score": brier,
            }
            summary.append(row)

        df_sum = pd.DataFrame(summary)
        df_sum.to_csv(os.path.join(out_dir, "subset_report.csv"), index=False)
    else:
        df_sum = pd.DataFrame()

    # === Calibration curves ===
    if n_classes > 0:
        plt.figure(figsize=(6,6))
        ece_scores = {}
        for i,c in enumerate(prob_cols):
            true_bin = (df["true"]==i).astype(int)
            frac_pos, mean_pred = calibration_curve(true_bin, df[c], n_bins=10, strategy="uniform")
            plt.plot(mean_pred, frac_pos, marker="o", label=c)
            ece = np.mean(np.abs(frac_pos - mean_pred))
            ece_scores[c] = ece
        plt.plot([0,1],[0,1],"k--")
        plt.title("Reliability Diagram (All Classes)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Fraction positive")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"calibration.png"))

        with open(os.path.join(out_dir,"ece_scores.json"),"w") as f:
            json.dump(ece_scores,f,indent=2)

    # === Diversity metrics (Q, double fault) ===
    models = df["model"].unique() if "model" in df.columns else None
    if models is not None:
        from itertools import combinations
        div_metrics = {}
        for m1,m2 in combinations(models,2):
            d1 = df[df["model"]==m1]; d2 = df[df["model"]==m2]
            common = d1.merge(d2,on="Title",suffixes=("_1","_2"))
            if len(common)==0: continue
            N = len(common)
            N11 = np.sum((common["true_1"]==common["pred_1"]) & (common["true_2"]==common["pred_2"]))
            N00 = np.sum((common["true_1"]!=common["pred_1"]) & (common["true_2"]!=common["pred_2"]))
            N10 = np.sum((common["true_1"]==common["pred_1"]) & (common["true_2"]!=common["pred_2"]))
            N01 = np.sum((common["true_1"]!=common["pred_1"]) & (common["true_2"]==common["pred_2"]))
            Q = (N11*N00 - N10*N01) / max(1e-6,(N11*N00+N10*N01))
            double_fault = N00/N
            div_metrics[f"{m1} vs {m2}"]={"Q":Q,"double_fault":double_fault}
        with open(os.path.join(out_dir,"diversity.json"),"w") as f:
            json.dump(div_metrics,f,indent=2)

    # === Disagreement map ===
    if n_classes > 0:
        df["disagreement"] = -np.sum(df[prob_cols]*np.log(df[prob_cols]+1e-12),axis=1)
        plt.figure()
        plt.hist(df["disagreement"],bins=40,color="orange",alpha=0.7)
        plt.xlabel("Prediction entropy"); plt.ylabel("Count")
        plt.title("Ensemble Disagreement Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"disagreement.png"))

    # === Graph-based ensemble flowchart ===
    if models is not None:
        import networkx as nx
        from itertools import combinations
        G = nx.Graph()
        for m in models: G.add_node(m)
        for m1,m2 in combinations(models,2):
            a = np.mean(df[df["model"]==m1]["pred"].values==
                        df[df["model"]==m2]["pred"].values)
            G.add_edge(m1,m2,weight=a)
        nx.write_gml(G, os.path.join(out_dir,"ensemble_graph.gml"))
        with open(os.path.join(out_dir,"flowchart.txt"),"w") as f:
            f.write("=== Rule + Graph-Based Ensemble Logic ===\n")
            f.write("1. If all models agree → accept consensus.\n")
            f.write("2. If clique ≥3 models agree → trust clique.\n")
            f.write("3. If disagreement confined to carcinoma subtypes → defer to subtype-strong model.\n")
            f.write("4. If entropy > threshold or no consensus → mark as Uncertain.\n")

    # === Case study panels ===
    if "true" in df.columns and "pred" in df.columns:
        errors = df[df["true"]!=df["pred"]].sample(min(3,len(df)),random_state=42)
        with open(os.path.join(out_dir,"case_studies.txt"),"w") as f:
            for _,row in errors.iterrows():
                f.write(f"Image {row['Title']} | True={row['true']} | Pred={row['pred']}\n")
                for c in prob_cols:
                    f.write(f"  {c}: {row[c]:.3f}\n")
                f.write("\n")

    print(f"[ANALYSIS] Reports, plots, and flowchart saved to {out_dir}")

# ------------------------
# Main
# ------------------------
def main(args):
    set_seed(42); safe_mkdir(args.out_dir)

    # Decide models
    expected_models = args.models if args.models else DEFAULT_MODELS

    print("[INFO] Reading CSVs...")
    # Auto-load CSVs from ./data/<model>/
    if not args.train_csvs and not args.val_csvs and not args.test_csvs:
        base_dir = Path("data")
        train_csvs = [str(base_dir/m/f"train_{m}.csv") for m in args.models or DEFAULT_MODELS]
        val_csvs   = [str(base_dir/m/f"val_{m}.csv")   for m in args.models or DEFAULT_MODELS]
        test_csvs  = [str(base_dir/m/f"test_{m}.csv")  for m in args.models or DEFAULT_MODELS]
    else:
        train_csvs, val_csvs, test_csvs = args.train_csvs, args.val_csvs, args.test_csvs

    train_df = read_split_csvs(train_csvs)
    val_df   = read_split_csvs(val_csvs)
    test_df  = read_split_csvs(test_csvs)

    # Optionally, enforce that CSV 'model' column contains only expected names
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        uniq = sorted(df["model"].unique())
        missing = [m for m in expected_models if m not in uniq]
        if missing:
            print(f"[WARN] {df_name} split missing models: {missing} (images lacking full 5 models will be skipped)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- COST MEASUREMENT ----------
    print("[INFO] Measuring per-model costs (latency, memory, power, weight size)...")
    cost_table = measure_all_costs(
        expected_models,
        out_dir=args.out_dir,
        img_size=args.cost_img_size,
        passes=args.cost_passes,
        batch_size=args.cost_batch,
        latency_w=args.cost_w_latency,
        memory_w=args.cost_w_memory,
        power_w=args.cost_w_power,
        weights_w=args.cost_w_weights,
        device=device
    )
    print("[INFO] Cost table:", cost_table)

    # ---------- Build graphs ----------
    print("[INFO] Building graphs...")
    train_graphs = make_graphs(train_df, cost_table, expected_models)
    val_graphs   = make_graphs(val_df,   cost_table, expected_models)
    test_graphs  = make_graphs(test_df,  cost_table, expected_models)

    if not train_graphs or not val_graphs or not test_graphs:
        raise RuntimeError("No graphs built for one or more splits. Check CSV format and model names.")

    max_cost_tr = max(g.subset_cost for g in train_graphs)

    # ---------- Optuna ----------
    def optuna_objective(trial):
        return objective(
            trial,
            train_graphs,
            val_graphs,
            device,
            max_cost=max_cost_tr,
            out_dir=args.out_dir
        )

    storage = args.storage if args.storage else None
    if storage:
        study = optuna.create_study(direction="maximize",
                                    storage=storage,
                                    study_name=args.study_name,
                                    load_if_exists=True)
    else:
        study = optuna.create_study(direction="maximize")

    study.optimize(optuna_objective, n_trials=args.trials)
    print("[INFO] Best trial:", study.best_trial.params,
          "F1:", study.best_trial.value)

    # ---------- Load best checkpoint only ----------
    best_params = study.best_trial.params
    best_ckpt = study.best_trial.user_attrs.get("best_ckpt", None)

    if not best_ckpt or not os.path.exists(best_ckpt):
        raise RuntimeError("[ERROR] Best checkpoint not found after Optuna tuning.")
    else:
        print(f"[INFO] Using best checkpoint from tuning: {best_ckpt}")

    model = CAGNNE(hid=best_params["hidden_dim"],
                   drop=best_params["dropout"]).to(device)
    model.lr = best_params["lr"]
    load_checkpoint(model, best_ckpt, map_location=device)

    # model is now ready to use on test set or save directly
    save_checkpoint(model, os.path.join(args.out_dir, "final.pt"))
    print(f"[INFO] Final model saved to {os.path.join(args.out_dir, 'final.pt')}")

def FullRetrain(params, train_graphs, val_graphs, test_graphs, out_dir="outputs_gnne"):
    print("[FullRetrain] Starting full retrain with given parameters...")
    safe_mkdir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_path = os.path.join(out_dir, "final.pt")

    model = CAGNNE(hid=params["hidden_dim"], drop=params["dropout"]).to(device)
    model.lr = params["lr"]

    # -------- Check if final.pt exists --------
    if os.path.exists(final_path):
        print(f"[FullRetrain] Found existing final.pt → skipping training.")
        load_checkpoint(model, final_path, map_location=device)
    else:
        # Combine train+val
        full_graphs = train_graphs + val_graphs
        max_cost_full = max(g.subset_cost for g in full_graphs)

        tr_loader = DataLoader(GraphDataset(full_graphs, max_cost=max_cost_full),
                               batch_size=params["batch_size"], shuffle=True)
        va_loader = None  # no separate validation

        # Train for 30 epochs with patience=10
        print("[FullRetrain] Training on train+val for 30 epochs (patience=10)...")
        best_f1, model = train_model(
            model,
            tr_loader,
            va_loader,
            device,
            alpha=params["alpha"],
            beta=params["beta"],
            lam=params["lam"],
            epochs=30,
            patience=10,
            checkpoint_path=os.path.join(out_dir, "best.pt")
        )

        if best_f1 is not None:
            print(f"[FullRetrain] Training complete. Best F1={best_f1:.4f} (on val).")
        else:
            print("[FullRetrain] Training complete on train+val (no validation F1).")

        # Save final checkpoint
        save_checkpoint(model, final_path)
        print(f"[FullRetrain] Final model saved → {final_path}")

    # ---- Inference on test ----
    print("[FullRetrain] Running inference on test set...")
    te_loader = DataLoader(GraphDataset(test_graphs, max_cost=max(g.subset_cost for g in test_graphs)),
                           batch_size=params["batch_size"], shuffle=False)

    all_rows = []
    model.eval()
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(device)
            _, out_multi = model(batch.x, batch.edge_index, batch.batch)
            prob = torch.softmax(out_multi, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)

            for i in range(len(pred)):
                subset_size = getattr(batch, "subset_size", torch.tensor([-1]))
                subset_cost = getattr(batch, "subset_cost", torch.tensor([-1.0]))
                all_rows.append({
                    "image_id": getattr(batch, "image_id", ["NA"])[i] if hasattr(batch, "image_id") else "NA",
                    "subset": getattr(batch, "subset", ["NA"])[i] if hasattr(batch, "subset") else "NA",
                    "subset_size": int(subset_size[i].item() if isinstance(subset_size, torch.Tensor) else subset_size),
                    "subset_cost": float(subset_cost[i].item() if isinstance(subset_cost, torch.Tensor) else subset_cost),
                    "true": int(batch.y_multi.cpu().numpy()[i]),
                    "pred": int(pred[i]),
                    **{f"p_{c}": float(prob[i, j]) for j, c in enumerate(CLASSES)}
                })

    preds_csv = os.path.join(out_dir, "test_predictions.csv")
    pd.DataFrame(all_rows).to_csv(preds_csv, index=False)
    print(f"[FullRetrain] Test predictions saved → {preds_csv}")

    # ---- Analysis ----
    print("[FullRetrain] Analyzing test predictions...")
    analyze_predictions(preds_csv, out_dir)
    print(f"[FullRetrain] Analysis report saved in {out_dir}/subset_report.csv")
    print("[FullRetrain] Done.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # CSVs (optional, auto-discovered from ./data if not provided)
    parser.add_argument("--train_csvs", nargs="+", default=None)
    parser.add_argument("--val_csvs",   nargs="+", default=None)
    parser.add_argument("--test_csvs",  nargs="+", default=None)

    # Models
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names in CSVs (default 5): convnext resnet efficientnet deit regnety")

    # Cost measurement
    parser.add_argument("--cost_img_size", type=int, default=224)
    parser.add_argument("--cost_passes", type=int, default=30)
    parser.add_argument("--cost_batch", type=int, default=1)
    parser.add_argument("--cost_w_latency", type=float, default=1.0)
    parser.add_argument("--cost_w_memory", type=float, default=1.0)
    parser.add_argument("--cost_w_power", type=float, default=1.0)
    parser.add_argument("--cost_w_weights", type=float, default=0.5)

    # Optuna
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL, e.g., sqlite:///orchid_optuna.db")
    parser.add_argument("--study-name", type=str, default="cagne_study")

    # Retrain
    parser.add_argument("--retrain_epochs", type=int, default=20)
    parser.add_argument("--params", type=str, default=None,
                        help="JSON dict of params for FullRetrain (skip Optuna if provided)")

    # Output
    parser.add_argument("--out_dir", type=str, default="outputs_cagne")

    args = parser.parse_args()

    if args.params:
        # Parse params as JSON
        params = json.loads(args.params.replace("'", '"'))

        # Load CSVs
        base_dir = Path("data")
        train_csvs = args.train_csvs or [str(base_dir/m/f"train_{m}.csv") for m in args.models or DEFAULT_MODELS]
        val_csvs   = args.val_csvs   or [str(base_dir/m/f"val_{m}.csv")   for m in args.models or DEFAULT_MODELS]
        test_csvs  = args.test_csvs  or [str(base_dir/m/f"test_{m}.csv")  for m in args.models or DEFAULT_MODELS]

        train_df = read_split_csvs(train_csvs)
        val_df   = read_split_csvs(val_csvs)
        test_df  = read_split_csvs(test_csvs)

        # Always use existing costs.json
        cost_path = os.path.join(args.out_dir, "costs.json")
        if not os.path.exists(cost_path):
            raise RuntimeError(f"[ERROR] Expected {cost_path} not found.")
        with open(cost_path, "r") as f:
            costs = json.load(f)
        cost_table = {k: v["composite_cost"] for k, v in costs.items()}

        # Build graphs
        train_graphs = make_graphs(train_df, cost_table, args.models or DEFAULT_MODELS)
        val_graphs   = make_graphs(val_df,   cost_table, args.models or DEFAULT_MODELS)
        test_graphs  = make_graphs(test_df,  cost_table, args.models or DEFAULT_MODELS)

        # Run full retrain with provided params
        FullRetrain(params, train_graphs, val_graphs, test_graphs, out_dir=args.out_dir)
    else:
        main(args)