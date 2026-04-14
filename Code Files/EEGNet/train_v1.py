# ============================================================
# train.py  —  EEGNet PD vs CTL training (local VS Code version)
# ============================================================
# Before running, install dependencies in your terminal:
#   pip install torch torchvision scikit-learn mne tqdm matplotlib seaborn pandas numpy
# ============================================================

import json
import logging
import multiprocessing
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — swap to "TkAgg" or
import matplotlib.pyplot as plt  # "Qt5Agg" if you want pop-up windows locally
import seaborn as sns
from tqdm import tqdm

import mne

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)


# ============================================================
# Paths  —  must match preprocessing.py
# ============================================================
OUT_DIR     = Path("Signal Code/Outputs")
EPOCHS_DIR  = OUT_DIR / "cache_epochs_binary"
SUMMARY_TSV = OUT_DIR / "preprocessing_summary.tsv"
MODEL_DIR   = OUT_DIR / "eegnet_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("eegnet_pd_ctl")


# ============================================================
# Pretty-print helpers
# ============================================================

def _banner(title: str, width: int = 60) -> None:
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _section(title: str, width: int = 60) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ============================================================
# Reproducibility
# ============================================================
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# set_seed is called inside main() — NOT at module level —
# so DataLoader worker processes don't re-execute it and flood stdout.


# ============================================================
# Environment helpers
# ============================================================

def _safe_num_workers() -> int:
    """
    Returns a safe DataLoader worker count.
    Uses 0 inside Jupyter/Colab kernels to avoid fork deadlocks;
    otherwise uses min(4, cpu_count // 2).
    """
    in_notebook = "ipykernel" in sys.modules or "google.colab" in sys.modules
    if in_notebook:
        return 0
    return min(4, max(1, multiprocessing.cpu_count() // 2))


# ============================================================
# Config dataclass  —  all hyperparameters in one place
# ============================================================
@dataclass
class EEGNetConfig:
    # ── Data ────────────────────────────────────────────────
    sfreq:         float = 250.0
    n_channels:    int   = 16
    epoch_len_sec: float = 1.0

    # ── EEGNet architecture  (Lawhern et al. 2018) ──────────
    F1:              int   = 8
    D:               int   = 2
    F2:              int   = 16
    kernel_temporal: int   = 125
    dropout:         float = 0.5

    # ── Training ────────────────────────────────────────────
    batch_size:     int   = 64
    lr:             float = 3e-4      # FIXED: was 1e-3 — too high, caused immediate overfitting
    weight_decay:   float = 1e-4
    max_epochs:     int   = 200       # raised from 150 to give ReduceLROnPlateau more room
    min_epochs:     int   = 25        # NEW: early stopping cannot fire before this epoch
    patience:       int   = 30        # raised from 20
    grad_clip_norm: float = 1.0

    # ── LR scheduler ────────────────────────────────────────
    # ReduceLROnPlateau: halves the LR when val loss hasn't improved
    # for lr_patience epochs. Much more robust than cosine on small datasets.
    lr_patience:    int   = 8
    lr_factor:      float = 0.5
    lr_min:         float = 1e-6

    # ── Cross-validation ────────────────────────────────────
    n_folds: int = 5

    # ── Normalisation ───────────────────────────────────────
    normalize: bool = True

    # ── Class imbalance ─────────────────────────────────────
    imbalance_threshold: float = 1.5

    # ── DataLoader ──────────────────────────────────────────
    num_workers: int = field(default_factory=_safe_num_workers)

    # ── Device ──────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    @property
    def n_timepoints(self) -> int:
        return int(self.sfreq * self.epoch_len_sec)

    def save(self, path: Path):
        d = asdict(self)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)
        logger.info(f"Config saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "EEGNetConfig":
        with open(path) as fh:
            d = json.load(fh)
        return cls(**d)


# CFG lives at module level so helper functions can access it,
# but ALL print/banner calls are inside main() — this is what
# prevents DataLoader worker processes from reprinting the config
# banner 4x every fold.
CFG = EEGNetConfig()


# ============================================================
# Data loading
# ============================================================

def load_all_epochs(
    epochs_dir:  Path,
    summary_tsv: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _banner("Step 1 / 4 — Loading preprocessed epochs")

    label_map: Dict[str, int] = {}
    if summary_tsv and summary_tsv.exists():
        df = pd.read_csv(summary_tsv, sep="\t", dtype={"subject": str})
        for _, row in df.iterrows():
            label_map[str(row["subject"])] = 1 if row["group"] == "PD" else 0
        n_pd_tsv  = sum(v == 1 for v in label_map.values())
        n_ctl_tsv = sum(v == 0 for v in label_map.values())
        print(f"  Label map loaded : {len(label_map)} subjects  "
              f"(PD={n_pd_tsv}, CTL={n_ctl_tsv})")
    else:
        raise FileNotFoundError(
            f"Preprocessing summary not found at {summary_tsv}. "
            "Re-run preprocessing.py first."
        )

    fif_files = sorted(epochs_dir.glob("sub-*_epochs_binary-epo.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No epoch files found in {epochs_dir}")

    print(f"  Epoch files found: {len(fif_files)}")

    X_list: List[np.ndarray]  = []
    y_list: List[np.ndarray]  = []
    g_list: List[np.ndarray]  = []
    subject_index_map: Dict[str, int] = {}
    unlabelled:        List[str]      = []
    skipped_channels:  List[str]      = []

    t0   = time.time()
    pbar = tqdm(
        fif_files,
        desc="  Loading .fif files",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for fif in pbar:
        subj = fif.name.split("_")[0].replace("sub-", "")

        if subj not in label_map:
            unlabelled.append(subj)
            continue

        label  = label_map[subj]
        epochs = mne.read_epochs(str(fif), preload=True, verbose="warning")

        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        if len(eeg_picks) < 8:
            skipped_channels.append(subj)
            pbar.write(f"  sub-{subj}: only {len(eeg_picks)} EEG channels — skipping")
            continue

        data = epochs.get_data(picks=eeg_picks)   # (n_epochs, C, T)

        T = CFG.n_timepoints
        if data.shape[2] > T:
            data = data[:, :, :T]
        elif data.shape[2] < T:
            data = np.pad(data, ((0, 0), (0, 0), (0, T - data.shape[2])))

        if subj not in subject_index_map:
            subject_index_map[subj] = len(subject_index_map)
        grp = subject_index_map[subj]

        n = data.shape[0]
        X_list.append(data.astype(np.float32))
        y_list.append(np.full(n, label, dtype=np.int64))
        g_list.append(np.full(n, grp,   dtype=np.int64))

        group_str = "PD " if label == 1 else "CTL"
        pbar.set_postfix({"last": f"sub-{subj} [{group_str}] {n} epochs"})

    if unlabelled:
        raise KeyError(
            f"{len(unlabelled)} subject(s) found in the epoch directory but "
            f"absent from the summary TSV: {unlabelled}. "
            "Add them to the TSV or remove their .fif files."
        )

    if not X_list:
        raise RuntimeError("No usable epochs loaded.")

    X      = np.concatenate(X_list, axis=0)
    y      = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    n_pd  = int((y == 1).sum())
    n_ctl = int((y == 0).sum())
    ratio = max(n_pd, n_ctl) / max(min(n_pd, n_ctl), 1)

    print(f"\n  Loading complete in {_fmt_time(time.time() - t0)}")
    print(f"  Subjects loaded  : {len(subject_index_map)}"
          + (f"  ({len(skipped_channels)} skipped — too few channels)"
             if skipped_channels else ""))
    print(f"  Total epochs     : {X.shape[0]}  (PD={n_pd}, CTL={n_ctl})")
    print(f"  Imbalance ratio  : {ratio:.2f}  "
          f"({'balanced' if ratio < 1.5 else 'imbalanced — sampler will be used'})")
    print(f"  Data tensor      : {X.shape}  ->  {X.nbytes / 1e6:.1f} MB")

    return X, y, groups


# ============================================================
# Normalisation
# ============================================================

def channel_wise_zscore(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Fit per-channel z-score on X_train; apply to val and (optionally) test."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, \
           (X_test  - mean) / std if X_test is not None else None, \
           mean, std


# ============================================================
# PyTorch Dataset
# ============================================================

class EEGDataset(Dataset):
    """Wraps (X, y) arrays. Adds channel dim so EEGNet receives (B, 1, C, T)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts            = np.bincount(y)
    weights_per_class = 1.0 / counts
    sample_weights    = weights_per_class[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(y),
        replacement=True,
    )


def compute_class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    counts  = np.bincount(y)
    weights = 1.0 / counts.astype(np.float32)
    weights /= weights.mean()
    return torch.tensor(weights, device=device)


def _use_weighted_sampler(y: np.ndarray, threshold: float) -> bool:
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() == 0:
        return False
    return (counts.max() / counts.min()) > threshold


# ============================================================
# EEGNet architecture
# ============================================================

class EEGNet(nn.Module):
    """EEGNet — Lawhern et al. (2018), J. Neural Eng."""

    def __init__(self, cfg: EEGNetConfig, n_classes: int = 2):
        super().__init__()
        T  = cfg.n_timepoints
        C  = cfg.n_channels
        F1 = cfg.F1
        D  = cfg.D
        F2 = cfg.F2
        p  = cfg.dropout

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, cfg.kernel_temporal),
                      padding=(0, cfg.kernel_temporal // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            dummy = self.temporal_conv(dummy)
            dummy = self.depthwise_conv(dummy)
            dummy = self.separable_conv(dummy)
            flat  = dummy.flatten(1).shape[1]

        self.classifier = nn.Linear(flat, n_classes)
        self._flat_size = flat
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        return F.log_softmax(self.classifier(x.flatten(1)), dim=1)

    def apply_max_norm_(self, max_norm_spatial: float = 1.0, max_norm_clf: float = 0.25):
        with torch.no_grad():
            dw   = self.depthwise_conv[0]
            norm = dw.weight.data.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            dw.weight.data.mul_(norm.clamp(max=max_norm_spatial) / norm)
            norm_c = self.classifier.weight.data.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
            self.classifier.weight.data.mul_(norm_c.clamp(max=max_norm_clf) / norm_c)


# ============================================================
# Training utilities
# ============================================================

class EarlyStopping:
    """
    Stops training when val loss has not improved for `patience` epochs,
    but never before `min_epochs` have completed.
    """
    def __init__(self, patience: int, save_path: Path,
                 min_epochs: int = 0, min_delta: float = 1e-4):
        self.patience   = patience
        self.save_path  = save_path
        self.min_epochs = min_epochs   # NEW: grace period before stopping is allowed
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
        # Only allow early stopping after the grace period
        return (epoch >= self.min_epochs) and (self.counter >= self.patience)


def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    device:       str,
    epoch_pbar:   tqdm,
    class_weight: Optional[torch.Tensor] = None,
    grad_clip:    float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        log_probs = model(X_batch)
        loss      = F.nll_loss(log_probs, y_batch, weight=class_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        model.apply_max_norm_()
        total_loss += loss.item() * len(y_batch)

        epoch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"}, refresh=True)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    all_y:     List[int]   = []
    all_probs: List[float] = []

    for X_batch, y_batch in loader:
        X_batch   = X_batch.to(device, non_blocking=True)
        y_batch   = y_batch.to(device, non_blocking=True)
        log_probs = model(X_batch)
        loss      = F.nll_loss(log_probs, y_batch)
        probs     = log_probs.exp()

        total_loss += loss.item() * len(y_batch)
        correct    += (probs.argmax(1) == y_batch).sum().item()
        all_y.extend(y_batch.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    n = len(loader.dataset)
    return (
        total_loss / n,
        correct / n,
        np.array(all_y,     dtype=np.int64),
        np.array(all_probs, dtype=np.float32),
    )


@torch.no_grad()
def predict_proba(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    """Inference-only forward pass — no labels or loss needed."""
    model.eval()
    all_probs: List[float] = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        all_probs.extend(model(X_batch).exp()[:, 1].cpu().tolist())
    return np.array(all_probs, dtype=np.float32)


# ============================================================
# Subject-level aggregation
# ============================================================

def subject_level_predictions(
    y_epoch:     np.ndarray,
    probs_epoch: np.ndarray,
    groups:      np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_subjects = np.unique(groups)
    subj_y     = np.empty(len(unique_subjects), dtype=np.int64)
    subj_probs = np.empty(len(unique_subjects), dtype=np.float32)
    for i, s in enumerate(unique_subjects):
        mask          = groups == s
        subj_y[i]    = int(np.round(y_epoch[mask].mean()))
        subj_probs[i] = probs_epoch[mask].mean()
    return subj_y, subj_probs, unique_subjects


# ============================================================
# Shared builder helpers
# ============================================================

def _make_loaders(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    cfg:  EEGNetConfig,
    use_sampler: bool,
) -> Tuple[DataLoader, DataLoader]:
    sampler      = make_weighted_sampler(y_tr) if use_sampler else None
    train_loader = DataLoader(
        EEGDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        EEGDataset(X_va, y_va),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    return train_loader, val_loader


def _build_model_and_optimiser(
    cfg: EEGNetConfig,
) -> Tuple[EEGNet, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    model     = EEGNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # FIXED: ReduceLROnPlateau replaces CosineAnnealingLR.
    # Cosine dropped the LR too aggressively in the first few epochs on
    # this small dataset. ReduceLROnPlateau only reduces when val loss
    # actually stalls, giving the model time to learn first.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.lr_min,
    )
    return model, optimizer, scheduler


def _run_training_loop(
    model:        EEGNet,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          EEGNetConfig,
    ckpt_path:    Path,
    class_weight: Optional[torch.Tensor],
    fold_label:   str = "",
) -> Tuple[EEGNet, EarlyStopping, List[float], List[float]]:
    early_stop = EarlyStopping(
        patience=cfg.patience,
        save_path=ckpt_path,
        min_epochs=cfg.min_epochs,   # grace period before stopping is allowed
    )
    train_losses: List[float] = []
    val_losses:   List[float] = []

    epoch_pbar = tqdm(
        range(1, cfg.max_epochs + 1),
        desc=f"  {fold_label}",
        unit="ep",
        bar_format=(
            "{desc}{n_fmt:>3}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        ),
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, cfg.device,
            epoch_pbar=epoch_pbar,
            class_weight=class_weight,
            grad_clip=cfg.grad_clip_norm,
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, cfg.device)

        # ReduceLROnPlateau steps on val loss
        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        improved    = va_loss < early_stop.best_loss - early_stop.min_delta
        best_marker = "best" if improved else ""
        grace_str   = f"grace {epoch}/{cfg.min_epochs}" if epoch < cfg.min_epochs else ""

        epoch_pbar.set_postfix({
            "train": f"{tr_loss:.4f}",
            "val":   f"{va_loss:.4f}",
            "acc":   f"{va_acc:.3f}",
            "lr":    f"{current_lr:.2e}",
            "pat":   f"{early_stop.counter}/{cfg.patience}",
            "":      best_marker or grace_str,
        }, refresh=True)

        if early_stop(va_loss, model, epoch):
            epoch_pbar.write(
                f"  ⏹  {fold_label}early stop at epoch {epoch} "
                f"(best={early_stop.best_epoch}, "
                f"val_loss={early_stop.best_loss:.4f})"
            )
            break

    # Restore best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    return model, early_stop, train_losses, val_losses


# ============================================================
# Cross-validation
# ============================================================

def run_cross_validation(
    X:      np.ndarray,
    y:      np.ndarray,
    groups: np.ndarray,
    cfg:    EEGNetConfig,
) -> Tuple[pd.DataFrame, List[List[float]], List[List[float]]]:
    """StratifiedGroupKFold cross-validation with live per-fold progress."""

    _banner("Step 2 / 4 — Cross-validation")
    print(f"  Strategy   : {cfg.n_folds}-fold StratifiedGroupKFold")
    print(f"  Subjects   : {len(np.unique(groups))}")
    print(f"  Epochs     : {len(y)}  (PD={int((y==1).sum())}, CTL={int((y==0).sum())})")
    print(f"  Min epochs : {cfg.min_epochs}  (early stopping grace period)\n")

    cv     = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=SEED)
    device = cfg.device

    fold_results:      List[Dict]        = []
    fold_train_losses: List[List[float]] = []
    fold_val_losses:   List[List[float]] = []
    cv_t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):

        _section(f"Fold {fold} / {cfg.n_folds}")

        X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_va, y_va, g_va = X[val_idx],   y[val_idx],   groups[val_idx]

        print(f"  Train : {len(y_tr):>5} epochs / {len(np.unique(g_tr))} subjects"
              f"  (PD={int((y_tr==1).sum())}, CTL={int((y_tr==0).sum())})")
        print(f"  Val   : {len(y_va):>5} epochs / {len(np.unique(g_va))} subjects"
              f"  (PD={int((y_va==1).sum())}, CTL={int((y_va==0).sum())})")

        if cfg.normalize:
            X_tr, X_va, _, _, _ = channel_wise_zscore(X_tr, X_va)
            print("  Normalisation : channel-wise z-score (fitted on train)")

        use_sampler  = _use_weighted_sampler(y_tr, cfg.imbalance_threshold)
        class_weight = None if use_sampler else compute_class_weights(y_tr, device)

        if use_sampler:
            counts = np.bincount(y_tr)
            print(f"  Imbalance     : ratio={counts.max()/counts.min():.2f} "
                  f"-> WeightedRandomSampler")
        else:
            print("  Imbalance     : mild -> class-weighted NLL loss")

        train_loader, val_loader = _make_loaders(X_tr, y_tr, X_va, y_va, cfg, use_sampler)
        model, optimizer, scheduler = _build_model_and_optimiser(cfg)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params  : {n_params:,}\n")

        ckpt_path  = MODEL_DIR / f"best_fold{fold}.pt"
        fold_t0    = time.time()

        model, early_stop, train_losses, val_losses = _run_training_loop(
            model, optimizer, scheduler,
            train_loader, val_loader, cfg,
            ckpt_path, class_weight,
            fold_label=f"Fold {fold} | ",
        )

        fold_elapsed = time.time() - fold_t0
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        # ── Per-fold metrics ──────────────────────────────────────────────
        _, ep_acc, ep_y, ep_probs = evaluate(model, val_loader, device)
        ep_auc = roc_auc_score(ep_y, ep_probs)
        ep_bal = balanced_accuracy_score(ep_y, (ep_probs >= 0.5).astype(int))

        subj_y, subj_probs, _ = subject_level_predictions(ep_y, ep_probs, g_va)
        subj_acc = (subj_y == (subj_probs >= 0.5).astype(int)).mean()
        subj_auc = (
            roc_auc_score(subj_y, subj_probs)
            if len(np.unique(subj_y)) > 1 else float("nan")
        )
        subj_bal = balanced_accuracy_score(subj_y, (subj_probs >= 0.5).astype(int))

        print(f"\n  ── Fold {fold} results {'─' * 30}")
        print(f"  {'Metric':<18} {'Epoch-level':>12} {'Subject-level':>14}")
        print(f"  {'─'*18} {'─'*12} {'─'*14}")
        print(f"  {'Accuracy':<18} {ep_acc:>12.4f} {subj_acc:>14.4f}")
        print(f"  {'AUC-ROC':<18} {ep_auc:>12.4f} {subj_auc:>14.4f}")
        print(f"  {'Balanced Acc':<18} {ep_bal:>12.4f} {subj_bal:>14.4f}")
        print(f"  Best epoch : {early_stop.best_epoch}  |  "
              f"Fold time : {_fmt_time(fold_elapsed)}")

        fold_results.append({
            "fold":         fold,
            "best_epoch":   early_stop.best_epoch,
            "n_val_epochs": len(ep_y),
            "n_val_subjs":  len(subj_y),
            "ep_acc":       ep_acc,
            "ep_auc":       ep_auc,
            "ep_bal_acc":   ep_bal,
            "subj_acc":     subj_acc,
            "subj_auc":     subj_auc,
            "subj_bal_acc": subj_bal,
        })

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(MODEL_DIR / "cv_results.tsv", sep="\t", index=False)
    print(f"\n  Cross-validation complete in {_fmt_time(time.time() - cv_t0)}")
    return results_df, fold_train_losses, fold_val_losses


# ============================================================
# Final model
# ============================================================

def train_final_model(
    X:            np.ndarray,
    y:            np.ndarray,
    groups:       np.ndarray,
    cfg:          EEGNetConfig,
    fixed_epochs: int,
) -> "EEGNet":
    """Train on all data for fixed_epochs (floor-clipped to cfg.min_epochs)."""

    _banner("Step 3 / 4 — Final model training")
    print(f"  Subjects : {len(np.unique(groups))}")
    print(f"  Epochs   : {len(y)}  (PD={int((y==1).sum())}, CTL={int((y==0).sum())})")
    print(f"  Duration : {fixed_epochs} epochs  (CV mean best_epoch, >= min_epochs)\n")

    X_tr, y_tr = X, y

    if cfg.normalize:
        mean = X_tr.mean(axis=(0, 2), keepdims=True)
        std  = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
        X_tr = (X_tr - mean) / std
        np.save(MODEL_DIR / "norm_mean.npy", mean)
        np.save(MODEL_DIR / "norm_std.npy",  std)
        print("  Normalisation statistics saved.\n")

    use_sampler  = _use_weighted_sampler(y_tr, cfg.imbalance_threshold)
    class_weight = None if use_sampler else compute_class_weights(y_tr, cfg.device)
    sampler      = make_weighted_sampler(y_tr) if use_sampler else None

    train_loader = DataLoader(
        EEGDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model, optimizer, _ = _build_model_and_optimiser(cfg)
    ckpt_path = MODEL_DIR / "final_model.pt"
    t0        = time.time()

    epoch_pbar = tqdm(
        range(1, fixed_epochs + 1),
        desc="  Final model",
        unit="ep",
        bar_format=(
            "{desc} {n_fmt:>3}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        ),
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, cfg.device,
            epoch_pbar=epoch_pbar,
            class_weight=class_weight,
            grad_clip=cfg.grad_clip_norm,
        )
        epoch_pbar.set_postfix({"train_loss": f"{tr_loss:.4f}"}, refresh=True)

    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Final model trained in {_fmt_time(time.time() - t0)}")
    print(f"  Checkpoint : {ckpt_path}")
    cfg.save(MODEL_DIR / "config.json")
    return model


# ============================================================
# Plotting utilities
# ============================================================

def plot_learning_curves(
    train_losses_per_fold: List[List[float]],
    val_losses_per_fold:   List[List[float]],
    save_dir: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=False)
    palette   = sns.color_palette("muted", n_colors=len(train_losses_per_fold))

    for ax, (losses_per_fold, title) in zip(
        axes,
        [(train_losses_per_fold, "Train NLL Loss"),
         (val_losses_per_fold,   "Validation NLL Loss")]
    ):
        max_len = max(len(l) for l in losses_per_fold)
        matrix  = np.full((len(losses_per_fold), max_len), np.nan)
        for i, lv in enumerate(losses_per_fold):
            matrix[i, : len(lv)] = lv
            ax.plot(range(1, len(lv) + 1), lv,
                    color=palette[i], alpha=0.45, linewidth=1.2,
                    label=f"Fold {i + 1}")
        mean_curve = np.nanmean(matrix, axis=0)
        ax.plot(range(1, max_len + 1), mean_curve,
                color="black", linewidth=2.2, label="Mean", zorder=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL Loss")
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle("Learning curves — all folds", fontsize=13)
    fig.tight_layout()
    out = save_dir / "learning_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Learning curves -> {out}")


def plot_cv_summary(results_df: pd.DataFrame, save_dir: Path):
    """Strip plot: per-fold points + mean+/-SD for each metric."""
    metrics = {
        "Accuracy":          ("ep_acc",     "subj_acc"),
        "AUC-ROC":           ("ep_auc",     "subj_auc"),
        "Balanced Accuracy": ("ep_bal_acc", "subj_bal_acc"),
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    colours   = {"Epoch-level": "#4C72B0", "Subject-level": "#DD8452"}

    for ax, (title, (ep_col, subj_col)) in zip(axes, metrics.items()):
        for x_pos, (col, level) in enumerate(
            [(ep_col, "Epoch-level"), (subj_col, "Subject-level")]
        ):
            vals   = results_df[col].values
            colour = colours[level]
            jitter = (np.random.default_rng(SEED).random(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                       color=colour, s=60, zorder=3, alpha=0.85, label=level)
            m, s = np.nanmean(vals), np.nanstd(vals)
            ax.errorbar(x_pos, m, yerr=s, fmt="D", color=colour,
                        markersize=7, capsize=5, linewidth=2, zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Epoch-level", "Subject-level"])
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.45, label="Chance")
        ax.set_title(title)
        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle("Cross-validation results — EEGNet PD vs CTL", fontsize=13)
    fig.tight_layout()
    out = save_dir / "cv_summary.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  CV summary plot -> {out}")


def plot_confusion_matrix(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    save_dir: Path,
    title:    str = "Confusion Matrix",
):
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["CTL", "PD"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def print_cv_summary(results_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(f"{'CROSS-VALIDATION SUMMARY':^70}")
    print("=" * 70)
    header = (
        f"{'Fold':>5} | {'Ep Acc':>7} {'Ep AUC':>7} {'Ep Bal':>7} | "
        f"{'Su Acc':>7} {'Su AUC':>7} {'Su Bal':>7}"
    )
    print(header)
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(
            f"{int(row['fold']):>5} | "
            f"{row['ep_acc']:.4f}  {row['ep_auc']:.4f}  {row['ep_bal_acc']:.4f} | "
            f"{row['subj_acc']:.4f}  {row['subj_auc']:.4f}  {row['subj_bal_acc']:.4f}"
        )
    print("-" * 70)
    for col_ep, col_su, label in [
        ("ep_acc",     "subj_acc",     "Accuracy     "),
        ("ep_auc",     "subj_auc",     "AUC-ROC      "),
        ("ep_bal_acc", "subj_bal_acc", "Balanced Acc "),
    ]:
        ep_m, ep_s = results_df[col_ep].mean(), results_df[col_ep].std()
        su_m, su_s = results_df[col_su].mean(), results_df[col_su].std()
        print(
            f"{'Mean+/-SD':>5}   {label}: "
            f"epoch={ep_m:.4f}+/-{ep_s:.4f}  "
            f"subject={su_m:.4f}+/-{su_s:.4f}"
        )
    print("=" * 70 + "\n")


# ============================================================
# Inference utility
# ============================================================

def predict_subject(
    model:      "EEGNet",
    epochs_fif: Path,
    cfg:        EEGNetConfig,
    norm_mean:  Optional[np.ndarray] = None,
    norm_std:   Optional[np.ndarray] = None,
) -> Dict:
    """Classify a single subject from their .fif epoch file."""
    epochs = mne.read_epochs(str(epochs_fif), preload=True, verbose="warning")
    data   = epochs.get_data(picks="eeg").astype(np.float32)

    T = cfg.n_timepoints
    if data.shape[2] > T:
        data = data[:, :, :T]
    elif data.shape[2] < T:
        data = np.pad(data, ((0, 0), (0, 0), (0, T - data.shape[2])))

    if norm_mean is not None and norm_std is not None:
        data = (data - norm_mean) / norm_std

    dummy_labels = np.zeros(len(data), dtype=np.int64)
    loader = DataLoader(
        EEGDataset(data, dummy_labels),
        batch_size=64, shuffle=False, num_workers=cfg.num_workers,
    )
    probs = predict_proba(model, loader, cfg.device)

    subj_prob  = float(probs.mean())
    prediction = "PD" if subj_prob >= 0.5 else "CTL"
    confidence = abs(subj_prob - 0.5)

    return {
        "epoch_probs":  probs,
        "subject_prob": subj_prob,
        "prediction":   prediction,
        "confidence":   confidence,
    }


# ============================================================
# MAIN  —  all top-level side-effects live here so that
#          DataLoader worker processes (which re-import this
#          module) do not accidentally re-execute them.
# ============================================================

def main() -> None:
    set_seed(SEED)
    wall_t0 = time.time()

    # Print config once, here, not at module level
    _banner("EEGNet PD vs CTL — Configuration")
    print(f"  Device       : {CFG.device}")
    print(f"  Channels     : {CFG.n_channels}")
    print(f"  Timepoints   : {CFG.n_timepoints}  ({CFG.epoch_len_sec}s @ {CFG.sfreq}Hz)")
    print(f"  Architecture : F1={CFG.F1}  D={CFG.D}  F2={CFG.F2}")
    print(f"  Training     : lr={CFG.lr}  wd={CFG.weight_decay}  batch={CFG.batch_size}")
    print(f"  Epochs       : max={CFG.max_epochs}  min={CFG.min_epochs}  patience={CFG.patience}")
    print(f"  Scheduler    : ReduceLROnPlateau  factor={CFG.lr_factor}  patience={CFG.lr_patience}")
    print(f"  CV folds     : {CFG.n_folds}")
    print(f"  num_workers  : {CFG.num_workers}")

    # ── 1. Load ──────────────────────────────────────────────────────────
    X, y, groups = load_all_epochs(EPOCHS_DIR, SUMMARY_TSV)

    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        raise RuntimeError(
            "Only one class present in the dataset — cannot train a classifier."
        )

    # ── 2. Cross-validation ──────────────────────────────────────────────
    results_df, fold_train_losses, fold_val_losses = run_cross_validation(
        X, y, groups, CFG
    )
    print_cv_summary(results_df)
    plot_cv_summary(results_df, MODEL_DIR)
    plot_learning_curves(fold_train_losses, fold_val_losses, MODEL_DIR)

    # ── 3. Final model ───────────────────────────────────────────────────
    # FIXED: clamp to cfg.min_epochs so the final model always trains
    # for at least as long as the grace period. Previously mean best_epoch=2
    # produced a basically untrained model.
    raw_best     = max(1, int(round(results_df["best_epoch"].mean())))
    fixed_epochs = max(raw_best, CFG.min_epochs)
    print(f"\n  CV mean best_epoch = {results_df['best_epoch'].mean():.1f} "
          f"-> training final model for {fixed_epochs} epoch(s).")
    final_model = train_final_model(X, y, groups, CFG, fixed_epochs=fixed_epochs)

    # ── 4. Full-dataset evaluation ───────────────────────────────────────
    _banner("Step 4 / 4 — Final evaluation  (training set — use CV for reporting)")

    norm_mean = np.load(MODEL_DIR / "norm_mean.npy") if CFG.normalize else None
    norm_std  = np.load(MODEL_DIR / "norm_std.npy")  if CFG.normalize else None
    X_norm    = (X - norm_mean) / norm_std if norm_mean is not None else X

    full_loader = DataLoader(
        EEGDataset(X_norm, y),
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
    )
    print("  Running inference on full dataset...", end=" ", flush=True)
    _, _, ep_y, ep_probs = evaluate(final_model, full_loader, CFG.device)
    print("done.")

    ep_pred = (ep_probs >= 0.5).astype(int)
    print("\n  Epoch-level classification report:")
    print(classification_report(ep_y, ep_pred, target_names=["CTL", "PD"]))

    subj_y, subj_probs, _ = subject_level_predictions(ep_y, ep_probs, groups)
    subj_pred = (subj_probs >= 0.5).astype(int)
    print("  Subject-level classification report:")
    print(classification_report(subj_y, subj_pred, target_names=["CTL", "PD"]))

    plot_confusion_matrix(
        subj_y, subj_pred, MODEL_DIR,
        title="Subject-level Confusion Matrix (final model, training set)"
    )

    _banner("Pipeline complete")
    print(f"  Total wall time  : {_fmt_time(time.time() - wall_t0)}")
    print(f"  Model checkpoint : {MODEL_DIR / 'final_model.pt'}")
    print(f"  Config           : {MODEL_DIR / 'config.json'}")
    print(f"  CV results       : {MODEL_DIR / 'cv_results.tsv'}")
    if CFG.normalize:
        print(f"  Norm stats       : {MODEL_DIR / 'norm_mean.npy'}, norm_std.npy")


if __name__ == "__main__":
    main()
