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
matplotlib.use("Agg")          # non-interactive backend — swap to "TkAgg" or
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
OUT_DIR           = Path("Signal Code/Outputs")
EPOCHS_DIR        = OUT_DIR / "cache_epochs_binary"
SUMMARY_TSV       = OUT_DIR / "preprocessing_summary.tsv"
MODEL_DIR         = OUT_DIR / "eegnet_model"
# CHANGED: singletrial dataset built by preprocessing.py (Step4 equivalent)
SINGLETRIAL_PATH  = OUT_DIR / "SINGLETRIAL_ODDBALL.npz"
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

# set_seed() and banners are called inside main() only.
# DataLoader workers re-import this module; any module-level side-effects
# (prints, seed calls) would execute once per worker per fold.


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
# Config dataclass
# ============================================================
@dataclass
class EEGNetConfig:
    # ── Data ────────────────────────────────────────────────
    sfreq:          float = 100.0   # CHANGED: match MATLAB Step4 100 Hz
    n_channels:     int   = 16      # will be overridden when loading data
    epoch_len_sec:  float = 1.0

    # ── Group comparison (MATLAB Step5: subgrp) ──────────────
    # 'ON_CTL'  -> PD-ON  vs CTL  (sessioni 1 vs 3)
    # 'OFF_CTL' -> PD-OFF vs CTL  (sessioni 2 vs 3)
    # 'ON_OFF'  -> PD-ON  vs PD-OFF (sessioni 1 vs 2)
    comparison: str = "ON_CTL"

    # ── Stimulus condition (MATLAB Step5: Condi) ─────────────
    # 'N'=Novelty  'T'=Target  'S'=Standard
    condition: str = "N"

    # ── FFT feature extraction (MATLAB Step5: FFTlabel='FFT') ─
    # If True: classify on FFT power of the 250-1000ms window
    # If False: classify on raw EEG (legacy EEGNet mode)
    use_fft_features: bool = True

    # ── Use last third of trials (MATLAB Step8: FFT_Last3rd) ──
    # Set True to replicate the decremental / last-third analysis
    use_last_third: bool = False

    # ── Classification time window (MATLAB Step5: T1=250, T2=1000 ms) ──
    classify_t1_ms: int = 250
    classify_t2_ms: int = 1000

    # ── EEGNet architecture  (Lawhern et al. 2018) ──────────
    F1:              int   = 4      # CHANGED: was 8
    D:               int   = 2
    F2:              int   = 8      # CHANGED: was 16
    kernel_temporal: int   = 125
    dropout:         float = 0.7   # CHANGED: was 0.5

    # ── Training ────────────────────────────────────────────
    batch_size:     int   = 64
    lr:             float = 1e-4   # CHANGED: was 1e-3
    weight_decay:   float = 3e-3   # CHANGED: was 1e-4; folds 3+4 still diverge
    max_epochs:     int   = 200
    min_epochs:     int   = 25     # NEW: grace period before early stopping fires
    patience:       int   = 20     # CHANGED: was 30; stop diverging folds earlier
    grad_clip_norm: float = 1.0

    # ── LR scheduler: ReduceLROnPlateau ──────────────────────
    # CHANGED: replaces CosineAnnealingLR; only reduces LR when val stalls
    lr_patience:    int   = 8
    lr_factor:      float = 0.5
    lr_min:         float = 1e-6

    # ── Cross-validation ────────────────────────────────────
    n_folds: int = 5

    # ── Normalisation ────────────────────────────────────────
    # CHANGED: False — subject_wise_zscore() replaces per-fold channel z-score
    normalize: bool = False

    # ── Augmentation ─────────────────────────────────────────
    # NEW: Gaussian noise + channel dropout on training data only
    augment_noise_std:  float = 0.1   # CHANGED: was 0.05
    channel_drop_prob:  float = 0.1   # NEW: randomly zero entire channels per batch

    # ── Label smoothing ──────────────────────────────────────
    # NEW: prevents the model becoming overconfident on training subjects
    label_smoothing:    float = 0.1

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


# CFG at module level so helpers can reference it; banners only in main().
CFG = EEGNetConfig()


# ============================================================
# Data loading
# ============================================================

def load_all_epochs(
    epochs_dir: Path,
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

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    g_list: List[np.ndarray] = []
    subject_index_map: Dict[str, int] = {}
    unlabelled:       List[str] = []
    skipped_channels: List[str] = []

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
            pbar.write(f"  ⚠  sub-{subj}: only {len(eeg_picks)} EEG channels — skipping")
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

    print(f"\n  ✅ Loading complete in {_fmt_time(time.time() - t0)}")
    print(f"  Subjects loaded  : {len(subject_index_map)}"
          + (f"  ({len(skipped_channels)} skipped — too few channels)"
             if skipped_channels else ""))
    print(f"  Total epochs     : {X.shape[0]}  (PD={n_pd}, CTL={n_ctl})")
    print(f"  Imbalance ratio  : {ratio:.2f}  "
          f"({'balanced' if ratio < 1.5 else 'imbalanced — sampler will be used'})")
    print(f"  Data tensor      : {X.shape}  →  {X.nbytes / 1e6:.1f} MB")

    return X, y, groups


# ============================================================
# Normalisation
# ============================================================

def channel_wise_zscore(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
    X_train_z = (X_train - mean) / std
    X_val_z   = (X_val   - mean) / std
    X_test_z  = (X_test  - mean) / std if X_test is not None else None
    return X_train_z, X_val_z, X_test_z, mean, std


# ============================================================
# Subject-wise normalisation (NEW)
# ============================================================

def subject_wise_zscore(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Normalise each subject independently before any train/val split.

    Why: the old per-fold channel z-score left between-subject amplitude
    differences intact, giving the model a shortcut — it could classify by
    recognising a subject's amplitude level rather than PD vs CTL signal.
    Per-subject normalisation removes that shortcut entirely.

    Applied once in main() before the CV loop.
    """
    X_out = X.copy()
    for sid in np.unique(groups):
        mask        = groups == sid
        mean        = X[mask].mean(axis=(0, 2), keepdims=True)
        std         = X[mask].std(axis=(0, 2),  keepdims=True) + 1e-8
        X_out[mask] = (X[mask] - mean) / std
    return X_out


# ============================================================
# Loss with label smoothing (NEW)
# ============================================================

def label_smoothed_nll_loss(
    log_probs: torch.Tensor,
    targets:   torch.Tensor,
    smoothing: float = 0.1,
    weight:    Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NLL loss with label smoothing.

    Why: with only 50 subjects the model can become overconfident on training
    subjects, pushing log-probabilities to near-zero for the true class and
    producing a very small loss even when generalisation is poor.  Label
    smoothing replaces hard 0/1 targets with:
        smooth_target = (1 - eps) * one_hot + eps / n_classes
    which prevents the loss from collapsing and acts as regularisation.

    Implemented as a convex combination of:
      - standard NLL (true class)
      - uniform NLL (average over all classes)
    so no change to the model architecture is needed.
    """
    n_classes   = log_probs.size(1)
    nll         = F.nll_loss(log_probs, targets, weight=weight, reduction="none")
    smooth      = -log_probs.mean(dim=1)
    loss        = (1.0 - smoothing) * nll + smoothing * smooth
    return loss.mean()


# ============================================================
# PyTorch Datasets
# ============================================================

class EEGDataset(Dataset):
    """Clean dataset — used for validation and inference (no augmentation)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class EEGDatasetAugmented(Dataset):
    """
    Training dataset with two complementary augmentations:

    1. Gaussian noise (noise_std): adds small amplitude jitter to every sample.
       Prevents memorising exact amplitude patterns of individual subjects.

    2. Channel dropout (channel_drop_prob): randomly zeroes entire EEG channels
       per forward pass.  Forces the model to rely on distributed spatial
       patterns rather than a fixed subset of channels — critical for folds
       3 and 4 where the model latches onto channel-specific artefacts.

    Neither augmentation is applied to the validation or inference loaders.
    """
    def __init__(
        self,
        X:                np.ndarray,
        y:                np.ndarray,
        noise_std:        float = 0.1,
        channel_drop_prob: float = 0.1,
    ):
        self.X                 = torch.from_numpy(X).unsqueeze(1)  # (N,1,C,T)
        self.y                 = torch.from_numpy(y).long()
        self.noise_std         = noise_std
        self.channel_drop_prob = channel_drop_prob

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()                   # (1, C, T)

        # 1. Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # 2. Channel dropout: zero out entire channels independently
        if self.channel_drop_prob > 0:
            n_ch = x.shape[1]                     # C
            keep = torch.bernoulli(
                torch.full((n_ch,), 1.0 - self.channel_drop_prob)
            )                                     # (C,)
            x = x * keep.view(1, n_ch, 1)         # broadcast over time

        return x, self.y[idx]


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts            = np.bincount(y)
    weights_per_class = 1.0 / counts
    sample_weights    = weights_per_class[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(y),
        replacement=True
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
    Stops when val loss has not improved for `patience` epochs,
    but never before `min_epochs` have completed (grace period).
    """
    def __init__(self, patience: int, save_path: Path,
                 min_epochs: int = 0, min_delta: float = 1e-4):
        self.patience   = patience
        self.save_path  = save_path
        self.min_epochs = min_epochs
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
        return (epoch >= self.min_epochs) and (self.counter >= self.patience)


def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    device:       str,
    epoch_pbar:   tqdm,
    class_weight: Optional[torch.Tensor] = None,
    grad_clip:    float = 1.0,
    cfg:          Optional["EEGNetConfig"] = None,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        log_probs = model(X_batch)
        loss      = label_smoothed_nll_loss(
            log_probs, y_batch,
            smoothing=cfg.label_smoothing if hasattr(cfg, "label_smoothing") else 0.1,
            weight=class_weight,
        )
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
    groups:      np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_subjects = np.unique(groups)
    subj_y     = np.empty(len(unique_subjects), dtype=np.int64)
    subj_probs = np.empty(len(unique_subjects), dtype=np.float32)
    for i, s in enumerate(unique_subjects):
        mask           = groups == s
        subj_y[i]     = int(np.round(y_epoch[mask].mean()))
        subj_probs[i]  = probs_epoch[mask].mean()
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
    sampler = make_weighted_sampler(y_tr) if use_sampler else None
    # CHANGED: train uses EEGDatasetAugmented; val stays clean.
    train_loader = DataLoader(
        EEGDatasetAugmented(
            X_tr, y_tr,
            noise_std=cfg.augment_noise_std,
            channel_drop_prob=cfg.channel_drop_prob,
        ),
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        EEGDataset(X_va, y_va),   # no augmentation on validation
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    return train_loader, val_loader


def _build_model_and_optimiser(
    cfg: EEGNetConfig,
) -> Tuple[EEGNet, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model     = EEGNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # CHANGED: ReduceLROnPlateau — only reduces LR when val loss actually stalls.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor,
        patience=cfg.lr_patience, min_lr=cfg.lr_min,
    )
    return model, optimizer, scheduler


def _run_training_loop(
    model:        EEGNet,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.LRScheduler,
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
        min_epochs=cfg.min_epochs,
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
            cfg=cfg,
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, cfg.device)
        scheduler.step(va_loss)      # ReduceLROnPlateau steps on val loss
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
                f"  stop  {fold_label}epoch {epoch} "
                f"(best={early_stop.best_epoch}, "
                f"val_loss={early_stop.best_loss:.4f})"
            )
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    return model, early_stop, train_losses, val_losses


# ============================================================
# Cross-validation
# ============================================================

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: EEGNetConfig
) -> Tuple[pd.DataFrame, List[List[float]], List[List[float]]]:
    _banner("Step 2 / 4 — Cross-validation")
    print(f"  Strategy : {cfg.n_folds}-fold StratifiedGroupKFold")
    print(f"  Subjects : {len(np.unique(groups))}")
    print(f"  Epochs   : {len(y)}  (PD={int((y==1).sum())}, CTL={int((y==0).sum())})\n")

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
            print(f"  Normalisation : channel-wise z-score (fitted on train)")

        use_sampler  = _use_weighted_sampler(y_tr, cfg.imbalance_threshold)
        class_weight = None if use_sampler else compute_class_weights(y_tr, device)

        if use_sampler:
            counts = np.bincount(y_tr)
            print(f"  Imbalance     : ratio={counts.max()/counts.min():.2f} "
                  f"→ WeightedRandomSampler")
        else:
            print(f"  Imbalance     : mild → class-weighted NLL loss")

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
            "fold":          fold,
            "best_epoch":    early_stop.best_epoch,
            "n_val_epochs":  len(ep_y),
            "n_val_subjs":   len(subj_y),
            "ep_acc":        ep_acc,
            "ep_auc":        ep_auc,
            "ep_bal_acc":    ep_bal,
            "subj_acc":      subj_acc,
            "subj_auc":      subj_auc,
            "subj_bal_acc":  subj_bal,
        })

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(MODEL_DIR / "cv_results.tsv", sep="\t", index=False)
    print(f"\n  ✅ Cross-validation complete in {_fmt_time(time.time() - cv_t0)}")
    return results_df, fold_train_losses, fold_val_losses


# ============================================================
# Final model
# ============================================================

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: EEGNetConfig,
    fixed_epochs: int,
) -> "EEGNet":
    _banner("Step 3 / 4 — Final model training")
    print(f"  Subjects : {len(np.unique(groups))}")
    print(f"  Epochs   : {len(y)}  (PD={int((y==1).sum())}, CTL={int((y==0).sum())})")
    print(f"  Duration : {fixed_epochs} epochs  (CV mean best_epoch)\n")

    X_tr, y_tr = X, y

    # Save norm stats for inference (subject_wise_zscore already applied to X)
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std  = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
    np.save(MODEL_DIR / "norm_mean.npy", mean)
    np.save(MODEL_DIR / "norm_std.npy",  std)
    print("  Normalisation statistics saved.\n")

    if cfg.normalize:
        X_tr = (X_tr - mean) / std

    use_sampler  = _use_weighted_sampler(y_tr, cfg.imbalance_threshold)
    class_weight = None if use_sampler else compute_class_weights(y_tr, cfg.device)
    sampler      = make_weighted_sampler(y_tr) if use_sampler else None

    train_loader = DataLoader(
        EEGDatasetAugmented(
            X_tr, y_tr,
            noise_std=cfg.augment_noise_std,
            channel_drop_prob=cfg.channel_drop_prob,
        ),
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model, optimizer, scheduler = _build_model_and_optimiser(cfg)
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
            cfg=cfg,
        )
        scheduler.step()
        epoch_pbar.set_postfix({"train_loss": f"{tr_loss:.4f}"}, refresh=True)

    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  ✅ Final model trained in {_fmt_time(time.time() - t0)}")
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
    print(f"  Learning curves → {out}")


def plot_cv_summary(results_df: pd.DataFrame, save_dir: Path):
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
    print(f"  CV summary plot → {out}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
    title: str = "Confusion Matrix"
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
            f"{'Mean±SD':>5}   {label}: "
            f"epoch={ep_m:.4f}±{ep_s:.4f}  "
            f"subject={su_m:.4f}±{su_s:.4f}"
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
# MAIN
# ============================================================

def main() -> None:
    # All side-effects here — not at module level — so DataLoader workers
    # (which re-import this module) do not repeat them.
    set_seed(SEED)
    wall_t0 = time.time()

    _banner("EEGNet PD vs CTL — Configuration")
    print(f"  Device          : {CFG.device}")
    print(f"  Architecture    : F1={CFG.F1}  D={CFG.D}  F2={CFG.F2}  dropout={CFG.dropout}")
    print(f"  Training        : lr={CFG.lr}  wd={CFG.weight_decay}  batch={CFG.batch_size}")
    print(f"  Epochs          : max={CFG.max_epochs}  min={CFG.min_epochs}  patience={CFG.patience}")
    print(f"  Scheduler       : ReduceLROnPlateau  factor={CFG.lr_factor}  patience={CFG.lr_patience}")
    print(f"  Augmentation    : noise={CFG.augment_noise_std}  ch_drop={CFG.channel_drop_prob}")
    print(f"  Label smoothing : {CFG.label_smoothing}")
    print(f"  Normalisation   : subject-wise z-score (pre-split)")
    print(f"  CV folds        : {CFG.n_folds}")
    print(f"  num_workers     : {CFG.num_workers}")

    # ── 1. Load ─────────────────────────────────────────────────────────
    X, y, groups = load_all_epochs(EPOCHS_DIR, SUMMARY_TSV)

    # Update model dims to match loaded data
    CFG.n_channels    = X.shape[1]
    CFG.epoch_len_sec = X.shape[2] / CFG.sfreq

    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        raise RuntimeError(
            "Only one class present in the dataset — cannot train a classifier."
        )

    # ── Subject-wise normalisation (once, before any splitting) ──────────
    # Removes between-subject amplitude differences so the model cannot use
    # subject identity as a classification shortcut.
    print("\n  Applying subject-wise z-score normalisation...")
    X = subject_wise_zscore(X, groups)
    print(f"  Done. Data mean: {X.mean():.4f}  std: {X.std():.4f}")

    # ── 2. Cross-validation ─────────────────────────────────────────────
    results_df, fold_train_losses, fold_val_losses = run_cross_validation(
        X, y, groups, CFG
    )
    print_cv_summary(results_df)
    plot_cv_summary(results_df, MODEL_DIR)
    plot_learning_curves(fold_train_losses, fold_val_losses, MODEL_DIR)

    # ── 3. Final model ──────────────────────────────────────────────────
    # Clamp to min_epochs so the final model is never effectively untrained.
    raw_best     = max(1, int(round(results_df["best_epoch"].mean())))
    fixed_epochs = max(raw_best, CFG.min_epochs)
    print(f"\n  CV mean best_epoch = {results_df['best_epoch'].mean():.1f} "
          f"-> training final model for {fixed_epochs} epoch(s).")
    final_model = train_final_model(X, y, groups, CFG, fixed_epochs=fixed_epochs)

    # ── 4. Full-dataset evaluation (training set — use CV for reporting) ──
    _banner("Step 4 / 4 — Final evaluation  (training set — use CV for reporting)")

    # X is already subject-wise normalised
    full_loader = DataLoader(
        EEGDataset(X, y),   # clean dataset, no augmentation
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
    print(f"  Norm stats       : {MODEL_DIR / 'norm_mean.npy'}, norm_std.npy")


if __name__ == "__main__":
    main()


# ============================================================
# NEW: FFT feature extraction — MATLAB Step5 / Step8 equivalent
# ============================================================

def extract_fft_features(
    data: np.ndarray,
    tx_ms: np.ndarray,
    t1_ms: int,
    t2_ms: int,
    sfreq: float = 100.0,
) -> np.ndarray:
    """
    Compute FFT power of each trial in the [t1_ms, t2_ms] window.
    Replicates MATLAB Step5 / Step8 FFT feature extraction.

    MATLAB Step8 pattern:
        fft_coeff = fft(elem{si}(mi,:,ni), dims(2));
        EndLim    = min(dims(2), samplingrate/2);
        pwr       = abs(fft_coeff(1:EndLim)) .* 2;

    Args:
        data   : (n_channels, n_times, n_trials)
        tx_ms  : time axis in ms (length n_times)
        t1_ms  : start of classification window in ms
        t2_ms  : end   of classification window in ms
        sfreq  : sampling rate in Hz

    Returns:
        features : (n_channels, n_freqs, n_trials)  float32
                   n_freqs = int(sfreq / 2)
    """
    t1_idx = int(np.argmin(np.abs(tx_ms - t1_ms)))
    t2_idx = int(np.argmin(np.abs(tx_ms - t2_ms)))
    window = data[:, t1_idx:t2_idx+1, :]   # (C, T_win, N)

    n_win   = window.shape[1]
    n_freqs = int(sfreq / 2)               # Step8: EndLim = min(dims(2), samplingrate/2)

    features = np.empty((window.shape[0], n_freqs, window.shape[2]), dtype=np.float32)
    for ci in range(window.shape[0]):
        for ni in range(window.shape[2]):
            fft_c = np.fft.rfft(window[ci, :, ni], n=n_win)
            pwr   = np.abs(fft_c[:n_freqs]) * 2   # Step8: abs(...) .* 2
            features[ci, :, ni] = pwr.astype(np.float32)

    return features   # (C, n_freqs, N)


# ============================================================
# NEW: SVM-based classifier — MATLAB Step5 CLASSIFY() equivalent
# ============================================================

def classify_svm(
    elem1: np.ndarray,
    elem2: np.ndarray,
    n_splits: int = 5,
) -> Tuple[float, float]:
    """
    Train a linear SVM to discriminate elem1 vs elem2 trials for one
    subject pair.  Mirrors the MATLAB CLASSIFY() function called in Step5.

    MATLAB result: per-subject accuracy for each condition (elem1_acc, elem2_acc).

    Args:
        elem1 : (n_features, n_trials_1)  — flattened FFT features for group 1
        elem2 : (n_features, n_trials_2)  — flattened FFT features for group 2

    Returns:
        (acc_elem1, acc_elem2) — mean sensitivity and specificity
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    X = np.concatenate([elem1.T, elem2.T], axis=0)   # (n1+n2, n_features)
    y = np.concatenate([np.zeros(elem1.shape[1]),
                        np.ones(elem2.shape[1])]).astype(int)

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])

    cv   = StratifiedKFold(n_splits=min(n_splits, min(np.bincount(y))))
    acc1 = []
    acc2 = []
    for train_idx, test_idx in cv.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        y_test = y[test_idx]
        # Sensitivity (elem1 accuracy) and specificity (elem2 accuracy)
        acc1.append(float((preds[y_test==0]==0).mean()) if (y_test==0).any() else float("nan"))
        acc2.append(float((preds[y_test==1]==1).mean()) if (y_test==1).any() else float("nan"))

    return float(np.nanmean(acc1)), float(np.nanmean(acc2))


# ============================================================
# NEW: Load singletrial dataset — mirrors MATLAB Step5 data load
# ============================================================

def load_singletrial_data(
    path:        Path,
    comparison:  str,
    condition:   str,
    use_last_third: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Load condition + comparison subset from SINGLETRIAL_ODDBALL.npz.

    comparison : 'ON_CTL' | 'OFF_CTL' | 'ON_OFF'
    condition  : 'N' (Novelty) | 'T' (Target) | 'S' (Standard)
    use_last_third : if True, only use the last third of trials (Step8)

    Returns:
        elem1 : list of length n_subjects, each (C, T, n_trials_e1)
        elem2 : list of length n_subjects, each (C, T, n_trials_e2)
        tx    : time axis in ms
    """
    _banner(f"Loading singletrial data  |  {comparison}  |  condition={condition}"
            + ("  [last 1/3 trials]" if use_last_third else ""))

    data = np.load(str(path), allow_pickle=True)
    slot_labels = list(data["slot_labels"])   # ["ON","OFF","CTL"]
    tx          = data["tx"]

    # Slot index mapping (mirrors MATLAB Step5 e1/e2 logic)
    if comparison == "ON_CTL":
        e1_slot, e2_slot = slot_labels.index("ON"),  slot_labels.index("CTL")
    elif comparison == "OFF_CTL":
        e1_slot, e2_slot = slot_labels.index("OFF"), slot_labels.index("CTL")
    elif comparison == "ON_OFF":
        e1_slot, e2_slot = slot_labels.index("ON"),  slot_labels.index("OFF")
    else:
        raise ValueError(f"Unknown comparison: {comparison}")

    # Condition arrays — TRI_ version used when use_last_third=True
    cond_key = {"N": "N", "T": "T", "S": "S"}[condition]
    if use_last_third:
        # Last third = third_idx=2 (Step8: A_THIRD*2+1:end)
        base_arr   = data[f"TRI_{cond_key}"].item()
        def _get(store, pair_idx, slot_idx):
            if pair_idx not in store or slot_idx not in store[pair_idx]:
                return None
            return store[pair_idx][slot_idx].get(2, None)   # third_idx=2
    else:
        base_arr   = data[cond_key].item()
        def _get(store, pair_idx, slot_idx):
            if pair_idx not in store or slot_idx not in store[pair_idx]:
                return None
            return store[pair_idx][slot_idx]

    n_pairs = len(base_arr)
    elem1: List[np.ndarray] = []
    elem2: List[np.ndarray] = []
    skipped = 0

    for pair_idx in range(n_pairs):
        d1 = _get(base_arr, pair_idx, e1_slot)
        d2 = _get(base_arr, pair_idx, e2_slot)
        if d1 is None or d2 is None or d1.shape[2] == 0 or d2.shape[2] == 0:
            skipped += 1
            continue
        elem1.append(d1)
        elem2.append(d2)

    print(f"  Subject pairs loaded : {len(elem1)}  ({skipped} skipped — missing data)")
    print(f"  Shape example        : {elem1[0].shape if elem1 else 'N/A'}")
    print(f"  Time axis            : {tx[0]:.0f}ms to {tx[-1]:.0f}ms  ({len(tx)} samples at {1000/(tx[1]-tx[0]):.0f} Hz)")

    return elem1, elem2, tx


# ============================================================
# NEW: Run FFT + SVM classification — MATLAB Step5 CLASSIFY() loop
# ============================================================

def run_fft_svm_classification(
    elem1:  List[np.ndarray],
    elem2:  List[np.ndarray],
    tx:     np.ndarray,
    cfg:    EEGNetConfig,
) -> Tuple[List[float], List[float]]:
    """
    For each subject pair: extract FFT features then classify with SVM.
    Replicates MATLAB Step5 [elem1_acc, elem2_acc] = CLASSIFY(...).

    Returns:
        acc1 : list of sensitivity  values (one per subject pair)
        acc2 : list of specificity  values
    """
    _banner("FFT + SVM classification  (Step5 equivalent)")

    t1 = cfg.classify_t1_ms
    t2 = cfg.classify_t2_ms
    print(f"  Comparison   : {cfg.comparison}")
    print(f"  Condition    : {cfg.condition}")
    print(f"  Time window  : {t1}–{t2} ms")
    print(f"  N pairs      : {len(elem1)}")

    acc1_list: List[float] = []
    acc2_list: List[float] = []

    for si in tqdm(range(len(elem1)), desc="  Classifying pairs"):
        d1 = elem1[si]   # (C, T, N1)
        d2 = elem2[si]   # (C, T, N2)

        # FFT features (Step5 FFT extraction)
        f1 = extract_fft_features(d1, tx, t1, t2, sfreq=cfg.sfreq)  # (C, F, N1)
        f2 = extract_fft_features(d2, tx, t1, t2, sfreq=cfg.sfreq)  # (C, F, N2)

        # Flatten channels × freqs for SVM input (Step8: FeatureKiller is C×F)
        n_c, n_f, n1 = f1.shape
        n_c2, n_f2, n2 = f2.shape
        feat1 = f1.reshape(n_c * n_f, n1)   # (n_features, N1)
        feat2 = f2.reshape(n_c2 * n_f2, n2) # (n_features, N2)

        a1, a2 = classify_svm(feat1, feat2)
        acc1_list.append(a1)
        acc2_list.append(a2)

    # Summary
    a1_arr = np.array([a for a in acc1_list if not np.isnan(a)])
    a2_arr = np.array([a for a in acc2_list if not np.isnan(a)])
    print(f"\n  Sensitivity  (elem1): {np.mean(a1_arr):.4f} +/- {np.std(a1_arr):.4f}")
    print(f"  Specificity  (elem2): {np.mean(a2_arr):.4f} +/- {np.std(a2_arr):.4f}")
    print(f"  Mean accuracy       : {(np.mean(a1_arr)+np.mean(a2_arr))/2:.4f}")

    return acc1_list, acc2_list


# ============================================================
# NEW: Decremental feature importance — MATLAB Step8 equivalent
# ============================================================

def run_feature_importance(
    elem1:      List[np.ndarray],
    elem2:      List[np.ndarray],
    tx:         np.ndarray,
    cfg:        EEGNetConfig,
    n_perms:    int = 50,
    keep_thresh: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decremental feature selection across N permutations.
    Replicates MATLAB Step8 FeatureKiller logic.

    For each permutation:
      1. Start with all features active.
      2. Shuffle feature order.
      3. Try removing each feature in that order; keep it removed only
         if accuracy stays >= keep_thresh * full_accuracy.
      4. Record the surviving feature mask and final accuracy.

    The mean feature mask across permutations shows which channel x freq
    combinations are most important for classification.

    Returns:
        META_FEATURES : (n_perms, n_features)  bool
        META_ACCURACY : (n_perms,)              float
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    _banner(f"Decremental feature importance  ({n_perms} permutations)  [Step8]")

    t1 = cfg.classify_t1_ms
    t2 = cfg.classify_t2_ms

    # Pre-compute FFT features for all subject pairs
    print("  Pre-computing FFT features...")
    feats1 = []
    feats2 = []
    for si in range(len(elem1)):
        f1 = extract_fft_features(elem1[si], tx, t1, t2, sfreq=cfg.sfreq)
        f2 = extract_fft_features(elem2[si], tx, t1, t2, sfreq=cfg.sfreq)
        n_c, n_f = f1.shape[0], f1.shape[1]
        feats1.append(f1.reshape(n_c * n_f, -1))
        feats2.append(f2.reshape(n_c * n_f, -1))
    n_features = feats1[0].shape[0]
    print(f"  Features per subject : {n_features}  "
          f"({n_c} channels x {n_f} freqs)")

    def _classify_with_mask(mask: np.ndarray) -> float:
        """Average accuracy across all subject pairs with feature mask applied."""
        accs = []
        for feat1, feat2 in zip(feats1, feats2):
            X = np.concatenate([feat1[mask].T, feat2[mask].T], axis=0)
            y = np.concatenate([np.zeros(feat1.shape[1]),
                                 np.ones(feat2.shape[1])]).astype(int)
            if len(np.unique(y)) < 2 or X.shape[1] == 0:
                continue
            pipe = Pipeline([("sc", StandardScaler()),
                              ("sv", LinearSVC(C=1.0, max_iter=1000, random_state=42))])
            cv   = StratifiedKFold(n_splits=min(5, min(np.bincount(y))))
            fold_acc = []
            for tr, te in cv.split(X, y):
                pipe.fit(X[tr], y[tr])
                fold_acc.append((pipe.predict(X[te]) == y[te]).mean())
            accs.append(np.mean(fold_acc))
        return float(np.mean(accs)) if accs else 0.0

    rng = np.random.default_rng(42)
    META_FEATURES = np.ones((n_perms, n_features), dtype=bool)
    META_ACCURACY = np.zeros(n_perms, dtype=float)

    for permi in tqdm(range(n_perms), desc="  Permutations"):
        perm_order  = rng.permutation(n_features)
        feat_killer = np.ones(n_features, dtype=bool)
        total_acc   = _classify_with_mask(feat_killer)

        for fki in perm_order:
            feat_killer[fki] = False
            new_acc = _classify_with_mask(feat_killer)
            if new_acc < keep_thresh * total_acc:
                feat_killer[fki] = True   # put it back

        final_acc               = _classify_with_mask(feat_killer)
        META_FEATURES[permi]    = feat_killer
        META_ACCURACY[permi]    = final_acc

    print(f"\n  Mean predictors kept : {META_FEATURES.sum(1).mean():.1f} / {n_features}")
    print(f"  Mean accuracy        : {META_ACCURACY.mean():.4f} +/- {META_ACCURACY.std():.4f}")

    return META_FEATURES, META_ACCURACY


def plot_feature_importance(
    META_FEATURES: np.ndarray,
    META_ACCURACY: np.ndarray,
    n_channels:    int,
    sfreq:         float,
    save_dir:      Path,
):
    """
    Visualise decremental feature importance by frequency band.
    Replicates MATLAB Step8 MAPS / band plots.

    Bands (Step8): Delta 0-3Hz, Theta 4-7Hz, Alpha 8-12Hz, Beta 13-30Hz, Gamma 31-50Hz
    """
    n_freqs      = int(sfreq / 2)
    hz           = np.linspace(0, sfreq / 2, n_freqs, endpoint=False)
    ForMaps      = META_FEATURES.T.reshape(n_channels, n_freqs, -1).mean(axis=2)  # (C, F)

    # Frequency band masks (Step8)
    bands = {
        "Delta":  (hz >= 0)  & (hz <= 3),
        "Theta":  (hz >  3)  & (hz <= 7),
        "Alpha":  (hz >  7)  & (hz <= 12),
        "Beta":   (hz >  12) & (hz <= 30),
        "Gamma":  (hz >  30) & (hz <= 50),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].boxplot(META_FEATURES.sum(axis=1))
    axes[0].set_title("# Predictors kept")
    axes[0].set_ylim(0, META_FEATURES.shape[1])

    axes[1].boxplot(META_ACCURACY)
    axes[1].set_title("Accuracy per permutation")
    axes[1].set_ylim(0.5, 1.0)

    band_means = np.array([ForMaps[:, mask].mean(axis=1) for mask in bands.values()])
    for bi, (band_name, _) in enumerate(bands.items()):
        axes[2].plot(band_means[bi], label=band_name)
    axes[2].set_xlabel("Channel index")
    axes[2].set_ylabel("Mean feature retention")
    axes[2].set_title("Feature importance by band")
    axes[2].legend(fontsize=7)

    fig.suptitle("Decremental feature importance (Step8)")
    fig.tight_layout()
    out = save_dir / "feature_importance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Feature importance plot -> {out}")
