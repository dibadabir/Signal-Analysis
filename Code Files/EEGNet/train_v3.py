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
    balanced_accuracy_score, f1_score, precision_score, recall_score
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

    # ── Group comparison ─────────────────────────────────────
    # 'ON_CTL'  -> PD-ON  vs CTL
    # 'OFF_CTL' -> PD-OFF vs CTL   ← active
    # 'ON_OFF'  -> PD-ON  vs PD-OFF
    comparison: str = "OFF_CTL"

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
    F1:              int   = 4
    D:               int   = 2
    F2:              int   = 8
    kernel_temporal: int   = 125
    dropout:         float = 0.7

    # ── Training ────────────────────────────────────────────
    batch_size:     int   = 64
    lr:             float = 3e-4   # FIXED: was 1e-4 — too slow; train barely improved
    weight_decay:   float = 1e-3   # FIXED: was 3e-3 — too strong; train stuck at ~chance
    max_epochs:     int   = 200
    min_epochs:     int   = 20     # FIXED: was 25
    patience:       int   = 25     # FIXED: was 20 — give good folds more room
    grad_clip_norm: float = 1.0

    # ── LR scheduler: ReduceLROnPlateau ──────────────────────
    lr_patience:    int   = 6      # FIXED: was 8 — respond faster to stalling
    lr_factor:      float = 0.5
    lr_min:         float = 1e-6

    # ── Cross-validation ────────────────────────────────────
    n_folds: int = 5

    # ── Normalisation ────────────────────────────────────────
    normalize: bool = False   # subject_wise_zscore applied pre-split in main()

    # ── Augmentation ─────────────────────────────────────────
    # FIXED: reduced noise and channel-drop — previous values were too aggressive,
    # preventing folds 3-5 from learning any signal at all.
    augment_noise_std:  float = 0.05  # was 0.1
    channel_drop_prob:  float = 0.05  # was 0.1

    # ── Mixup ────────────────────────────────────────────────
    # NEW: EEG Mixup — interpolates pairs of training epochs to create
    # virtual subjects never seen before.  Particularly effective for EEG
    # because neural signals are linear superpositions (physically meaningful).
    # alpha=0.2 -> Beta(0.2,0.2) mostly produces λ near 0 or 1 (mild mixing).
    # Set to 0.0 to disable.
    mixup_alpha:        float = 0.2

    # ── Label smoothing ──────────────────────────────────────
    # FIXED: was 0.1 — too high; made training loss plateau too early.
    label_smoothing:    float = 0.05

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
    epochs_dir:  Path,
    summary_tsv: Optional[Path] = None,
    comparison:  str = "OFF_CTL",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load epoch files for the requested comparison.

    comparison='OFF_CTL'  ->  sub-XXX_OFF-epo.fif (label 1) + sub-XXX_CTL-epo.fif (label 0)
    comparison='ON_CTL'   ->  sub-XXX_ON-epo.fif  (label 1) + sub-XXX_CTL-epo.fif (label 0)
    comparison='ON_OFF'   ->  sub-XXX_ON-epo.fif  (label 1) + sub-XXX_OFF-epo.fif (label 0)

    Only files whose med_status matches the comparison are loaded.
    """
    _banner(f"Step 1 / 4 -- Loading preprocessed epochs  [{comparison}]")

    _label_for = {
        "OFF_CTL": {"OFF": 1, "CTL": 0},
        "ON_CTL":  {"ON":  1, "CTL": 0},
        "ON_OFF":  {"ON":  1, "OFF": 0},
    }
    if comparison not in _label_for:
        raise ValueError(
            f"Unknown comparison '{comparison}'. Choose from: {list(_label_for)}"
        )
    med_label_map = _label_for[comparison]
    wanted_meds   = set(med_label_map.keys())
    print(f"  Comparison       : {comparison}")
    print(f"  Label mapping    : {med_label_map}")

    if not (summary_tsv and summary_tsv.exists()):
        raise FileNotFoundError(
            f"Preprocessing summary not found at {summary_tsv}. "
            "Re-run preprocessing.py first."
        )
    df = pd.read_csv(summary_tsv, sep="\t", dtype={"subject": str})
    missing_cols = {"subject", "group", "med_status"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"preprocessing_summary.tsv missing columns: {missing_cols}. "
            "Re-run preprocessing.py to regenerate it with med_status."
        )

    subj_med_to_label: Dict[Tuple[str, str], int] = {}
    for _, row in df.iterrows():
        med = str(row["med_status"]).strip().upper()
        if med in wanted_meds:
            subj_med_to_label[(str(row["subject"]), med)] = med_label_map[med]

    if not subj_med_to_label:
        raise RuntimeError(
            f"No rows in summary TSV matched comparison '{comparison}'. "
            "Check that med_status values are ON, OFF, or CTL."
        )

    # Glob only the relevant .fif files
    fif_files: List[Path] = []
    for med in wanted_meds:
        fif_files.extend(sorted(epochs_dir.glob(f"sub-*_{med}-epo.fif")))
    fif_files = sorted(set(fif_files))

    if not fif_files:
        raise FileNotFoundError(
            f"No epoch files found for comparison '{comparison}' in {epochs_dir}.\n"
            f"Expected: sub-*_[{'|'.join(sorted(wanted_meds))}]-epo.fif\n"
            "Re-run preprocessing.py first."
        )
    print(f"  Epoch files found: {len(fif_files)}  ({', '.join(sorted(wanted_meds))})")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    g_list: List[np.ndarray] = []
    subject_index_map: Dict[str, int] = {}
    skipped_channels:  List[str]      = []

    t0   = time.time()
    pbar = tqdm(
        fif_files,
        desc="  Loading .fif files",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for fif in pbar:
        # Filename: sub-{subj}_{MED}-epo.fif  e.g. sub-804_OFF-epo.fif
        stem  = fif.stem                              # "sub-804_OFF-epo"
        parts = stem.replace("-epo", "").split("_")   # ["sub-804", "OFF"]
        if len(parts) < 2:
            pbar.write(f"  Unexpected filename: {fif.name} -- skipping")
            continue
        subj = parts[0].replace("sub-", "")
        med  = parts[1].upper()

        if (subj, med) not in subj_med_to_label:
            continue

        label  = subj_med_to_label[(subj, med)]
        epochs = mne.read_epochs(str(fif), preload=True, verbose="warning")

        eeg_picks = mne.pick_types(epochs.info, eeg=True)
        if len(eeg_picks) < 8:
            skipped_channels.append(f"{subj}_{med}")
            pbar.write(f"  sub-{subj} [{med}]: only {len(eeg_picks)} EEG channels -- skipping")
            continue

        data = epochs.get_data(picks=eeg_picks)   # (n_epochs, C, T)

        T = CFG.n_timepoints
        if data.shape[2] > T:
            data = data[:, :, :T]
        elif data.shape[2] < T:
            data = np.pad(data, ((0, 0), (0, 0), (0, T - data.shape[2])))

        # Group key = subject + med so ON/OFF sessions of the same subject
        # are kept separate in StratifiedGroupKFold
        group_key = f"{subj}_{med}"
        if group_key not in subject_index_map:
            subject_index_map[group_key] = len(subject_index_map)
        grp = subject_index_map[group_key]

        n = data.shape[0]
        X_list.append(data.astype(np.float32))
        y_list.append(np.full(n, label, dtype=np.int64))
        g_list.append(np.full(n, grp,   dtype=np.int64))

        cls_name = [k for k, v in med_label_map.items() if v == label][0]
        pbar.set_postfix({"last": f"sub-{subj} [{med}={cls_name}] {n} ep"})

    if not X_list:
        raise RuntimeError(f"No usable epochs loaded for comparison '{comparison}'.")

    X      = np.concatenate(X_list, axis=0)
    y      = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    n_cls1    = int((y == 1).sum())
    n_cls0    = int((y == 0).sum())
    ratio     = max(n_cls1, n_cls0) / max(min(n_cls1, n_cls0), 1)
    cls1_name = [k for k, v in med_label_map.items() if v == 1][0]
    cls0_name = [k for k, v in med_label_map.items() if v == 0][0]

    print(f"\n  Loading complete in {_fmt_time(time.time() - t0)}")
    print(f"  Sessions loaded  : {len(subject_index_map)}"
          + (f"  ({len(skipped_channels)} skipped)" if skipped_channels else ""))
    print(f"  Total epochs     : {X.shape[0]}  "
          f"({cls1_name}={n_cls1}, {cls0_name}={n_cls0})")
    print(f"  Imbalance ratio  : {ratio:.2f}  "
          f"('balanced' if ratio < 1.5 else 'imbalanced -- sampler will be used')")
    print(f"  Data tensor      : {X.shape}  ->  {X.nbytes / 1e6:.1f} MB")

    return X, y, groups


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

# ============================================================
# Loss with label smoothing
# ============================================================

def label_smoothed_nll_loss(
    log_probs: torch.Tensor,
    targets:   torch.Tensor,
    smoothing: float = 0.05,
    weight:    Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    NLL loss with label smoothing.
    Replaces hard 0/1 targets with (1-eps)*one_hot + eps/n_classes,
    preventing overconfidence on training subjects.
    """
    n_classes  = log_probs.size(1)
    nll        = F.nll_loss(log_probs, targets, weight=weight, reduction="none")
    smooth     = -log_probs.mean(dim=1)
    loss       = (1.0 - smoothing) * nll + smoothing * smooth
    return loss.mean()


# ============================================================
# EEG Mixup  (NEW)
# ============================================================

def mixup_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    EEG Mixup: interpolate pairs of training epochs.

    Why this works for EEG:
      Neural signals are linear superpositions of underlying sources.
      λ·epoch_A + (1-λ)·epoch_B is physically plausible — it resembles
      an epoch from a subject whose brain state lies between A and B.
      This creates an infinite supply of virtual subjects the model has
      never seen, which is the most direct solution to the 50-subject
      dataset limitation.

    Implementation:
      λ ~ Beta(alpha, alpha).  With alpha=0.2 the distribution is U-shaped,
      so λ is usually close to 0 or 1 (mild mixing), which avoids creating
      confusing intermediate examples while still providing regularisation.

    Returns (mixed_X, y_a, y_b, lam) so the caller can compute:
        loss = lam * loss(mixed_X, y_a) + (1-lam) * loss(mixed_X, y_b)
    """
    lam        = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = X.size(0)
    idx        = torch.randperm(batch_size, device=X.device)
    mixed_X    = lam * X + (1.0 - lam) * X[idx]
    return mixed_X, y, y[idx], lam


def mixup_loss(
    log_probs:  torch.Tensor,
    y_a:        torch.Tensor,
    y_b:        torch.Tensor,
    lam:        float,
    smoothing:  float = 0.05,
    weight:     Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combined label-smoothed loss for mixup pairs."""
    loss_a = label_smoothed_nll_loss(log_probs, y_a, smoothing=smoothing, weight=weight)
    loss_b = label_smoothed_nll_loss(log_probs, y_b, smoothing=smoothing, weight=weight)
    return lam * loss_a + (1.0 - lam) * loss_b


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
    total_loss   = 0.0
    correct      = 0          # track train accuracy
    n_total      = 0
    use_mixup    = cfg is not None and getattr(cfg, "mixup_alpha", 0.0) > 0
    smoothing    = cfg.label_smoothing if cfg is not None else 0.05
    alpha        = cfg.mixup_alpha     if cfg is not None else 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_mixup:
            X_mix, y_a, y_b, lam = mixup_batch(X_batch, y_batch, alpha=alpha)
            log_probs = model(X_mix)
            loss      = mixup_loss(log_probs, y_a, y_b, lam,
                                   smoothing=smoothing, weight=class_weight)
            # Accuracy vs the dominant label (argmax of mixed labels)
            # lam is a Python float; torch.where needs a tensor condition
            dom_labels = y_a if lam >= 0.5 else y_b
            correct   += (log_probs.argmax(1) == dom_labels).sum().item()
        else:
            log_probs = model(X_batch)
            loss      = label_smoothed_nll_loss(
                log_probs, y_batch, smoothing=smoothing, weight=class_weight)
            correct  += (log_probs.argmax(1) == y_batch).sum().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        model.apply_max_norm_()
        total_loss += loss.item() * len(y_batch)
        n_total    += len(y_batch)
        epoch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"}, refresh=True)

    return total_loss / n_total, correct / n_total   # (loss, accuracy)


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
    train_accs:   List[float] = []
    val_accs:     List[float] = []

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
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, cfg.device,
            epoch_pbar=epoch_pbar,
            class_weight=class_weight,
            grad_clip=cfg.grad_clip_norm,
            cfg=cfg,
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, cfg.device)
        scheduler.step(va_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        improved    = va_loss < early_stop.best_loss - early_stop.min_delta
        best_marker = "best" if improved else ""
        grace_str   = f"grace {epoch}/{cfg.min_epochs}" if epoch < cfg.min_epochs else ""

        epoch_pbar.set_postfix({
            "tr_loss": f"{tr_loss:.4f}",
            "va_loss": f"{va_loss:.4f}",
            "tr_acc":  f"{tr_acc:.3f}",
            "va_acc":  f"{va_acc:.3f}",
            "lr":      f"{current_lr:.2e}",
            "pat":     f"{early_stop.counter}/{cfg.patience}",
            "":        best_marker or grace_str,
        }, refresh=True)

        if early_stop(va_loss, model, epoch):
            epoch_pbar.write(
                f"  stop  {fold_label}epoch {epoch} "
                f"(best={early_stop.best_epoch}, "
                f"val_loss={early_stop.best_loss:.4f})"
            )
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
    return model, early_stop, train_losses, val_losses, train_accs, val_accs


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
    fold_train_accs:   List[List[float]] = []
    fold_val_accs:     List[List[float]] = []
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

        model, early_stop, train_losses, val_losses, train_accs, val_accs = _run_training_loop(
            model, optimizer, scheduler,
            train_loader, val_loader, cfg,
            ckpt_path, class_weight,
            fold_label=f"Fold {fold} | ",
        )

        fold_elapsed = time.time() - fold_t0
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_train_accs.append(train_accs)
        fold_val_accs.append(val_accs)

        _, ep_acc, ep_y, ep_probs = evaluate(model, val_loader, device)
        ep_pred = (ep_probs >= 0.5).astype(int)
        ep_auc  = roc_auc_score(ep_y, ep_probs)
        ep_bal  = balanced_accuracy_score(ep_y, ep_pred)
        ep_f1   = f1_score(ep_y, ep_pred, zero_division=0)
        ep_prec = precision_score(ep_y, ep_pred, zero_division=0)
        ep_rec  = recall_score(ep_y, ep_pred, zero_division=0)

        subj_y, subj_probs, _ = subject_level_predictions(ep_y, ep_probs, g_va)
        subj_pred = (subj_probs >= 0.5).astype(int)
        subj_acc  = (subj_y == subj_pred).mean()
        subj_auc  = (
            roc_auc_score(subj_y, subj_probs)
            if len(np.unique(subj_y)) > 1 else float("nan")
        )
        subj_bal  = balanced_accuracy_score(subj_y, subj_pred)
        subj_f1   = f1_score(subj_y, subj_pred, zero_division=0)
        subj_prec = precision_score(subj_y, subj_pred, zero_division=0)
        subj_rec  = recall_score(subj_y, subj_pred, zero_division=0)

        print(f"\n  ── Fold {fold} results {'─' * 32}")
        print(f"  {'Metric':<20} {'Epoch-level':>12} {'Subject-level':>14}")
        print(f"  {'─'*20} {'─'*12} {'─'*14}")
        print(f"  {'Accuracy':<20} {ep_acc:>12.4f} {subj_acc:>14.4f}")
        print(f"  {'Balanced Acc':<20} {ep_bal:>12.4f} {subj_bal:>14.4f}")
        print(f"  {'AUC-ROC':<20} {ep_auc:>12.4f} {subj_auc:>14.4f}")
        print(f"  {'F1 Score':<20} {ep_f1:>12.4f} {subj_f1:>14.4f}")
        print(f"  {'Precision':<20} {ep_prec:>12.4f} {subj_prec:>14.4f}")
        print(f"  {'Recall':<20} {ep_rec:>12.4f} {subj_rec:>14.4f}")
        print(f"  Best epoch : {early_stop.best_epoch}  |  "
              f"Fold time : {_fmt_time(fold_elapsed)}")

        fold_results.append({
            "fold":          fold,
            "best_epoch":    early_stop.best_epoch,
            "n_val_epochs":  len(ep_y),
            "n_val_subjs":   len(subj_y),
            "ep_acc":        ep_acc,  "subj_acc":      subj_acc,
            "ep_auc":        ep_auc,  "subj_auc":      subj_auc,
            "ep_bal_acc":    ep_bal,  "subj_bal_acc":  subj_bal,
            "ep_f1":         ep_f1,   "subj_f1":       subj_f1,
            "ep_prec":       ep_prec, "subj_prec":     subj_prec,
            "ep_rec":        ep_rec,  "subj_rec":      subj_rec,
        })

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(MODEL_DIR / "cv_results.tsv", sep="\t", index=False)
    print(f"\n  Cross-validation complete in {_fmt_time(time.time() - cv_t0)}")
    return results_df, fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs


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
    save_dir:              Path,
    train_accs_per_fold:   Optional[List[List[float]]] = None,
    val_accs_per_fold:     Optional[List[List[float]]] = None,
):
    """
    Plot training and validation loss (and accuracy if provided) across all folds.
    Saves two files:
        learning_curves_loss.png   — NLL loss (train + val, all folds)
        learning_curves_acc.png    — Accuracy (train + val, all folds)
    and one combined 4-panel figure:
        learning_curves.png
    """
    palette   = sns.color_palette("muted", n_colors=len(train_losses_per_fold))
    has_acc   = (train_accs_per_fold is not None and val_accs_per_fold is not None
                 and len(train_accs_per_fold) > 0)
    n_cols    = 4 if has_acc else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 4), sharex=False)
    if n_cols == 2:
        axes = list(axes)

    curve_specs = [
        (train_losses_per_fold, "Train NLL Loss",       "NLL Loss",  axes[0]),
        (val_losses_per_fold,   "Validation NLL Loss",  "NLL Loss",  axes[1]),
    ]
    if has_acc:
        curve_specs += [
            (train_accs_per_fold, "Train Accuracy",       "Accuracy",  axes[2]),
            (val_accs_per_fold,   "Validation Accuracy",  "Accuracy",  axes[3]),
        ]

    for curves_per_fold, title, ylabel, ax in curve_specs:
        max_len = max(len(l) for l in curves_per_fold)
        matrix  = np.full((len(curves_per_fold), max_len), np.nan)
        for i, lv in enumerate(curves_per_fold):
            matrix[i, : len(lv)] = lv
            ax.plot(range(1, len(lv) + 1), lv,
                    color=palette[i], alpha=0.45, linewidth=1.2,
                    label=f"Fold {i + 1}")
        mean_curve = np.nanmean(matrix, axis=0)
        ax.plot(range(1, max_len + 1), mean_curve,
                color="black", linewidth=2.2, label="Mean", zorder=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        if "Accuracy" in title:
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

    fig.suptitle("Learning curves — all folds", fontsize=13)
    fig.tight_layout()
    out = save_dir / "learning_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Learning curves  -> {out}")

    # Also save standalone loss-only figure for quick reference
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    for curves_per_fold, title, ax in [
        (train_losses_per_fold, "Train NLL Loss",      axes2[0]),
        (val_losses_per_fold,   "Validation NLL Loss", axes2[1]),
    ]:
        max_len = max(len(l) for l in curves_per_fold)
        matrix  = np.full((len(curves_per_fold), max_len), np.nan)
        for i, lv in enumerate(curves_per_fold):
            matrix[i, : len(lv)] = lv
            ax.plot(range(1, len(lv) + 1), lv,
                    color=palette[i], alpha=0.45, linewidth=1.2, label=f"Fold {i+1}")
        ax.plot(range(1, max_len + 1), np.nanmean(matrix, axis=0),
                color="black", linewidth=2.2, label="Mean", zorder=5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("NLL Loss")
        ax.set_title(title); ax.legend(fontsize=7)
    fig2.suptitle("Loss curves — all folds", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(save_dir / "learning_curves_loss.png", dpi=120)
    plt.close(fig2)


def plot_cv_summary(results_df: pd.DataFrame, save_dir: Path):
    """
    Strip-plot with mean±SD for every metric, epoch-level and subject-level.
    Includes Accuracy, Balanced Acc, AUC-ROC, F1, Precision, Recall.
    """
    metrics = {
        "Accuracy":       ("ep_acc",     "subj_acc"),
        "Balanced Acc":   ("ep_bal_acc", "subj_bal_acc"),
        "AUC-ROC":        ("ep_auc",     "subj_auc"),
        "F1 Score":       ("ep_f1",      "subj_f1"),
        "Precision":      ("ep_prec",    "subj_prec"),
        "Recall":         ("ep_rec",     "subj_rec"),
    }
    # Only plot columns that actually exist in the dataframe
    metrics = {k: v for k, v in metrics.items()
               if v[0] in results_df.columns and v[1] in results_df.columns}

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    colours = {"Epoch-level": "#4C72B0", "Subject-level": "#DD8452"}

    for ax, (title, (ep_col, subj_col)) in zip(axes, metrics.items()):
        for x_pos, (col, level) in enumerate(
            [(ep_col, "Epoch-level"), (subj_col, "Subject-level")]
        ):
            vals   = results_df[col].dropna().values
            colour = colours[level]
            jitter = (np.random.default_rng(SEED).random(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                       color=colour, s=50, zorder=3, alpha=0.85, label=level)
            m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
            ax.errorbar(x_pos, m, yerr=s, fmt="D", color=colour,
                        markersize=7, capsize=5, linewidth=2, zorder=4)
            ax.text(x_pos, -0.10, f"{m:.3f}", ha="center", fontsize=8,
                    color=colour, transform=ax.get_xaxis_transform())

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Epoch", "Subject"], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.35, linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle("Cross-validation results — EEGNet PD-OFF vs CTL", fontsize=12)
    fig.tight_layout()
    out = save_dir / "cv_summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  CV summary plot  -> {out}")


def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    save_dir:  Path,
    title:     str  = "Confusion Matrix",
    filename:  str  = "confusion_matrix.png",
    labels:    List[str] = None,
):
    """
    Plot a confusion matrix showing both raw counts and row percentages.
    Each cell displays:
        N
       xx%
    where N is the count and xx% is that row's recall (true-class rate).
    """
    if labels is None:
        labels = ["CTL", "PD-OFF"]

    cm    = confusion_matrix(y_true, y_pred)
    n_cls = cm.shape[0]

    # Build annotation array: "N\nxx%"
    annot = np.empty_like(cm, dtype=object)
    for i in range(n_cls):
        row_sum = cm[i].sum()
        for j in range(n_cls):
            pct = 100.0 * cm[i, j] / row_sum if row_sum > 0 else 0.0
            annot[i, j] = f"{cm[i, j]}\n{pct:.1f}%"

    # Normalised version drives the colour scale (0..1 per row)
    cm_norm = cm.astype(float)
    for i in range(n_cls):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] /= row_sum

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    # Manually fix annotation font size and alignment
    for text in ax.texts:
        text.set_fontsize(11)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=12)

    # Add overall accuracy as a subtitle
    overall_acc = (np.array(y_true) == np.array(y_pred)).mean()
    ax.text(
        0.5, -0.12, f"Overall accuracy: {overall_acc:.1%}",
        transform=ax.transAxes, ha="center", fontsize=10, color="gray"
    )

    fig.tight_layout()
    out = save_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix -> {out}")


def print_cv_summary(results_df: pd.DataFrame):
    W = 82
    print("\n" + "=" * W)
    print(f"{'CROSS-VALIDATION SUMMARY':^{W}}")
    print("=" * W)
    print(
        f"{'Fold':>5} | "
        f"{'Ep Acc':>6} {'Ep Bal':>6} {'Ep F1':>6} {'Ep Pre':>6} {'Ep Rec':>6} | "
        f"{'Su Acc':>6} {'Su Bal':>6} {'Su F1':>6} {'Su Pre':>6} {'Su Rec':>6}"
    )
    print("-" * W)
    for _, row in results_df.iterrows():
        def _g(col, default=float("nan")):
            return row[col] if col in row.index else default
        print(
            f"{int(row['fold']):>5} | "
            f"{_g('ep_acc'):.4f} {_g('ep_bal_acc'):.4f} "
            f"{_g('ep_f1'):.4f} {_g('ep_prec'):.4f} {_g('ep_rec'):.4f} | "
            f"{_g('subj_acc'):.4f} {_g('subj_bal_acc'):.4f} "
            f"{_g('subj_f1'):.4f} {_g('subj_prec'):.4f} {_g('subj_rec'):.4f}"
        )
    print("-" * W)
    metric_pairs = [
        ("ep_acc",     "subj_acc",     "Accuracy    "),
        ("ep_bal_acc", "subj_bal_acc", "Balanced Acc"),
        ("ep_auc",     "subj_auc",     "AUC-ROC     "),
        ("ep_f1",      "subj_f1",      "F1 Score    "),
        ("ep_prec",    "subj_prec",    "Precision   "),
        ("ep_rec",     "subj_rec",     "Recall      "),
    ]
    for col_ep, col_su, label in metric_pairs:
        if col_ep not in results_df.columns:
            continue
        ep_m = results_df[col_ep].mean(); ep_s = results_df[col_ep].std()
        su_m = results_df[col_su].mean(); su_s = results_df[col_su].std()
        print(
            f"  Mean±SD   {label}: "
            f"epoch={ep_m:.4f}±{ep_s:.4f}   "
            f"subject={su_m:.4f}±{su_s:.4f}"
        )
    print("=" * W + "\n")


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
# Held-out subject split  (NEW)
# ============================================================

def split_holdout_subjects(
    X:             np.ndarray,
    y:             np.ndarray,
    groups:        np.ndarray,
    n_per_class:   int = 3,
    seed:          int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
    """
    Reserve n_per_class subjects from each class as a held-out test set
    BEFORE any training occurs.

    Strategy:
      - Find all unique subject-group IDs for each class.
      - Randomly select n_per_class from each class (fixed seed).
      - Return train split (remaining) and test split (held-out).

    This is called BEFORE subject_wise_zscore so that normalisation for
    each subject uses only that subject's own epochs — no leakage.

    Returns:
        X_tr, y_tr, g_tr   : training data  (all epochs except held-out)
        X_te, y_te, g_te   : held-out test  (epochs from held-out subjects)
        held_out_ids        : 1-D array of group IDs that were held out
    """
    rng            = np.random.default_rng(seed)
    unique_groups  = np.unique(groups)

    # Map each unique group to its class label (all epochs of a group share the same label)
    group_label: Dict[int, int] = {}
    for gid in unique_groups:
        mask              = groups == gid
        group_label[gid]  = int(y[mask][0])

    # Separate group IDs by class
    class_groups: Dict[int, List[int]] = {0: [], 1: []}
    for gid, lbl in group_label.items():
        class_groups[lbl].append(gid)

    for lbl, gids in class_groups.items():
        if len(gids) < n_per_class:
            raise ValueError(
                f"Class {lbl} only has {len(gids)} subjects but "
                f"n_per_class={n_per_class} were requested for holdout."
            )

    # Sample held-out group IDs
    held_out_ids = np.concatenate([
        rng.choice(sorted(class_groups[lbl]), size=n_per_class, replace=False)
        for lbl in sorted(class_groups)
    ])

    test_mask  = np.isin(groups, held_out_ids)
    train_mask = ~test_mask

    return (
        X[train_mask],  y[train_mask],  groups[train_mask],
        X[test_mask],   y[test_mask],   groups[test_mask],
        held_out_ids,
    )


def evaluate_held_out_subjects(
    model:       "EEGNet",
    X_te:        np.ndarray,
    y_te:        np.ndarray,
    g_te:        np.ndarray,
    held_out_ids: np.ndarray,
    cfg:          "EEGNetConfig",
    save_dir:     Path,
    comparison:   str = "OFF_CTL",
) -> pd.DataFrame:
    """
    Run the final model on held-out subjects and report per-subject results.

    Returns a DataFrame with one row per held-out subject, containing:
        group_id, true_label, pred_label, subject_prob, correct
    """
    _banner("Held-out subject evaluation")
    print(f"  Held-out subjects : {len(held_out_ids)}")
    print(f"  Held-out epochs   : {len(y_te)}")

    _label_for = {
        "OFF_CTL": {1: "PD-OFF", 0: "CTL"},
        "ON_CTL":  {1: "PD-ON",  0: "CTL"},
        "ON_OFF":  {1: "PD-ON",  0: "PD-OFF"},
    }
    label_names = _label_for.get(comparison, {1: "Class 1", 0: "Class 0"})

    loader = DataLoader(
        EEGDataset(X_te, y_te),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    print("  Running inference...", end=" ", flush=True)
    _, _, ep_y, ep_probs = evaluate(model, loader, cfg.device)
    print("done.")

    # Aggregate to subject level
    subj_y, subj_probs, subj_ids = subject_level_predictions(ep_y, ep_probs, g_te)
    subj_pred = (subj_probs >= 0.5).astype(int)

    # Build results table
    rows = []
    for i, gid in enumerate(subj_ids):
        true_lbl = label_names[int(subj_y[i])]
        pred_lbl = label_names[int(subj_pred[i])]
        correct  = bool(subj_y[i] == subj_pred[i])
        rows.append({
            "group_id":    int(gid),
            "true_label":  true_lbl,
            "pred_label":  pred_lbl,
            "prob_cls1":   float(subj_probs[i]),
            "correct":     correct,
        })
    results = pd.DataFrame(rows)

    # Console report
    print()
    print(f"  {'Group':>8}  {'True':>8}  {'Pred':>8}  {'Prob':>6}  {'OK':>4}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*4}")
    for _, row in results.iterrows():
        ok_str = "YES" if row["correct"] else " NO"
        print(f"  {row['group_id']:>8}  {row['true_label']:>8}  "
              f"{row['pred_label']:>8}  {row['prob_cls1']:>6.3f}  {ok_str:>4}")

    n_correct = results["correct"].sum()
    n_total   = len(results)
    print(f"\n  Subject accuracy : {n_correct}/{n_total} = {n_correct/n_total:.1%}")

    # Full metrics (only meaningful if both classes present)
    if len(np.unique(subj_y)) > 1:
        auc  = roc_auc_score(subj_y, subj_probs)
        bal  = balanced_accuracy_score(subj_y, subj_pred)
        f1   = f1_score(subj_y, subj_pred, zero_division=0)
        prec = precision_score(subj_y, subj_pred, zero_division=0)
        rec  = recall_score(subj_y, subj_pred, zero_division=0)
        print(f"  AUC-ROC          : {auc:.4f}")
        print(f"  Balanced acc.    : {bal:.4f}")
        print(f"  F1 Score         : {f1:.4f}")
        print(f"  Precision        : {prec:.4f}")
        print(f"  Recall           : {rec:.4f}")
        # Save a classification report to disk too
        report_path = save_dir / "held_out_classification_report.txt"
        with open(report_path, "w") as fh:
            label_names_list = sorted(label_names.values(),
                key=lambda v: list(label_names.values()).index(v))
            fh.write(f"Held-out subject evaluation — {comparison}\n")
            fh.write("=" * 50 + "\n")
            fh.write(classification_report(
                subj_y, subj_pred, target_names=label_names_list
            ))
        print(f"  Report saved  -> {report_path}")

    # Save results TSV
    tsv_path = save_dir / "held_out_results.tsv"
    results.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Results saved -> {tsv_path}")

    # Confusion matrix for held-out subjects
    plot_confusion_matrix(
        subj_y, subj_pred, save_dir,
        title=f"Held-out subject confusion matrix\n({comparison})",
        filename="confusion_matrix_held_out.png",
        labels=sorted(label_names.values(), key=lambda v: list(label_names.values()).index(v)),
    )

    return results


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
    print(f"  Augmentation    : noise={CFG.augment_noise_std}  ch_drop={CFG.channel_drop_prob}  mixup_alpha={CFG.mixup_alpha}")
    print(f"  Label smoothing : {CFG.label_smoothing}")
    print(f"  Normalisation   : subject-wise z-score (pre-split)")
    print(f"  CV folds        : {CFG.n_folds}")
    print(f"  num_workers     : {CFG.num_workers}")

    # ── 1. Load ─────────────────────────────────────────────────────────
    X, y, groups = load_all_epochs(EPOCHS_DIR, SUMMARY_TSV, comparison=CFG.comparison)

    # Update model dims to match loaded data (use full X before split)
    CFG.n_channels    = X.shape[1]
    CFG.epoch_len_sec = X.shape[2] / CFG.sfreq

    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        raise RuntimeError(
            "Only one class present in the dataset — cannot train a classifier."
        )

    # ── 1b. Hold out 3 PD + 3 CTL subjects BEFORE any training ────────────
    # These subjects are completely invisible during training and CV.
    # subject_wise_zscore uses per-subject statistics, so it is safe to
    # apply it after the split without any normalisation leakage.
    print("\n  Splitting held-out test subjects (3 PD-OFF + 3 CTL)...")
    X_tr, y_tr, g_tr, X_te, y_te, g_te, held_out_ids = split_holdout_subjects(
        X, y, groups, n_per_class=3
    )
    n_held_pd  = int((y_te == 1).sum())
    n_held_ctl = int((y_te == 0).sum())
    print(f"  Training pool    : {len(np.unique(g_tr))} subjects  "
          f"({int((y_tr==1).sum())} PD epochs, {int((y_tr==0).sum())} CTL epochs)")
    print(f"  Held-out pool    : {len(held_out_ids)} subjects  "
          f"({n_held_pd} PD epochs, {n_held_ctl} CTL epochs)")
    print(f"  Held-out IDs     : {sorted(held_out_ids.tolist())}")

    # ── Subject-wise normalisation (applied independently to train and test) ──
    # Using subject_wise_zscore on each split separately ensures each subject's
    # normalisation uses only their own epochs — no cross-split leakage.
    print("\n  Applying subject-wise z-score normalisation...")
    X_tr = subject_wise_zscore(X_tr, g_tr)
    X_te = subject_wise_zscore(X_te, g_te)
    print(f"  Train — mean: {X_tr.mean():.4f}  std: {X_tr.std():.4f}")
    print(f"  Test  — mean: {X_te.mean():.4f}  std: {X_te.std():.4f}")

    # ── 2. Cross-validation (on training pool only) ──────────────────────
    results_df, fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs =         run_cross_validation(X_tr, y_tr, g_tr, CFG)
    print_cv_summary(results_df)
    plot_cv_summary(results_df, MODEL_DIR)
    plot_learning_curves(
        fold_train_losses, fold_val_losses, MODEL_DIR,
        train_accs_per_fold=fold_train_accs,
        val_accs_per_fold=fold_val_accs,
    )

    # ── 3. Final model (trained on full training pool) ───────────────────
    raw_best     = max(1, int(round(results_df["best_epoch"].mean())))
    fixed_epochs = max(raw_best, CFG.min_epochs)
    print(f"\n  CV mean best_epoch = {results_df['best_epoch'].mean():.1f} "
          f"-> training final model for {fixed_epochs} epoch(s).")
    final_model = train_final_model(X_tr, y_tr, g_tr, CFG, fixed_epochs=fixed_epochs)

    # ── 4a. Training-set evaluation (for reference only) ─────────────────
    _banner("Step 4a — Training-set evaluation  (not a valid performance estimate)")
    full_loader = DataLoader(
        EEGDataset(X_tr, y_tr),
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
    )
    print("  Running inference on training set...", end=" ", flush=True)
    _, _, ep_y, ep_probs = evaluate(final_model, full_loader, CFG.device)
    print("done.")
    subj_y, subj_probs, _ = subject_level_predictions(ep_y, ep_probs, g_tr)
    subj_pred = (subj_probs >= 0.5).astype(int)
    print("  Subject-level classification report (training set):")
    print(classification_report(subj_y, subj_pred, target_names=["CTL", "PD-OFF"]))
    plot_confusion_matrix(
        subj_y, subj_pred, MODEL_DIR,
        title=f"Training-set confusion matrix ({CFG.comparison})",
        filename="confusion_matrix_train.png",
    )

    # ── 4b. Held-out subject evaluation (the unbiased result) ────────────
    held_out_results = evaluate_held_out_subjects(
        model=final_model,
        X_te=X_te,
        y_te=y_te,
        g_te=g_te,
        held_out_ids=held_out_ids,
        cfg=CFG,
        save_dir=MODEL_DIR,
        comparison=CFG.comparison,
    )

    _banner("Pipeline complete")
    print(f"  Total wall time      : {_fmt_time(time.time() - wall_t0)}")
    print(f"  Model checkpoint     : {MODEL_DIR / 'final_model.pt'}")
    print(f"  CV results           : {MODEL_DIR / 'cv_results.tsv'}")
    print(f"  Held-out results     : {MODEL_DIR / 'held_out_results.tsv'}")
    print(f"  Confusion matrices   : confusion_matrix_train.png  |  confusion_matrix_held_out.png")
    print(f"  Norm stats           : {MODEL_DIR / 'norm_mean.npy'}, norm_std.npy")
    print()
    print("  NOTE: Use CV results for method comparison.")
    print("        Use held-out results as the unbiased estimate of real-world performance.")


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
