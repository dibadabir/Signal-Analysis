# ============================================================
# preprocessing_knn_target_novelty_concat.py
# ============================================================
# Oddball-Aware KNN Preprocessing for Target + Novelty Concatenation
# Incorporates: filtering, ICA, bad channel handling, epoch rejection
# Separate condition-specific (target + novelty) feature extraction & concatenation
#
# DESIGN NOTE:
# This script saves the FULL feature matrix without feature selection or
# normalization. These steps MUST be performed inside CV training folds to
# avoid data leakage and ensure unbiased performance estimates. Downstream
# training scripts should apply:
#   1. Feature selection (only on fold training data)
#   2. Subject-wise z-score normalization (only on fold training data)
# ============================================================

import os
import re
import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA
from mne import Epochs, pick_types
from scipy import signal
from scipy.stats import mannwhitneyu, kurtosis

from bids import BIDSLayout
from mne_bids import BIDSPath, read_raw_bids

try:
    from sklearn.feature_selection import mutual_info_classif
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False


# ============================================================
# Paths
# ============================================================
BIDS_ROOT = Path(r"C:\Users\User\Desktop\New folder\OneDrive\Desktop\Year 3\EEG Signalling\ds003490")
OUT_DIR = Path(r"C:\Users\User\Desktop\New folder\OneDrive\Desktop\Year 3\EEG Signalling\outputs")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUT_DIR / "qc_plots_knn_target_novelty"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("preprocess_knn_target_novelty")


# ============================================================
# Preprocessing Settings
# ============================================================

# Channel selection — standard 10-20 system (60 channels)
TARGET_CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6","FCz",
    "T7","C3","Cz","C4","T8",
    "CP5","CP1","CP2","CP6",
    "P7","P3","Pz","P4","P8",
    "PO7","PO3","POz","PO4","PO8",
    "O1","Oz","O2",
    "AF7","AF3","AFz","AF4","AF8",
    "F5","F1","F2","F6",
    "FT7","FC3","FC4","FT8",
    "C5","C1","C2","C6",
    "TP7","CP3","CP4","TP8",
    "P5","P1","P2","P6",
]

# Sampling
RESAMPLE_SFREQ = 100.0
BANDPASS = (0.5, 40.0)
ICA_HP = 1.0
ICA_NCOMP = 0.999
DEFAULT_NOTCH_FREQ = 60.0

# Oddball event codes
EVENT_CODE_TARGET = 200    # Target stimulus
EVENT_CODE_STANDARD = 201  # Standard stimulus
EVENT_CODE_NOVELTY = 202   # Novelty stimulus

# Oddball epoching
EPOCH_TMIN = -0.25   # -250ms
EPOCH_TMAX = 1.25    # +1250ms
BASELINE = (-0.25, 0.0)  # -250ms to 0ms

# Artifact rejection thresholds
REJECT_PTP_EPOCH = 150e-6  # 150µV peak-to-peak

# Bad channel detection
BAD_CHAN_FLAT_THRESH = 1e-6
BAD_CHAN_NOISY_Z_THRESH = 3.0
BAD_CHAN_EOG_PROTECT = {"Fp1", "Fp2", "FP1", "FP2"}
BAD_CHAN_MAX_FRACTION = 0.3

# Feature extraction
FFT_FREQ_BANDS = {
    "delta":   (0.5, 4),
    "theta":   (4, 8),
    "alpha":   (8, 12),
    "beta":    (12, 30),
    "gamma":   (30, 40),
}
WELCH_NPERSEG = 100
WELCH_NOVERLAP = 50

# Feature selection
N_SELECTED_FEATURES = 120

# Trial matching
RANDOM_SEED_MATCHING = 42

# QC settings
SHOW_PLOTS = False
MAX_INITIAL_SUBJECTS_WITHOUT_SAVE = 3


# ============================================================
# Epoch Strategy Enum
# ============================================================

class EpochStrategy(Enum):
    EVENT_BASED = auto()
    FIXED_LENGTH = auto()
    FAILED = auto()


# ============================================================
# Sidecar Reading Functions
# ============================================================

def read_sidecar_notch_freq_direct(eeg_dir: Path) -> float:
    """Read PowerLineFrequency from sidecar JSON in directory."""
    json_file = eeg_dir / next((f.name for f in eeg_dir.glob("*_eeg.json") if f.is_file()), None)
    if json_file and json_file.exists():
        try:
            with open(json_file) as fh:
                meta = json.load(fh)
            freq = float(meta.get("PowerLineFrequency", DEFAULT_NOTCH_FREQ))
            logger.info(f"  Sidecar PowerLineFrequency = {freq} Hz")
            return freq
        except Exception as e:
            logger.warning(f"  Could not read PowerLineFrequency: {e}")
    logger.info(f"  Using default notch frequency: {DEFAULT_NOTCH_FREQ} Hz")
    return DEFAULT_NOTCH_FREQ


def read_sidecar_ref_and_ground_direct(eeg_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Read EEGReference and EEGGround from sidecar JSON in directory."""
    ref_ch = ground_ch = None
    json_file = eeg_dir / next((f.name for f in eeg_dir.glob("*_eeg.json") if f.is_file()), None)
    if json_file and json_file.exists():
        try:
            with open(json_file) as fh:
                meta = json.load(fh)
            _null = {"N/A", "NONE", "NAN", ""}
            raw_ref = str(meta.get("EEGReference", "")).strip()
            raw_gnd = str(meta.get("EEGGround", "")).strip()
            if raw_ref.upper() not in _null:
                ref_ch = raw_ref
            if raw_gnd.upper() not in _null:
                ground_ch = raw_gnd
            logger.info(f"  Sidecar EEGReference='{ref_ch}', EEGGround='{ground_ch}'")
        except Exception as e:
            logger.warning(f"  Could not read reference/ground: {e}")
    return ref_ch, ground_ch


# ============================================================
# Bad Channel Detection & Interpolation
# ============================================================

def detect_and_interpolate_bad_channels(
    raw: mne.io.BaseRaw,
    subj: str
) -> Tuple[mne.io.BaseRaw, List[str]]:
    """Detect flat/noisy channels and interpolate them."""
    raw = raw.copy().load_data()
    
    # Pick only EEG channels, exclude coordinate/meta channels
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    data = raw.get_data(picks=eeg_picks)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    n_ch = len(ch_names)

    bad_channels: List[str] = []
    protected = {ch.upper() for ch in BAD_CHAN_EOG_PROTECT}

    # Detect flat channels
    channel_sds = np.std(data, axis=1)
    flat_mask = channel_sds < BAD_CHAN_FLAT_THRESH
    flat_chs = [ch_names[i] for i in range(n_ch) if flat_mask[i]]
    if flat_chs:
        logger.info(f"  ⚠️  Flat channels detected: {flat_chs}")
    bad_channels.extend(flat_chs)

    # Detect noisy channels (Z-score test)
    non_flat_sds = channel_sds[~flat_mask]
    if len(non_flat_sds) > 1:
        sd_mean = np.mean(non_flat_sds)
        sd_std = np.std(non_flat_sds)
        if sd_std > 0:
            z_scores = (channel_sds - sd_mean) / sd_std
            noisy_mask = (
                (z_scores > BAD_CHAN_NOISY_Z_THRESH)
                & (~flat_mask)
                & np.array([ch.upper() not in protected for ch in ch_names])
            )
            noisy_chs = [ch_names[i] for i in range(n_ch) if noisy_mask[i]]
            if noisy_chs:
                logger.info(f"  ⚠️  Noisy channels (z>{BAD_CHAN_NOISY_Z_THRESH}): {noisy_chs}")
            bad_channels.extend(noisy_chs)

    # Deduplicate
    bad_channels = list(dict.fromkeys(bad_channels))

    # Check fraction
    bad_fraction = len(bad_channels) / n_ch if n_ch > 0 else 0.0
    if bad_fraction > BAD_CHAN_MAX_FRACTION:
        raise ValueError(
            f"sub-{subj}: {len(bad_channels)}/{n_ch} channels flagged "
            f"({bad_fraction:.0%}) exceeds threshold ({BAD_CHAN_MAX_FRACTION:.0%})"
        )

    # Interpolate
    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose="warning")
        logger.info(f"  ✓ Interpolated {len(bad_channels)} bad channel(s)")
    else:
        logger.info(f"  ✓ No bad channels detected")

    return raw, bad_channels


# ============================================================
# Preprocessing Functions
# ============================================================

def force_eeg_channel_types(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Force appropriate channel types and exclude non-standard channels."""
    # Standard 10-20 channels
    eeg_like = {
        "FP1","FP2","F3","F4","FZ","FC1","FC2","FCZ","C3","C4","CZ","P3","P4","PZ","T7","T8",
        "F7","F8","C1","C2","C5","C6","P7","P8","O1","O2","OZ","AF3","AF4","AF7","AF8",
        "FT7","FT8","FT9","FT10","TP7","TP8","TP9","TP10","CP1","CP2","CP3","CP4","CP5","CP6",
        "PO3","PO4","PO7","PO8","POZ","P1","P2","P5","P6","F1","F2","F5","F6",
        "CPZ","FCZ","FPZ","AFZ","IZ","P9","P10","FC3","FC4","FC5","FC6",
    }
    
    # Mark non-standard channels as misc
    non_eeg = {ch: "misc" for ch in raw.ch_names if ch.strip().upper() not in eeg_like}
    
    # Mark standard channels as EEG
    mapping = {ch: "eeg" for ch in raw.ch_names if ch.strip().upper() in eeg_like}
    
    if mapping:
        raw.set_channel_types(mapping)
    if non_eeg:
        raw.set_channel_types(non_eeg)
    
    return raw


def safe_set_montage(raw: mne.io.BaseRaw) -> None:
    """Set standard 10-20 montage."""
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
        logger.info(f"  ✓ Standard 10-20 montage set")
    except Exception as e:
        logger.warning(f"  Could not set montage: {e}")


def safe_set_reference(
    raw: mne.io.BaseRaw,
    ref_ch: Optional[str] = None,
    ground_ch: Optional[str] = None
) -> None:
    """Apply average re-reference."""
    try:
        if ground_ch and ground_ch in raw.ch_names:
            raw.set_channel_types({ground_ch: "misc"})
            logger.info(f"  ✓ Ground electrode '{ground_ch}' excluded")

        if ref_ch and ref_ch not in raw.ch_names:
            raw.add_reference_channels(ref_ch)
            logger.info(f"  ✓ Added back original reference '{ref_ch}'")

        raw.set_eeg_reference("average", projection=False)
        logger.info(f"  ✓ Applied average re-reference")
    except Exception as e:
        logger.warning(f"  Re-referencing failed: {e}")


def ensure_sfreq(raw: mne.io.BaseRaw, target_sfreq: float) -> mne.io.BaseRaw:
    """Resample to target sampling frequency."""
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-6:
        logger.info(f"  Resampling {raw.info['sfreq']} Hz → {target_sfreq} Hz")
        raw = raw.copy().resample(target_sfreq)
    return raw


def select_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Select standard 10-20 channels."""
    picks = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
    if not picks:
        logger.warning(f"  ⚠️  No target channels found, keeping all")
        return raw
    logger.info(f"  ✓ Selected {len(picks)} standard 10-20 channels")
    return raw.copy().pick(picks)


def apply_filters(raw: mne.io.BaseRaw, notch_freq: float) -> mne.io.BaseRaw:
    """Apply notch and bandpass filters."""
    raw = raw.copy().load_data()
    nyq = raw.info["sfreq"] / 2.0
    
    # Notch filter
    freqs = [notch_freq]
    if notch_freq * 2 < nyq:
        freqs.append(notch_freq * 2)
    logger.info(f"  ✓ Notch filter: {freqs} Hz")
    raw.notch_filter(freqs, picks="eeg", verbose="warning")
    
    # Bandpass filter
    logger.info(f"  ✓ Bandpass filter: {BANDPASS[0]}-{BANDPASS[1]} Hz")
    raw.filter(BANDPASS[0], BANDPASS[1], picks="eeg", verbose="warning")
    
    return raw


def fit_ica(raw_filt: mne.io.BaseRaw, subj: str) -> ICA:
    """Fit ICA on highpass filtered data."""
    raw_ica = raw_filt.copy().load_data()
    raw_ica.filter(l_freq=ICA_HP, h_freq=None, picks="eeg", verbose="warning")
    
    ica = ICA(n_components=ICA_NCOMP, random_state=42, max_iter="auto")
    ica.fit(raw_ica, picks="eeg", verbose="warning")
    
    logger.info(f"  ✓ ICA fitted: {ica.n_components_} components")
    return ica


def apply_ica(
    ica: ICA,
    raw_filt: mne.io.BaseRaw,
    has_eog: bool = False
) -> Tuple[mne.io.BaseRaw, List[int]]:
    """Apply ICA and remove artifact components."""
    bad: List[int] = []

    # Try to detect EOG components
    if has_eog:
        eog_chs = [c for c in raw_filt.ch_names if "EOG" in c.upper()]
        if eog_chs:
            try:
                inds, _ = ica.find_bads_eog(raw_filt, ch_name=eog_chs[0])
                bad.extend(inds)
                logger.info(f"  ✓ Detected {len(inds)} EOG components")
            except Exception:
                pass

    # Use Fp1/Fp2 as proxy for EOG detection
    if not bad:
        for proxy in ["Fp1", "Fp2", "FP1", "FP2"]:
            if proxy in raw_filt.ch_names:
                try:
                    inds, _ = ica.find_bads_eog(raw_filt, ch_name=proxy)
                    bad.extend(inds)
                    logger.info(f"  ✓ Detected {len(inds)} EOG components (via {proxy})")
                    break
                except Exception:
                    continue

    bad = sorted(set(bad))
    raw_clean = raw_filt.copy()
    if bad:
        ica.apply(raw_clean, exclude=bad)
        logger.info(f"  ✓ Removed {len(bad)} ICA components")
    else:
        ica.apply(raw_clean)
        logger.info(f"  ✓ ICA applied (no components excluded)")

    return raw_clean, bad


# ============================================================
# Event Parsing (Oddball-Aware)
# ============================================================

def parse_oddball_event_code(value_str: str) -> Optional[int]:
    """
    Parse oddball event code from various formats.
    
    Handles:
    - Direct integers: "200", "201", "202"
    - BrainVision format: "S200", "S 200", "S  200"
    - Numeric strings with/without whitespace
    
    Returns:
    - 200 for target
    - 201 for standard
    - 202 for novelty
    - None if not recognized
    """
    if not isinstance(value_str, str):
        return None
    
    value_str = value_str.strip().upper()
    
    # Remove 'S' prefix if present
    if value_str.startswith('S'):
        value_str = value_str[1:].strip()
    
    try:
        code = int(value_str)
        if code in {200, 201, 202}:
            return code
    except (ValueError, TypeError):
        pass
    
    return None


def extract_oddball_conditions(events_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract target and novelty conditions from events dataframe.
    
    Returns dict with:
    - 'target': array of onset times for target (200) events
    - 'novelty': array of onset times for novelty (202) events
    - 'standard': array of onset times for standard (201) events
    """
    conditions = {'target': [], 'novelty': [], 'standard': []}
    
    for _, row in events_df.iterrows():
        value = str(row.get('value', '')).strip()
        code = parse_oddball_event_code(value)
        
        if code == EVENT_CODE_TARGET:
            conditions['target'].append(float(row['onset']))
        elif code == EVENT_CODE_NOVELTY:
            conditions['novelty'].append(float(row['onset']))
        elif code == EVENT_CODE_STANDARD:
            conditions['standard'].append(float(row['onset']))
    
    # Convert to numpy arrays
    for key in conditions:
        conditions[key] = np.array(conditions[key], dtype=float)
    
    return conditions


def create_oddball_events(conditions: Dict[str, np.ndarray], sfreq: float) -> Tuple[np.ndarray, Dict]:
    """
    Create MNE events array from oddball condition onsets.
    
    Returns:
    - events: (N, 3) array in MNE format [sample, 0, event_code]
    - event_id: dict mapping condition name to event code
    """
    all_samples = []
    all_codes = []
    
    code_map = {
        'target': EVENT_CODE_TARGET,
        'novelty': EVENT_CODE_NOVELTY,
        'standard': EVENT_CODE_STANDARD
    }
    
    for condition, onsets in conditions.items():
        if len(onsets) > 0:
            samples = (onsets * sfreq).astype(int)
            codes = [code_map[condition]] * len(samples)
            all_samples.extend(samples)
            all_codes.extend(codes)
    
    if len(all_samples) == 0:
        return np.empty((0, 3), dtype=int), {}
    
    # Sort by sample time
    sort_idx = np.argsort(all_samples)
    samples_sorted = np.array(all_samples)[sort_idx]
    codes_sorted = np.array(all_codes)[sort_idx]
    
    events = np.column_stack([
        samples_sorted,
        np.zeros(len(samples_sorted), dtype=int),
        codes_sorted
    ])
    
    event_id = {
        'target': EVENT_CODE_TARGET,
        'novelty': EVENT_CODE_NOVELTY,
        'standard': EVENT_CODE_STANDARD
    }
    
    return events, event_id


def epoch_and_reject_condition(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_code: int
) -> Optional[Epochs]:
    """
    Create epochs for a specific oddball condition and apply artifact rejection.
    
    Multi-stage artifact detection:
    1. Peak-to-peak amplitude check (>150µV = likely artifact)
    2. Kurtosis check (high kurtosis = non-Gaussian noise/artifacts)
    3. Muscle noise detection (high power in 20-40 Hz band)
    """
    picks = pick_types(raw.info, meg=False, eeg=True, eog=False, exclude="bads")
    
    # Filter events to only this condition
    condition_events = events[events[:, 2] == event_code]
    
    if len(condition_events) == 0:
        return None
    
    try:
        epochs = Epochs(
            raw, condition_events, event_id={str(event_code): event_code},
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=BASELINE,
            preload=True,
            picks=picks,
            verbose="warning"
        )
    except Exception as e:
        logger.warning(f"    Failed to create epochs for code {event_code}: {e}")
        return None

    if len(epochs) == 0:
        return None

    data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    sfreq = epochs.info["sfreq"]
    valid_mask = np.ones(len(epochs), dtype=bool)
    
    # ═ STAGE 1: Peak-to-Peak Amplitude Rejection ═
    ptp = np.ptp(data, axis=2)
    ptp_keep = ~np.any(ptp > REJECT_PTP_EPOCH, axis=1)
    n_reject_ptp = (~ptp_keep).sum()
    
    # ═ STAGE 2: Kurtosis-Based Rejection ═
    kurt_threshold = 3.0  # std deviations
    epoch_kurtoses = []
    for epoch_idx in range(len(epochs)):
        epoch = data[epoch_idx]
        epoch_kurt = np.mean([kurtosis(epoch[ch]) for ch in range(epoch.shape[0])])
        epoch_kurtoses.append(epoch_kurt)
    
    epoch_kurtoses = np.array(epoch_kurtoses)
    kurt_mean = np.mean(epoch_kurtoses)
    kurt_std = np.std(epoch_kurtoses) + 1e-8
    kurt_zscore = (epoch_kurtoses - kurt_mean) / kurt_std
    kurt_keep = np.abs(kurt_zscore) <= kurt_threshold
    n_reject_kurt = (~kurt_keep).sum()
    
    # ═ STAGE 3: Muscle Noise Detection (20-40 Hz) ═
    muscle_keep = np.ones(len(epochs), dtype=bool)
    n_reject_muscle = 0
    try:
        for epoch_idx in range(len(epochs)):
            epoch = data[epoch_idx]
            # Compute mean power spectrum
            freqs, psd = signal.welch(epoch, sfreq=sfreq, nperseg=int(sfreq*2), axis=-1)
            
            # Power in different bands
            brain_band = psd[:, (freqs >= 1) & (freqs <= 12)].mean()
            muscle_band = psd[:, (freqs >= 20) & (freqs <= 40)].mean()
            
            # If muscle band >> brain band, likely muscle artifact
            if brain_band > 0 and muscle_band > 3 * brain_band:
                muscle_keep[epoch_idx] = False
                n_reject_muscle += 1
    except Exception as e:
        logger.warning(f"    Could not compute muscle noise detection: {e}")
    
    # ═ COMBINE REJECTION CRITERIA ═
    valid_mask = ptp_keep & kurt_keep & muscle_keep
    n_reject_total = (~valid_mask).sum()
    
    if n_reject_total > 0:
        rejection_breakdown = []
        if n_reject_ptp > 0:
            rejection_breakdown.append(f"PTP:{n_reject_ptp}")
        if n_reject_kurt > 0:
            rejection_breakdown.append(f"Kurt:{n_reject_kurt}")
        if n_reject_muscle > 0:
            rejection_breakdown.append(f"Muscle:{n_reject_muscle}")
        
        breakdown_str = " + ".join(rejection_breakdown) if rejection_breakdown else "multiple"
        logger.info(f"    Rejected {n_reject_total} / {len(epochs)} epochs ({breakdown_str})")
    else:
        logger.info(f"    ✓ All {len(epochs)} epochs passed artifact detection")
    
    epochs = epochs[valid_mask]
    return epochs


# ============================================================
# Feature Extraction
# ============================================================

def extract_enhanced_features(
    eeg_data: np.ndarray,
    sfreq: float = RESAMPLE_SFREQ,
    freq_bands: dict = None
) -> Tuple[np.ndarray, List[str]]:
    """Extract 590 features from EEG epoch."""
    if freq_bands is None:
        freq_bands = FFT_FREQ_BANDS

    # Handle potential NaN/inf in input
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    freqs, psd = signal.welch(
        eeg_data, sfreq,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        axis=-1
    )
    
    # Ensure no NaN/inf in PSD
    psd = np.nan_to_num(psd, nan=1e-10, posinf=1e-10, neginf=1e-10)

    n_channels = eeg_data.shape[0]
    features = []
    feature_names = []

    # Band powers
    band_powers = {}
    for band_name, (f_low, f_high) in freq_bands.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        band_power = psd[:, band_mask].mean(axis=1)
        band_powers[band_name] = band_power

    total_power = psd[:, (freqs >= 0.5) & (freqs <= 40)].mean(axis=1)
    total_power = np.maximum(total_power, 1e-10)  # Avoid division by zero

    # 1. Absolute band power
    for band_name in freq_bands.keys():
        features.append(band_powers[band_name])
        for ch in range(n_channels):
            feature_names.append(f"{band_name}_abs_power_ch{ch}")

    # 2. Relative band power
    for band_name in freq_bands.keys():
        rel_power = band_powers[band_name] / (total_power + 1e-8)
        features.append(rel_power)
        for ch in range(n_channels):
            feature_names.append(f"{band_name}_rel_power_ch{ch}")

    # 3. Power ratios
    da_ratio = band_powers["delta"] / (band_powers["alpha"] + 1e-8)
    features.append(da_ratio)
    for ch in range(n_channels):
        feature_names.append(f"delta_alpha_ratio_ch{ch}")

    tb_ratio = band_powers["theta"] / (band_powers["beta"] + 1e-8)
    features.append(tb_ratio)
    for ch in range(n_channels):
        feature_names.append(f"theta_beta_ratio_ch{ch}")

    ta_ratio = band_powers["theta"] / (band_powers["alpha"] + 1e-8)
    features.append(ta_ratio)
    for ch in range(n_channels):
        feature_names.append(f"theta_alpha_ratio_ch{ch}")

    # 4. Spectral entropy
    normalized_psd = psd / (psd.sum(axis=1, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-10), axis=1)
    features.append(spectral_entropy)
    for ch in range(n_channels):
        feature_names.append(f"spectral_entropy_ch{ch}")

    # 5. Peak frequency per band
    for band_name, (f_low, f_high) in freq_bands.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        band_freqs = freqs[band_mask]
        band_psd = psd[:, band_mask]
        peak_freqs = band_freqs[np.argmax(band_psd, axis=1)]
        features.append(peak_freqs)
        for ch in range(n_channels):
            feature_names.append(f"{band_name}_peak_freq_ch{ch}")

    # 6. Central frequency per band
    for band_name, (f_low, f_high) in freq_bands.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        band_freqs = freqs[band_mask]
        band_psd = psd[:, band_mask]
        central_freqs = np.sum(band_freqs[np.newaxis, :] * band_psd, axis=1) / (np.sum(band_psd, axis=1) + 1e-8)
        features.append(central_freqs)
        for ch in range(n_channels):
            feature_names.append(f"{band_name}_central_freq_ch{ch}")

    features = np.concatenate(features).astype(np.float32)
    # Final NaN/inf cleanup
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, feature_names


def extract_features_from_epochs(epochs: mne.Epochs) -> Tuple[np.ndarray, List[str]]:
    """Extract features from all epochs."""
    eeg_picks = mne.pick_types(epochs.info, eeg=True)
    data = epochs.get_data(picks=eeg_picks)

    X_features = []
    all_feature_names = None

    for epoch_idx in range(data.shape[0]):
        eeg_epoch = data[epoch_idx]
        features, feature_names = extract_enhanced_features(eeg_epoch)
        X_features.append(features)
        if all_feature_names is None:
            all_feature_names = feature_names

    X_features = np.array(X_features, dtype=np.float32)
    return X_features, all_feature_names


def match_condition_trials(
    target_epochs: np.ndarray,
    novelty_epochs: np.ndarray,
    random_seed: int = RANDOM_SEED_MATCHING
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match target and novelty trial counts by random downsampling.
    
    Returns:
    - matched_target: (n_matched, n_features)
    - matched_novelty: (n_matched, n_features)
    - n_matched: number of matched trials per condition
    """
    n_target = len(target_epochs)
    n_novelty = len(novelty_epochs)
    n_matched = min(n_target, n_novelty)
    
    if n_matched == 0:
        return np.empty((0, target_epochs.shape[1]), dtype=np.float32), \
               np.empty((0, novelty_epochs.shape[1]), dtype=np.float32), \
               0
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(random_seed)
    
    # Randomly select n_matched trials from each condition
    if n_target > n_matched:
        target_indices = rng.choice(n_target, size=n_matched, replace=False)
        matched_target = target_epochs[target_indices]
    else:
        matched_target = target_epochs.copy()
    
    if n_novelty > n_matched:
        novelty_indices = rng.choice(n_novelty, size=n_matched, replace=False)
        matched_novelty = novelty_epochs[novelty_indices]
    else:
        matched_novelty = novelty_epochs.copy()
    
    return matched_target, matched_novelty, n_matched


def concatenate_condition_features(
    target_features: np.ndarray,
    novelty_features: np.ndarray,
    target_feature_names: List[str],
    novelty_feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate target and novelty features horizontally.
    
    Adds condition prefix to feature names for clarity.
    
    Returns:
    - concatenated_X: (n_trials, 2 * n_features_per_condition)
    - concatenated_names: condition-aware feature names
    """
    # Add condition prefix to feature names
    prefixed_target_names = [f"target_{name}" for name in target_feature_names]
    prefixed_novelty_names = [f"novelty_{name}" for name in novelty_feature_names]
    
    # Concatenate horizontally
    concatenated_X = np.concatenate([target_features, novelty_features], axis=1).astype(np.float32)
    concatenated_names = prefixed_target_names + prefixed_novelty_names
    
    return concatenated_X, concatenated_names


def select_knn_optimized_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: int = 120
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Select top features via Mutual Information."""
    if not HAS_SKLEARN:
        logger.warning("⚠️  sklearn not available, selecting via statistical test")
        p_values = []
        for feat_idx in range(X.shape[1]):
            pd_vals = X[y == 1, feat_idx]
            ctl_vals = X[y == 0, feat_idx]
            if np.std(pd_vals) < 1e-6 or np.std(ctl_vals) < 1e-6:
                p_values.append(1.0)
                continue
            stat, p_val = mannwhitneyu(pd_vals, ctl_vals)
            p_values.append(p_val)
        p_values = np.array(p_values)
        top_indices = np.argsort(p_values)[:n_features]
        importance = -np.log10(p_values[top_indices] + 1e-10)
    else:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        top_indices = np.argsort(mi_scores)[-n_features:][::-1]
        importance = mi_scores[top_indices]
        logger.info(f"  ✓ MI scores: {importance.min():.6f} to {importance.max():.6f}")

    X_selected = X[:, top_indices]
    selected_names = [feature_names[i] for i in top_indices]

    return X_selected, selected_names, importance


def subject_wise_zscore(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Normalize each subject's features independently."""
    X_norm = X.copy()
    for subj_id in np.unique(groups):
        mask = groups == subj_id
        mean = X[mask].mean(axis=0, keepdims=True)
        std = X[mask].std(axis=0, keepdims=True) + 1e-8
        X_norm[mask] = (X[mask] - mean) / std
    return X_norm


# ============================================================
# Main Preprocessing Pipeline
# ============================================================

def main():
    logger.info("=" * 75)
    logger.info("KNN PREPROCESSING — TARGET + NOVELTY CONCATENATION")
    logger.info("=" * 75)

    # Load dataset info
    summary_tsv = OUT_DIR / "preprocessing_summary.tsv"
    if not summary_tsv.exists():
        logger.error(f"preprocessing_summary.tsv not found: {summary_tsv}")
        raise FileNotFoundError(f"Missing: {summary_tsv}")

    summary_df = pd.read_csv(summary_tsv, sep="\t", dtype={"subject": str})
    logger.info(f"Loaded {len(summary_df)} subjects")

    logger.info("Using direct file access (skipping BIDSLayout for performance)")

    X_all = []
    y_all = []
    groups_all = []
    feature_names_all = None
    subject_index_map = {}
    preprocessing_stats = []

    # Process each subject
    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Preprocessing subjects"):
        subj = str(row["subject"]).strip().zfill(3)
        session = str(row.get("session", "ses-01")).strip()
        group = str(row["group"]).strip().upper()
        label = 1 if group == "PD" else 0

        try:
            # Build BIDS path
            raw_path = BIDS_ROOT / f"sub-{subj}" / session / "eeg" / f"sub-{subj}_{session}_task-Rest_eeg.set"
            events_path = BIDS_ROOT / f"sub-{subj}" / session / "eeg" / f"sub-{subj}_{session}_task-Rest_events.tsv"

            if not raw_path.exists() or not events_path.exists():
                logger.warning(f"sub-{subj}: files not found, skipping")
                continue

            logger.info(f"\nProcessing sub-{subj} ({group}, {session})...")

            # ─────────────────────────────────────────────────────────────
            # Load raw data
            raw = mne.io.read_raw_eeglab(str(raw_path), preload=True, verbose=False)
            logger.info(f"  ✓ Loaded: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels")

            # Setup channel types
            raw = force_eeg_channel_types(raw)
            safe_set_montage(raw)

            # Bad channel detection
            raw, bad_channels = detect_and_interpolate_bad_channels(raw, subj)

            # Reference
            ref_ch, ground_ch = read_sidecar_ref_and_ground_direct(raw_path.parent)
            safe_set_reference(raw, ref_ch=ref_ch, ground_ch=ground_ch)

            # Filtering
            notch_freq = read_sidecar_notch_freq_direct(raw_path.parent)
            raw = apply_filters(raw, notch_freq=notch_freq)

            # Resample
            raw = ensure_sfreq(raw, RESAMPLE_SFREQ)
            logger.info(f"  ✓ Resampled to {RESAMPLE_SFREQ} Hz")

            # ICA
            has_eog = any(ch.upper().startswith("EOG") for ch in raw.ch_names)
            has_ecg = any(ch.upper().startswith("ECG") for ch in raw.ch_names)
            ica = fit_ica(raw, subj)
            raw, ica_excluded = apply_ica(ica, raw, has_eog=has_eog)

            # Channel selection
            raw = select_channels(raw)
            n_eeg = len(raw.copy().pick_types(eeg=True).ch_names)
            if n_eeg < 8:
                logger.warning(f"sub-{subj}: too few EEG channels ({n_eeg}), skipping")
                continue

            # ─────────────────────────────────────────────────────────────
            # Extract oddball events
            events_df = pd.read_csv(events_path, sep="\t")
            
            # Extract target and novelty condition onsets
            oddball_conditions = extract_oddball_conditions(events_df)
            
            n_target_raw = len(oddball_conditions['target'])
            n_novelty_raw = len(oddball_conditions['novelty'])
            n_standard_raw = len(oddball_conditions['standard'])
            
            logger.info(f"  Found oddball events: target={n_target_raw}, novelty={n_novelty_raw}, standard={n_standard_raw}")
            
            if n_target_raw == 0 or n_novelty_raw == 0:
                logger.warning(f"sub-{subj}: missing target or novelty events, skipping")
                continue
            
            # Create MNE events array
            events, event_id = create_oddball_events(oddball_conditions, raw.info["sfreq"])
            
            if len(events) == 0:
                logger.warning(f"sub-{subj}: no oddball events found, skipping")
                continue

            # ─────────────────────────────────────────────────────────────
            # Create condition-specific epochs with artifact rejection
            logger.info(f"  Creating condition-specific epochs...")
            
            target_epochs = epoch_and_reject_condition(raw, events, EVENT_CODE_TARGET)
            novelty_epochs = epoch_and_reject_condition(raw, events, EVENT_CODE_NOVELTY)
            
            if target_epochs is None or novelty_epochs is None:
                logger.warning(f"sub-{subj}: failed to create epochs for one or both conditions, skipping")
                continue
            
            n_target_after = len(target_epochs)
            n_novelty_after = len(novelty_epochs)
            
            logger.info(f"    After artifact rejection: target={n_target_after}, novelty={n_novelty_after}")
            
            if n_target_after == 0 or n_novelty_after == 0:
                logger.warning(f"sub-{subj}: all epochs rejected for one or both conditions, skipping")
                continue

            # ─────────────────────────────────────────────────────────────
            # Extract features per condition
            logger.info(f"  Extracting features per condition...")
            
            X_target, target_feat_names = extract_features_from_epochs(target_epochs)
            X_novelty, novelty_feat_names = extract_features_from_epochs(novelty_epochs)
            
            if feature_names_all is None:
                feature_names_all = target_feat_names
                logger.info(f"  ✓ Extracted {len(target_feat_names)} features per condition")

            # ─────────────────────────────────────────────────────────────
            # Match trial counts and concatenate features
            logger.info(f"  Matching trial counts...")
            
            X_target_matched, X_novelty_matched, n_matched = match_condition_trials(
                X_target, X_novelty, random_seed=RANDOM_SEED_MATCHING
            )
            
            if n_matched == 0:
                logger.warning(f"sub-{subj}: no matched trials after downsampling, skipping")
                continue
            
            logger.info(f"    Matched trials: {n_matched} per condition")
            
            # Concatenate features
            X_subj_concat, concat_feat_names = concatenate_condition_features(
                X_target_matched, X_novelty_matched,
                target_feat_names, novelty_feat_names
            )
            
            logger.info(f"    Concatenated shape: {X_subj_concat.shape}")

            # ─────────────────────────────────────────────────────────────
            # Track subject
            if subj not in subject_index_map:
                subject_index_map[subj] = len(subject_index_map)

            subj_idx = subject_index_map[subj]

            X_all.append(X_subj_concat)
            y_all.extend([label] * len(X_subj_concat))
            groups_all.extend([subj_idx] * len(X_subj_concat))

            preprocessing_stats.append({
                "subject": subj,
                "group": group,
                "n_target_raw": n_target_raw,
                "n_novelty_raw": n_novelty_raw,
                "n_target_after_qc": n_target_after,
                "n_novelty_after_qc": n_novelty_after,
                "n_matched_per_condition": n_matched,
                "n_bad_channels": len(bad_channels),
                "n_ica_excluded": len(ica_excluded),
                "notch_hz": notch_freq,
            })

            logger.info(f"  ✓ sub-{subj} complete: {len(X_subj_concat)} epochs, {len(concat_feat_names)} total features")

        except Exception as e:
            logger.error(f"sub-{subj}: {str(e)[:100]}")
            continue

    # ─────────────────────────────────────────────────────────────
    # Combine features
    if not X_all:
        raise RuntimeError("No usable epochs extracted")

    X = np.vstack(X_all)
    y = np.array(y_all, dtype=int)
    groups = np.array(groups_all, dtype=int)

    logger.info(f"\n{'='*75}")
    logger.info(f"Feature extraction complete:")
    logger.info(f"  Total epochs: {X.shape[0]}")
    logger.info(f"  PD: {(y==1).sum()}, CTL: {(y==0).sum()}")
    logger.info(f"  Total features (target + novelty concatenated): {X.shape[1]}")
    logger.info(f"  Subjects: {len(np.unique(groups))}")

    # ═══════════════════════════════════════════════════════════════
    # NOTE: Feature selection and normalization must happen INSIDE
    # the training CV folds to avoid data leakage. Do NOT apply them
    # at the preprocessing stage. The full feature matrix is saved
    # here for downstream ML training.
    # ═══════════════════════════════════════════════════════════════

    # Save full feature matrix
    out_path = OUT_DIR / "FEATURES_KNN_TARGET_NOVELTY_CONCAT.npz"
    np.savez(
        out_path,
        X=X,
        y=y,
        groups=groups,
        feature_names=np.array(concat_feat_names, dtype=object),
        n_channels=len([ch for ch in TARGET_CHANNELS if ch]),
        feature_mode="target_novelty_concat"
    )
    logger.info(f"\n{'='*75}")
    logger.info(f"✅ Preprocessing complete!")
    logger.info(f"✅ Features saved to: {out_path}")
    logger.info(f"   → {X.shape[1]} features per subject")
    logger.info(f"   → Feature selection & normalization deferred to CV training")
    logger.info(f"{'='*75}")

    # Save detailed stats
    stats_df = pd.DataFrame(preprocessing_stats)
    stats_path = OUT_DIR / "preprocessing_knn_target_novelty_stats.tsv"
    stats_df.to_csv(stats_path, sep="\t", index=False)
    logger.info(f"✅ Stats saved to: {stats_path}")

    return X, y, groups, concat_feat_names


if __name__ == "__main__":
    main()
