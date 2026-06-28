# ============================================================
# preprocessing.py  —  EEG Preprocessing (local VS Code version)
# ============================================================
# Before running, install dependencies in your terminal:
#   pip install mne mne-bids pybids tqdm pandas numpy matplotlib
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
matplotlib.use("Agg")          # non-interactive backend — swap to "TkAgg" or
import matplotlib.pyplot as plt  # "Qt5Agg" if you want pop-up windows locally

import mne
from mne.preprocessing import ICA
from mne import Epochs, pick_types

from bids import BIDSLayout
from mne_bids import BIDSPath, read_raw_bids


# ============================================================
# Paths  —  UPDATE THESE to match your local folder layout
# ============================================================
BIDS_ROOT = Path("/Users/dibadabiransari/Desktop/EEG Project/data/ds003490")   # folder containing participants.tsv
OUT_DIR   = Path("/Users/dibadabiransari/Desktop/EEG Project/Signal Code/Outputs")          # all outputs go here

CACHE_EPOCHS = OUT_DIR / "cache_epochs_binary"
PLOTS_DIR    = OUT_DIR / "qc_plots_binary"

CACHE_EPOCHS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("preprocess_binary_pd_ctl")


# ============================================================
# Preprocessing settings
# ============================================================
# CHANGED: extended to full 60-channel set so the FFT classifier
# (train.py, Step5/Step8 equivalent) has full spatial resolution.
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

RESAMPLE_SFREQ = 100.0   # CHANGED: MATLAB Step4 uses 100 Hz (pop_resample(EEG,100))
BANDPASS       = (0.5, 40.0)
ICA_HP         = 1.0

ICA_NCOMP = 0.999

DEFAULT_NOTCH_FREQ = 60.0   # US-recorded dataset uses 60 Hz mains

# Epoch timing
TOTAL_EPOCH_SEC = 1.0
TMIN            = -0.2 * TOTAL_EPOCH_SEC
TMAX            =  0.8 * TOTAL_EPOCH_SEC
BASELINE        = (TMIN, 0)

REJECT_PTP_EVENT      = 150e-6   # 150 µV  — for event-locked epochs
REJECT_PTP_FIXED      = 200e-6   # 200 µV  — for resting-state / fixed-length epochs

FIXED_EPOCH_LEN     = 1.0
FIXED_OVERLAP_FRAC  = 0.2

# Bad channel detection thresholds
BAD_CHAN_FLAT_THRESH    = 1e-6
BAD_CHAN_NOISY_Z_THRESH = 3.0
BAD_CHAN_EOG_PROTECT    = {"Fp1", "Fp2", "FP1", "FP2"}
BAD_CHAN_MAX_FRACTION   = 0.3

SHOW_PLOTS                        = True   # set False to skip all plots
CLEAR_OUTPUT_EACH_SUBJ            = False  # no-op outside Jupyter; safe to leave
MAX_INITIAL_SUBJECTS_WITHOUT_SAVE = 3


# ============================================================
# Epoch strategy enum
# ============================================================

class EpochStrategy(Enum):
    EVENT_BASED  = auto()
    FIXED_LENGTH = auto()
    FAILED       = auto()


# ============================================================
# Helper: Read PowerLineFrequency from sidecar
# ============================================================

def _find_sidecar(bids_path: BIDSPath) -> Optional[Path]:
    candidate = bids_path.fpath.with_suffix("").with_suffix(".json")
    if candidate.exists():
        return candidate
    candidate2 = Path(str(bids_path.fpath).replace(bids_path.fpath.suffix, ".json"))
    if candidate2.exists():
        return candidate2
    return None


def read_sidecar_notch_freq(bids_path: BIDSPath) -> float:
    sidecar = _find_sidecar(bids_path)
    if sidecar:
        try:
            with open(sidecar) as fh:
                meta = json.load(fh)
            freq = float(meta.get("PowerLineFrequency", DEFAULT_NOTCH_FREQ))
            logger.info(f"Sidecar PowerLineFrequency = {freq} Hz (from {sidecar.name})")
            return freq
        except Exception as e:
            logger.warning(f"Could not read PowerLineFrequency from sidecar: {e}")
    logger.info(f"Using default notch frequency: {DEFAULT_NOTCH_FREQ} Hz")
    return DEFAULT_NOTCH_FREQ


def read_sidecar_ref_and_ground(
    bids_path: BIDSPath
) -> Tuple[Optional[str], Optional[str]]:
    ref_ch = ground_ch = None
    sidecar = _find_sidecar(bids_path)
    if sidecar:
        try:
            with open(sidecar) as fh:
                meta = json.load(fh)
            _null = {"N/A", "NONE", "NAN", ""}
            raw_ref = str(meta.get("EEGReference", "")).strip()
            raw_gnd = str(meta.get("EEGGround",    "")).strip()
            if raw_ref.upper() not in _null:
                ref_ch = raw_ref
            if raw_gnd.upper() not in _null:
                ground_ch = raw_gnd
            logger.info(f"Sidecar EEGReference='{ref_ch}', EEGGround='{ground_ch}'")
        except Exception as e:
            logger.warning(f"Could not read reference/ground from sidecar: {e}")
    return ref_ch, ground_ch


def read_sidecar_channel_counts(bids_path: BIDSPath) -> Tuple[bool, bool]:
    sidecar = _find_sidecar(bids_path)
    if sidecar:
        try:
            with open(sidecar) as fh:
                meta = json.load(fh)
            has_eog = int(meta.get("EOGChannelCount", 0)) > 0
            has_ecg = int(meta.get("ECGChannelCount", 0)) > 0
            return has_eog, has_ecg
        except Exception as e:
            logger.warning(f"Could not read channel counts from sidecar: {e}")
    return False, False


def read_coordsystem(bids_path: BIDSPath) -> Optional[str]:
    try:
        session_dir = bids_path.fpath.parent
        coord_files = list(session_dir.glob("*_coordsystem.json"))
        if coord_files:
            with open(coord_files[0]) as fh:
                coord = json.load(fh)
            sys_name = coord.get("EEGCoordinateSystem", None)
            sys_desc = coord.get("EEGCoordinateSystemDescription", "")
            logger.info(
                f"coordsystem.json: EEGCoordinateSystem='{sys_name}', "
                f"description='{sys_desc}'"
            )
            return sys_name
    except Exception as e:
        logger.warning(f"Could not read coordsystem.json: {e}")
    return None


# ============================================================
# Bad channel detection and interpolation
# ============================================================

def detect_and_interpolate_bad_channels(
    raw: mne.io.BaseRaw,
    subj: str,
    fname: str
) -> Tuple[mne.io.BaseRaw, List[str]]:
    raw      = raw.copy().load_data()
    data     = raw.copy().pick("eeg").get_data()
    ch_names = raw.copy().pick("eeg").ch_names
    n_ch     = len(ch_names)
    protected = {ch.upper() for ch in BAD_CHAN_EOG_PROTECT}

    bad_channels: List[str] = []

    channel_sds = np.std(data, axis=1)
    flat_mask   = channel_sds < BAD_CHAN_FLAT_THRESH
    flat_chs    = [ch_names[i] for i in range(n_ch) if flat_mask[i]]
    if flat_chs:
        logger.info(f"sub-{subj} ({fname}): flat channels detected: {flat_chs}")
    bad_channels.extend(flat_chs)

    non_flat_sds = channel_sds[~flat_mask]
    if len(non_flat_sds) > 1:
        sd_mean = np.mean(non_flat_sds)
        sd_std  = np.std(non_flat_sds)
        if sd_std > 0:
            z_scores   = (channel_sds - sd_mean) / sd_std
            noisy_mask = (
                (z_scores > BAD_CHAN_NOISY_Z_THRESH)
                & (~flat_mask)
                & np.array([ch.upper() not in protected for ch in ch_names])
            )
            noisy_chs = [ch_names[i] for i in range(n_ch) if noisy_mask[i]]
            if noisy_chs:
                logger.info(
                    f"sub-{subj} ({fname}): noisy channels "
                    f"(z > {BAD_CHAN_NOISY_Z_THRESH}): {noisy_chs}"
                )
            bad_channels.extend(noisy_chs)

    seen, bad_unique = set(), []
    for ch in bad_channels:
        if ch not in seen:
            bad_unique.append(ch)
            seen.add(ch)
    bad_channels = bad_unique

    bad_fraction = len(bad_channels) / n_ch if n_ch > 0 else 0.0
    if bad_fraction > BAD_CHAN_MAX_FRACTION:
        raise ValueError(
            f"sub-{subj} ({fname}): {len(bad_channels)}/{n_ch} channels "
            f"({bad_fraction:.0%}) flagged as bad — exceeds "
            f"BAD_CHAN_MAX_FRACTION ({BAD_CHAN_MAX_FRACTION:.0%}). "
            f"Skipping file to avoid unreliable interpolation."
        )

    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose="warning")
        logger.info(
            f"sub-{subj} ({fname}): interpolated {len(bad_channels)} "
            f"bad channel(s): {bad_channels}"
        )
    else:
        logger.info(f"sub-{subj} ({fname}): no bad channels detected.")

    return raw, bad_channels


# ============================================================
# Sanity checks
# ============================================================

def assert_path_ok():
    if not BIDS_ROOT.exists():
        raise FileNotFoundError(f"BIDS_ROOT does not exist: {BIDS_ROOT}")
    participants_path = BIDS_ROOT / "participants.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found in: {BIDS_ROOT}")
    test_file = OUT_DIR / "_write_test.txt"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except Exception as e:
        raise RuntimeError(
            f"Cannot write to OUT_DIR: {OUT_DIR}. "
            "Check that the folder exists and you have write permissions."
        ) from e


assert_path_ok()
logger.info("✅ Paths look OK.")


# ============================================================
# Subject / session selection
# ============================================================

def load_participants_df(bids_root: Path) -> pd.DataFrame:
    return pd.read_csv(bids_root / "participants.tsv", sep="\t")


def build_subject_selection_map(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    CHANGED (guided by MATLAB Step4):
    Now tracks the medication label (ON / OFF / CTL) for every session so
    preprocessing saves separate .fif files per medication status:
        sub-XXX_ON-epo.fif    sub-XXX_OFF-epo.fif    sub-XXX_CTL-epo.fif

    This mirrors Step4's three sessioni slots:
        sessioni=1 -> PD ON   sessioni=2 -> PD OFF   sessioni=3 -> CTL
    which train.py uses for COMPARISON in {'ON_CTL', 'OFF_CTL', 'ON_OFF'}.
    """
    req = {"participant_id", "Group", "sess1_Med", "sess2_Med"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"participants.tsv missing required columns: {missing}")

    selection_map: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        pid   = str(row["participant_id"]).strip()
        subj  = pid.replace("sub-", "")
        group = str(row["Group"]).strip().upper()
        s1    = str(row["sess1_Med"]).strip().upper()
        s2    = str(row["sess2_Med"]).strip().upper()

        if group == "PD":
            # Keep BOTH sessions with their ON/OFF label so train.py can
            # load the correct session for each comparison group.
            s1_med = s1 if s1 in {"ON", "OFF"} else "UNKNOWN"
            s2_med = s2 if s2 in {"ON", "OFF"} else None
            sessions = {
                "sess1": {"keep": True,                "med": s1_med},
                "sess2": {"keep": s2_med is not None,  "med": s2_med},
            }
            selection_map[subj] = {"group": "PD", "sessions": sessions}

        elif group in {"CTL", "CONTROL"}:
            s2_keep = s2 not in {"NO S2", "NONE", "NAN"}
            sessions = {
                "sess1": {"keep": True,    "med": "CTL"},
                "sess2": {"keep": s2_keep, "med": "CTL"},
            }
            selection_map[subj] = {"group": "CTL", "sessions": sessions}

    return selection_map


_SESSION_RE = re.compile(r"(?:ses(?:sion)?[-_]?)?0*([12])\b", re.IGNORECASE)

def session_entity_to_index(session_val) -> int:
    if session_val is None:
        return -1
    m = _SESSION_RE.search(str(session_val).strip())
    if m:
        return int(m.group(1))
    return -1


def select_subject_files(
    layout: BIDSLayout,
    subj: str,
    selection_map: Dict[str, Dict]
) -> List[str]:
    info = selection_map.get(subj)
    if info is None:
        return []

    wanted = {
        session_name
        for session_name, session_info in info.get("sessions", {}).items()
        if session_info.get("keep")
    }
    files  = layout.get(
        subject=subj,
        extension=[".set", ".vhdr", ".edf", ".eeg", ".cnt", ".bdf"],
        return_type="filename"
    ) or []

    if not files:
        return []

    selected           = []
    has_session_entity = False

    for f in files:
        ent = layout.parse_file_entities(f)
        ses = ent.get("session")
        if ses is not None:
            has_session_entity = True
            idx = session_entity_to_index(ses)
            if idx == 1 and "sess1" in wanted:
                selected.append(f)
            if idx == 2 and "sess2" in wanted:
                selected.append(f)

    if not has_session_entity:
        tok_sess1 = ["ses-1", "ses-01", "s1", "sess1"]
        tok_sess2 = ["ses-2", "ses-02", "s2", "sess2"]
        for f in files:
            lf = f.lower()
            if "sess1" in wanted and any(t in lf for t in tok_sess1):
                selected.append(f)
            if "sess2" in wanted and any(t in lf for t in tok_sess2):
                selected.append(f)

    if not selected and info["group"] == "CTL":
        selected = files.copy()

    out, seen = [], set()
    for f in selected:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


# ============================================================
# Preprocessing steps
# ============================================================

def force_eeg_channel_types(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    eeg_like = {
        "FP1","FP2","F3","F4","FZ","FC1","FC2","FCZ","C3","C4","CZ","P3","P4","PZ","T7","T8",
        "F7","F8","C1","C2","C5","C6","P7","P8","O1","O2","OZ","AF3","AF4","AF7","AF8",
        "FT7","FT8","FT9","FT10","TP7","TP8","TP9","TP10","CP1","CP2","CP3","CP4","CP5","CP6",
        "PO3","PO4","PO7","PO8","POZ","P1","P2","P5","P6","F1","F2","F5","F6",
        "CPZ","FCZ","FPZ","AFZ","IZ","P9","P10","FC3","FC4","FC5","FC6",
    }

    mapping = {ch: "eeg" for ch in raw.ch_names if ch.strip().upper() in eeg_like}
    if mapping:
        raw.set_channel_types(mapping)

    eog_mapping = {
        ch: "eog" for ch in raw.ch_names
        if ch.strip().upper() in {"VEOG", "HEOG", "EOG"}
    }
    if eog_mapping:
        raw.set_channel_types(eog_mapping)

    return raw


def safe_set_montage(
    raw: mne.io.BaseRaw,
    coord_system: Optional[str] = None
) -> None:
    if coord_system and coord_system.upper() not in {
        "OTHER", "STANDARD_1020", "10-20", "ARS"
    }:
        logger.warning(
            f"coordsystem.json reports '{coord_system}' — verify that "
            f"standard_1020 montage is appropriate for this dataset."
        )
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass


def safe_set_reference(
    raw: mne.io.BaseRaw,
    ref_ch: Optional[str] = None,
    ground_ch: Optional[str] = None
) -> None:
    try:
        if ground_ch and ground_ch in raw.ch_names:
            raw.set_channel_types({ground_ch: "misc"})
            logger.info(f"Ground electrode '{ground_ch}' excluded from EEG picks.")

        if ref_ch and ref_ch not in raw.ch_names:
            raw.add_reference_channels(ref_ch)
            logger.info(f"Original reference '{ref_ch}' added back before avg re-ref.")

        raw.set_eeg_reference("average", projection=False)

    except Exception as e:
        logger.warning(f"safe_set_reference failed: {e}")


def ensure_sfreq(raw: mne.io.BaseRaw, target_sfreq: float) -> mne.io.BaseRaw:
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-6:
        logger.info(
            f"Resampling from {raw.info['sfreq']} Hz → {target_sfreq} Hz."
        )
        raw = raw.copy().resample(target_sfreq)
    return raw


def select_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    picks = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
    if not picks:
        return raw
    return raw.copy().pick(picks)


def apply_filters(raw: mne.io.BaseRaw, notch_freq: float) -> mne.io.BaseRaw:
    raw  = raw.copy().load_data()
    nyq  = raw.info["sfreq"] / 2.0
    freqs = [notch_freq]
    if notch_freq * 2 < nyq:
        freqs.append(notch_freq * 2)
    logger.info(f"Notch filter frequencies: {freqs} Hz")
    raw.notch_filter(freqs, picks="eeg", verbose="warning")
    raw.filter(BANDPASS[0], BANDPASS[1], picks="eeg", verbose="warning")
    return raw


def fit_ica(raw_filt: mne.io.BaseRaw, subj: str) -> ICA:
    raw_ica = raw_filt.copy().load_data()
    raw_ica.filter(l_freq=ICA_HP, h_freq=None, picks="eeg", verbose="warning")
    ica = ICA(n_components=ICA_NCOMP, random_state=42, max_iter="auto")
    ica.fit(raw_ica, picks="eeg")
    logger.info(
        f"sub-{subj}: ICA fitted — {ica.n_components_} components "
        f"(variance threshold: {ICA_NCOMP})"
    )
    return ica


def apply_ica(
    ica: ICA,
    raw_filt: mne.io.BaseRaw,
    has_eog: bool = False,
    has_ecg: bool = False
) -> Tuple[mne.io.BaseRaw, List[int]]:
    bad: List[int] = []

    if has_eog:
        eog_chs = [c for c in raw_filt.ch_names if "EOG" in c.upper()]
        if eog_chs:
            try:
                inds, _ = ica.find_bads_eog(raw_filt, ch_name=eog_chs)
                bad += list(inds)
            except Exception:
                pass

    if not bad:
        for proxy in ["Fp1", "Fp2", "FP1", "FP2"]:
            if proxy in raw_filt.ch_names:
                try:
                    inds, _ = ica.find_bads_eog(raw_filt, ch_name=proxy)
                    bad += list(inds)
                    break
                except Exception:
                    continue

    if has_ecg:
        ecg_chs = [c for c in raw_filt.ch_names if "ECG" in c.upper()]
        if ecg_chs:
            try:
                inds, _ = ica.find_bads_ecg(raw_filt, ch_name=ecg_chs[0])
                bad += list(inds)
            except Exception:
                pass

    bad       = sorted(set(bad))
    raw_clean = raw_filt.copy()
    ica.apply(raw_clean, exclude=bad)
    return raw_clean, bad


def get_events(raw: mne.io.BaseRaw):
    if raw.annotations is not None and len(raw.annotations) > 0:
        return mne.events_from_annotations(raw)
    try:
        events   = mne.find_events(raw, verbose="warning")
        event_id = {str(int(x)): int(x) for x in np.unique(events[:, 2])}
        return events, event_id
    except Exception:
        return np.empty((0, 3), dtype=int), {}


def epoch_and_reject(raw: mne.io.BaseRaw, events, event_id) -> Epochs:
    picks  = pick_types(raw.info, meg=False, eeg=True, eog=False, exclude="bads")
    epochs = Epochs(
        raw, events, event_id=event_id,
        tmin=TMIN, tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        picks=picks,
        verbose="warning"
    )
    if len(epochs) > 0:
        ptp     = np.ptp(epochs.get_data(), axis=2)
        keep    = ~np.any(ptp > REJECT_PTP_EVENT, axis=1)
        epochs  = epochs[np.where(keep)[0]]
    return epochs


def make_fixed_length_epochs(raw: mne.io.BaseRaw) -> Epochs:
    picks  = pick_types(raw.info, meg=False, eeg=True, eog=False, exclude="bads")
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=FIXED_EPOCH_LEN,
        overlap=FIXED_OVERLAP_FRAC * FIXED_EPOCH_LEN,
        preload=True,
        picks=picks,
        verbose="warning"
    )
    if len(epochs) > 0:
        ptp    = np.ptp(epochs.get_data(), axis=2)
        keep   = ~np.any(ptp > REJECT_PTP_FIXED, axis=1)
        epochs = epochs[np.where(keep)[0]]
    return epochs


# ============================================================
# QC plots
# ============================================================

def save_fig(fig, save_path: Path):
    """Save figure to disk. Displays it only if SHOW_PLOTS is True."""
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def qc_plots_for_subject(
    subj: str,
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    ica: ICA,
    exclude: List[int],
    bad_channels: List[str],
    strategy: EpochStrategy
):
    print(f"--- QC for sub-{subj} | epoch strategy: {strategy.name} ---")

    try:
        eeg_raw  = raw_before.copy().pick("eeg")
        ch_sds   = np.std(eeg_raw.get_data(), axis=1) * 1e6
        ch_names = eeg_raw.ch_names
        colours  = ["red" if ch in bad_channels else "steelblue" for ch in ch_names]

        fig0, ax = plt.subplots(figsize=(max(10, len(ch_names) * 0.35), 4))
        ax.bar(range(len(ch_names)), ch_sds, color=colours)
        ax.set_xticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax.set_ylabel("SD (µV)")
        ax.set_title(
            f"sub-{subj} channel SD — bad channels in red "
            f"({len(bad_channels)} flagged: {bad_channels or 'none'})"
        )
        ax.axhline(
            BAD_CHAN_FLAT_THRESH * 1e6, color="orange", linestyle="--",
            label=f"flat threshold ({BAD_CHAN_FLAT_THRESH*1e6:.1f} µV)"
        )
        ax.legend(fontsize=8)
        fig0.tight_layout()
        save_fig(fig0, PLOTS_DIR / f"sub-{subj}_bad_channels.png")
    except Exception as e:
        logger.info(f"sub-{subj}: bad channel plot failed (non-critical): {e}")

    fig1 = raw_before.copy().pick("eeg").plot_psd(fmin=1, fmax=80, show=False)
    fig1.suptitle(f"sub-{subj} PSD BEFORE filtering")
    save_fig(fig1, PLOTS_DIR / f"sub-{subj}_psd_before.png")

    fig2 = raw_after.copy().pick("eeg").plot_psd(fmin=1, fmax=80, show=False)
    fig2.suptitle(f"sub-{subj} PSD AFTER filtering ({int(DEFAULT_NOTCH_FREQ)} Hz notch)")
    save_fig(fig2, PLOTS_DIR / f"sub-{subj}_psd_after.png")

    try:
        fig3 = ica.plot_overlay(raw_after, exclude=exclude, picks="eeg", show=False)
        if isinstance(fig3, list):
            fig3 = fig3[0]
        fig3.suptitle(f"sub-{subj} ICA overlay (filtered vs cleaned)")
        save_fig(fig3, PLOTS_DIR / f"sub-{subj}_ica_overlay.png")
    except Exception as e:
        logger.info(f"sub-{subj}: ICA overlay plot failed (non-critical): {e}")


# ============================================================
# Save validation
# ============================================================

def assert_epochs_saved(out_path: Path):
    if not out_path.exists():
        raise RuntimeError(f"Epoch file was not saved: {out_path}")
    if out_path.stat().st_size < 10_000:
        raise RuntimeError(
            f"Epoch file is unexpectedly small: {out_path} "
            f"({out_path.stat().st_size} bytes)"
        )


# ============================================================
# RUN PREPROCESSING
# ============================================================

layout          = BIDSLayout(str(BIDS_ROOT), validate=False)
participants_df = load_participants_df(BIDS_ROOT)
selection_map   = build_subject_selection_map(participants_df)

subjects = sorted(selection_map.keys())
logger.info(f"Subjects selected for binary PD/CTL preprocessing: {len(subjects)}")

subject_summary: List[Dict] = []

saved_subjects:              List[str] = []
attempted_subjects                     = 0
unsaved_subjects_in_a_row              = 0

for subj in tqdm(subjects, desc="Preprocess binary PD/CTL subjects"):
    attempted_subjects += 1

    files = select_subject_files(layout, subj, selection_map)
    if not files:
        logger.warning(
            f"sub-{subj}: no EEG files matched selection rule "
            f"{selection_map[subj]}."
        )
        subject_summary.append({
            "subject": subj, "group": selection_map[subj]["group"],
            "strategy": EpochStrategy.FAILED.name,
            "n_epochs": 0, "n_channels": 0,
            "bad_channels": "", "ica_components": "",
            "notch_hz": "", "notes": "no files matched",
        })
        unsaved_subjects_in_a_row += 1
        if (
            attempted_subjects <= MAX_INITIAL_SUBJECTS_WITHOUT_SAVE
            and not saved_subjects
            and unsaved_subjects_in_a_row >= MAX_INITIAL_SUBJECTS_WITHOUT_SAVE
        ):
            raise RuntimeError(
                "The first few subjects produced no savable epochs. "
                "Stopping early so you do not waste time."
            )
        continue

    epochs_list:    List[mne.Epochs]     = []
    file_strategies: List[EpochStrategy] = []
    subj_bad_channels: List[str]         = []
    subj_ica_ncomp:    List[int]         = []
    last_notch_freq                      = DEFAULT_NOTCH_FREQ

    for f in files:
        try:
            ent       = layout.parse_file_entities(f)
            bids_path = BIDSPath(root=BIDS_ROOT, **ent)

            notch_freq            = read_sidecar_notch_freq(bids_path)
            ref_ch, ground_ch     = read_sidecar_ref_and_ground(bids_path)
            coord_system          = read_coordsystem(bids_path)
            has_eog, has_ecg      = read_sidecar_channel_counts(bids_path)
            last_notch_freq       = notch_freq

            raw = read_raw_bids(bids_path=bids_path, verbose="warning")
            raw = force_eeg_channel_types(raw)
            safe_set_montage(raw, coord_system=coord_system)

            raw, bad_channels = detect_and_interpolate_bad_channels(
                raw, subj=subj, fname=Path(f).name
            )
            subj_bad_channels.extend(bad_channels)

            safe_set_reference(raw, ref_ch=ref_ch, ground_ch=ground_ch)
            raw = ensure_sfreq(raw, RESAMPLE_SFREQ)

            n_eeg_full = len(raw.copy().pick_types(eeg=True).ch_names)
            if n_eeg_full < 8:
                logger.warning(
                    f"sub-{subj}: too few EEG channels ({n_eeg_full}) in "
                    f"{Path(f).name}; skipping file."
                )
                continue

            raw_before = raw.copy()
            raw_filt   = apply_filters(raw, notch_freq=notch_freq)

            ica = fit_ica(raw_filt, subj=subj)
            subj_ica_ncomp.append(ica.n_components_)

            raw_clean, exclude = apply_ica(
                ica, raw_filt, has_eog=has_eog, has_ecg=has_ecg
            )
            raw_clean      = select_channels(raw_clean)
            raw_before_sel = select_channels(raw_before)
            raw_filt_sel   = select_channels(raw_filt)

            n_eeg = len(raw_clean.copy().pick_types(eeg=True).ch_names)
            if n_eeg < 8:
                logger.warning(
                    f"sub-{subj}: too few EEG channels after selection "
                    f"({n_eeg}) in {Path(f).name}; skipping file."
                )
                continue

            events, event_id = get_events(raw_clean)

            if events.size > 0 and event_id:
                epochs   = epoch_and_reject(raw_clean, events, event_id)
                strategy = EpochStrategy.EVENT_BASED

                if len(epochs) == 0:
                    logger.warning(
                        f"sub-{subj}: event markers found but ALL epochs rejected "
                        f"(threshold={REJECT_PTP_EVENT*1e6:.0f} µV). "
                        f"Falling back to fixed-length — REVIEW THIS SUBJECT."
                    )
                    epochs   = make_fixed_length_epochs(raw_clean)
                    strategy = EpochStrategy.FIXED_LENGTH
            else:
                logger.info(
                    f"sub-{subj}: no event markers found in {Path(f).name} "
                    f"→ using fixed-length windows."
                )
                epochs   = make_fixed_length_epochs(raw_clean)
                strategy = EpochStrategy.FIXED_LENGTH

            file_strategies.append(strategy)

            qc_plots_for_subject(
                subj, raw_before_sel, raw_filt_sel, ica, exclude,
                bad_channels, strategy
            )

            if len(epochs) == 0:
                logger.warning(
                    f"sub-{subj}: 0 epochs after fallback in "
                    f"{Path(f).name}; skipping file."
                )
                continue

            epochs_list.append(epochs)

        except Exception as e:
            logger.warning(
                f"sub-{subj}: preprocessing failed for "
                f"{Path(f).name} → {e}"
            )

    if not epochs_list:
        logger.warning(f"sub-{subj}: no usable epochs; skipping subject.")
        subject_summary.append({
            "subject": subj, "group": selection_map[subj]["group"],
            "strategy": EpochStrategy.FAILED.name,
            "n_epochs": 0, "n_channels": 0,
            "bad_channels": "; ".join(sorted(set(subj_bad_channels))),
            "ica_components": "; ".join(map(str, subj_ica_ncomp)),
            "notch_hz": last_notch_freq, "notes": "no usable epochs",
        })
        unsaved_subjects_in_a_row += 1
        if (
            attempted_subjects <= MAX_INITIAL_SUBJECTS_WITHOUT_SAVE
            and not saved_subjects
            and unsaved_subjects_in_a_row >= MAX_INITIAL_SUBJECTS_WITHOUT_SAVE
        ):
            raise RuntimeError(
                "The first few subjects produced no savable epochs. "
                "Stopping early so you do not waste time."
            )
        continue

    merged   = mne.concatenate_epochs(epochs_list)
    out_path = CACHE_EPOCHS / f"sub-{subj}_epochs_binary-epo.fif"
    merged.save(str(out_path), overwrite=True)
    assert_epochs_saved(out_path)

    saved_subjects.append(subj)
    unsaved_subjects_in_a_row = 0

    subj_strategy = (
        EpochStrategy.EVENT_BASED
        if all(s == EpochStrategy.EVENT_BASED for s in file_strategies)
        else EpochStrategy.FIXED_LENGTH
    )

    subject_summary.append({
        "subject":        subj,
        "group":          selection_map[subj]["group"],
        "med_status":     (
            "/".join(sorted({
                session_info["med"]
                for session_info in selection_map[subj]["sessions"].values()
                if session_info.get("keep") and session_info.get("med")
            }))
            or "UNKNOWN"
        ),
        "strategy":       subj_strategy.name,
        "n_epochs":       len(merged),
        "n_channels":     len(merged.ch_names),
        "bad_channels":   "; ".join(sorted(set(subj_bad_channels))),
        "ica_components": "; ".join(map(str, subj_ica_ncomp)),
        "notch_hz":       last_notch_freq,
        "notes":          "",
    })

    print(
        f"✅ Saved: {out_path.name} | "
        f"group={selection_map[subj]['group']} | "
        f"strategy={subj_strategy.name} | "
        f"epochs={len(merged)} | "
        f"channels={len(merged.ch_names)} | "
        f"ICA components={subj_ica_ncomp} | "
        f"notch={last_notch_freq} Hz"
    )


# ============================================================
# Final summary
# ============================================================

epoch_files = sorted(CACHE_EPOCHS.glob("sub-*_epochs_binary-epo.fif"))

logger.info(f"✅ Saved subjects: {len(saved_subjects)}")
logger.info(f"✅ Epoch files: {len(epoch_files)}")
logger.info(f"Epochs folder: {CACHE_EPOCHS}")
logger.info(f"QC plots folder: {PLOTS_DIR}")

syncth_df   = pd.DataFrame(subject_summary)
syncth_path = OUT_DIR / "preprocessing_summary.tsv"
syncth_df.to_csv(syncth_path, sep="\t", index=False)
logger.info(f"✅ Preprocessing summary written to: {syncth_path}")

fixed_length_subjects = syncth_df[
    syncth_df["strategy"] == EpochStrategy.FIXED_LENGTH.name
]["subject"].tolist()
if fixed_length_subjects:
    logger.warning(
        f"⚠️  {len(fixed_length_subjects)} subject(s) used fixed-length epoching "
        f"(no event markers or all event epochs rejected) — review before "
        f"including in analysis: {fixed_length_subjects}"
    )

if len(epoch_files) == 0:
    raise RuntimeError(
        "No epoch files were saved. "
        "Stopping so you do not continue to modelling without data."
    )


# ============================================================
# NEW: Condition-specific epoch extraction — MATLAB Step4 equivalent
# ============================================================

# Condition time window: displayed as -250 to 1250 ms (Step4: tx2disp)
COND_TMIN = -0.25   # seconds
COND_TMAX =  1.25   # seconds

# Per-condition baseline windows in ms (Step4: Nb1:Nb2, STb1:STb2)
NOVEL_BASE_MS = (-250, 0)
STIM_BASE_MS  = (-250, 0)

# Stimulus event codes for ds003490 (Step1: S200=target, S201=standard, S202=novelty)
STIM_CODES = {200: "target", 201: "standard", 202: "novelty"}

# Number of random Standards per subject/session (Step4 line: tempnum(1:30))
N_RANDOM_STANDARDS = 30

# Classification window in ms (Step5: T1=250, T2=1000)
CLASSIFY_T1_MS = 250
CLASSIFY_T2_MS = 1000


def extract_condition_epochs(
    epochs_full: mne.Epochs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop to COND_TMIN..COND_TMAX and apply per-condition baseline
    correction.  Mirrors MATLAB Step4's windowing logic.

    Step4 windows (all displayed on tx2disp = -250:10:1250 ms):
      Standard (S201): tmin=-250ms tmax=1250ms
      Target   (S200): tmin=-250ms tmax=1250ms
      Novelty  (S202): tmin=-250ms tmax=1250ms

    Returns:
      std_data : float32 (n_channels, n_times, n_standard_trials)
      tgt_data : float32 (n_channels, n_times, n_target_trials)
      nov_data : float32 (n_channels, n_times, n_novelty_trials)
      tx       : time axis in ms
    """
    epochs_cond = epochs_full.copy().crop(tmin=COND_TMIN, tmax=COND_TMAX)
    tx          = epochs_cond.times * 1000   # ms

    # Stimulus codes from event array
    stim_vector = epochs_full.events[:, 2]

    data = epochs_cond.get_data()   # (n_epochs, n_channels, n_times)

    def _bidx(t_start, t_end):
        return (np.argmin(np.abs(tx - t_start)),
                np.argmin(np.abs(tx - t_end)))

    nb1, nb2   = _bidx(*NOVEL_BASE_MS)
    stb1, stb2 = _bidx(*STIM_BASE_MS)

    # Baseline correction per condition (Step4 pattern)
    data_bc = data.copy()
    for ai in range(len(data_bc)):
        if stim_vector[ai] == 202:   # Novelty
            base = data_bc[ai, :, nb1:nb2+1].mean(axis=1, keepdims=True)
        else:                         # Standard / Target
            base = data_bc[ai, :, stb1:stb2+1].mean(axis=1, keepdims=True)
        data_bc[ai] -= base

    # Split by stimulus code and transpose to (channels, times, trials)
    std_data = data_bc[stim_vector == 201].transpose(1, 2, 0).astype(np.float32)
    tgt_data = data_bc[stim_vector == 200].transpose(1, 2, 0).astype(np.float32)
    nov_data = data_bc[stim_vector == 202].transpose(1, 2, 0).astype(np.float32)

    return std_data, tgt_data, nov_data, tx


def _trinary_split(n_trials: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split n_trials into first / middle / last third.
    MATLAB Step2: S{1}, S{2}, S{3} — used for habituation analyses.
    """
    t1 = np.arange(0,             n_trials // 3)
    t2 = np.arange(n_trials // 3, 2 * (n_trials // 3))
    t3 = np.arange(2 * (n_trials // 3), n_trials)
    return t1, t2, t3


# ============================================================
# NEW: Build single-trial dataset — MATLAB Step4 / SINGLETRIAL_ODDBALL_AGG_100.mat
# ============================================================

def build_singletrial_dataset(participants_df: pd.DataFrame) -> None:
    """
    Load preprocessed .fif files, extract Standard/Target/Novelty epochs
    per subject and medication session, match trial counts, and save
    SINGLETRIAL_ODDBALL.npz.

    Dataset structure (mirrors MATLAB Step4 output):
      S / T / N [pair_idx][slot_idx]            -> (C, T, n_trials) float32
      TRI_S / TRI_T / TRI_N [pair_idx][slot_idx][third_idx]  -> same
      slot_labels  -> ['ON', 'OFF', 'CTL']   (indices 0, 1, 2)
      third_labels -> ['first', 'middle', 'last']
      tx           -> time axis in ms
      pd_subjs / ctl_subjs -> participant IDs

    slot_idx mapping:  0=ON  1=OFF  2=CTL
    Matches MATLAB Step4 sessioni=1(ON), 2(OFF), 3(CTL).
    """
    logger.info("Building singletrial dataset (Step4 equivalent)...")

    pd_ids = [
        str(r).replace("sub-","")
        for r in participants_df[
            participants_df["Group"].str.upper() == "PD"
        ]["participant_id"]
    ]
    ctl_ids = [
        str(r).replace("sub-","")
        for r in participants_df[
            participants_df["Group"].str.upper().isin(["CTL","CONTROL"])
        ]["participant_id"]
    ]

    SLOT_LABELS  = ["ON", "OFF", "CTL"]
    THIRD_LABELS = ["first", "middle", "last"]
    rng = np.random.default_rng(42)

    S_all: Dict  = {}
    T_all: Dict  = {}
    N_all: Dict  = {}
    TRI_S: Dict  = {}
    TRI_T: Dict  = {}
    TRI_N: Dict  = {}
    tx_saved     = None

    for pair_idx, pd_subj in enumerate(
        tqdm(pd_ids, desc="  Building singletrial dataset")
    ):
        S_all[pair_idx] = {}
        T_all[pair_idx] = {}
        N_all[pair_idx] = {}
        TRI_S[pair_idx] = {}
        TRI_T[pair_idx] = {}
        TRI_N[pair_idx] = {}

        ctl_subj = ctl_ids[pair_idx] if pair_idx < len(ctl_ids) else None

        for slot_idx, slot_label in enumerate(SLOT_LABELS):
            if slot_label in ("ON", "OFF"):
                fif_path = CACHE_EPOCHS / f"sub-{pd_subj}_{slot_label}-epo.fif"
            else:
                if ctl_subj is None:
                    continue
                fif_path = CACHE_EPOCHS / f"sub-{ctl_subj}_CTL-epo.fif"

            if not fif_path.exists():
                logger.warning(f"  Missing epoch file: {fif_path.name}")
                continue

            try:
                epochs_full = mne.read_epochs(str(fif_path), preload=True, verbose="warning")
                std_data, tgt_data, nov_data, tx = extract_condition_epochs(epochs_full)

                if tx_saved is None:
                    tx_saved = tx

                # Select N_RANDOM_STANDARDS random standards (Step4)
                if std_data.shape[2] > N_RANDOM_STANDARDS:
                    idx = rng.permutation(std_data.shape[2])[:N_RANDOM_STANDARDS]
                    std_data = std_data[:, :, idx]

                S_all[pair_idx][slot_idx] = std_data
                T_all[pair_idx][slot_idx] = tgt_data
                N_all[pair_idx][slot_idx] = nov_data

                # Trinary splits (Step2 habituation structure)
                for store, d in [(TRI_S, std_data), (TRI_T, tgt_data), (TRI_N, nov_data)]:
                    t1, t2, t3 = _trinary_split(d.shape[2])
                    store[pair_idx][slot_idx] = {0: d[:,:,t1], 1: d[:,:,t2], 2: d[:,:,t3]}

            except Exception as e:
                logger.warning(f"  Could not load {fif_path.name}: {e}")

    # Trial count matching — MATLAB Step4 MinEpochCt logic
    for pair_idx in S_all:
        slots = list(S_all[pair_idx].keys())
        if not slots:
            continue
        counts = [
            store[pair_idx][si].shape[2]
            for store in (S_all, T_all, N_all)
            for si in slots
            if si in store[pair_idx]
        ]
        if not counts:
            continue
        min_ct = min(counts)

        for si in slots:
            for store, tri_store in [
                (S_all, TRI_S), (T_all, TRI_T), (N_all, TRI_N)
            ]:
                if si not in store[pair_idx]:
                    continue
                d = store[pair_idx][si]
                if d.shape[2] > min_ct:
                    idx = rng.permutation(d.shape[2])[:min_ct]
                    d   = d[:, :, idx]
                    store[pair_idx][si] = d
                # Rebuild trinary splits after matching
                t1, t2, t3 = _trinary_split(d.shape[2])
                tri_store[pair_idx][si] = {0: d[:,:,t1], 1: d[:,:,t2], 2: d[:,:,t3]}

    out_path = OUT_DIR / "SINGLETRIAL_ODDBALL.npz"
    np.savez_compressed(
        str(out_path),
        pd_subjs     = np.array(pd_ids,       dtype=object),
        ctl_subjs    = np.array(ctl_ids,       dtype=object),
        slot_labels  = np.array(SLOT_LABELS,   dtype=object),
        third_labels = np.array(THIRD_LABELS,  dtype=object),
        tx           = tx_saved if tx_saved is not None else np.array([]),
        S    = np.array(S_all,  dtype=object),
        T    = np.array(T_all,  dtype=object),
        N    = np.array(N_all,  dtype=object),
        TRI_S= np.array(TRI_S,  dtype=object),
        TRI_T= np.array(TRI_T,  dtype=object),
        TRI_N= np.array(TRI_N,  dtype=object),
    )
    logger.info(f"Singletrial dataset saved -> {out_path}")
