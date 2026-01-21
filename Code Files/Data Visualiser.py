# car_montage_pipeline.py
from pathlib import Path
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids

# =========================
# EDIT THESE
# =========================
BIDS_ROOT = Path("/Users/dibadabiransari/Desktop/EEG Project/data/ds003490")
SUBJECT = "001"
SESSION = "01"
TASK = "Rest"   # "Rest" or "3AOB"

L_FREQ = 0.5
H_FREQ = 45.0
NOTCH = 50  # set 60 if needed

SAVE_DIR = BIDS_ROOT / "derivatives" / "montage_demo"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# FIX CHANNEL TYPES (DATASET-SPECIFIC)
# ---------------------------------------------------------------------
def fix_channel_types_for_ds003490(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    ds003490 uses type='n/a' in channels.tsv.
    We must manually assign EEG/EOG/MISC channel types.
    """
    channel_types = {ch: "eeg" for ch in raw.ch_names}

    for ch in raw.ch_names:
        ch_up = ch.upper()
        if ch_up == "VEOG":
            channel_types[ch] = "eog"
        if ch_up in {"X", "Y", "Z"}:
            channel_types[ch] = "misc"

    raw.set_channel_types(channel_types)

    # Pick EEG + EOG (version-safe)
    picks = mne.pick_types(
        raw.info,
        eeg=True,
        eog=True,
        stim=False,
        misc=False
    )
    raw.pick(picks)

    return raw


# ---------------------------------------------------------------------
# APPLY MONTAGE FROM electrodes.tsv (WITH ROTATION)
# ---------------------------------------------------------------------
def apply_electrodes_tsv_montage(raw, electrodes_tsv_path, unit="mm", rotate_ccw_90=True):
    """
    Build and apply a montage from BIDS electrodes.tsv.

    rotate_ccw_90=True:
        rotates electrode coordinates 90 degrees anti-clockwise in the XY plane:
            x' = -y
            y' =  x
            z' =  z
    """
    df = pd.read_csv(electrodes_tsv_path, sep="\t")

    ch_pos = {}
    for _, r in df.iterrows():
        name = str(r["name"]).strip()

        # Some TSVs may have n/a or empty rows; skip safely
        try:
            x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
        except Exception:
            continue

        # Convert mm -> meters
        if unit.lower() == "mm":
            x, y, z = x / 1000.0, y / 1000.0, z / 1000.0

        # Rotate 90° anti-clockwise in XY plane (optional)
        if rotate_ccw_90:
            x, y = -y, x

        ch_pos[name] = (x, y, z)

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame="head"
    )

    raw.set_montage(montage, match_case=False, on_missing="ignore")
    return raw


# ---------------------------------------------------------------------
# MAIN PIPELINE (COMMON AVERAGE REFERENCE)
# ---------------------------------------------------------------------
def main():
    # 1) Load EEG from BIDS
    bids_path = BIDSPath(
        root=BIDS_ROOT,
        subject=SUBJECT,
        session=SESSION,
        task=TASK,
        datatype="eeg",
        suffix="eeg",
    )

    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()
    print(raw)

    # 2) Fix channel types
    raw = fix_channel_types_for_ds003490(raw)

    # 3) Apply electrode positions (rotated to match MNE head orientation)
    electrodes_tsv = (
        BIDS_ROOT
        / f"sub-{SUBJECT}"
        / f"ses-{SESSION}"
        / "eeg"
        / f"sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_electrodes.tsv"
    )

    raw = apply_electrodes_tsv_montage(
        raw,
        electrodes_tsv_path=electrodes_tsv,
        unit="mm",
        rotate_ccw_90=True,   # <-- KEY CHANGE
    )

    # 4) Apply common average reference
    raw.set_eeg_reference("average", projection=False)

    # 5) Plot sensors
    raw.plot_sensors(
    show_names=True,
    sphere=(0., 0., 0., 0.09)
)

    # 6) Filtering
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)
    raw.notch_filter(NOTCH, fir_design="firwin", verbose=False)

    # 7) Visual inspection
    raw.plot(
        n_channels=min(20, len(raw.ch_names)),
        duration=10,
        scalings="auto",
        block=True
    )
    raw.plot_psd(fmin=L_FREQ, fmax=H_FREQ)

    # 8) Epoching (task only)
    if TASK.lower() != "rest":
        events, event_id = mne.events_from_annotations(raw)
        print("Event IDs:", event_id)

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=-0.2,
            tmax=0.8,
            baseline=(None, 0),
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )

        epochs.plot(n_epochs=min(10, len(epochs)), n_channels=20)
        epochs.plot_psd(fmin=L_FREQ, fmax=H_FREQ)

        out_epochs = SAVE_DIR / f"sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_CAR-epo.fif"
        epochs.save(out_epochs, overwrite=True)
        print("Saved epochs to:", out_epochs)

    # 9) Save processed raw (CAR)
    out_raw = SAVE_DIR / f"sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_CAR-raw.fif"
    raw.save(out_raw, overwrite=True)
    print("Saved raw to:", out_raw)


if __name__ == "__main__":
    main()
