#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-24 09:47:13 (ywatanabe)"

import re
import sqlite3

import mne
import mngs
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CONFIG = mngs.io.load("./config/global.yaml")


def load_eeg_file(filepath):
    with mngs.gen.suppress_output():
        # Loads
        try:
            raw = mne.io.read_raw(filepath, preload=True, verbose=False)
        except Exception as e:
            print(e)
            return None

        # Converts channel names to uppercase
        raw.rename_channels(lambda x: x.upper())

        # Maps channels
        try:
            raw.rename_channels(CONFIG["CHANNEL_MAPPING"])
        except Exception as e:
            print("Channel mapping was not conducted:", e)

        # Sets montages
        raw.set_montage(
            mngs.dsp.to_dig_montage(CONFIG["TGT_MONTAGE"]),
            on_missing="ignore",
            match_case=False,
        )

        # Picks predefined channels
        raw = raw.pick(CONFIG["TGT_MONTAGE"])

        # Re-reference
        try:
            raw = mne.set_bipolar_reference(
                raw,
                anode=[
                    bi.split("-")[0]
                    for bi in CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]
                ],
                cathode=[
                    bi.split("-")[1]
                    for bi in CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]
                ],
                ch_name=CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"],
                drop_refs=True,
            )
        except Exception as e:
            print(e)
            return None

        # Notch filtering at 50 and 60 Hz
        raw = raw.notch_filter(freqs=[50, 60], n_jobs=1)

        # Interpolate bad channels if any
        raw = raw.interpolate_bads(verbose=False)

        # Downsampling to CONFIG["TGT_FS"]
        raw.resample(CONFIG["TGT_FS"], n_jobs=1)

        # Convert to DataFrame and transpose
        df = raw.to_data_frame().set_index("time").T

        # Segmentation
        segs = mngs.dsp.crop(
            np.array(df),
            int(CONFIG["WINDOW_SIZE_SEC"] * CONFIG["TGT_FS"]),  # 256
            overlap_factor=CONFIG["OVERLAP_FACTOR"],
        )

        # Extract segments
        segs = segs[
            np.random.permutation(np.arange(len(segs)))[
                : CONFIG["MAX_N_WINDOWS_PER_SUB"]
            ]
        ]

        return segs


# Create and populate the database
def create_and_populate_db(files_list, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop the existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS eeg_data")

    # Recreate the table with the correct schema
    cursor.execute(
        """
        CREATE TABLE eeg_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,        
            subject_id TEXT NOT NULL,
            segment_number INTEGER NOT NULL,
            segment BLOB NOT NULL
        )
        """
    )
    for filepath in files_list:
        dataset_id, subject_id = extract_ids(filepath)
        eeg_segments = load_eeg_file(filepath)
        if eeg_segments is None:
            continue
        for i, segment in enumerate(eeg_segments):
            segment_bytes = segment.tobytes()
            cursor.execute(
                "INSERT INTO eeg_data (dataset_id, subject_id, segment_number, segment) VALUES (?, ?, ?, ?)",
                (dataset_id, subject_id, i, segment_bytes),
            )
    conn.commit()
    conn.close()


def extract_ids(filepath):
    pattern = (
        r".*/ds(?P<dataset_id>\d+).*?/sub-(?P<subject_id>[a-zA-Z0-9]+)/.*"
    )
    match = re.search(pattern, filepath)
    dataset_id = "ds" + match.group("dataset_id")
    subject_id = "sub-" + match.group("subject_id")
    return dataset_id, subject_id


class BidsDatasetSQLite(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM eeg_data")
        self.length = self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Adjust the SQL query to also select dataset_id and subject_id
        self.cursor.execute(
            "SELECT dataset_id, subject_id, segment FROM eeg_data WHERE id = ?",
            (index + 1,),
        )
        row = self.cursor.fetchone()
        dataset_id, subject_id, segment = row
        # Convert the segment from bytes to a NumPy array and then to a PyTorch tensor
        segment_tensor = torch.from_numpy(
            np.frombuffer(segment, dtype=np.float32)
            .reshape(len(CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]), -1)
            .copy()
        )
        # Return a tuple with dataset_id, subject_id, and the segment tensor
        return dataset_id, subject_id, segment_tensor

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    # Usage
    import os

    import mngs

    if os.path.exists(CONFIG["DB_PATH"]):
        os.remove(CONFIG["DB_PATH"])

    # Initializing EEG data file paths
    LPATHS = np.hstack(
        [
            mngs.gen.find(CONFIG["DATA_DIR"], "*" + ext)
            for ext in [".edf", ".bdf", ".set"]
        ]
    )
    # comment here
    create_and_populate_db(LPATHS, CONFIG["DB_PATH"])
