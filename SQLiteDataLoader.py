#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-24 10:06:21 (ywatanabe)"

import re
import sqlite3

import mne
import mngs
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CONFIG = mngs.io.load("./config/global.yaml")


class SQLiteDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM eeg_data")
        self.length = self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Execute the SQL query against the database
        self.cursor.execute(
            "SELECT dataset_id, subject_id, segment FROM eeg_data WHERE id = ?",
            (index + 1,),
        )
        row = self.cursor.fetchone()
        dataset_id, subject_id, segment = row

        # Converts the segment from bytes to a NumPy array and then to a PyTorch tensor
        segment_tensor = torch.from_numpy(
            np.frombuffer(segment, dtype=np.float32)
            .reshape(len(CONFIG["TGT_TRANVERSE_BIPOLAR_MONTAGE"]), -1)
            .copy()
        )
        return dataset_id, subject_id, segment_tensor

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    from datetime import datetime

    # Starts timer
    start_time = datetime.now()

    # Loads the database iteratively using PyTorch Dataloader
    bids_dataset_sqlite = SQLiteDataset(db_path=CONFIG["DB_PATH"])
    data_loader = DataLoader(
        bids_dataset_sqlite,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    ds_all = []
    batch_counter = 0
    for batch in data_loader:
        dataset_ids, subject_ids, X = batch
        print(X.shape)
        ds_all.append(dataset_ids)
        batch_counter += 1
    print(f"\nLoaded datasets: {np.unique(np.hstack(ds_all))}")

    # Calculates time spent
    end_time = datetime.now()
    diff_time = end_time - start_time
    print(f"\nTime taken: {diff_time}")

    # Don't forget to close the database connection when done
    bids_dataset_sqlite.close()

    # Prints the loading performance
    n_samples, _, seq_len = X.shape
    loaded_sec = batch_counter * n_samples * seq_len / CONFIG["TGT_FS"]
    print(f"\n{loaded_sec} sec of EEG data was loaded in {diff_time}")
    # 148224.0 sec of EEG data was loaded in 0:00:00.294323

## OF
