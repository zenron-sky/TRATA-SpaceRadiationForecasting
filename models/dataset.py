"""
TRATA- Sliding Window Dataset Builder
-----------------------------------------------
Converts the flat feature table into (X, y) sequences for the CNN-LSTM:
  X: (n_samples, seq_len, n_features) - past `seq_len` timesteps
  y: (n_samples, n_horizons)          - future log(flux) at each horizon
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SpaceWeatherSequenceDataset(Dataset):
    def __init__(self, feature_array: np.ndarray, target_array: np.ndarray, seq_len: int = 48):
        self.seq_len = seq_len
        self.X = feature_array
        self.y = target_array
        self.valid_starts = np.arange(0, len(self.X) - seq_len)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len
        x_seq = self.X[start:end]
        y_target = self.y[end - 1]  # target aligned to the last timestep of the window
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)
