import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Data loader that will handle loading, normalizing, and formatting data
# Shape should be (4, S)
class MIMICIVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len=60, scaler=None, is_train=True):
        """
        X_df: Dataframe of FINAL_MEASUREMENT_TABLE.parquet
        y_df: Dataframe of LABELS_33DAY.parquet
        max_len: Maximum sequence length (bins) to truncate/pad to.
        scaler: sklearn StandardScaler. If None, fits on this data (use for Train).
        """

        sorted_df = df.sort_values(["stay_id", "time_bin_index"]).reset_index(drop=True)

        self.df = sorted_df.drop(columns=["event_time_bin", "event_flag"]).copy()
        self.labels_df = sorted_df[["stay_id", "event_time_bin", "event_flag"]].drop_duplicates(subset="stay_id").copy()
        self.max_len = max_len
        
        # Merge Labels to ensure alignment
        # Group by stay_id -> need unique labels per stay
        self.labels_map = self.labels_df.set_index('stay_id')[['event_time_bin', 'event_flag']].to_dict('index')
        
        # Filter df to only include stays with labels
        self.df = self.df[self.df['stay_id'].isin(self.labels_map.keys())]
        self.stay_ids = self.df['stay_id'].unique()

        # Identify Feature Columns: ToDo Change these as we add more features
        # Exclude stay_id, time_bin_index, and the _missing columns for scaling
        self.value_cols = [c for c in self.df.columns if '_value' in c]
        self.mask_cols  = [c for c in self.df.columns if '_missing' in c]
        self.static_cols = ['admission_age', 'gender_M', 'gender_F']
        
        # Treat static features as "always observed" variables.
        self.feature_cols = self.static_cols + self.value_cols
        
        # Normalization
        if is_train and scaler is None:
            self.scaler = StandardScaler()
            # Fit only on the dynamic value columns and static cols
            self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        else:
            self.scaler = scaler
            if self.scaler:
                self.df[self.feature_cols] = self.scaler.transform(self.df[self.feature_cols])

        # Pre-group by stay_id for fast access in __getitem__
        self.df = self.df.sort_values(['stay_id', 'time_bin_index'])
        self.grouped = self.df.groupby('stay_id')

    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]
        group = self.grouped.get_group(stay_id)
        
        # Shape: (seq_len, num_features)
        X = group[self.feature_cols].values.astype(np.float32)
        
        # Extract Masks (M)
        # Static features are never missing (Mask = 1)
        # Dynamic features: Pipeline uses 1 for missing, 0 for present. 
        # GRU-D expects 1 for present, 0 for missing. INVERT valid dynamic masks.
        mask_dynamic = 1.0 - group[self.mask_cols].values.astype(np.float32) 
        mask_static = np.ones((len(group), len(self.static_cols)), dtype=np.float32)
        Mask = np.concatenate([mask_static, mask_dynamic], axis=1)
        
        # Calculate Delta (Time since last observation)
        # ToDo: maybe do in DuckDB pipeline
        Delta = np.zeros_like(X)
        for t in range(1, len(X)):
            # If missing (Mask=0), Delta = Prev_Delta + Time_Step_Size
            # If present (Mask=1), Delta = 0
            # GRU-D Logic: Delta[t] = Time[t] - Time[last_obsv]
            Delta[t] = Delta[t-1] + 1 # +1 bin (4 hours)
            # Reset delta to 0 where data is observed
            Delta[t] = Delta[t] * (1 - Mask[t]) 

        # GRU-D needs X_last
        # Use raw X values and Mask
        X_vals = group[self.feature_cols].values.astype(np.float32)
        X_last = np.zeros_like(X_vals)
        last_observation = np.zeros(X_vals.shape[1]) # TODO: Try init with mean?

        for t in range(len(X_vals)):
            # Mask is 1, update the last observation
            # Mask is 0, keep the previous last observation
            observed_indices = np.where(Mask[t] == 1)[0]
            last_observation[observed_indices] = X_vals[t, observed_indices]
            X_last[t] = last_observation

        # Get Label
        label_info = self.labels_map[stay_id]
        # Target: (Event_Flag, Time_Bin)
        y = np.array([label_info['event_flag'], label_info['event_time_bin']], dtype=np.float32)

        # Truncate if longer than max_len
        if len(X) > self.max_len:
            X = X[:self.max_len]
            Mask = Mask[:self.max_len]
            Delta = Delta[:self.max_len]
            X_last = X_last[:self.max_len]

        return {
            'X': X, 
            'X_last': X_last, 
            'Mask': Mask, 
            'Delta': Delta, 
            'y': y,
            'length': len(X)
        }
