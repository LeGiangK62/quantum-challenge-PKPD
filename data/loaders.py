from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import math

# =========================================================
# prepare data
# =========================================================
def load_estdata(path: str = "EstData.csv", *, standardize_cols: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (df_all, df_obs, df_dose).
    - df_all: original
    - df_obs: observed (EVID==0 & MDV==0)
    - df_dose: dosing (EVID==1)
    """
    df = pd.read_csv(path)
    if standardize_cols:
        df.columns = [c.strip().upper() for c in df.columns]
    df_obs = df[(df['EVID'] == 0) & (df['MDV'] == 0)].copy()
    df_dose = df[df['EVID'] == 1].copy()
    return df, df_obs, df_dose

# =========================================================
# FE: dose history -> additional features (leakage-safe)
# =========================================================
def features_from_dose_history(
    obs_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    add_pk_baseline: bool = False,
    add_pd_delta: bool = False,   
    target: str = "dv",           
    allow_future_dose: bool = False  
) -> pd.DataFrame:
    required_obs = {"ID", "TIME", "DV", "DVID"}
    required_dose = {"ID", "TIME", "AMT"}
    miss_obs = required_obs - set(obs_df.columns)
    miss_dose = required_dose - set(dose_df.columns)
    if miss_obs:
        raise ValueError(f"obs_df missing columns: {sorted(miss_obs)}")
    if miss_dose:
        raise ValueError(f"dose_df missing columns: {sorted(miss_dose)}")

    if str(target).lower() == "dv" and add_pd_delta:
        raise ValueError("Target is 'DV' but add_pd_delta=True -> data leakage risk.")

    out = obs_df.copy()

    # Pre-group doses per subject (sorted by time)
    dose_by_id = {
        i: d.sort_values("TIME")[["TIME", "AMT"]].to_numpy()
        for i, d in dose_df.groupby("ID", sort=False)
    }

    tsld_list, last_amt_list = [], []
    ndose_list, cumdose_list = [], []
    ttnext_list = []
    sum24_list, sum48_list, sum168_list = [], [], []

    def window_sum(d_times, csum, t_start, t_end):
        """Cumulative dose in the window (t_start, t_end], inclusive on right."""
        r = np.searchsorted(d_times, t_end, side="right") - 1
        l = np.searchsorted(d_times, t_start, side="right") - 1
        if r < 0:
            return 0.0
        left_cum = 0.0 if l < 0 else float(csum[l])
        right_cum = float(csum[r])
        return right_cum - left_cum

    # Build row-wise features
    for i, g in out.groupby("ID", sort=False):
        times = g["TIME"].to_numpy()
        if i not in dose_by_id:
            n = len(times)
            tsld_list += [np.nan] * n
            last_amt_list += [0.0] * n
            ndose_list += [0] * n
            cumdose_list += [0.0] * n

            ttnext_list += [np.nan] * n
            sum24_list += [0.0] * n
            sum48_list += [0.0] * n
            sum168_list += [0.0] * n
            continue

        dmat = dose_by_id[i]
        d_times = dmat[:, 0]
        d_amts = dmat[:, 1]
        csum = np.cumsum(d_amts)

        for t in times:
            # last dose info up to time t (inclusive)
            idx_last = np.searchsorted(d_times, t, side="right") - 1
            if idx_last >= 0:
                t_last = d_times[idx_last]
                last_amt = float(d_amts[idx_last])
                ndoses = int(idx_last + 1)
                cumdose = float(csum[idx_last])
                tsld = float(t - t_last)
            else:
                last_amt = 0.0
                ndoses = 0
                cumdose = 0.0
                tsld = np.nan

            if allow_future_dose:
                idx_next = idx_last + 1
                if idx_next < len(d_times):
                    t_next = d_times[idx_next]
                    ttnext = float(t_next - t)
                else:
                    ttnext = np.nan
            else:
                ttnext = np.nan

            # rolling dose sums over past windows
            sum24 = window_sum(d_times, csum, t - 24.0, t)
            sum48 = window_sum(d_times, csum, t - 48.0, t)
            sum168 = window_sum(d_times, csum, t - 168.0, t)

            tsld_list.append(tsld)
            last_amt_list.append(last_amt)
            ndose_list.append(ndoses)
            cumdose_list.append(cumdose)
            ttnext_list.append(ttnext)
            sum24_list.append(sum24)
            sum48_list.append(sum48)
            sum168_list.append(sum168)

    # Assign engineered columns
    out["TSLD"] = tsld_list
    out["LAST_DOSE_AMT"] = last_amt_list
    out["N_DOSES_UP_TO_T"] = ndose_list
    out["CUM_DOSE_UP_TO_T"] = cumdose_list
    out["DOSE_SUM_PREV24H"] = sum24_list
    out["DOSE_SUM_PREV48H"] = sum48_list
    out["DOSE_SUM_PREV168H"] = sum168_list
    out["TIME_TO_NEXT_DOSE"] = ttnext_list  # may be NaN if allow_future_dose=False

    # Baselines: computed as first DV per (ID, DVID) after sorting by time
    base_by_id_dvid = (
        out.sort_values(["ID", "TIME"])
           .groupby(["ID", "DVID"])["DV"]
           .transform(lambda s: s.iloc[0] if len(s) else np.nan)
    )

    # PD baseline/delta only on PD rows (DVID==2)
    out["PD_BASELINE"] = np.where(out["DVID"].eq(2), base_by_id_dvid, np.nan)
    if add_pd_delta:
        # Safe because we assert target!='dv' above
        out["PD_DELTA"] = np.where(out["DVID"].eq(2), out["DV"] - out["PD_BASELINE"], np.nan)

    # Optional PK baseline/delta only on PK rows (DVID==1) — not used as features by default
    if add_pk_baseline:
        out["PK_BASELINE"] = np.where(out["DVID"].eq(1), base_by_id_dvid, np.nan)
        out["PK_DELTA"] = np.where(out["DVID"].eq(1), out["DV"] - out["PK_BASELINE"], np.nan)

    return out


# =========================================================
# Feature Engineering (build feature lists with leakage guards)
# =========================================================
def use_feature_engineering(
    df_obs: pd.DataFrame,
    df_dose: pd.DataFrame,
    use_perkg: bool,
    *,
    target: str = "dv", # 'dv' or 'delta'
    use_pd_baseline_for_dv: bool = True,
    allow_future_dose: bool = False
):
    """
    Build engineered dataframe and return (df_final, pk_features, pd_features).
    """
    print("applying feature engineering (leakage-safe).")
    df_final = features_from_dose_history(
        obs_df=df_obs,
        dose_df=df_dose,
        add_pk_baseline=False,
        add_pd_delta=(str(target).lower() == "delta"),   # compute PD_DELTA only if target='delta'
        target=target,
        allow_future_dose=allow_future_dose
    )

    # Base feature pool (no target columns here)
    base_feats = [
        'BW', 'COMED', 'DOSE', 'TIME',
        'TSLD', 'LAST_DOSE_AMT', 'N_DOSES_UP_TO_T', 'CUM_DOSE_UP_TO_T',
        'DOSE_SUM_PREV24H', 'DOSE_SUM_PREV48H', 'DOSE_SUM_PREV168H'
    ]
    if allow_future_dose:
        base_feats.append('TIME_TO_NEXT_DOSE')

    pk_features = base_feats.copy()
    pd_features = base_feats.copy()

    if str(target).lower() == 'dv' and use_pd_baseline_for_dv:
        if 'PD_BASELINE' in df_final.columns:
            pd_features.append('PD_BASELINE')

    # Optional per-kg features for both PK/PD
    if use_perkg:
        bw = df_final['BW'].replace(0, np.nan)
        perkg_cols = [
            'DOSE', 'LAST_DOSE_AMT', 'CUM_DOSE_UP_TO_T',
            'DOSE_SUM_PREV24H', 'DOSE_SUM_PREV48H', 'DOSE_SUM_PREV168H'
        ]
        added = []
        for col in perkg_cols:
            if col in df_final.columns:
                df_final[f'{col}_PER_KG'] = (df_final[col] / bw).fillna(0.0)
                added.append(f'{col}_PER_KG')
        if added:
            pk_features += added
            pd_features += added
            print("  + per-kg features added:", added)

    forbidden = {'DV', 'PD_DELTA', 'PK_DELTA'}
    assert not (forbidden & set(pk_features)), f"Leakage risk in PK features: {forbidden & set(pk_features)}"
    assert not (forbidden & set(pd_features)), f"Leakage risk in PD features: {forbidden & set(pd_features)}"

    df_final.fillna(0, inplace=True)
    return df_final, pk_features, pd_features

# =========================================================
# PK/PD dataframe separation
# =========================================================
def separate_pkpd(df_obs: pd.DataFrame, df_dose: pd.DataFrame, use_fe: bool, use_perkg: bool):
    """Return (df_final, pk_df, pd_df, pd_features, pk_features)."""
    if use_fe:
        df_final, pk_feats, pd_feats = use_feature_engineering(df_obs, df_dose, use_perkg)
    else:
        print("not applying feature engineering.")
        df_final = df_obs.copy()
        pk_feats = ['BW', 'COMED', 'DOSE', 'TIME']
        pd_feats = ['BW', 'COMED', 'DOSE', 'TIME']

    print(df_final.head(3))
    pk_df = df_final[df_final['DVID'] == 1].copy()
    pd_df = df_final[df_final['DVID'] == 2].copy()
    return df_final, pk_df, pd_df, pd_feats, pk_feats

# =========================================================
# Custom Dataset
# =========================================================
class CustomDataset(Dataset):
    def __init__(self, X, y, ids=None, times=None, mask=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ids = None if ids is None else torch.tensor(ids, dtype=torch.long)
        self.times = None if times is None else torch.tensor(times, dtype=torch.float32)
        self.mask = None if mask is None else torch.tensor(mask, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        items = [self.X[idx], self.y[idx]]
        if self.mask is not None:  items.append(self.mask[idx])
        if self.ids is not None:   items.append(self.ids[idx])
        if self.times is not None: items.append(self.times[idx])
        return tuple(items)