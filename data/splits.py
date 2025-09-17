from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

# =========================================================
# Split strategy
# =========================================================
STRATEGY_MAP = {
    0: "random_subject",
    1: "stratify_dose_even",
    2: "leave_one_dose_out",
    3: "only_bw_range",
    4: "stratify_dose_even_no_placebo_test",
}

def normalize_split_strategy(x) -> str:
    s = str(x).strip().lower()
    if s.isdigit():
        code = int(s)
        if code in STRATEGY_MAP:
            return STRATEGY_MAP[code]
        raise ValueError(f"Unknown split strategy code: {code}")
    if s in STRATEGY_MAP.values():
        return s
    raise ValueError(f"Unknown split strategy: {x}")

def parse_float_list(val) -> List[float]:
    if val is None or val == "":
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [float(x) for x in val]
    return [float(x.strip()) for x in str(val).split(",") if str(x).strip()]

# =========================================================
# Utilities
# =========================================================
def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan

def _detect_id_col(df: pd.DataFrame) -> str:
    return _first_existing(df, ["ID", "SUBJ", "SUBJECT", "USUBJID"]) or "ID"

def _subject_primary_dose(df: pd.DataFrame, id_col: str, dose_col: str) -> pd.Series:
    """
    Calculate representative dose: EVID==1 first, if not available use mode(>0) from all data.
    """
    d = df.loc[df.get("EVID", 0).eq(1), [id_col, dose_col]].dropna()
    if d.empty:
        d = df[[id_col, dose_col]].dropna()

    def per_id(s: pd.Series) -> float:
        vals = s.values.astype(float)
        nz = vals[vals > 0]
        if nz.size:
            uniq, cnt = np.unique(nz, return_counts=True)
            return float(uniq[np.argmax(cnt)])
        return 0.0

    return d.groupby(id_col)[dose_col].apply(per_id).astype(float)

def _split_ids(ids: np.ndarray, test_size: float, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly shuffle ids and split into train/test.
    - test_size: ratio (0~1) or integer (count)
    - Ensure at least 1 subject remains in train.
    """
    ids = np.array(ids)
    if ids.size == 0:
        return np.array([], dtype=ids.dtype), np.array([], dtype=ids.dtype)

    perm = rng.permutation(ids)
    if 0 < test_size < 1:
        n_test = max(0, int(round(len(ids) * test_size)))
    else:
        n_test = int(max(0, test_size))

    n_test = min(n_test, len(ids) - 1)  # At least 1 subject in train
    n_test = max(n_test, 0)

    test_ids = np.sort(perm[:n_test])
    train_ids = np.sort(perm[n_test:])
    return train_ids, test_ids

# =========================================================
# Core splitting
# =========================================================
def split_dataset(
    df: pd.DataFrame,
    *,
    id_col: Optional[str] = None,
    strategy: str = "random_subject",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 123,
    dose_col: Optional[str] = None,
    n_dose_bins: int = 4,
    leaveout_doses: Optional[Iterable[float]] = None,
    bw_col: Optional[str] = None,
    bw_range: Optional[Tuple[float, float]] = None,
    quantile_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Create subject-pure train/val/test splits under various strategies.

    df: Input frame containing at least ID, DOSE (or AMT), BW.
    """
    df = df.copy()
    id_col = id_col or _detect_id_col(df)
    if id_col not in df.columns:
        raise ValueError(f"ID column not found: '{id_col}'")

    rng = np.random.RandomState(random_state)
    all_ids = np.array(sorted(df[id_col].unique()))
    strategy = normalize_split_strategy(strategy)

    if strategy == "random_subject":
        tr_ids, te_ids = _split_ids(all_ids, test_size, rng)
        # Maintain global ratio: adjust val ratio in train set to (val_size / (1 - test_size))
        denom = max(1e-9, (1.0 - (len(te_ids) / max(1, len(all_ids)))))
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(tr_ids, adj_val, np.random.RandomState(random_state + 1))

    elif strategy == "stratify_dose_even":
        dose_col = dose_col or _first_existing(df, ["DOSE", "AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)
        zero_mask = subj_dose.eq(0.0)
        bins = pd.Series(index=subj_dose.index, dtype=object)
        bins[zero_mask] = "dose=0"
        nonzero = subj_dose[~zero_mask]
        if nonzero.size:
            q = min(n_dose_bins, max(1, nonzero.nunique()))
            qbins = pd.qcut(nonzero, q=q, duplicates="drop")
            bins.loc[nonzero.index] = qbins.astype(str)

        tr_ids, va_ids, te_ids = [], [], []
        # Apply same adjustment per bin to maintain global ratio
        for _, idx in bins.groupby(bins):
            ids_b = np.array(sorted(idx.index))
            tr_b, te_b = _split_ids(ids_b, test_size, rng)
            # Adjust global val ratio within bin as well
            denom_b = max(1e-9, (1.0 - (len(te_b) / max(1, len(ids_b)))))
            adj_val_b = min(0.99, max(0.0, val_size / denom_b))
            tr_b, va_b = _split_ids(tr_b, adj_val_b, np.random.RandomState(random_state + 2))
            tr_ids += tr_b.tolist(); va_ids += va_b.tolist(); te_ids += te_b.tolist()
        tr_ids, va_ids, te_ids = np.array(sorted(tr_ids)), np.array(sorted(va_ids)), np.array(sorted(te_ids))

    elif strategy == "leave_one_dose_out":
        dose_col = dose_col or _first_existing(df, ["DOSE", "AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)
        excl = set(parse_float_list(leaveout_doses))
        if not excl:
            raise ValueError("leave_one_dose_out requires non-empty leaveout_doses.")

        te_ids = np.array(sorted([i for i, v in subj_dose.items()
                                  if any(np.isclose(v, e, atol=1e-6) for e in excl)]))
        remain = np.array(sorted(np.setdiff1d(all_ids, te_ids)))

        # Global ratio adjustment: adjust val ratio in remaining considering test ratio
        test_ratio = len(te_ids) / max(1, len(all_ids))
        denom = max(1e-9, 1.0 - test_ratio)
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(remain, adj_val, np.random.RandomState(random_state + 3))

    elif strategy == "stratify_dose_even_no_placebo_test":
        # Same as stratify_dose_even but exclude dose=0 from test set
        dose_col = dose_col or _first_existing(df, ["DOSE", "AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)
        zero_mask = subj_dose.eq(0.0)
        bins = pd.Series(index=subj_dose.index, dtype=object)
        bins[zero_mask] = "dose=0"
        nonzero = subj_dose[~zero_mask]
        if nonzero.size:
            q = min(n_dose_bins, max(1, nonzero.nunique()))
            qbins = pd.qcut(nonzero, q=q, duplicates="drop")
            bins.loc[nonzero.index] = qbins.astype(str)

        tr_ids, va_ids, te_ids = [], [], []
        # Apply same adjustment per bin to maintain global ratio
        for bin_name, bin_ids in bins.groupby(bins):
            bin_ids = np.array(sorted(bin_ids.index))
            if len(bin_ids) < 2:
                # Too few subjects in this bin, assign to train
                tr_ids.extend(bin_ids)
                continue
            
            # For dose=0 bin, exclude from test set
            if bin_name == "dose=0":
                # Split dose=0 subjects between train and validation only
                n_test = 0  # No test subjects for placebo
                n_val = max(1, int(len(bin_ids) * val_size))
                n_tr = len(bin_ids) - n_val
                
                # Shuffle and assign
                np.random.RandomState(random_state + hash(bin_name) % 1000).shuffle(bin_ids)
                tr_ids.extend(bin_ids[:n_tr])
                va_ids.extend(bin_ids[n_tr:n_tr + n_val])
            else:
                # For non-placebo doses, use normal split
                n_test = max(1, int(len(bin_ids) * test_size))
                n_val = max(1, int(len(bin_ids) * val_size))
                n_tr = len(bin_ids) - n_test - n_val
                
                # Shuffle and assign
                np.random.RandomState(random_state + hash(bin_name) % 1000).shuffle(bin_ids)
                tr_ids.extend(bin_ids[:n_tr])
                va_ids.extend(bin_ids[n_tr:n_tr + n_val])
                te_ids.extend(bin_ids[n_tr + n_val:])

    elif strategy == "only_bw_range":
        bw_col = bw_col or _first_existing(df, ["BW", "WT", "WEIGHT", "BODYWEIGHT"]) or "BW"
        if bw_col not in df.columns:
            raise ValueError("bw_col not found.")
        subj_bw = df.groupby(id_col)[bw_col].median()
        if quantile_range:
            lo, hi = subj_bw.quantile(quantile_range[0]), subj_bw.quantile(quantile_range[1])
        elif bw_range:
            lo, hi = bw_range
        else:
            raise ValueError("only_bw_range requires bw_range or quantile_range.")
        keep = np.array(sorted(subj_bw[(subj_bw >= lo) & (subj_bw <= hi)].index))
        tr_ids, te_ids = _split_ids(keep, test_size, rng)

        # Global ratio adjustment
        denom = max(1e-9, (1.0 - (len(te_ids) / max(1, len(keep)))))
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(tr_ids, adj_val, np.random.RandomState(random_state + 4))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    train_mask = df[id_col].isin(tr_ids)
    val_mask = df[id_col].isin(va_ids)
    test_mask = df[id_col].isin(te_ids)
    return {"train": df[train_mask].copy(), "val": df[val_mask].copy(), "test": df[test_mask].copy()}

# =========================================================
# Universe & split-ready DF
# =========================================================
def choose_universe_and_build_split_df(
    df_final: pd.DataFrame,
    df_dose: Optional[pd.DataFrame] = None,
    *,
    id_universe: str = 'union'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Select PK/PD universe from df_final and construct representative DOSE for splitting.
    - Priority: AMT/DOSE mode from df_dose(EVID==1) → if not available, DOSE mode from df_final
    Return: (df_universe, pk_df, pd_df, dose_grp, df_for_split)
    """
    pk_df = df_final[df_final['DVID'] == 1].copy()
    pd_df = df_final[df_final['DVID'] == 2].copy()

    pk_ids = set(pk_df['ID'].unique())
    pd_ids = set(pd_df['ID'].unique())
    matched = pk_ids & pd_ids
    union = pk_ids | pd_ids

    used = matched if id_universe == 'intersection' else union
    if not used:
        raise ValueError("No subjects in chosen universe!")

    pk_df = pk_df[pk_df['ID'].isin(used)].copy()
    pd_df = pd_df[pd_df['ID'].isin(used)].copy()
    df_universe = df_final[df_final['ID'].isin(used)].copy()

    # Calculate representative DOSE: df_dose first
    if df_dose is not None and not df_dose.empty:
        id_col = _detect_id_col(df_dose)
        dose_name = _first_existing(df_dose, ["AMT", "DOSE"]) or "AMT"
        tmp = df_dose[df_dose[id_col].isin(used)].copy()
        rep_from_dose = _subject_primary_dose(tmp, id_col=id_col, dose_col=dose_name)
        rep_from_dose.name = "DOSE_SPLIT"
    else:
        rep_from_dose = pd.Series(dtype=float, name="DOSE_SPLIT")

    # Auxiliary: observation-based (mode)
    rep_from_obs = df_universe.groupby("ID")["DOSE"].agg(_mode_or_nan)
    rep_from_obs.name = "DOSE_SPLIT"

    # Merge: dosing-based first, if not available use observation-based
    dose_grp = rep_from_dose.reindex(list(used))
    dose_grp = dose_grp.fillna(rep_from_obs)
    dose_grp = dose_grp.fillna(0.0)

    df_universe = df_universe.merge(dose_grp.rename("DOSE_SPLIT"), left_on='ID', right_index=True, how='left')

    # DF for splitting: drop original DOSE and use DOSE_SPLIT as DOSE
    df_for_split = df_universe.copy()
    df_for_split.drop(columns=['DOSE'], inplace=True, errors='ignore')
    df_for_split.rename(columns={'DOSE_SPLIT': 'DOSE'}, inplace=True)
    return df_universe, pk_df, pd_df, dose_grp, df_for_split

def project_split(sub_df: pd.DataFrame, global_splits: Dict[str, pd.DataFrame], id_col: str = "ID") -> Dict[str, pd.DataFrame]:
    """Project a global (train/val/test) split onto PK/PD subset."""
    idsets = {k: set(global_splits[k][id_col].unique()) for k in ("train","val","test")}
    return {k: sub_df[sub_df[id_col].isin(idsets[k])].copy() for k in ("train","val","test")}

# =========================================================
# Reporting
# =========================================================
def _col_as_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] if exists else a NaN series with proper name."""
    if name in df.columns:
        s = df[name]; s.name = name
        return s
    return pd.Series(np.nan, index=df.index, name=name)

def report_dose_and_bw(df: pd.DataFrame, tag: str, topn: int = 5):
    print(f"\n=== [{tag}] brief summary ===")
    dose_col = _col_as_series(df, 'DOSE')
    uniq = sorted(pd.unique(dose_col.dropna()))
    head = uniq[:topn]
    more = f" (+{len(uniq)-len(head)} more)" if len(uniq) > len(head) else ""
    zero_ratio = float((dose_col.fillna(0) == 0).mean())

    print(f"Rows={len(df):,} | IDs={df['ID'].nunique():,}")
    print(f"DOSE levels (row, non-null): {head}{more}")
    print(f"DOSE==0 (row): {zero_ratio:.3%}")

    subj_dose = df.groupby("ID")[dose_col.name].apply(_mode_or_nan)

    bw_name = _first_existing(df, ["BW","WT","WEIGHT","BODYWEIGHT"])
    if bw_name is not None:
        subj_bw = df.groupby("ID")[bw_name].first()
    else:
        subj_bw = pd.Series(np.nan, index=subj_dose.index, name="BW")

    idx = subj_dose.index.intersection(subj_bw.index)
    subj_dose = subj_dose.loc[idx]
    subj_bw   = subj_bw.loc[idx]

    dose_perkg = (subj_dose / subj_bw).replace([np.inf, -np.inf], np.nan)

    vc = subj_dose.value_counts(dropna=False)
    vc_head = vc.head(topn).to_dict()
    more2 = f" (+{len(vc)-len(vc_head)} more)" if len(vc) > len(vc_head) else ""

    if bw_name is not None and subj_bw.notna().any():
        bw_mean, bw_std = float(np.nanmean(subj_bw)), float(np.nanstd(subj_bw))
        bw_min, bw_med, bw_max = map(float, (np.nanmin(subj_bw), np.nanmedian(subj_bw), np.nanmax(subj_bw)))
        print(f"BW by ID: mean={bw_mean:.2f}±{bw_std:.2f}, min|med|max={bw_min:.1f}|{bw_med:.1f}|{bw_max:.1f}")
    else:
        print("BW by ID: (missing)")

    if dose_perkg.notna().any():
        dpk_med = float(np.nanmedian(dose_perkg))
        q25 = float(np.nanpercentile(dose_perkg.dropna(), 25))
        q75 = float(np.nanpercentile(dose_perkg.dropna(), 75))
        print(f"Primary DOSE/BW by ID: median={dpk_med:.4f} [IQR {q25:.4f}–{q75:.4f}]")
    else:
        print("Primary DOSE/BW by ID: (insufficient data)")

    print(f"Primary DOSE by ID (top): {vc_head}{more2}")

# =========================================================
# High-level helper: prepare_for_split
# =========================================================
def prepare_for_split(
    df_final: pd.DataFrame,
    df_dose: pd.DataFrame,
    pk_df: pd.DataFrame,
    pd_df: pd.DataFrame,
    *,
    split_strategy,
    test_size: float,
    val_size: float,
    random_state: int,
    dose_bins: int = 4,
    leaveout_doses: Optional[Iterable[float]] = None,
    bw_range: Optional[Tuple[float, float]] = None,
    bw_quantiles: Optional[Tuple[float, float]] = None,
    id_universe: str = 'intersection',
    verbose: bool = True,
):
    """Prepare global PK/PD splits under a chosen universe and strategy."""
    df_univ, pk_df_u, pd_df_u, dose_grp_obs, df_for_split = choose_universe_and_build_split_df(
        df_final, df_dose=df_dose, id_universe=id_universe
    )

    strategy = normalize_split_strategy(split_strategy)
    leaveout = parse_float_list(leaveout_doses)

    split_kwargs = dict(
        strategy=strategy,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        dose_col='DOSE',
        bw_col='BW',
    )
    if strategy == "stratify_dose_even":
        split_kwargs["n_dose_bins"] = dose_bins
    elif strategy == "leave_one_dose_out":
        if not leaveout:
            raise ValueError("leave_one_dose_out requires leaveout_doses (e.g., [0,10,30]).")
        split_kwargs["leaveout_doses"] = leaveout
    elif strategy == "only_bw_range":
        if bw_range:
            split_kwargs["bw_range"] = bw_range
        elif bw_quantiles:
            split_kwargs["quantile_range"] = bw_quantiles
        else:
            raise ValueError("only_bw_range requires bw_range or bw_quantiles.")

    global_splits = split_dataset(df_for_split, **split_kwargs)
    pk_splits = project_split(pk_df_u, global_splits)
    pd_splits = project_split(pd_df_u, global_splits)

    pk_ids_u = set(pk_df_u["ID"].unique())
    pd_ids_u = set(pd_df_u["ID"].unique())
    used_ids = pk_ids_u | pd_ids_u if id_universe == 'union' else (pk_ids_u & pd_ids_u)
    for k in ("train","val","test"):
        pk_ids_k = set(pk_splits[k]["ID"].unique())
        pd_ids_k = set(pd_splits[k]["ID"].unique())
        assert pk_ids_k.issubset(used_ids) and pd_ids_k.issubset(used_ids)
        if id_universe == 'intersection':
            assert pk_ids_k == pd_ids_k, f"{k} IDs not aligned: PK {len(pk_ids_k)} vs PD {len(pd_ids_k)}"

    if verbose:
        for split in ("train","val","test"):
            ids = global_splits[split]['ID'].unique()
            levels = sorted(dose_grp_obs.loc[ids].unique().tolist())
            print(f"[{split}] representative DOSE levels: {levels}")
        report_dose_and_bw(global_splits['train'], "Train (matched IDs)")
        report_dose_and_bw(global_splits['val'],   "Val (matched IDs)")
        report_dose_and_bw(global_splits['test'],  "Test (matched IDs)")

    return pk_splits, pd_splits, global_splits, df_for_split
