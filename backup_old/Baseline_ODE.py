#!/usr/bin/env python3
# ODE_baseline.py — single-file PK/PD ODE baseline with fair splits
# - PK: 1C + tlag + BW allometry
# - PD: effect-site (ke0) + Emax ± Hill ± COMED
# - Fair split strategies inlined (random / dose-stratified / leave-one-dose-out / BW-range)
# - Plots & bootstrap support

import os, math, argparse
import numpy as np
import pandas as pd

# headless plot backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

# =========================
# Defaults / Globals
# =========================
CSV_PATHS_DEFAULT = ["EstData.csv"]
LOG_EPS    = 1e-3
BW_REF     = 73.0
BW_COLS    = ["WT","BW","Weight","WEIGHT","BODYWEIGHT"]
LOG_PD_EPS = 1e-3
RNG = np.random.default_rng(42)

# =========================================================
# Split helpers (inlined from split.py-based design)
# =========================================================
STRATEGY_MAP = {
    0: "random_subject",
    1: "stratify_dose_even",
    2: "leave_one_dose_out",
    3: "only_bw_range",
}

def _first_existing(df: pd.DataFrame, candidates):
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
    return _first_existing(df, ["ID","SUBJ","SUBJECT","USUBJID"]) or "ID"

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

def parse_float_list(val):
    if val is None or val == "":
        return []
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        return [float(x) for x in val]
    return [float(x.strip()) for x in str(val).split(",") if str(x).strip()]

def _subject_primary_dose(df: pd.DataFrame, id_col: str, dose_col: str) -> pd.Series:
    # If EVID is present, prefer EVID==1 rows; otherwise use whole frame
    if "EVID" in df.columns:
        d = df.loc[df["EVID"].eq(1), [id_col, dose_col]].dropna()
    else:
        d = df[[id_col, dose_col]].dropna()

    def per_id(s: pd.Series) -> float:
        vals = s.values.astype(float)
        nz = vals[vals > 0]
        if nz.size:
            uniq, cnt = np.unique(nz, return_counts=True)
            return float(uniq[np.argmax(cnt)])
        return 0.0

    return d.groupby(id_col)[dose_col].apply(per_id).astype(float)

def _split_ids(ids: np.ndarray, test_size: float, rng: np.random.RandomState):
    ids = np.array(ids)
    if ids.size == 0:
        return np.array([], dtype=ids.dtype), np.array([], dtype=ids.dtype)
    perm = rng.permutation(ids)
    if 0 < test_size < 1:
        n_test = max(0, int(round(len(ids) * test_size)))
    else:
        n_test = int(max(0, test_size))
    n_test = min(n_test, len(ids) - 1)  # keep ≥1 in train
    n_test = max(n_test, 0)
    test_ids = np.sort(perm[:n_test])
    train_ids = np.sort(perm[n_test:])
    return train_ids, test_ids

def split_dataset(
    df: pd.DataFrame,
    *,
    id_col: str | None = None,
    strategy: str = "random_subject",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 123,
    dose_col: str | None = None,
    n_dose_bins: int = 4,
    leaveout_doses = None,
    bw_col: str | None = None,
    bw_range: tuple[float,float] | None = None,
    quantile_range: tuple[float,float] | None = None,
) -> dict[str, pd.DataFrame]:
    df = df.copy()
    id_col = id_col or _detect_id_col(df)
    if id_col not in df.columns:
        raise ValueError(f"ID column not found: {id_col}")

    rng = np.random.RandomState(random_state)
    all_ids = np.array(sorted(df[id_col].unique()))
    strategy = normalize_split_strategy(strategy)

    if strategy == "random_subject":
        tr_ids, te_ids = _split_ids(all_ids, test_size, rng)
        denom = max(1e-9, (1.0 - (len(te_ids) / max(1, len(all_ids)))))
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(tr_ids, adj_val, np.random.RandomState(random_state + 1))

    elif strategy == "stratify_dose_even":
        dose_col = dose_col or _first_existing(df, ["DOSE","AMT"]) or "DOSE"
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
        for _, idx in bins.groupby(bins):
            ids_b = np.array(sorted(idx.index))
            tr_b, te_b = _split_ids(ids_b, test_size, rng)
            denom_b = max(1e-9, (1.0 - (len(te_b) / max(1, len(ids_b)))))
            adj_val_b = min(0.99, max(0.0, val_size / denom_b))
            tr_b, va_b = _split_ids(tr_b, adj_val_b, np.random.RandomState(random_state + 2))
            tr_ids += tr_b.tolist(); va_ids += va_b.tolist(); te_ids += te_b.tolist()
        tr_ids, va_ids, te_ids = np.array(sorted(tr_ids)), np.array(sorted(va_ids)), np.array(sorted(te_ids))

    elif strategy == "leave_one_dose_out":
        dose_col = dose_col or _first_existing(df, ["DOSE","AMT"]) or "DOSE"
        if dose_col not in df.columns:
            raise ValueError("dose_col not found.")
        subj_dose = _subject_primary_dose(df, id_col=id_col, dose_col=dose_col)
        excl = set(parse_float_list(leaveout_doses))
        if not excl:
            raise ValueError("leave_one_dose_out requires non-empty leaveout_doses.")
        te_ids = np.array(sorted([i for i, v in subj_dose.items()
                                  if any(np.isclose(v, e, atol=1e-6) for e in excl)]))
        remain = np.array(sorted(np.setdiff1d(all_ids, te_ids)))
        test_ratio = len(te_ids) / max(1, len(all_ids))
        denom = max(1e-9, 1.0 - test_ratio)
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(remain, adj_val, np.random.RandomState(random_state + 3))

    elif strategy == "only_bw_range":
        bw_col = bw_col or _first_existing(df, BW_COLS) or "BW"
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
        denom = max(1e-9, (1.0 - (len(te_ids) / max(1, len(keep)))))
        adj_val = min(0.99, max(0.0, val_size / denom))
        tr_ids, va_ids = _split_ids(tr_ids, adj_val, np.random.RandomState(random_state + 4))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    train_mask = df[id_col].isin(tr_ids)
    val_mask   = df[id_col].isin(va_ids)
    test_mask  = df[id_col].isin(te_ids)
    return {"train": df[train_mask].copy(), "val": df[val_mask].copy(), "test": df[test_mask].copy()}

def choose_universe_and_build_split_df(
    df_final: pd.DataFrame,
    df_dose: pd.DataFrame | None = None,
    *,
    id_universe: str = "intersection"
):
    pk_df = df_final[df_final["DVID"]==1].copy()
    pd_df = df_final[df_final["DVID"]==2].copy()

    pk_ids = set(pk_df["ID"].unique())
    pd_ids = set(pd_df["ID"].unique())
    used = (pk_ids & pd_ids) if id_universe == "intersection" else (pk_ids | pd_ids)
    if not used:
        raise ValueError("No subjects in chosen universe!")

    pk_df = pk_df[pk_df["ID"].isin(used)].copy()
    pd_df = pd_df[pd_df["ID"].isin(used)].copy()
    df_universe = df_final[df_final["ID"].isin(used)].copy()

    # representative DOSE
    if df_dose is not None and not df_dose.empty:
        id_col = _detect_id_col(df_dose)
        dose_name = _first_existing(df_dose, ["AMT","DOSE"]) or "AMT"
        tmp = df_dose[df_dose[id_col].isin(used)].copy()
        rep_from_dose = _subject_primary_dose(tmp, id_col=id_col, dose_col=dose_name)
        rep_from_dose.name = "DOSE_SPLIT"
    else:
        rep_from_dose = pd.Series(dtype=float, name="DOSE_SPLIT")

    # fallback from observations (mode) if DOSE present; else zeros
    if "DOSE" in df_universe.columns:
        rep_from_obs = df_universe.groupby("ID")["DOSE"].agg(_mode_or_nan)
        rep_from_obs.name = "DOSE_SPLIT"
    else:
        rep_from_obs = pd.Series(0.0, index=sorted(df_universe["ID"].unique()), name="DOSE_SPLIT")

    dose_grp = rep_from_dose.reindex(list(used))
    dose_grp = dose_grp.fillna(rep_from_obs).fillna(0.0)

    df_universe = df_universe.merge(dose_grp.rename("DOSE_SPLIT"), left_on="ID", right_index=True, how="left")
    df_for_split = df_universe.copy()
    df_for_split.drop(columns=["DOSE"], inplace=True, errors="ignore")
    df_for_split.rename(columns={"DOSE_SPLIT":"DOSE"}, inplace=True)

    return df_universe, pk_df, pd_df, dose_grp, df_for_split

def project_split(sub_df: pd.DataFrame, global_splits: dict[str,pd.DataFrame], id_col: str = "ID"):
    idsets = {k: set(global_splits[k][id_col].unique()) for k in ("train","val","test")}
    return {k: sub_df[sub_df[id_col].isin(idsets[k])].copy() for k in ("train","val","test")}

def report_dose_and_bw(df: pd.DataFrame, tag: str, topn: int = 5):
    print(f"\n=== [{tag}] brief summary ===")
    dose_col = "DOSE" if "DOSE" in df.columns else ("AMT" if "AMT" in df.columns else None)
    print(f"Rows={len(df):,} | IDs={df['ID'].nunique():,}")
    if dose_col is None:
        print("DOSE levels: (missing)")
        return
    uniq = sorted(pd.unique(df[dose_col].dropna()))
    head = uniq[:topn]
    more = f" (+{len(uniq)-len(head)} more)" if len(uniq) > len(head) else ""
    zero_ratio = float((df[dose_col].fillna(0) == 0).mean())
    print(f"DOSE levels (row, non-null): {head}{more}")
    print(f"DOSE==0 (row): {zero_ratio:.3%}")

    subj_dose = df.groupby("ID")[dose_col].apply(_mode_or_nan)
    bw_name = _first_existing(df, BW_COLS)
    if bw_name is not None:
        subj_bw = df.groupby("ID")[bw_name].first()
        bw_mean, bw_std = float(np.nanmean(subj_bw)), float(np.nanstd(subj_bw))
        bw_min, bw_med, bw_max = map(float, (np.nanmin(subj_bw), np.nanmedian(subj_bw), np.nanmax(subj_bw)))
        print(f"BW by ID: mean={bw_mean:.2f}±{bw_std:.2f}, min|med|max={bw_min:.1f}|{bw_med:.1f}|{bw_max:.1f}")
        dose_perkg = (subj_dose / subj_bw).replace([np.inf, -np.inf], np.nan)
        if dose_perkg.notna().any():
            dpk_med = float(np.nanmedian(dose_perkg))
            q25 = float(np.nanpercentile(dose_perkg.dropna(), 25))
            q75 = float(np.nanpercentile(dose_perkg.dropna(), 75))
            print(f"Primary DOSE/BW by ID: median={dpk_med:.4f} [IQR {q25:.4f}–{q75:.4f}]")
    else:
        print("BW by ID: (missing)")

    vc = subj_dose.value_counts(dropna=False).head(topn).to_dict()
    more2 = f" (+{len(subj_dose.value_counts())-len(vc)} more)" if len(subj_dose.value_counts()) > len(vc) else ""
    print(f"Primary DOSE by ID (top): {vc}{more2}")

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
    leaveout_doses=None,
    bw_range=None,
    bw_quantiles=None,
    id_universe: str = "intersection",
    verbose: bool = True,
):
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
        dose_col="DOSE",
        bw_col="BW",
    )
    if strategy == "stratify_dose_even":
        split_kwargs["n_dose_bins"] = dose_bins
    elif strategy == "leave_one_dose_out":
        if not leaveout:
            raise ValueError("leave_one_dose_out requires leaveout_doses (e.g., 0,10,30).")
        split_kwargs["leaveout_doses"] = leaveout
    elif strategy == "only_bw_range":
        if bw_range:
            split_kwargs["bw_range"] = tuple(map(float, bw_range))
        elif bw_quantiles:
            split_kwargs["quantile_range"] = tuple(map(float, bw_quantiles))
        else:
            raise ValueError("only_bw_range requires bw_range or bw_quantiles.")

    global_splits = split_dataset(df_for_split, **split_kwargs)
    pk_splits = project_split(pk_df_u, global_splits)
    pd_splits = project_split(pd_df_u, global_splits)

    pk_ids_u = set(pk_df_u["ID"].unique())
    pd_ids_u = set(pd_df_u["ID"].unique())
    used_ids = pk_ids_u | pd_ids_u if id_universe == "union" else (pk_ids_u & pd_ids_u)
    for k in ("train","val","test"):
        pk_ids_k = set(pk_splits[k]["ID"].unique())
        pd_ids_k = set(pd_splits[k]["ID"].unique())
        assert pk_ids_k.issubset(used_ids) and pd_ids_k.issubset(used_ids)
        if id_universe == "intersection":
            assert pk_ids_k == pd_ids_k, f"{k} IDs not aligned: PK {len(pk_ids_k)} vs PD {len(pd_ids_k)}"

    if verbose:
        for split in ("train","val","test"):
            ids = global_splits[split]['ID'].unique()
            levels = sorted(pd.Series(dose_grp_obs).loc[ids].unique().tolist())
            print(f"[{split}] representative DOSE levels: {levels}")
        report_dose_and_bw(global_splits['train'], "Train (matched IDs)")
        report_dose_and_bw(global_splits['val'],   "Val (matched IDs)")
        report_dose_and_bw(global_splits['test'],  "Test (matched IDs)")

    return pk_splits, pd_splits, global_splits, df_for_split

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="PK(1C+tlag+BW) → PD(effect-site ke0 + Emax ± Hill) with fair splits (single file)")
    # I/O & run
    ap.add_argument("--csv", default='EstData.csv', help="CSV path (takes priority when specified)")
    ap.add_argument("--plots", action="store_true", help="Save individual ID plots")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations (0=disabled)")
    ap.add_argument("--verbose", action="store_true", help="Detailed logging")

    # model toggles
    ap.add_argument("--no-hill", action="store_true", help="Disable Hill exponent")
    ap.add_argument("--no-comed", action="store_true", help="Disable COMED covariate")
    ap.add_argument("--no-log-pk", action="store_true", help="Disable PK log error")
    ap.add_argument("--no-log-pd", action="store_true", help="Disable PD log error")

    # split (fair)
    ap.add_argument("--split-strategy", default="stratify_dose_even",
                    help="random_subject | stratify_dose_even | leave_one_dose_out | only_bw_range | (or 0/1/2/3)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test ratio or integer (count)")
    ap.add_argument("--val-size", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--dose-bins", type=int, default=4, help="Number of dose quantiles in stratify_dose_even")
    ap.add_argument("--leaveout-doses", default="", help="Doses to exclude in leave_one_dose_out (e.g., '0,10,30')")
    ap.add_argument("--bw-quantiles", default="", help="BW quantile range in only_bw_range (e.g., '0.1,0.9')")
    ap.add_argument("--bw-range", default="", help="BW absolute range in only_bw_range (e.g., '50,90')")
    ap.add_argument("--random-state", type=int, default=123, help="Split seed")
    ap.add_argument("--id-universe", default="intersection", choices=["intersection","union"],
                    help="PK and PD ID universe selection")

    # legacy convenience (optional): test IDs override
    ap.add_argument("--test-ids", default="", help="Test IDs comma-separated (ignores split when set) e.g., 4,7,20")
    return ap.parse_args()

# =========================
# Small helpers
# =========================
def resolve_csv_path(args):
    if args.csv:
        if os.path.exists('./data/'+args.csv):
            return './data/'+args.csv
        if os.path.exists(args.csv):
            return args.csv
        raise FileNotFoundError(f"CSV not found: {args.csv} (also tried ./data/{args.csv})")
    for p in CSV_PATHS_DEFAULT:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("EstData.csv not found in default search paths.")

def parse_id_set(s):
    if not s: return set()
    return {int(x.strip()) for x in s.split(",") if x.strip()}

def get_bw(df_any, sid):
    for c in BW_COLS:
        if c in df_any.columns:
            v = df_any.loc[df_any["ID"]==sid, c]
            if len(v):
                x = float(v.iloc[0])
                if np.isfinite(x) and x > 0:
                    return x
    return BW_REF

def get_comed(df_any, sid):
    return int(df_any.loc[df_any["ID"]==sid, "COMED"].max()) if "COMED" in df_any.columns else 0

def fallback_doses(df_any, sid, t_end):
    d = float(max(df_any.loc[df_any["ID"]==sid, "DOSE"].max(), 0)) if "DOSE" in df_any.columns else 0.0
    if d > 0:
        times = list(range(0, int(np.ceil(float(t_end)))+1, 24))
        return pd.DataFrame({"ID":sid, "TIME":times, "AMT":[d]*len(times), "CMT":[1]*len(times)})
    return pd.DataFrame(columns=["ID","TIME","AMT","CMT"])

# =========================
# PK model: 1C + tlag + BW
# =========================
def sim_1c_tlag(times, ka, CL, V, doses, tlag=0.0):
    times = np.asarray(sorted(times), float)
    ke = CL / V
    out = np.zeros_like(times)
    for _, r in doses.iterrows():
        td, D = float(r.TIME), float(r.AMT)
        dt = np.clip(times - (td + tlag), 0, None)
        if np.isclose(ka, ke):
            out += (D / V) * (ka * dt) * np.exp(-ka * dt)
        else:
            sc = ka / (ka - ke)
            out += (D / V) * sc * (np.exp(-ke*dt) - np.exp(-ka*dt))
    return out

def coarse_tlag_scan(pk_tr, doses_tr, df_tr, LOG_PK_FIT, ka=0.7, CL=6.0, V=60.0, tgrid=np.linspace(0.05, 1.0, 20)):
    obs = pk_tr[pk_tr["EVID"]==0]
    ids = obs["ID"].unique()
    best_t, best_val = None, np.inf
    for tl in tgrid:
        err=[]
        for sid in ids:
            g = obs[obs["ID"]==sid]
            t = g["TIME"].values; y=g["DV"].values
            d = doses_tr[doses_tr["ID"]==sid][["TIME","AMT","CMT"]]
            if d.empty: d=fallback_doses(df_tr, sid, t_end=t.max() if len(t) else 0)
            bw = get_bw(df_tr, sid); s=(bw/BW_REF)
            c = sim_1c_tlag(t, ka, CL*s**0.75, V*s, d, tl)
            if LOG_PK_FIT:
                r = np.log(c+LOG_EPS)-np.log(np.maximum(y,0)+LOG_EPS)
            else:
                r = c - y
            err.append(np.mean(r*r))
        val = np.mean(err) if err else np.inf
        if val < best_val:
            best_val, best_t = val, tl
    return best_t if best_t is not None else 0.3

def fit_pk_1c_tlag_bw(pk_tr, df_tr, doses_tr, LOG_PK_FIT, VERBOSE):
    obs = pk_tr[pk_tr["EVID"]==0].copy()
    ids = list(obs["ID"].unique())
    tlag_init = coarse_tlag_scan(pk_tr, doses_tr, df_tr, LOG_PK_FIT)
    if VERBOSE:
        print(f"[tlag init] coarse ≈ {tlag_init:.3f} h")
    x0 = np.array([np.log(0.7), np.log(6.0), np.log(60.0), np.log(max(tlag_init, 1e-2))], float)
    lb = np.array([-np.inf, -np.inf, -np.inf, np.log(1e-3)], float)
    ub = np.array([ np.inf,  np.inf,  np.inf, np.log(48.0)], float)

    def unpack(theta):
        ka = np.exp(theta[0]); CL = np.exp(theta[1]); V = np.exp(theta[2]); tlag = np.exp(theta[3])
        return ka, CL, V, tlag

    def resid(theta):
        ka,CL,V,tlag = unpack(theta)
        rs=[]
        for sid in ids:
            g = obs[obs["ID"]==sid]; t=g["TIME"].values; y=g["DV"].values
            d = doses_tr[doses_tr["ID"]==sid][["TIME","AMT","CMT"]]
            if d.empty: d=fallback_doses(df_tr, sid, t_end=t.max() if len(t) else 0)
            bw = get_bw(df_tr, sid); s=(bw/BW_REF)
            CLi = CL*s**0.75; Vi = V*s
            c = sim_1c_tlag(t, ka, CLi, Vi, d, tlag)
            r = np.log(c+LOG_EPS)-np.log(np.maximum(y,0)+LOG_EPS) if LOG_PK_FIT else (c-y)
            rs.append(r)
        return np.concatenate(rs) if rs else np.array([0.0])

    sol = least_squares(resid, x0, loss="soft_l1", bounds=(lb,ub), max_nfev=400, verbose=0)
    ka,CL,V,tlag = unpack(sol.x)
    return dict(ka=ka, CL=CL, V=V, tlag=tlag)

def predict_pk_on(pd_like_df, df_any, doses_any, pk_pars):
    rows=[]
    for sid, g in pd_like_df.groupby("ID"):
        t = g["TIME"].values
        d = doses_any[doses_any["ID"]==sid][["TIME","AMT","CMT"]]
        if d.empty: d=fallback_doses(df_any, sid, t_end=t.max() if len(t) else 0)
        bw = get_bw(df_any, sid); s=(bw/BW_REF)
        CLi = pk_pars["CL"]*s**0.75; Vi = pk_pars["V"]*s
        c = sim_1c_tlag(t, pk_pars["ka"], CLi, Vi, d, pk_pars["tlag"])
        rows.append(pd.DataFrame({"ID":sid,"TIME":t,"C_pred":c}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ID","TIME","C_pred"])

# =========================
# PD: effect-site + Emax(+Hill)
# =========================
def sim_effect_site(times, C_times, C_vals, ke0):
    order = np.argsort(times)
    times = np.asarray(times)[order]
    C_vals = np.asarray(C_vals)[order]
    C_times = np.asarray(C_times)[order]
    tgrid = np.asarray(C_times, float)
    cgrid = np.asarray(C_vals, float)

    def Cfun(t):
        if t <= tgrid[0]: return float(cgrid[0])
        if t >= tgrid[-1]: return float(cgrid[-1])
        return float(np.interp(t, tgrid, cgrid))

    y0 = [Cfun(times[0])]
    sol = solve_ivp(lambda t,y: [ke0*(Cfun(t)-y[0])],
                    [times[0], times[-1]], y0,
                    t_eval=times,
                    method="BDF", rtol=1e-6, atol=1e-8, max_step=0.5)
    return sol.y[0] if sol.status >= 0 else np.interp(times, tgrid, cgrid)

def fit_pd_emax_effectsite(pd_tr, pk_pars, df_tr, doses_tr, USE_HILL, USE_COMED, LOG_PD_FIT):
    pd_tr2 = pd_tr[pd_tr["EVID"]==0][["ID","TIME","DV"]].copy()
    c_pred = predict_pk_on(pd_tr2.rename(columns={"DV":"ignore"}), df_tr, doses_tr, pk_pars)
    pd_tr2 = pd_tr2.merge(c_pred, on=["ID","TIME"], how="left")

    x = [pd_tr2["DV"].median(),
         - (pd_tr2["DV"].median()*0.5),
         np.log(np.percentile(pd_tr2["C_pred"], 60)+1e-3),
         np.log(0.1),
         np.log(1.0)]
    if USE_COMED: x += [0.0]
    x = np.array(x, float)

    def unpack(theta):
        i=0
        E0   = theta[i]; i+=1
        Emax = theta[i]; i+=1
        EC50 = np.exp(theta[i]); i+=1
        ke0  = np.exp(theta[i]); i+=1
        h    = np.exp(theta[i]) if USE_HILL else 1.0
        if USE_HILL: i+=1
        beta = theta[i] if USE_COMED else 0.0
        return E0,Emax,EC50,ke0,h,beta

    def resid(theta):
        E0,Emax,EC50,ke0,h,beta = unpack(theta)
        rs=[]
        for sid, gs in pd_tr2.groupby("ID"):
            t = gs["TIME"].values
            y = gs["DV"].values
            C = gs["C_pred"].values
            Ce = sim_effect_site(t, t, C, ke0)
            EC50_eff = EC50 * math.exp(beta*get_comed(df_tr, sid))
            if USE_HILL and (h != 1.0):
                Ce_h = Ce**h
                eff = E0 + Emax * (Ce_h/(EC50_eff**h + Ce_h))
            else:
                eff = E0 + Emax * (Ce/(EC50_eff + Ce))
            if LOG_PD_FIT:
                r = np.log(eff + LOG_PD_EPS) - np.log(np.maximum(y,0)+LOG_PD_EPS)
            else:
                r = eff - y
            rs.append(r)
        return np.concatenate(rs) if rs else np.array([0.0])

    sol = least_squares(resid, x, loss="soft_l1", max_nfev=400, verbose=0)
    return unpack(sol.x)

def predict_pd_on(pd_df, df_any, doses_any, pk_pars, pd_pars, USE_HILL):
    E0,Emax,EC50,ke0,h,beta = pd_pars
    rows=[]
    for sid, gs in pd_df[pd_df["EVID"]==0].groupby("ID"):
        t = gs["TIME"].values; y = gs["DV"].values
        c_pred = predict_pk_on(gs[["ID","TIME"]].copy(), df_any, doses_any, pk_pars)
        C = c_pred["C_pred"].values
        Ce = sim_effect_site(t, t, C, ke0)
        EC50_eff = EC50 * math.exp(beta*get_comed(df_any, sid))
        if USE_HILL and (h != 1.0):
            Ce_h = Ce**h
            yhat = E0 + Emax*(Ce_h/(EC50_eff**h + Ce_h))
        else:
            yhat = E0 + Emax*(Ce/(EC50_eff+Ce))
        rows.append(pd.DataFrame({"ID":sid,"TIME":t,"B_obs":y,"B_pred":yhat}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ID","TIME","B_obs","B_pred"])

def score_pd(df_pred):
    if df_pred.empty: return np.nan, np.nan
    y, yhat = df_pred["B_obs"].values, df_pred["B_pred"].values
    rmse = float(np.sqrt(np.mean((y-yhat)**2)))
    r2   = float(1 - np.sum((y-yhat)**2)/(np.sum((y-y.mean())**2)+1e-12))
    return rmse, r2

# =========================
# Main
# =========================
def main():
    args = parse_args()
    VERBOSE = bool(args.verbose)
    USE_HILL   = not args.no_hill
    USE_COMED  = not args.no_comed
    LOG_PK_FIT = not args.no_log_pk
    LOG_PD_FIT = not args.no_log_pd

    csv_path = resolve_csv_path(args)
    df = pd.read_csv(csv_path).sort_values(["ID","TIME"]).reset_index(drop=True)

    pk = df[df["DVID"]==1].copy()
    pd_data = df[df["DVID"]==2].copy()
    doses = df[(df["EVID"]==1) & (df["AMT"]>0)][["ID","TIME","AMT","CMT"]].copy()

    # --- Split selection ---
    fixed_test = parse_id_set(args.test_ids)
    if fixed_test:
        # legacy override path
        train_ids = sorted(set(df["ID"].unique()) - fixed_test)
        test_ids  = sorted(fixed_test)
        df_tr = df[df["ID"].isin(train_ids)].copy()
        pk_tr = pk[pk["ID"].isin(train_ids)].copy()
        pd_tr = pd_data[pd_data["ID"].isin(train_ids)].copy()
        df_te = df[df["ID"].isin(test_ids)].copy()
        pd_te = pd_data[pd_data["ID"].isin(test_ids)].copy()
        doses_tr = doses[doses["ID"].isin(train_ids)].copy()
        doses_te = doses[doses["ID"].isin(test_ids)].copy()
        if VERBOSE:
            print(f"[override] fixed TEST IDs used: n_test={len(test_ids)}")
    else:
        # fair split via inlined splitter
        # parse optional ranges
        bw_quantiles = tuple(map(float, args.bw_quantiles.split(","))) if args.bw_quantiles else None
        bw_range = tuple(map(float, args.bw_range.split(","))) if args.bw_range else None

        pk_splits, pd_splits, global_splits, _ = prepare_for_split(
            df_final=df,
            df_dose=doses,
            pk_df=pk,
            pd_df=pd_data,
            split_strategy=args.split_strategy,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            dose_bins=args.dose_bins,
            leaveout_doses=args.leaveout_doses,
            bw_range=bw_range,
            bw_quantiles=bw_quantiles,
            id_universe=args.id_universe,
            verbose=True,
        )
        df_tr  = global_splits["train"].copy()
        df_te  = global_splits["test"].copy()
        pk_tr  = pk_splits["train"].copy()
        pd_tr  = pd_splits["train"].copy()
        pd_te  = pd_splits["test"].copy()
        doses_tr = doses[doses["ID"].isin(df_tr["ID"].unique())].copy()
        doses_te = doses[doses["ID"].isin(df_te["ID"].unique())].copy()

    # --- Fit & evaluate ---
    pk_pars = fit_pk_1c_tlag_bw(pk_tr, df_tr, doses_tr, LOG_PK_FIT, VERBOSE)
    if VERBOSE:
        print(f"[PK] ka={pk_pars['ka']:.3f}, CL={pk_pars['CL']:.3f}, V={pk_pars['V']:.3f}, tlag={pk_pars['tlag']:.3f} h")

    pd_pars = fit_pd_emax_effectsite(pd_tr, pk_pars, df_tr, doses_tr, USE_HILL, USE_COMED, LOG_PD_FIT)
    if VERBOSE:
        E0,Emax,EC50,ke0,h,beta = pd_pars
        print(f"[PD] E0={E0:.3g}, Emax={Emax:.3g}, EC50={EC50:.3g}, ke0={ke0:.3g}, h={h:.3g}, beta={beta:.3g}")

    pd_pred_test = predict_pd_on(pd_te, df_te, doses_te, pk_pars, pd_pars, USE_HILL)
    rm, r2 = score_pd(pd_pred_test)
    print(f"\n=== PD TEST performance ===\nRMSE={rm:.3f}, R^2={r2:.3f}")

    # --- Plots ---
    if args.plots:
        os.makedirs("plots", exist_ok=True)
        for sid, g in pd_pred_test.groupby("ID"):
            g = g.sort_values("TIME")
            plt.figure()
            plt.plot(g["TIME"], g["B_pred"], label="Pred")
            plt.scatter(g["TIME"], g["B_obs"], s=40, label="Obs")
            plt.xlabel("Time (h)"); plt.ylabel("Biomarker")
            plt.title(f"Effect-site+Emax — ID={sid} (R2={score_pd(g)[1]:.3f})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/pd_overlay_id{sid}.png", dpi=150)
            plt.close()
        if VERBOSE:
            print("Plots saved to ./plots")

    # --- Bootstrap (ID-resampling on TRAIN for PD params) ---
    if args.bootstrap > 0:
        ids = sorted(pd_tr[pd_tr["EVID"]==0]["ID"].unique())
        boot_rows=[]
        for b in range(1, args.bootstrap+1):
            samp = RNG.choice(ids, size=len(ids), replace=True)
            pd_boot = pd.concat([pd_tr[pd_tr["ID"]==i] for i in samp], ignore_index=True)
            try:
                pd_pars_b = fit_pd_emax_effectsite(pd_boot, pk_pars, df_tr, doses_tr, USE_HILL, USE_COMED, LOG_PD_FIT)
                pred_b = predict_pd_on(pd_te, df_te, doses_te, pk_pars, pd_pars_b, USE_HILL)
                rm_b, r2_b = score_pd(pred_b)
                boot_rows.append([*pd_pars_b, rm_b, r2_b])
            except Exception as e:
                print(f"[bootstrap error] {b}-th bootstrap failed: {e}")
                continue
            if VERBOSE and (b % 10 == 0):
                print(f"[bootstrap] {b}/{args.bootstrap} done")
        if boot_rows:
            boot = pd.DataFrame(boot_rows, columns=["E0","Emax","EC50","ke0","h","beta","RMSE","R2"])
            qq = boot.quantile([0.025, 0.5, 0.975]).T
            print("\n=== PD bootstrap (ID-resampling) — param & metrics quantiles ===")
            print(qq)
            boot.to_csv("pd_bootstrap_results.csv", index=False)
            if VERBOSE:
                print("Bootstrap results saved to pd_bootstrap_results.csv")

if __name__ == "__main__":
    main()
