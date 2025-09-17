#!/usr/bin/env python3
"""
Improved Competition Solver - Based on real.ipynb's proven PD classification approach
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

warnings.filterwarnings('ignore')

class ImprovedCompetitionSolver:
    """Competition solver based on proven PD classification approach from real.ipynb"""
    
    def __init__(self, csv_path: str = "EstData.csv"):
        self.csv_path = csv_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.config = {
            'PD_THRESHOLD': 3.3,
            'EPOCHS': 60,
            'LR': 5e-2,
            'N_SUBJ': 300,
            'MLP_HIDDEN': 36,
            'MLP_LAYERS': 2,
            'MLP_DROPOUT': 0.2,
            'SEED': 42
        }
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['SEED'])
        torch.manual_seed(self.config['SEED'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config['SEED'])
        
        print(f"Improved Competition Solver initialized")
        print(f"Device: {self.device}")
        print(f"CSV path: {csv_path}")
    
    def _find_col_like(self, df, name_opts):
        """Find column by name options"""
        low = {c.lower(): c for c in df.columns}
        for n in name_opts:
            if n in low: 
                return low[n]
        return None
    
    def _load_and_preprocess_data(self):
        """Load and preprocess data from CSV"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.csv_path)
        
        # Detect columns
        col_ID = self._find_col_like(df, ["id"])
        col_TIME = self._find_col_like(df, ["time"])
        col_DVID = self._find_col_like(df, ["dvid"])
        col_DV = self._find_col_like(df, ["dv", "pd", "value"])
        col_EVID = self._find_col_like(df, ["evid"])
        col_AMT = self._find_col_like(df, ["amt", "dose", "dosen", "doses"])
        col_BW = self._find_col_like(df, ["bw", "weight", "bodyweight"])
        col_COMED = self._find_col_like(df, ["comed", "conmed", "concom"])
        
        print("Detected columns:", {
            'ID': col_ID, 'TIME': col_TIME, 'DV': col_DV, 'EVID': col_EVID, 
            'AMT': col_AMT, 'BW': col_BW, 'COMED': col_COMED
        })
        
        # Extract PD and dose data
        if col_DVID is not None and col_DV is not None:
            pdf = df[df[col_DVID] == 2].copy()
        else:
            pdf = df.copy()
        
        if col_EVID is not None:
            dose_df = df[df[col_EVID] == 1].copy()
        else:
            dose_df = df[df[col_AMT].notna()].copy()
        
        # Convert to numeric
        for c in [col_TIME, col_DV, col_AMT, col_BW, col_COMED]:
            if c is not None:
                pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
                dose_df[c] = pd.to_numeric(dose_df[c], errors="coerce")
        
        # Keep only necessary columns
        keep_pd = [c for c in [col_ID, col_TIME, col_DV, col_BW, col_COMED] if c is not None]
        keep_dose = [c for c in [col_ID, col_TIME, col_AMT] if c is not None]
        pdf = pdf[keep_pd].dropna().sort_values([col_ID, col_TIME])
        dose_df = dose_df[keep_dose].dropna().sort_values([col_ID, col_TIME])
        
        print(f"PD rows: {len(pdf)}, Dose rows: {len(dose_df)}")
        
        return pdf, dose_df, {
            'ID': col_ID, 'TIME': col_TIME, 'DV': col_DV, 'AMT': col_AMT,
            'BW': col_BW, 'COMED': col_COMED
        }
    
    class PDSamples(Dataset):
        """PD samples dataset"""
        def __init__(self, pd_df: pd.DataFrame, dose_df: pd.DataFrame,
                     col_ID: str, col_TIME: str, col_DV: str,
                     col_BW: Optional[str], col_COMED: Optional[str], 
                     pd_threshold: float = 3.3):
            self.col_ID, self.col_TIME, self.col_DV = col_ID, col_TIME, col_DV
            self.col_BW, self.col_COMED = col_BW, col_COMED
            self.pd_threshold = pd_threshold
            
            # ID별 투약 히스토리
            d_groups = defaultdict(list)
            for _, row in dose_df.iterrows():
                d_groups[row[col_ID]].append((float(row[col_TIME]), float(row[col_AMT])))
            self.dose_map = {
                k: (np.array([t for t, a in v], dtype=np.float32),
                    np.array([a for t, a in v], dtype=np.float32)) 
                for k, v in d_groups.items()
            }
            
            # 샘플 생성
            feats = []
            for _, row in pd_df.iterrows():
                sid = row[col_ID]
                if sid not in self.dose_map:
                    continue
                t = float(row[col_TIME])
                val = float(row[col_DV])
                y = 1.0 if (val <= pd_threshold) else 0.0
                bw = float(row[col_BW]) if col_BW is not None else 70.0
                cm = float(row[col_COMED]) if col_COMED is not None else 0.0
                feats.append((sid, t, y, bw, cm))
            self.samples = feats
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sid, t, y, bw, cm = self.samples[idx]
            dose_t, dose_a = self.dose_map.get(sid, 
                (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)))
            return {
                "t": torch.tensor(t, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.float32),
                "bw": torch.tensor(bw, dtype=torch.float32),
                "cm": torch.tensor(cm, dtype=torch.float32),
                "dose_t": torch.tensor(dose_t, dtype=torch.float32),
                "dose_a": torch.tensor(dose_a, dtype=torch.float32),
            }
    
    class PDMLPClassifier(nn.Module):
        """PD MLP Classifier with exposure modeling"""
        def __init__(self, bw_mean: float = 70.0, hidden: int = 32, 
                     n_layers: int = 3, dropout_p: float = 0.2):
            super().__init__()
            # tau 링크 파라미터
            self.b0 = nn.Parameter(torch.tensor(math.log(24.0)))
            self.b1 = nn.Parameter(torch.tensor(0.0))
            self.b2 = nn.Parameter(torch.tensor(0.0))
            self.bw_mean = float(bw_mean)
            
            layers = []
            in_dim = 3
            for i in range(n_layers):
                layers += [
                    nn.Linear(in_dim if i == 0 else hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity(),
                ]
            layers += [nn.Linear(hidden if n_layers > 0 else in_dim, 1)]
            self.mlp = nn.Sequential(*layers)
        
        def _tau(self, bw: torch.Tensor, comed: torch.Tensor):
            bwc = (bw - self.bw_mean) / 10.0
            log_tau = self.b0 + self.b1 * bwc + self.b2 * comed
            return torch.exp(log_tau).clamp_min(1.0)
        
        @torch.no_grad()
        def tau_from_cov_np(self, bw_np: np.ndarray, cm_np: np.ndarray, device=None):
            device = device or next(self.parameters()).device
            bw = torch.tensor(bw_np, dtype=torch.float32, device=device)
            cm = torch.tensor(cm_np, dtype=torch.float32, device=device)
            return self._tau(bw, cm).detach().cpu().numpy()
        
        def forward_single(self, t: torch.Tensor, dose_t: torch.Tensor, dose_a: torch.Tensor,
                          bw: float, comed: float):
            tau = self._tau(torch.tensor(bw, dtype=torch.float32, device=t.device),
                          torch.tensor(comed, dtype=torch.float32, device=t.device))
            dt = t - dose_t
            mask = (dt >= 0).float()
            exposure = (dose_a * torch.exp(-dt.clamp_min(0) / tau) * mask).sum()
            x = torch.stack([
                exposure,
                (torch.tensor(bw, dtype=torch.float32, device=t.device) - self.bw_mean) / 10.0,
                torch.tensor(comed, dtype=torch.float32, device=t.device)
            ])
            return self.mlp(x).squeeze()
    
    def _compute_metrics(self, y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5):
        """Compute classification metrics"""
        pred = (prob >= thr).astype(int)
        acc = accuracy_score(y_true, pred) if len(y_true) else float("nan")
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        try:
            roc = roc_auc_score(y_true, prob) if (len(np.unique(y_true)) > 1) else float("nan")
        except Exception:
            roc = float("nan")
        try:
            ap = average_precision_score(y_true, prob) if (len(np.unique(y_true)) > 1) else float("nan")
        except Exception:
            ap = float("nan")
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        except Exception:
            tn = fp = fn = tp = 0
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "roc_auc": roc, "pr_auc": ap,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    
    def _collate(self, batch):
        return batch
    
    def _split_data(self, df, dataset_all, col_ID):
        """Split data by ID groups"""
        rng = self.config['SEED']
        placebo_ids = set(range(1, 13))     # 0 mg (제외)
        ids_1mg = list(range(13, 25))       # 1 mg
        ids_3mg = list(range(25, 37))       # 3 mg
        ids_10mg = list(range(37, 49))      # 10 mg
        
        ids_in_data = set(df[col_ID].unique())
        ids_1mg = sorted(ids_in_data.intersection(ids_1mg))
        ids_3mg = sorted(ids_in_data.intersection(ids_3mg))
        ids_10mg = sorted(ids_in_data.intersection(ids_10mg))
        
        def split_70_15_15(ids, seed=42):
            if len(ids) == 0:
                return [], [], []
            tr_ids, temp_ids = train_test_split(ids, test_size=0.30, random_state=seed, shuffle=True)
            va_ids, te_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed, shuffle=True)
            return list(tr_ids), list(va_ids), list(te_ids)
        
        tr_1, va_1, te_1 = split_70_15_15(ids_1mg, seed=rng)
        tr_3, va_3, te_3 = split_70_15_15(ids_3mg, seed=rng)
        tr_10, va_10, te_10 = split_70_15_15(ids_10mg, seed=rng)
        
        ids_tr = set(tr_1 + tr_3 + tr_10)
        ids_va = set(va_1 + va_3 + va_10)
        ids_te = set(te_1 + te_3 + te_10)
        
        sid_list = [s[0] for s in dataset_all.samples]
        tr_idx = [i for i, sid in enumerate(sid_list) if sid in ids_tr]
        va_idx = [i for i, sid in enumerate(sid_list) if sid in ids_va]
        te_idx = [i for i, sid in enumerate(sid_list) if sid in ids_te]
        
        train_ds = Subset(dataset_all, tr_idx)
        valid_ds = Subset(dataset_all, va_idx)
        test_ds = Subset(dataset_all, te_idx)
        
        print("[ID split by fixed dose groups] (70/15/15)")
        print(f" train IDs: {len(ids_tr)} | valid IDs: {len(ids_va)} | test IDs: {len(ids_te)}")
        print(f" 1mg -> train/valid/test: {len(tr_1)} {len(va_1)} {len(te_1)}")
        print(f" 3mg -> train/valid/test: {len(tr_3)} {len(va_3)} {len(te_3)}")
        print(f"10mg -> train/valid/test: {len(tr_10)} {len(va_10)} {len(te_10)}")
        print(f"#samples -> train: {len(train_ds)} | valid: {len(valid_ds)} | test: {len(test_ds)}")
        
        return train_ds, valid_ds, test_ds
    
    def train_model(self):
        """Train the PD classification model"""
        print("Training PD classification model...")
        
        # Load and preprocess data
        pdf, dose_df, cols = self._load_and_preprocess_data()
        
        # Create dataset
        dataset_all = self.PDSamples(pdf, dose_df, cols['ID'], cols['TIME'], cols['DV'],
                                   cols['BW'], cols['COMED'], pd_threshold=self.config['PD_THRESHOLD'])
        
        # Split data
        train_ds, valid_ds, test_ds = self._split_data(pdf, dataset_all, cols['ID'])
        
        # Create model
        bw_mean = float(np.mean([b["bw"].item() for b in train_ds]))
        self.model = self.PDMLPClassifier(
            bw_mean=bw_mean,
            hidden=self.config['MLP_HIDDEN'],
            n_layers=self.config['MLP_LAYERS'],
            dropout_p=self.config['MLP_DROPOUT']
        ).to(self.device)
        
        # Training setup
        g = torch.Generator().manual_seed(self.config['SEED'])
        tr_loader = DataLoader(train_ds, batch_size=128, shuffle=True, 
                              collate_fn=self._collate, generator=g)
        va_loader = DataLoader(valid_ds, batch_size=256, shuffle=False, 
                              collate_fn=self._collate)
        
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config['LR'])
        loss_fn = nn.BCEWithLogitsLoss()
        
        best = (-1.0, None)
        best_state = None
        
        # Training loop
        for ep in range(1, self.config['EPOCHS'] + 1):
            self.model.train()
            for batch in tr_loader:
                opt.zero_grad()
                loss = 0.0
                for b in batch:
                    logit = self.model.forward_single(
                        b["t"].to(self.device), b["dose_t"].to(self.device), 
                        b["dose_a"].to(self.device), float(b["bw"]), float(b["cm"])
                    )
                    loss = loss + loss_fn(logit.view(()), b["y"].to(self.device).view(()))
                (loss / len(batch)).backward()
                opt.step()
            
            # Validation
            self.model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for batch in va_loader:
                    for b in batch:
                        logit = self.model.forward_single(
                            b["t"].to(self.device), b["dose_t"].to(self.device),
                            b["dose_a"].to(self.device), float(b["bw"]), float(b["cm"])
                        )
                        ys.append(float(b["y"]))
                        ps.append(torch.sigmoid(logit).item())
            
            y = np.array(ys)
            p = np.array(ps)
            metrics = self._compute_metrics(y, p, thr=0.5)
            
            # Save best model
            if metrics["f1"] > best[0]:
                best = (metrics["f1"], {"epoch": ep, **metrics})
                best_state = deepcopy(self.model.state_dict())
            
            if ep % 10 == 0 or ep <= 3:
                print(f"[ep {ep:03d}] acc={metrics['acc']:.3f} f1={metrics['f1']:.3f} "
                      f"prec={metrics['prec']:.3f} rec={metrics['rec']:.3f} "
                      f"roc_auc={metrics['roc_auc']:.3f} pr_auc={metrics['pr_auc']:.3f}")
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        print("best(valid):", best[1])
        
        # Test evaluation
        test_metrics = self._evaluate_dataset(test_ds)
        print("\n=== Final TEST metrics (unseen IDs) ===")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"{k:>8s}: {v:.4f}")
            else:
                print(f"{k:>8s}: {v}")
        
        return best[1], test_metrics
    
    def _evaluate_dataset(self, dataset, thr=0.5, batch_size=512):
        """Evaluate dataset"""
        device = next(self.model.parameters()).device
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate)
        ys, ps = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                for b in batch:
                    logit = self.model.forward_single(
                        b["t"].to(device), b["dose_t"].to(device), b["dose_a"].to(device),
                        float(b["bw"]), float(b["cm"])
                    )
                    ys.append(float(b["y"]))
                    ps.append(torch.sigmoid(logit).item())
        y = np.array(ys)
        p = np.array(ps)
        return self._compute_metrics(y, p, thr=thr)
    
    def subject_cov_sampler(self, pd_df: pd.DataFrame, col_BW: Optional[str], 
                           col_COMED: Optional[str], n: int, scenario: str = "base", 
                           rng_seed: int = 123):
        """Sample subject covariates for different scenarios"""
        obs_bw = pd_df[col_BW].dropna().to_numpy(dtype=float) if col_BW is not None else np.full(len(pd_df), 80.0)
        obs_cm = pd_df[col_COMED].dropna().to_numpy(dtype=float) if col_COMED is not None else np.zeros(len(pd_df))
        rng = np.random.default_rng(rng_seed)
        
        if scenario == "base":
            idx = rng.integers(0, len(obs_bw), size=n)
            return obs_bw[idx], obs_cm[idx]
        elif scenario == "bw_wide":
            bw = rng.uniform(70.0, 140.0, size=n)
            cm = obs_cm[rng.integers(0, len(obs_cm), size=n)] if len(obs_cm) > 0 else np.zeros(n)
            return bw, cm
        elif scenario == "no_comed":
            idx = rng.integers(0, len(obs_bw), size=n)
            return obs_bw[idx], np.zeros(n)
        else:
            raise ValueError("unknown scenario")
    
    @torch.no_grad()
    def success_fraction_for_dose_ss(self, dose_mg: float, freq_h: int, last_window_h: int,
                                   pd_df, col_BW: Optional[str], col_COMED: Optional[str],
                                   Nsubj=300, scenario="base", decision_threshold=0.5,
                                   grid_step_h=1.0, agg="min", alpha=0.9) -> float:
        """Calculate success fraction for steady-state dosing"""
        bw_arr, cm_arr = self.subject_cov_sampler(pd_df, col_BW, col_COMED, Nsubj, scenario=scenario)
        tau = self.model.tau_from_cov_np(bw_arr, cm_arr, device=next(self.model.parameters()).device)
        tgrid = np.arange(0.0, last_window_h + 1e-6, grid_step_h, dtype=float)
        ok = 0
        
        for i in range(Nsubj):
            denom = (1.0 - np.exp(-float(freq_h) / max(tau[i], 1e-6)))
            denom = max(denom, 1e-6)
            e_t = dose_mg * np.exp(-tgrid / max(tau[i], 1e-6)) / denom
            bwc = (bw_arr[i] - self.model.bw_mean) / 10.0
            cm = cm_arr[i]
            X = torch.tensor(
                np.stack([e_t, np.full_like(e_t, bwc), np.full_like(e_t, cm)], axis=1),
                dtype=torch.float32, device=next(self.model.parameters()).device
            )
            probs = torch.sigmoid(self.model.mlp(X).squeeze(1)).detach().cpu().numpy()
            
            if agg == "min":
                success = probs.min() >= decision_threshold
            elif agg == "trough":
                success = probs[-1] >= decision_threshold
            elif agg == "mean":
                success = probs.mean() >= decision_threshold
            elif agg == "coverage":
                success = (probs >= decision_threshold).mean() >= alpha
            else:
                raise ValueError("Unknown agg")
            
            ok += int(success)
        return ok / Nsubj
    
    def search_min_dose_ss(self, grid, freq_h, last_window_h, pd_df,
                          col_BW: Optional[str], col_COMED: Optional[str],
                          Nsubj=300, scenario="base", target=0.9, decision_threshold=0.5,
                          agg="min", alpha=0.9):
        """Search minimum dose for steady-state"""
        rows = []
        for d in grid:
            frac = self.success_fraction_for_dose_ss(
                d, freq_h, last_window_h, pd_df, col_BW, col_COMED,
                Nsubj=Nsubj, scenario=scenario, decision_threshold=decision_threshold,
                agg=agg, alpha=alpha
            )
            rows.append({"dose": d, "fraction": frac})
        df_res = pd.DataFrame(rows).sort_values("dose")
        feas = df_res[df_res["fraction"] >= target]
        best = feas.iloc[0]["dose"] if len(feas) > 0 else None
        return df_res, best
    
    def solve_all_tasks(self) -> Dict:
        """Solve all competition tasks"""
        print("Improved Competition Solver - Using Proven PD Classification Approach")
        print("=" * 70)
        
        # Train model first
        valid_best, test_metrics = self.train_model()
        
        # Load data for dose optimization
        pdf, dose_df, cols = self._load_and_preprocess_data()
        
        # Find optimal threshold
        # (This would need validation set predictions - simplified for now)
        thr_opt = 0.6  # From real.ipynb results
        
        # Define dose grids
        daily_grid = [0.5 * i for i in range(0, 121)]  # 0..60 mg, 0.5 mg 단위
        weekly_grid = [5 * i for i in range(0, 41)]    # 0..200 mg, 5 mg 단위
        
        results = {}
        
        # Task 1: Base scenario (90% coverage)
        print("Solving Task 1: Base scenario (90% coverage)")
        daily_base, best_daily_base = self.search_min_dose_ss(
            daily_grid, 24, 24, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="base", target=0.90, 
            decision_threshold=thr_opt, agg="min"
        )
        weekly_base, best_weekly_base = self.search_min_dose_ss(
            weekly_grid, 168, 168, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="base", target=0.90, 
            decision_threshold=thr_opt, agg="trough"
        )
        
        results['task_1'] = {
            "scenario": "Base (Phase 1-like)",
            "target_coverage": 90,
            "daily_dose_mg": round(best_daily_base, 1) if best_daily_base else None,
            "weekly_dose_mg": round(best_weekly_base, 1) if best_weekly_base else None,
            "method": "Proven PD classification with exposure modeling"
        }
        
        # Task 2: BW 70-140kg (90% coverage)
        print("Solving Task 2: BW 70-140kg (90% coverage)")
        daily_bw, best_daily_bw = self.search_min_dose_ss(
            daily_grid, 24, 24, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="bw_wide", target=0.90, 
            decision_threshold=thr_opt, agg="min"
        )
        weekly_bw, best_weekly_bw = self.search_min_dose_ss(
            weekly_grid, 168, 168, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="bw_wide", target=0.90, 
            decision_threshold=thr_opt, agg="trough"
        )
        
        results['task_2'] = {
            "scenario": "BW 70–140 kg",
            "target_coverage": 90,
            "daily_dose_mg": round(best_daily_bw, 1) if best_daily_bw else None,
            "weekly_dose_mg": round(best_weekly_bw, 1) if best_weekly_bw else None,
            "method": "Proven PD classification with exposure modeling"
        }
        
        # Task 3: No COMED allowed (90% coverage)
        print("Solving Task 3: No COMED allowed (90% coverage)")
        daily_nocm, best_daily_nocm = self.search_min_dose_ss(
            daily_grid, 24, 24, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="no_comed", target=0.90, 
            decision_threshold=thr_opt, agg="min"
        )
        weekly_nocm, best_weekly_nocm = self.search_min_dose_ss(
            weekly_grid, 168, 168, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="no_comed", target=0.90, 
            decision_threshold=thr_opt, agg="trough"
        )
        
        results['task_3'] = {
            "scenario": "No COMED allowed",
            "target_coverage": 90,
            "daily_dose_mg": round(best_daily_nocm, 1) if best_daily_nocm else None,
            "weekly_dose_mg": round(best_weekly_nocm, 1) if best_weekly_nocm else None,
            "method": "Proven PD classification with exposure modeling"
        }
        
        # Task 4: Base scenario (75% coverage)
        print("Solving Task 4: Base scenario (75% coverage)")
        daily_75, best_daily_75 = self.search_min_dose_ss(
            daily_grid, 24, 24, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="base", target=0.75, 
            decision_threshold=thr_opt, agg="min"
        )
        weekly_75, best_weekly_75 = self.search_min_dose_ss(
            weekly_grid, 168, 168, pdf, cols['BW'], cols['COMED'],
            Nsubj=self.config['N_SUBJ'], scenario="base", target=0.75, 
            decision_threshold=thr_opt, agg="trough"
        )
        
        results['task_4'] = {
            "scenario": "Base (Phase 1-like)",
            "target_coverage": 75,
            "daily_dose_mg": round(best_daily_75, 1) if best_daily_75 else None,
            "weekly_dose_mg": round(best_weekly_75, 1) if best_weekly_75 else None,
            "method": "Proven PD classification with exposure modeling"
        }
        
        # Task 5: Weekly dosing (90% coverage) - same as Task 1 weekly
        results['task_5'] = {
            "scenario": "Weekly dosing",
            "target_coverage": 90,
            "daily_dose_mg": round(best_weekly_base / 7, 1) if best_weekly_base else None,
            "weekly_dose_mg": round(best_weekly_base, 1) if best_weekly_base else None,
            "method": "Proven PD classification with exposure modeling"
        }
        
        # Save results
        self._save_results(results, "improved_competition_results.json")
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: Dict, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def _print_summary(self, results: Dict):
        """Print results summary"""
        print("\n" + "=" * 70)
        print("DOSE RECOMMENDATIONS SUMMARY (IMPROVED APPROACH)")
        print("=" * 70)
        print(f"{'scenario':<20} {'target':<8} {'once-daily (mg)':<15} {'once-weekly (mg)':<15}")
        print("-" * 70)
        
        for task_key, result in results.items():
            scenario = result['scenario']
            target = f"{result['target_coverage']}%"
            daily = result['daily_dose_mg']
            weekly = result['weekly_dose_mg']
            print(f"{scenario:<20} {target:<8} {daily:<15} {weekly:<15}")


if __name__ == "__main__":
    # Test the improved solver
    solver = ImprovedCompetitionSolver("EstData.csv")
    results = solver.solve_all_tasks()
