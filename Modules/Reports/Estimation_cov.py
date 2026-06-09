# -*- coding: utf-8 -*-
"""
Rapport PDF pour l'évaluation des estimateurs de covariance.
"""


from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # AVANT tout import de pyplot


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


# Palette couleurs
_PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#6366F1",]


def _model_color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]

# Extracteurs robustes depuis un BacktestResult (dataclass ou dict)
def _get(obj: Any, *names: str) -> Any:
    """Cherche les attributs dans obj (dict ou objet)."""
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


def _as_series(x: Any) -> Optional[pd.Series]:
    return x if isinstance(x, pd.Series) and len(x) > 0 else None


def _nav_from_returns(ret: pd.Series, base: float = 1.0) -> pd.Series:
    ret = ret.dropna()
    nav = (1.0 + ret).cumprod() * base
    nav.name = "NAV"
    return nav


def _extract(result: Any) -> Tuple[
    Optional[pd.Series],   # nav_port
    Optional[pd.Series],   # nav_bench
    Optional[pd.Series],   # ret_port
    Optional[pd.Series],   # ret_bench
]:
    
    ret_p = _as_series(_get(result, "portfolio_returns", "returns", "ptf_returns", "port_returns"))
    ret_b = _as_series(_get(result, "benchmark_returns", "bench_returns", "bm_returns"))
    nav_p = _as_series(_get(result, "nav", "portfolio_nav", "nav_series", "nav_portfolio"))
    nav_b = _as_series(_get(result, "benchmark_nav", "bench_nav", "nav_benchmark"))

    if nav_p is None and ret_p is not None:
        nav_p = _nav_from_returns(ret_p)
    if nav_b is None and ret_b is not None:
        nav_b = _nav_from_returns(ret_b)

    return nav_p, nav_b, ret_p, ret_b



# TE ex-post rolling
def _compute_te_rolling(ret_p: pd.Series, ret_b: pd.Series, window: int = 252) -> pd.Series:
    """TE ex-post = std rolling(window) des active returns, annualisée (x sqrt(252))."""
    
    idx = ret_p.index.intersection(ret_b.index)
    
    if len(idx) < window + 1:
        return pd.Series(dtype=float)
    
    active = ret_p.loc[idx] - ret_b.loc[idx]
    te = active.rolling(window).std() * np.sqrt(252)
    te.name = "TE_rolling"
    
    return te.dropna()



# Attribution sectorielle 
def _sector_attribution(
    ret_port: pd.Series,
    ret_bench: pd.Series,
    weights_port: pd.DataFrame,    # (T x N) poids port — début de période (t-1)
    weights_bench: pd.DataFrame,   # (T x N) poids bench — début de période (t-1)
    returns_assets: pd.DataFrame,  # (T x N) rendements journaliers par actif
    sector_map: Dict[str, str],    # ticker -> secteur
) -> pd.DataFrame:
    
    """
    Attribution sectorielle exactement selon Bloomberg PORT :
    Modèle Brinson-Fachler + liaison géométrique Menchero multi-périodes.
    """

    # Alignement 
    common_idx = (
        ret_port.index
        .intersection(ret_bench.index)
        .intersection(weights_port.index)
        .intersection(weights_bench.index)
        .intersection(returns_assets.index)
    )
    if len(common_idx) < 5:
        return pd.DataFrame()

    wp = weights_port.loc[common_idx].fillna(0.0)   # (T x N)
    wb = weights_bench.loc[common_idx].fillna(0.0)
    ra = returns_assets.loc[common_idx].fillna(0.0)
    rp = ret_port.loc[common_idx].fillna(0.0)       # (T,)
    rb = ret_bench.loc[common_idx].fillna(0.0)      # (T,)
    T  = len(common_idx)

    sectors = sorted(set(sector_map.values()))

    # ÉTAPE 1 : effets journaliers par secteur 

    # Rendement total benchmark journalier
    # On utilise ret_bench directement (rendement total du benchmark fourni)
    Rb_daily = rb.values  # (T,)

    # Pré-calculer pour chaque secteur
    sec_wp  = {}   # poids secteur port (T,)
    sec_wb  = {}   # poids secteur bench (T,)
    sec_rp  = {}   # return secteur port (T,)
    sec_rb  = {}   # return secteur bench (T,)
    #Calcul de la time series du ttr pour chaque secteur
    for sec in sectors:
        tickers_sec = [t for t in ra.columns if sector_map.get(t) == sec]
        if not tickers_sec:
            continue

        wp_s = wp[tickers_sec].values   # (T x K)
        wb_s = wb[tickers_sec].values
        ra_s = ra[tickers_sec].values

         # Sommes de poids par jour t, poids total du secteur
        sum_w_p = wp_s.sum(axis=1, keepdims=True)  # (T, 1)
        sum_w_b = wb_s.sum(axis=1, keepdims=True)  # (T, 1)
        
        # Si un secteur n'existe pas ce jour, on met NaN
        sum_w_p = np.where(sum_w_p > 1e-12, sum_w_p, 0)
        sum_w_b = np.where(sum_w_b > 1e-12, sum_w_b,0)
        
        # Poids normalisés intra-secteur
        wp_norm = wp_s / sum_w_p   # (T, K)
        wb_norm = wb_s / sum_w_b   # (T, K)
        
        # Rendements journaliers du secteur
        r_p_s = np.nansum(wp_norm * ra_s, axis=1)  # (T,)
        r_b_s = np.nansum(wb_norm * ra_s, axis=1)  # (T,)

        
        # Stockage des poids totaux du secteur 
        sec_wp[sec] = sum_w_p.squeeze()
        sec_wb[sec] = sum_w_b.squeeze()
        sec_rp[sec] = r_p_s
        sec_rb[sec] = r_b_s

    if not sec_wp:
        return pd.DataFrame()

    # ÉTAPE 2 : facteurs de liaison Menchero
    Rp_total = float((1.0 + rp).prod() - 1.0)
    Rb_total = float((1.0 + rb).prod() - 1.0)

    # Produits cumulatifs jusqu'à t inclus
    cum_p = (1.0 + rp).cumprod().values   # (T,)  — ∏_{k=1}^{t}(1+rp_k)
    cum_b = (1.0 + rb).cumprod().values

    # Facteur Menchero A_t (T,)
    A = ((1.0 + Rp_total) / cum_p) * (cum_b / (1.0 + Rb_total))
    
    
    rows = []
    for sec in sectors:
        if sec not in sec_wp:
            continue

        w_p_s = sec_wp[sec]   # (T,)
        w_b_s = sec_wb[sec]
        r_p_s = sec_rp[sec]   # returns journaliers du secteur
        r_b_s = sec_rb[sec]

        # Poids moyens
        w_p_mean = float(w_p_s.mean())
        w_b_mean = float(w_b_s.mean())

        # Total return sectoriel
        mask_p = w_p_s > 1e-10
        mask_b = w_b_s > 1e-10
        ret_p_cumul = float((1.0 + r_p_s[mask_p]).prod() - 1.0) if mask_p.any() else 0.0
        ret_b_cumul = float((1.0 + r_b_s[mask_b]).prod() - 1.0) if mask_b.any() else 0.0

        # CTR secteur façon “buy and hold”
        ctr_p = w_p_mean * ret_p_cumul
        ctr_b = w_b_mean * ret_b_cumul
        ctr_active = ctr_p - ctr_b

        rows.append({
            "sector":          sec,
            "w_port":          w_p_mean,
            "w_bench":         w_b_mean,
            "w_active":        w_p_mean - w_b_mean,
            "ret_port":        ret_p_cumul,
            "ret_bench":       ret_b_cumul,
            "ret_active":      ret_p_cumul - ret_b_cumul,
            "contrib_port":    ctr_p,
            "contrib_bench":   ctr_b,
            "contrib_active":  ctr_active,
        })
        
    
    """
    # ÉTAPE 3 : agrége avec liaison géométrique 
    rows = []
    for sec in sectors:
        if sec not in sec_wp:
            continue

        w_p_s = sec_wp[sec]   # (T,)
        w_b_s = sec_wb[sec]
        r_p_s = sec_rp[sec]
        r_b_s = sec_rb[sec]


        # Effets journaliers (Brinson-Fachler)
        alloc_daily      = (w_p_s - w_b_s) * (r_b_s - Rb_daily)
        selection_daily  = w_b_s           * (r_p_s - r_b_s)
        interaction_daily= (w_p_s - w_b_s) * (r_p_s - r_b_s)

        # CTR journalière
        ctr_p_daily = w_p_s * r_p_s
        ctr_b_daily = w_b_s * r_b_s

        # Liaison Menchero : multiplier par A_t et sommer
        alloc       = float((alloc_daily       * A).sum())
        selection   = float((selection_daily   * A).sum())
        interaction = float((interaction_daily * A).sum())
        ctr_p = float((ctr_p_daily * A).sum())
        ctr_b = float((ctr_b_daily * A).sum())
        ctr_active = ctr_p - ctr_b
        
        # Performance sectorielle cumulée (pour affichage)
        # = produit des rendements journaliers pondérés intra-secteur
        mask_p = w_p_s > 1e-10
        mask_b = w_b_s > 1e-10
        ret_p_cumul = float((1.0 + r_p_s[mask_p]).prod() - 1.0) if mask_p.any() else 0.0
        ret_b_cumul = float((1.0 + r_b_s[mask_b]).prod() - 1.0) if mask_b.any() else 0.0

        # Poids moyen (affiché, pas utilisé dans le calcul)
        w_p_mean = float(w_p_s.mean())
        w_b_mean = float(w_b_s.mean())

        rows.append({
            "sector":           sec,
            "w_port":           w_p_mean,
            "w_bench":          w_b_mean,
            "w_active":         w_p_mean - w_b_mean,
            "ret_port":         ret_p_cumul,
            "ret_bench":        ret_b_cumul,
            "ret_active":       ret_p_cumul - ret_b_cumul,
            "contrib_port":     ctr_p,
            "contrib_bench":    ctr_b,
            "contrib_active":   ctr_active,
            "allocation_effect":alloc,
            "selection_effect": selection,
            "interaction_effect":interaction,
            "total_active_effect": alloc + selection + interaction,
        })
    """
    
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("sector")

    # test
    rp_reconstruit = sum(sec_wp[s] * sec_rp[s] for s in sectors)
    print(np.max(np.abs(rp_reconstruit - rp.values)))


    """
    # Ligne TOTAL
    # Par construction Menchero :
    #   Σ_s (alloc_s + selection_s + interaction_s) = Rp_total - Rb_total
    total = {
        "w_port":            float(wp.sum(axis=1).mean()),
        "w_bench":           float(wb.sum(axis=1).mean()),
        "w_active":          float(wp.sum(axis=1).mean() - wb.sum(axis=1).mean()),
        "ret_port":          Rp_total,
        "ret_bench":         Rb_total,
        "ret_active":        Rp_total - Rb_total,
        "contrib_port":      float(df["contrib_port"].sum()),
        "contrib_bench":     float(df["contrib_bench"].sum()),
        "contrib_active":    float(df["contrib_active"].sum()),
        "allocation_effect": float(df["allocation_effect"].sum()),
        "selection_effect":  float(df["selection_effect"].sum()),
        "interaction_effect":float(df["interaction_effect"].sum()),
        "total_active_effect":float(df["total_active_effect"].sum()),  
    }
    df.loc["TOTAL"] = total
    """
    
    total = {
        "w_port":          float(wp.sum(axis=1).mean()),
        "w_bench":         float(wb.sum(axis=1).mean()),
        "w_active":        float(wp.sum(axis=1).mean() - wb.sum(axis=1).mean()),
        "ret_port":        float((1.0 + rp).prod() - 1.0),
        "ret_bench":       float((1.0 + rb).prod() - 1.0),
        "ret_active":      float((1.0 + rp).prod() - 1.0) - float((1.0 + rb).prod() - 1.0),
        "contrib_port":    sum(r["contrib_port"] for r in rows),
        "contrib_bench":   sum(r["contrib_bench"] for r in rows),
        "contrib_active":  sum(r["contrib_active"] for r in rows),
    }
    df.loc["TOTAL"] = total

    return df



# Commentaires automatiques sur les métriques statistiques
_METRIC_META = {
    "frobenius": {
        "label": "Erreur de Frobenius",
        "direction": "min",
        "desc": (
            "Distance euclidienne moyenne ||Σ̂(t) - Σ_true(t)||_F. "
            "Plus la valeur est BASSE, meilleure est l'estimation."
        ),
    },
    "spectral": {
        "label": "Erreur Spectrale",
        "direction": "min",
        "desc": (
            "Norme spectrale de l'erreur Σ̂ - Σ_true. "
            "Capture les erreurs sur les directions principales (facteurs dominants). "
            "Une valeur BASSE indique que les grandes directions du risque sont bien capturées."
        ),
    },
    "stein": {
        "label": "Perte de Stein",
        "direction": "min",
        "desc": (
            "Perte entropique : tr(Σ̂ Σ⁻¹) - log det(Σ̂ Σ⁻¹) - N. "
            "Critère théorique d'optimalité de Ledoit-Wolf. "
            "Pénalise les erreurs sur tout le spectre, y compris les petites valeurs propres. "
            "Une valeur BASSE indique un estimateur bien calibré spectralement."
        ),
    },
    "precision": {
        "label": "Erreur Précision (Frobenius)",
        "direction": "min",
        "desc": (
            "Frobenius sur la matrice de précision : ||Σ̂⁻¹ - Σ⁻¹||_F. "
            "Directement pertinent pour l'optimisation de portefeuille — "
            "les poids optimaux dépendent de la matrice de précision. "
            "Une valeur BASSE signifie de meilleures allocations."
        ),
    },
}



"""    "stein": {
        "label": "Perte de Stein",
        "direction": "min",
        "desc": (
            "Perte de Stein : tr(Σ̂ Σ_true⁻¹) - log det(Σ̂ Σ_true⁻¹) - p. "
            "Invariante aux transformations linéaires, pénalise les erreurs sur l'ensemble du spectre "
            "y compris les petites valeurs propres. Une valeur BASSE indique un estimateur bien calibré spectralement."
        ),
    },
    "precision": {
        "label": "Erreur sur la Précision (Frobenius)",
        "direction": "min",
        "desc": (
            "Erreur de Frobenius sur la matrice de précision : ||Σ̂⁻¹ - Σ_true⁻¹||_F. "
            "Directement pertinent pour l'optimisation de portefeuille — c'est la précision qui détermine "
            "les poids optimaux. Une valeur BASSE signifie de meilleures allocations."
        ),
    },
"""


def _auto_comments(metrics: pd.DataFrame) -> List[str]:
    """Génère automatiquement des phrases d'analyse pour chaque métrique présente."""
    lines = []
    models = [m for m in metrics.index if str(m) != "error"]

    for col, meta in _METRIC_META.items():
        if col not in metrics.columns:
            continue
        vals = pd.to_numeric(metrics.loc[models, col], errors="coerce").dropna()
        if vals.empty:
            continue

        best_m = vals.idxmin() if meta["direction"] == "min" else vals.idxmax()
        worst_m = vals.idxmax() if meta["direction"] == "min" else vals.idxmin()
        best_v = vals[best_m]
        worst_v = vals[worst_m]

        arrow = "↓ minimiser" if meta["direction"] == "min" else "↑ maximiser"
        line = (
            f"• {meta['label']} ({arrow}) — {meta['desc']} "
            f"Meilleur modèle : {best_m} ({best_v:.4f}). "
            f"Modèle le moins performant : {worst_m} ({worst_v:.4f})."
        )
        lines.append(line)

    # Conclusion : vote majoritaire
    if lines:
        scores: Dict[str, int] = {m: 0 for m in models}
        for col, meta in _METRIC_META.items():
            if col not in metrics.columns:
                continue
            vals = pd.to_numeric(metrics.loc[models, col], errors="coerce").dropna()
            if vals.empty:
                continue
            winner = vals.idxmin() if meta["direction"] == "min" else vals.idxmax()
            if winner in scores:
                scores[winner] += 1
        best_overall = max(scores, key=scores.get)
        n_wins = scores[best_overall]
        total_metrics = sum(1 for c in _METRIC_META if c in metrics.columns)
        lines.append(
            f"\n CONCLUSION : Sur {total_metrics} métriques statistiques, "
            f"{best_overall} est le meilleur estimateur avec {n_wins} critère(s) remporté(s). "
            "Ce modèle offre le meilleur compromis biais-variance dans l'estimation de la matrice de covariance."
        )

    return lines


def compute_economic_summary(bt_results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    all_years = set()

    for model, res in bt_results.items():
        if isinstance(res, dict) and "error" in res:
            continue

        _, _, ret_p, ret_b = _extract(res)
        if ret_p is None or ret_b is None:
            continue

        df = pd.concat({"p": ret_p, "b": ret_b}, axis=1).dropna()
        df["a"] = df["p"] - df["b"]  # active return
        df["year"] = df.index.year
        
        #first_year = df["year"].min()
        first_year = 2019 #Si on veut forcer la premiere année à 2020
        df_clean = df[df["year"] > first_year].copy()

        out = {"Model": model}

        for yr, sub in df_clean.groupby("year"):
            all_years.add(yr)
            ar = (1 + sub["a"]).prod() - 1.0
            te = sub["a"].std() * np.sqrt(252)
            out[f"AR_{yr}"] = ar
            out[f"TE_{yr}"] = te
            
            ir = ar / te if te > 0 else np.nan
            out[f"IR_{yr}"] = ir
        
        if df_clean.empty:
            continue
        
        # totals
        ar_total = (1 + df_clean["a"]).prod() - 1.0
        te_total = df_clean["a"].std() * np.sqrt(252)
        ir_total = ar_total / te_total if te_total > 0 else np.nan


        # annualized
        #n_years = (df.index[-1] - df.index[0]).days / 365.25
        n_years = (df_clean.index[-1] - df_clean.index[0]).days / 365.25
        ar_ann = (1 + ar_total) ** (1/n_years) - 1
        te_ann = te_total  # TE annualisé = déjà annualisé
        ir_ann = ar_ann / te_ann if te_ann >0 else np.nan

        out["AR_total"] = ar_total
        out["TE_total"] = te_total
        out["IR_total"] = ir_total
        out["AR_annual"] = ar_ann
        out["TE_annual"] = te_ann
        out["IR_annual"] = ir_ann


        rows.append(out)

    df_final = pd.DataFrame(rows).set_index("Model")

    # Convertit toutes les colonnes numériques en float proprement
    for c in df_final.columns:
        df_final[c] = pd.to_numeric(df_final[c], errors="coerce")

    
    df_final.attrs["years"] = sorted(all_years)
    return df_final

def compute_variance_calibration(
    bt_results: Dict[str, Any],
    ann_factor: int = 252,
) -> Dict[str, pd.DataFrame]:
    """
    Deux tests de calibration de variance en parallèle.

    TEST 1 - Rendement ACTIF (TE)
      sigma2_ante_active = te_ex_ante**2   (depuis rebal_diagnostics, daily)
      sigma2_post_active = Var(r_ptf - r_bench) sur la periode inter-rebal
      Verifie si le modele predit bien le risque de deviation vs benchmark.
      Colonnes : te_ante_ann, te_post_ann, ratio_active, bias_active_ann

    TEST 2 - Rendement PORTEFEUILLE pur
      sigma2_ante_ptf = var_ptf_ante  (depuis rebal_diagnostics, daily)
                      = w_ptf' Sigma w_ptf
      sigma2_post_ptf = Var(r_ptf) sur la periode inter-rebal
      Test pur sur la qualite de Sigma, independant de l'optimizer.
      Colonnes : vol_ptf_ante_ann, vol_ptf_post_ann, ratio_ptf, bias_ptf_ann

    Pour les deux :
      ratio  = sigma2_ante / sigma2_post  (ideal = 1)
      bias   = (sigma_ante - sigma_post) * sqrt(ann_factor)
      h >= 5 : sigma2_post = Var empirique ddof=1
      h <  5 : sigma2_post = r2_cumule / h  (proxy periode courte)

    TEST 2 necessite var_ptf_ante dans rebal_diagnostics (core_replication a jour).
    """

    result: Dict[str, pd.DataFrame] = {}

    def _var_post(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) == 0:
            return float("nan")
        if len(s) >= 5:
            return float(s.var(ddof=1))
        r_cum = float((1.0 + s).prod() - 1.0)
        return (r_cum ** 2) / len(s)

    for model, res in bt_results.items():
        if isinstance(res, dict) and "error" in res:
            continue

        rd = _get(res, "rebal_diagnostics")
        if rd is None or not isinstance(rd, pd.DataFrame) or rd.empty:
            continue
        if "te_ex_ante" not in rd.columns:
            continue

        ret_p = _as_series(_get(res, "portfolio_returns"))
        ret_b = _as_series(_get(res, "benchmark_returns"))
        if ret_p is None or ret_b is None:
            continue

        common_idx = ret_p.index.intersection(ret_b.index)
        ret_p  = ret_p.reindex(common_idx)
        ret_b  = ret_b.reindex(common_idx)
        active = (ret_p - ret_b).dropna()

        has_ptf = "var_ptf_ante" in rd.columns

        rebal_dates = rd.index.sort_values()
        rows = []

        for i, t_k in enumerate(rebal_dates):

            # Delimitation de la periode inter-rebal
            if i + 1 < len(rebal_dates):
                t_next = rebal_dates[i + 1]
                mask_a = (active.index >= t_k) & (active.index < t_next)
                mask_p = (ret_p.index  >= t_k) & (ret_p.index  < t_next)
            else:
                mask_a = active.index >= t_k
                mask_p = ret_p.index  >= t_k

            period_active = active.loc[mask_a].dropna()
            period_port   = ret_p.loc[mask_p].dropna()
            h = len(period_active)

            if h == 0:
                continue

            row: Dict[str, Any] = {"rebal_date": t_k, "h": h}

            # TEST 1 : rendement actif
            te_ante_daily = rd.loc[t_k, "te_ex_ante"]
            if np.isfinite(te_ante_daily) and te_ante_daily > 0:
                var_ante_a = float(te_ante_daily) ** 2
                var_post_a = _var_post(period_active)
                if np.isfinite(var_post_a) and var_post_a > 0:
                    row["te_ante_ann"]     = np.sqrt(var_ante_a) * np.sqrt(ann_factor)
                    row["te_post_ann"]     = np.sqrt(var_post_a) * np.sqrt(ann_factor)
                    row["ratio_active"]    = var_ante_a / var_post_a
                    row["bias_active_ann"] = (np.sqrt(var_ante_a) - np.sqrt(var_post_a)) * np.sqrt(ann_factor)

            # TEST 2 : portefeuille pur
            if has_ptf:
                var_ptf_ante = rd.loc[t_k, "var_ptf_ante"]
                if np.isfinite(var_ptf_ante) and var_ptf_ante > 0:
                    var_post_p = _var_post(period_port)
                    if np.isfinite(var_post_p) and var_post_p > 0:
                        row["vol_ptf_ante_ann"] = np.sqrt(var_ptf_ante) * np.sqrt(ann_factor)
                        row["vol_ptf_post_ann"] = np.sqrt(var_post_p)   * np.sqrt(ann_factor)
                        row["ratio_ptf"]        = var_ptf_ante / var_post_p
                        row["bias_ptf_ann"]     = (np.sqrt(var_ptf_ante) - np.sqrt(var_post_p)) * np.sqrt(ann_factor)

            if len(row) > 2:
                rows.append(row)

        if not rows:
            continue

        df = pd.DataFrame(rows).set_index("rebal_date")
        df.index = pd.DatetimeIndex(df.index)
        result[model] = df

    return result


# Classe principale
@dataclass
class CovarianceEstimationReport:
    title: str = "Covariance Estimation — Model Evaluation Report"
    figsize: Tuple[float, float] = (13.0, 8.5)
    dpi: int = 150
    te_window: int = 252   # fenêtre TE ex-post (jours)
    include_attribution: bool = False
    include_optim_landscape: bool = False
    optim_landscape_n_steps: int = 200   # résolution de la courbe (nb de points α)
    optim_landscape_alpha_max: float = 2.0  # distance max de déplacement depuis w_slsqp
    include_multistart: bool = False
    multistart_n_perturbations: int = 200 #nb pertubation aleatoire testé
    plot_start_date = pd.Timestamp("2019-12-31")

    def to_pdf(
        self,
        pdf_path,
        *,
        index_name: str,
        date_range: Tuple[str, str],
        metrics_table: Optional[pd.DataFrame],
        backtest_results: Dict[str, Any],
        model_specs: Optional[List[Any]] = None,   
        sector_map: Optional[Dict[str, str]] = None,
        universe_returns: Optional[pd.DataFrame] = None, 
        notes: str = "",
        ann_factor: int = 252,
        loss_std_table: Optional[pd.DataFrame] = None,
        n_scenarios_stats: int = 1,
        stat_sim_cfg = None,
    ) -> Path:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(pdf_path) as pdf:

            # 1. Page de garde
            self._page_title(pdf, index_name=index_name, date_range=date_range,
                             model_specs=model_specs, notes=notes)

            # 2. Évaluation statistique
            if metrics_table is not None and not metrics_table.empty:
                self._page_stats(pdf,metrics_table,loss_std=loss_std_table,n_scenarios=n_scenarios_stats,stat_sim_cfg=stat_sim_cfg,)
                
            if backtest_results:
                eco_summary = compute_economic_summary(backtest_results)
                self._page_economic_table(pdf, eco_summary)

            # Évaluation économique - période totale
            if backtest_results:
                self._page_nav(pdf, backtest_results, title_prefix="[TOTAL]")
                self._page_te(pdf, backtest_results, title_prefix="[TOTAL]")

            # TE ex-ante : graphique + tableau par rebal
            self._page_te_exante(pdf, backtest_results, title_prefix="[TOTAL]")

            # Test de calibration de variance
            calib_data = compute_variance_calibration(backtest_results, ann_factor=ann_factor)
            if calib_data:
                self._page_variance_calibration(pdf, calib_data, title_prefix="[TOTAL]")
                
            if self.include_multistart:
                self._page_solver_convergence(pdf,backtest_results)

            if self.include_attribution and sector_map:
                self._page_attribution(pdf, backtest_results, sector_map,
                                        universe_returns=universe_returns,
                                       title_prefix="[TOTAL]")

            # 4. Évaluation économique — par année
            years = self._get_years(backtest_results)
            for yr in years:
                self._page_nav(pdf, backtest_results,
                               title_prefix=f"[{yr}]", year=yr)
                self._page_te(pdf, backtest_results,
                              title_prefix=f"[{yr}]", year=yr)
                if self.include_attribution and sector_map:
                    self._page_attribution(pdf, backtest_results, sector_map,
                                            universe_returns=universe_returns,
                                           title_prefix=f"[{yr}]", year=yr)

        return pdf_path


    # Helpers
    def _new_fig(self, nrows=1, ncols=1, **kwargs):
        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize, dpi=self.dpi, **kwargs)
        return fig, axes

    def _save(self, pdf: PdfPages, fig, tight: bool = True):
        if tight:
            try:
                fig.tight_layout()
            except Exception:
                # ax.table() est incompatible avec tight_layout — on ignore silencieusement
                pass
        pdf.savefig(fig)
        plt.close(fig)

    def _get_years(self, bt: Dict[str, Any]) -> List[int]:
        years: set = set()
        for res in bt.values():
            if isinstance(res, dict) and "error" in res:
                continue
            nav_p, _, ret_p, _ = _extract(res)
            src = nav_p if nav_p is not None else ret_p
            if src is not None and hasattr(src.index, "year"):
                years.update(src.index.year.unique().tolist())
        return sorted(years)


    # Page 1 — Garde
    def _page_title(self, pdf, *, index_name, date_range, model_specs, notes):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.axis("off")

        y = 0.93
        ax.text(0.04, y, self.title, fontsize=22, fontweight="bold",
                transform=ax.transAxes, color="#1E3A5F")
        y -= 0.07
        ax.text(0.04, y, f"Index : {index_name}    |    Période : {date_range[0]} → {date_range[1]}",
                fontsize=13, transform=ax.transAxes, color="#555555")
        y -= 0.06

        # Modèles et paramètres
        if model_specs:
            ax.text(0.04, y, "Modèles testés :", fontsize=13, fontweight="bold",
                    transform=ax.transAxes, color="#1E3A5F")
            y -= 0.04
            for i, spec in enumerate(model_specs):
                name = getattr(spec, "name", str(spec))
                cfg = getattr(spec, "cov_cfg", None)
                if cfg is not None:
                    # Dump des paramètres non-None et non-vides
                    params = {}
                    for attr in dir(cfg):
                        if attr.startswith("_"):
                            continue
                        val = getattr(cfg, attr, None)
                        if val is not None and not callable(val):
                            params[attr] = val
                    params_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:10])
                else:
                    params_str = "—"
                color = _model_color(i)
                ax.text(0.06, y, f"▪ {name}", fontsize=11, fontweight="bold",
                        transform=ax.transAxes, color=color)
                # Wrap params sur 120 chars
                wrapped = textwrap.fill(params_str, width=120)
                for line in wrapped.split("\n"):
                    y -= 0.033
                    ax.text(0.09, y, line, fontsize=9, transform=ax.transAxes,
                            color="#444444", fontstyle="italic")
                y -= 0.018

        # Notes supplémentaires
        if notes:
            y -= 0.015
            ax.text(0.04, y, "Notes :", fontsize=11, fontweight="bold",
                    transform=ax.transAxes)
            y -= 0.03
            for line in textwrap.wrap(notes, width=130):
                ax.text(0.06, y, line, fontsize=9, transform=ax.transAxes, color="#555555")
                y -= 0.028

        ax.text(0.04, 0.02, "Généré via matplotlib PdfPages",
                fontsize=8, alpha=0.6, transform=ax.transAxes)

        self._save(pdf, fig)

    # Page 2 — Statistiques
    def _page_stats(self, pdf, metrics: pd.DataFrame,loss_std: Optional[pd.DataFrame] = None,n_scenarios: int = 1,stat_sim_cfg=None):
        # PAGE 2a : Tableau des métriques
        fig_tab = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax_tab = fig_tab.add_subplot(111)
        ax_tab.axis("off")

        ax_tab.text(0.0, 0.99, "Évaluation statistique — Simulation",
                    fontsize=16, fontweight="bold", transform=ax_tab.transAxes,
                    color="#1E3A5F", va="top")
        
    
        # Bloc paramètres DGP 
        if stat_sim_cfg is not None:
            cfg = stat_sim_cfg
            dgp = getattr(cfg, "dgp_type", "factor_shock")

            # Ligne 1 : paramètres communs
            line1 = (
                f"DGP : {dgp}   |   "
                f"N_sim : {getattr(cfg, 'N_sim', '—')}   |   "
                f"Scénarios : {getattr(cfg, 'n_scenarios', n_scenarios)}   |   "
                f"Facteurs K : {getattr(cfg, 'n_factors', '—')}   |   "
                f"Innovation : {getattr(cfg, 'innovation', '—')}   |   "
                f"Drift : {'oui' if getattr(cfg, 'add_drift', False) else 'non'}"
            )

            # Ligne 2 : paramètres spécifiques au DGP
            if dgp == "static_oracle":
                p_ratio = getattr(cfg, "p_ratio", None)
                t_sim   = getattr(cfg, "T_sim", None)
                line2 = (
                    f"p_ratio (N/T) : {p_ratio if p_ratio is not None else '—'}   |   "
                    f"T_sim : {t_sim if t_sim is not None else 'auto'}"
                )
            else:  # factor_shock
                line2 = (
                    f"rho_B : {getattr(cfg, 'rho_B', '—')}   |   "
                    f"sigma_B : {getattr(cfg, 'sigma_B', '—')}   |   "
                    f"rho_d : {getattr(cfg, 'rho_d', '—')}   |   "
                    f"sigma_d : {getattr(cfg, 'sigma_d', '—')}   |   "
                    f"T_sim : {getattr(cfg, 'T_sim', 'auto')}"
                )

            ax_tab.text(0.0, 0.93, line1, fontsize=8.5, transform=ax_tab.transAxes,
                        color="#333333", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF2FA",
                                edgecolor="#C0C8DC", linewidth=0.8))
            ax_tab.text(0.0, 0.87, line2, fontsize=8.5, transform=ax_tab.transAxes,
                        color="#333333", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF2FA",
                                edgecolor="#C0C8DC", linewidth=0.8))

            # Ajuste le bbox du tableau pour laisser la place aux deux lignes
            table_bottom = 0.13
            table_top    = 0.82
        else:
            table_bottom = 0.20
            table_top    = 0.72


        # Colonnes à afficher (ordre préféré)
        preferred_cols = [
            "frobenius", "spectral",
            "stein", "precision",
            "T_effective", "N_assets",
        ]


        cols_show = [c for c in preferred_cols if c in metrics.columns]
        cols_show += [c for c in metrics.columns if c not in cols_show and c != "error"]
        df_show = metrics[cols_show].copy()

        # Si multi-scénarios : affiche "mean ± std" dans chaque cellule
        if loss_std is not None and n_scenarios > 1:
            for c in cols_show:
                if c in loss_std.columns and c in df_show.columns:
                    for idx_val in df_show.index:
                        mu_val = df_show.loc[idx_val, c]
                        sd_val = loss_std.loc[idx_val, c] if idx_val in loss_std.index else float("nan")
                        if pd.notna(mu_val) and pd.notna(sd_val):
                            df_show.loc[idx_val, c] = f"{float(mu_val):.4f} ± {float(sd_val):.4f}"

        # Ajoute une note sur le nombre de scénarios
        _n_scen_note = f"Résultats moyennés sur {n_scenarios} scénario(s) indépendant(s)." if n_scenarios > 1 else ""

        # Arrondir les numériques
        for c in df_show.columns:
            if pd.api.types.is_numeric_dtype(df_show[c]):
                df_show[c] = pd.to_numeric(df_show[c], errors="coerce").round(5)

        col_labels = [metrics.index.name or "Model"] + list(df_show.columns)
        cell_text = []
        for idx, row in df_show.iterrows():
            cell_text.append([str(idx)] + [("" if pd.isna(v) else str(v)) for v in row.values])

        tab = ax_tab.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
                bbox=[0.0, table_bottom, 1.0, table_top - table_bottom], 
            )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1.05, 1.4)

        for (r, c), cell in tab.get_celld().items():
            cell.set_linewidth(0.4)
            if r == 0:
                cell.set_facecolor("#1E3A5F")
                cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F0F4FA")

        if _n_scen_note:
            ax_tab.text(0.0, 0.14, _n_scen_note,fontsize=8, transform=ax_tab.transAxes,color="#555555", style="italic",)

        self._save(pdf, fig_tab, tight=False)

        # PAGE 2b : Commentaires automatiques
        fig_com = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax_com = fig_com.add_subplot(111)
        ax_com.axis("off")

        ax_com.text(0.0, 0.99, "Évaluation statistique — Analyse & Interprétation",
                    fontsize=14, fontweight="bold", transform=ax_com.transAxes,
                    color="#1E3A5F", va="top")

        comments = _auto_comments(metrics)

        # Si aucun commentaire généré, afficher un message d'info sur les colonnes présentes
        if not comments:
            ax_com.text(0.01, 0.90,
                        f"Aucune métrique reconnue dans le tableau.\n"
                        f"Colonnes présentes : {list(metrics.columns)}\n"
                        f"Colonnes attendues : {list(_METRIC_META.keys())}",
                        fontsize=10, transform=ax_com.transAxes, va="top", color="#AA0000")
        else:
            y = 0.92
            for line in comments:
                is_conclusion = "➤" in line
                weight = "bold" if is_conclusion else "normal"
                color = "#1E3A5F" if is_conclusion else "#222222"
                size = 9.5 if is_conclusion else 8.8
                if is_conclusion:
                    y -= 0.025

                clean = line.strip()
                wrapped = textwrap.fill(clean, width=150)
                for sub in wrapped.split("\n"):
                    ax_com.text(0.01, y, sub, fontsize=size, transform=ax_com.transAxes,
                                va="top", color=color, fontweight=weight)
                    y -= 0.058
                    if y < 0.02:
                        break
                y -= 0.015

        self._save(pdf, fig_com, tight=False)

    def _page_economic_table(self, pdf, eco_df: pd.DataFrame):
        import textwrap as _tw

        # Données de base
        years   = sorted({int(c.split("_")[1]) for c in eco_df.columns if c.startswith("AR_20")})
        models  = list(eco_df.index)
        n_models = len(models)

        def fmt(x):
            return "—" if pd.isna(x) else f"{x*100:.2f}%"

        # Lignes de données
        rows_annual = []
        for yr in years:
            row = [str(yr)]
            for model in models:
                mrow = eco_df.loc[model]
                row.append(fmt(mrow.get(f"AR_{yr}", np.nan)))
                row.append(fmt(mrow.get(f"TE_{yr}", np.nan)))
                row.append(f"{mrow.get(f'IR_{yr}', np.nan):.2f}")
            rows_annual.append(row)

        row_total = ["Total cumulé"]
        row_ann   = ["Annualisé"]
        for model in models:
            mrow = eco_df.loc[model]
            row_total.append(fmt(mrow.get("AR_total",  np.nan)))
            row_total.append(fmt(mrow.get("TE_total",  np.nan)))
            row_total.append(f"{mrow.get('IR_total', np.nan):.2f}")

            row_ann.append(fmt(mrow.get("AR_annual", np.nan)))
            row_ann.append(fmt(mrow.get("TE_annual", np.nan)))
            row_ann.append(f"{mrow.get('IR_annual', np.nan):.2f}")

        all_rows    = rows_annual + [row_total, row_ann]
        n_body_rows = len(all_rows)
        n_annual    = len(rows_annual)

        # Dimensions
        cell_h_header = 0.068
        cell_h_subhdr = 0.062
        body_h        = 0.072
        header_total  = cell_h_header + cell_h_subhdr
        content_h     = header_total + n_body_rows * body_h
        # Hauteur figure : on veut que le contenu tienne dans ylim [0,1]
        # On fixe un ratio : fig_height = base * (content_h / 0.82) pour laisser 0.10 de marge titre
        fig_h = max(self.figsize[1], self.figsize[1] * (content_h + 0.15) / 0.85)

        fig, ax = plt.subplots(figsize=(self.figsize[0], fig_h), dpi=self.dpi)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.0, 0.98,
                "Évaluation économique — Synthèse par année",
                fontsize=14, fontweight="bold",
                transform=ax.transAxes, color="#1E3A5F", va="top")

        # Largeurs de colonnes 
        year_col_w = 0.10
        n_data_cols = 3 * n_models
        data_col_w  = (1.0 - year_col_w) / n_data_cols

        def col_x(j):
            return 0.0 if j == 0 else year_col_w + (j - 1) * data_col_w

        def col_w(j):
            return year_col_w if j == 0 else data_col_w

        y_top = 0.91

        # ---------- Header ligne 1 : noms de modèles ----------
        # Cellule "Année" (hauteur = 2 lignes header)
        ax.add_patch(plt.Rectangle(
            (col_x(0), y_top - cell_h_subhdr), col_w(0), cell_h_header + cell_h_subhdr,
            color="#1E3A5F", ec="black", lw=0.5, clip_on=False,
        ))
        ax.text(col_x(0) + col_w(0)/2, y_top - cell_h_subhdr/2,
                "Année", color="white", weight="bold",
                ha="center", va="center", fontsize=9, clip_on=False)

        for m_idx, model_name in enumerate(models):
            j0    = 1 + 3 * m_idx
            x_blk = col_x(j0)
            w_blk = 3 * data_col_w
            wrapped = "\n".join(_tw.wrap(model_name, width=22))
            ax.add_patch(plt.Rectangle(
                (x_blk, y_top), w_blk, cell_h_header,
                color="#1E3A5F", ec="black", lw=0.5, clip_on=False,
            ))
            ax.text(x_blk + w_blk/2, y_top + cell_h_header/2,
                    wrapped, color="white", weight="bold",
                    ha="center", va="center", fontsize=8, clip_on=False)

        # ---------- Header ligne 2 : sous-labels ----------
        y_sub = y_top - cell_h_subhdr
        for m_idx in range(n_models):
            for k, lbl in enumerate(["Act. Rtn", "TE ex-post"]):
                j = 1 + 3 * m_idx + k
                ax.add_patch(plt.Rectangle(
                    (col_x(j), y_sub), col_w(j), cell_h_subhdr,
                    color="#2A4F8A", ec="black", lw=0.5, clip_on=False,
                ))
                ax.text(col_x(j) + col_w(j)/2, y_sub + cell_h_subhdr/2,
                        lbl, color="white",
                        ha="center", va="center", fontsize=7.5, clip_on=False)

        # ---------- Corps ----------
        start_y = y_top - header_total
        for row_i, row in enumerate(all_rows):
            row_y = start_y - row_i * body_h
            is_summary = row_i >= n_annual
            bg = "#D0E4FF" if is_summary else ("#F6F8FB" if row_i % 2 == 0 else "white")
            fw = "bold" if is_summary else "normal"

            for j, val in enumerate(row):
                ax.add_patch(plt.Rectangle(
                    (col_x(j), row_y), col_w(j), body_h,
                    color=bg, ec="black", lw=0.3, clip_on=False,
                ))
                color = "black"
                if j > 0 and val not in ("—", ""):
                    try:
                        num = float(val.replace("%", "").replace(",", "."))
                        color = "#C0392B" if num < 0 else ("#1A7A3A" if num > 0 else "black")
                    except ValueError:
                        pass
                ax.text(col_x(j) + col_w(j)/2, row_y + body_h/2,
                        val, ha="center", va="center",
                        fontsize=8.5, fontweight=fw, color=color, clip_on=False)

        self._save(pdf, fig, tight=False)


    # Page NAV
    def _page_nav(self, pdf, bt: Dict[str, Any], *, title_prefix: str, year: Optional[int] = None):
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_title(f"{title_prefix} NAV comparison", fontsize=14, fontweight="bold",color="#1E3A5F", pad=12)
                
        any_line = False
        bench_plotted = False

        # Passage 1 : collecte des NAV portfolio (filtrées par année si besoin) 
        navs: Dict[str, pd.Series] = {}
        nav_b_ref: Optional[pd.Series] = None

        for model, res in bt.items():
            if isinstance(res, dict) and "error" in res:
                continue
            nav_p, nav_b, _, _ = _extract(res)

            if year is not None:
                nav_p = self._filter_year(nav_p, year, rebase=True)
                if nav_b_ref is None:
                    nav_b_ref = self._filter_year(nav_b, year, rebase=True)
            else:
                if nav_b_ref is None:
                    nav_b_ref = nav_b

            if nav_p is not None and len(nav_p) > 2:
                navs[model] = nav_p

        # Identification de la NAV de référence (démarrage le plus tôt), On cherche la série dont le premier index est le plus ancien.
        ref_nav: Optional[pd.Series] = None
        if navs:
            ref_model = min(navs, key=lambda m: navs[m].index[0])
            ref_nav = navs[ref_model]

        # Passage 2 : rebasage et tracé 
        for i, (model, res) in enumerate(bt.items()):
            if model not in navs:
                continue
            nav_p = navs[model]

            # Rebasage : ancrer sur la valeur de ref_nav au premier jour de nav_p
            if ref_nav is not None and nav_p.index[0] > ref_nav.index[0]:
                # Trouve la valeur de la référence au premier jour de cette série
                start_dt = nav_p.index[0]
                if start_dt in ref_nav.index:
                    anchor = float(ref_nav.loc[start_dt])
                else:
                    # Snap avant (ffill)
                    ref_before = ref_nav.loc[ref_nav.index <= start_dt]
                    anchor = float(ref_before.iloc[-1]) if len(ref_before) > 0 else 1.0
                # Rescale : nav_p démarre à 1 : on la fait démarrer à anchor
                nav_p = nav_p * anchor

            ax.plot(nav_p.index, nav_p.values,
                    label=model, color=_model_color(i), linewidth=1.6)
            any_line = True

        # Benchmark (non rebasé, déjà en base 1 ou filtrée par année)
        if nav_b_ref is not None and len(nav_b_ref) > 2:
            ax.plot(nav_b_ref.index, nav_b_ref.values,label="Benchmark", color="black", linewidth=2.2, linestyle="--")
            bench_plotted = True
            any_line = True

        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.grid(True, alpha=0.3)
        if any_line:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "NAV indisponible", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)

        self._save(pdf, fig)


    def _page_te(self, pdf, bt: Dict[str, Any],title_prefix: str, year: Optional[int] = None):
        """Page TE ex-post rolling"""
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_title( f"{title_prefix} Tracking Error ex-post — rolling {self.te_window} jours (annualisée)",fontsize=13, fontweight="bold", color="#1E3A5F", pad=12,)


        any_line = False
        for i, (model, res) in enumerate(bt.items()):
            if isinstance(res, dict) and "error" in res:
                continue
            _, _, ret_p, ret_b = _extract(res)
            if ret_p is None or ret_b is None:
                continue

            te = _compute_te_rolling(ret_p, ret_b, window=self.te_window)
            if te.empty:
                continue
            
            if self.plot_start_date is not None:
                te =te[te.index >= self.plot_start_date]

            if year is not None:
                te = self._filter_year(te, year, rebase=False)
            if te is None or te.empty:
                continue

            ax.plot(te.index, te.values, label=model, color=_model_color(i), linewidth=1.6)
            any_line = True

        ax.set_xlabel("Date")
        ax.set_ylabel("TE annualisée")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.grid(True, alpha=0.3)
        if any_line:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5,
                    f"TE indisponible\n(returns portfolio/benchmark non trouvés ou fenêtre {self.te_window}j insuffisante)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)

        self._save(pdf, fig)

    

    # Page TE ex-ante par rebal (scatter + ligne)
    def _page_te_exante(self, pdf, bt: Dict[str, Any], *, title_prefix: str, year = None):
        """
        Graphique de la TE ex-ante à chaque date de rebalancement.
        """

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_title(
            f"{title_prefix} TE ex-ante par rebalancement (objectif optimiseur)",
            fontsize=13, fontweight="bold", color="#1E3A5F", pad=12,
        )

        any_line = False
        for i, (model, res) in enumerate(bt.items()):
            if isinstance(res, dict) and "error" in res:
                continue

            rd = _get(res, "rebal_diagnostics")
            if rd is None or not isinstance(rd, pd.DataFrame) or rd.empty:
                continue
            if "te_ex_ante" not in rd.columns:
                continue

            te_ante = rd["te_ex_ante"].dropna()
            if te_ante.empty:
                continue
            
            te_ante = te_ante.iloc[4:] #On supprime la première année pour etre cohérent avec la ex post
            if year is not None:
                te_ante = self._filter_year(te_ante, year, rebase=False)
            if te_ante is None or te_ante.empty:
                continue

            # Annualise : TE² → TE → * sqrt(252)
            # te_ex_ante est déjà une TE (pas TE²) d'après le diag de _allocate_returns
            te_ann = te_ante * np.sqrt(252.0)

            color = _model_color(i)
            ax.plot(te_ann.index, te_ann.values,
                    label=model, color=color, linewidth=1.4,
                    marker="o", markersize=4, markerfacecolor=color)
            any_line = True

        ax.set_xlabel("Date de rebalancement")
        ax.set_ylabel("TE ex-ante annualisée")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))
        ax.grid(True, alpha=0.3)
        if any_line:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5,
                    "TE ex-ante indisponible\n"
                    "(result.rebal_diagnostics absent — vérifier engine.py)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)

        self._save(pdf, fig)

    

    # Tableau TE ex-ante par date de rebal (style identique au yearly table ex-post)
    def _page_te_exante_table(self, pdf, bt: Dict[str, Any]):
        """
        Tableau récapitulatif de la TE ex-ante par date de rebalancement.
        """
        models = [m for m, res in bt.items()
                  if not (isinstance(res, dict) and "error" in res)]
        if not models:
            return

        # Collecte des TE ex-ante par modèle
        te_by_model: Dict[str, pd.Series] = {}
        for model in models:
            res = bt[model]
            rd  = _get(res, "rebal_diagnostics")
            if rd is None or not isinstance(rd, pd.DataFrame) or rd.empty:
                continue
            if "te_ex_ante" not in rd.columns:
                continue
            te_by_model[model] = (rd["te_ex_ante"].dropna() * np.sqrt(252.0))

        if not te_by_model:
            return

        # Index commun : toutes les dates de rebal
        all_dates = sorted(set().union(*[s.index for s in te_by_model.values()]))
        if not all_dates:
            return

        def fmt(x):
            return "—" if pd.isna(x) else f"{float(x)*100:.4f}%"

        # Données : une ligne par date de rebal
        rows = []
        for dt in all_dates:
            dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            row = [dt_str]
            for model in models:
                s = te_by_model.get(model)
                val = s.get(dt, np.nan) if s is not None else np.nan
                row.append(fmt(val))
            rows.append(row)

        # Totaux / moyennes
        row_mean = ["Moyenne"]
        row_min  = ["Minimum"]
        row_max  = ["Maximum"]
        for model in models:
            s = te_by_model.get(model)
            if s is None or s.empty:
                row_mean.append("—"); row_min.append("—"); row_max.append("—")
            else:
                row_mean.append(fmt(s.mean()))
                row_min.append(fmt(s.min()))
                row_max.append(fmt(s.max()))
        all_rows   = rows + [row_mean, row_min, row_max]
        n_body     = len(all_rows)
        n_summary  = 3
        n_annual   = n_body - n_summary
        n_models   = len(models)
        n_cols     = 1 + n_models

        # Dimensions
        cell_h_header = 0.068
        cell_h_subhdr = 0.062
        body_h        = 0.052
        header_total  = cell_h_header + cell_h_subhdr
        content_h     = header_total + n_body * body_h
        fig_h = max(self.figsize[1], self.figsize[1] * (content_h + 0.15) / 0.85)
        fig_h = min(fig_h, 20.0)   # cap pour éviter les figures géantes

        import textwrap as _tw
        fig, ax = plt.subplots(figsize=(self.figsize[0], fig_h), dpi=self.dpi)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.0, 0.98,
                "TE ex-ante par date de rebalancement",
                fontsize=14, fontweight="bold",
                transform=ax.transAxes, color="#1E3A5F", va="top")

        date_col_w = 0.14
        data_col_w = (1.0 - date_col_w) / n_models

        def col_x(j):
            return 0.0 if j == 0 else date_col_w + (j - 1) * data_col_w

        def col_w(j):
            return date_col_w if j == 0 else data_col_w

        y_top = 0.91

        # Header ligne 1 : noms des modèles
        ax.add_patch(plt.Rectangle(
            (col_x(0), y_top - cell_h_subhdr), col_w(0), cell_h_header + cell_h_subhdr,
            color="#1E3A5F", ec="black", lw=0.5, clip_on=False,
        ))
        ax.text(col_x(0) + col_w(0)/2, y_top,
                "Date rebal", color="white", weight="bold",
                ha="center", va="center", fontsize=8, clip_on=False)

        for m_idx, model_name in enumerate(models):
            j = 1 + m_idx
            wrapped = "\n".join(_tw.wrap(model_name, width=20))
            ax.add_patch(plt.Rectangle(
                (col_x(j), y_top), col_w(j), cell_h_header,
                color="#1E3A5F", ec="black", lw=0.5, clip_on=False,
            ))
            ax.text(col_x(j) + col_w(j)/2, y_top + cell_h_header/2,
                    wrapped, color="white", weight="bold",
                    ha="center", va="center", fontsize=7.5, clip_on=False)

        # Header ligne 2 : sous-label "TE ex-ante"
        y_sub = y_top - cell_h_subhdr
        for m_idx in range(n_models):
            j = 1 + m_idx
            ax.add_patch(plt.Rectangle(
                (col_x(j), y_sub), col_w(j), cell_h_subhdr,
                color="#2A4F8A", ec="black", lw=0.5, clip_on=False,
            ))
            ax.text(col_x(j) + col_w(j)/2, y_sub + cell_h_subhdr/2,
                    "TE ex-ante ann.", color="white",
                    ha="center", va="center", fontsize=7, clip_on=False)

        # Corps
        start_y = y_top - header_total
        for row_i, row in enumerate(all_rows):
            row_y = start_y - row_i * body_h
            is_summary = row_i >= n_annual
            bg = "#D0E4FF" if is_summary else ("#F6F8FB" if row_i % 2 == 0 else "white")
            fw = "bold" if is_summary else "normal"

            for j, val in enumerate(row):
                ax.add_patch(plt.Rectangle(
                    (col_x(j), row_y), col_w(j), body_h,
                    color=bg, ec="black", lw=0.3, clip_on=False,
                ))
                ax.text(col_x(j) + col_w(j)/2, row_y + body_h/2,
                        val, ha="center", va="center",
                        fontsize=7, fontweight=fw, clip_on=False)

        self._save(pdf, fig, tight=False)
        
        
 
    def _page_solver_convergence(self, pdf, bt: Dict[str, Any]) -> None:
        """
        [AJOUT v2.0] Graphique simple : chemin du solveur dans l'espace (distance_w0, TE).

        Axe X = TE ex-ante annualisée (%)
        Axe Y = ||w - w0|| distance euclidienne au point de départ
        Chaque point = une évaluation de l'objectif pendant l'optimisation

        SLSQP : tous les points intermédiaires visibles → chemin complet
        Clarabel : seulement départ + arrivée (n'itère pas sur value())

        Si SLSQP et Clarabel atterrissent au même point → minimum global
        Si écart → SLSQP est dans un minimum local
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_title(
            "Chemin du solveur — espace (TE ex-ante, distance au point de départ)"
            "\n(dernier rebalancement)",
            fontsize=13, fontweight="bold", color="#1E3A5F", pad=12,
        )

        any_data = False

        for i, (model, res) in enumerate(bt.items()):
            if isinstance(res, dict) and "error" in res:
                continue

            log = _get(res, "solver_eval_log")
            if log is None or len(log) == 0:
                continue

            # Dépaquète : list of (w_array, te_ann)
            try:
                ws     = [np.asarray(e[0], dtype=float) for e in log]
                te_ann = [float(e[1]) for e in log]
            except (IndexError, TypeError):
                continue

            if not ws:
                continue

            color  = _model_color(i)
            w0     = ws[0]

            # Distance euclidienne de chaque point au w0
            dists = [float(np.linalg.norm(w - w0)) for w in ws]

            n_pts = len(ws)

            if n_pts <= 2:
                # Clarabel : seulement départ + arrivée
                # Départ
                ax.scatter(
                    [te_ann[0] * 100], [dists[0]],
                    color=color, s=120, marker="o", zorder=5,
                    edgecolors="white", linewidths=0.8,
                )
                # Arrivée
                ax.scatter(
                    [te_ann[-1] * 100], [dists[-1]],
                    color=color, s=180, marker="*", zorder=6,
                    edgecolors="white", linewidths=0.8,
                    label=f"{model}  ★ TE={te_ann[-1]*100:.3f}%",
                )
                # Flèche départ → arrivée
                if n_pts == 2:
                    ax.annotate(
                        "",
                        xy=(te_ann[-1]*100, dists[-1]),
                        xytext=(te_ann[0]*100, dists[0]),
                        arrowprops=dict(
                            arrowstyle="->", color=color,
                            lw=1.2, alpha=0.6,
                        ),
                    )
            else:
                # SLSQP : chemin complet
                # Points intermédiaires avec dégradé (clair → foncé)
                alphas = np.linspace(0.15, 0.8, n_pts)
                sizes  = np.linspace(15, 50, n_pts)

                for j in range(n_pts):
                    ax.scatter(
                        [te_ann[j] * 100], [dists[j]],
                        color=color, alpha=float(alphas[j]),
                        s=float(sizes[j]), zorder=3,
                    )

                # Ligne du chemin
                ax.plot(
                    [v * 100 for v in te_ann], dists,
                    color=color, linewidth=0.8, alpha=0.35, zorder=2,
                )

                # Point de départ
                ax.scatter(
                    [te_ann[0] * 100], [dists[0]],
                    color=color, s=120, marker="o", zorder=5,
                    edgecolors="white", linewidths=0.8,
                )

                # Solution finale
                ax.scatter(
                    [te_ann[-1] * 100], [dists[-1]],
                    color=color, s=200, marker="*", zorder=6,
                    edgecolors="white", linewidths=0.8,
                    label=f"{model}  ★ TE={te_ann[-1]*100:.3f}%  ({n_pts} éval.)",
                )

            any_data = True

        ax.set_xlabel("TE ex-ante annualisée (%)")
        ax.set_ylabel("Distance euclidienne au point de départ ||w - w0||")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.3f}%")
        )
        ax.grid(True, alpha=0.3)

        if any_data:
            ax.legend(fontsize=9, loc="upper right")
            ax.text(
                0.01, 0.02,
                "● = point de départ (w0)  ★ = solution finale\n"
                "SLSQP : chemin complet (points de plus en plus foncés)\n"
                "Clarabel : départ → arrivée uniquement (flèche)\n"
                "Si ★ SLSQP ≈ ★ Clarabel → minimum global trouvé",
                transform=ax.transAxes, fontsize=8,
                color="gray", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray", alpha=0.85),
            )
        else:
            ax.text(
                0.5, 0.5,
                "Log indisponible\n"
                "(vérifier que _last_rebal_dt et log_evaluations sont actifs)",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11,
            )

        self._save(pdf, fig)
  


    # Page — Test de calibration de variance
    # ── Helpers privés partagés par les 3 slides de calibration ────────────
    def _calib_has_ptf(self, calib_data: Dict[str, pd.DataFrame]) -> bool:
        return any("ratio_ptf" in df.columns for df in calib_data.values() if not df.empty)

    def _calib_scatter(self, ax, calib_data, x_col, y_col, xlabel, ylabel, title):
        """Scatter ante vs post + diagonale calibration parfaite."""
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1E3A5F")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.grid(True, alpha=0.25)
        all_vals = []
        for i, (model, df) in enumerate(calib_data.items()):
            if df.empty or x_col not in df.columns or y_col not in df.columns:
                continue
            x = df[x_col].dropna().values
            y = df[y_col].dropna().values
            n = min(len(x), len(y))
            if n == 0:
                continue
            ax.scatter(x[:n], y[:n], label=model, color=_model_color(i),
                       s=35, alpha=0.75, edgecolors="none")
            all_vals.extend(x[:n].tolist() + y[:n].tolist())
        if all_vals:
            lo = min(all_vals) * 0.9
            hi = max(all_vals) * 1.1
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2,
                    linestyle="--", label="Calibration parfaite")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.legend(fontsize=8, loc="upper left")

    def _calib_ratio_line(self, ax, calib_data, ratio_col, title):
        """Ratio sigma2_ante / sigma2_post au fil du temps."""
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1E3A5F")
        ax.set_xlabel("Date de rebalancement", fontsize=9)
        ax.set_ylabel("sigma2_ante / sigma2_post", fontsize=9)
        ax.axhline(1.0, color="black", linewidth=1.3, linestyle="--", label="Ideal (=1)")
        ax.axhline(2.0, color="#DC2626", linewidth=0.8, linestyle=":", alpha=0.6,
                   label="Seuil x2 / /2")
        ax.axhline(0.5, color="#DC2626", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.grid(True, alpha=0.25)
        for i, (model, df) in enumerate(calib_data.items()):
            if df.empty or ratio_col not in df.columns:
                continue
            ratio = df[ratio_col].dropna().clip(0, 5)
            if ratio.empty:
                continue
            ax.plot(ratio.index, ratio.values, label=model,
                    color=_model_color(i), linewidth=1.4,
                    marker="o", markersize=4,
                    markerfacecolor=_model_color(i))
        ax.legend(fontsize=8)

    def _calib_table(self, ax, calib_data, ratio_col, bias_col, note):
        """Tableau recap : ratio moyen, std, biais, % sous/sur-estimation."""
        ax.axis("off")
        ax.set_title("Resume statistique", fontsize=11, fontweight="bold", color="#1E3A5F")

        def _fp(x): return "—" if (x is None or not np.isfinite(x)) else f"{x:.1%}"
        def _ff(x): return "—" if (x is None or not np.isfinite(x)) else f"{x:.2f}"

        col_labels = ["Modele", "Ratio moy.", "Ratio std",
                      "Biais moy. (ann.)", "Sous-est. %", "Sur-est. %"]
        rows_tab = []
        for model, df in calib_data.items():
            if df.empty or ratio_col not in df.columns:
                continue
            ratio = df[ratio_col].dropna()
            bias  = df[bias_col].dropna() if bias_col in df.columns else pd.Series(dtype=float)
            if ratio.empty:
                continue
            rows_tab.append([
                model,
                _ff(float(ratio.mean())),
                _ff(float(ratio.std(ddof=1)) if len(ratio) > 1 else float("nan")),
                _fp(float(bias.mean()) if not bias.empty else float("nan")),
                _fp(float((ratio < 1.0).mean())),
                _fp(float((ratio > 1.0).mean())),
            ])

        if rows_tab:
            n_rows = len(rows_tab)
            row_h  = min(0.10, 0.70 / max(n_rows, 1))
            tab_h  = row_h * (n_rows + 1)
            bbox   = [0.02, 0.20, 0.96, min(tab_h + 0.05, 0.72)]
            tab = ax.table(
                cellText=rows_tab, colLabels=col_labels,
                loc="center", cellLoc="center",
                bbox=bbox,
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(8)
            tab.scale(1.0, 1.8)
            for (r, c), cell in tab.get_celld().items():
                cell.set_linewidth(0.4)
                if r == 0:
                    cell.set_facecolor("#1E3A5F")
                    cell.set_text_props(color="white", weight="bold", fontsize=7.5)
                elif r % 2 == 0:
                    cell.set_facecolor("#F0F4FA")

        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize=7.5, color="#555555", va="bottom")

    # ── SLIDE 1 : scatter TE ante vs post (ligne 1) + scatter Vol ante vs post (ligne 2)

    def _page_calib_scatter(self, pdf, calib_data, *, title_prefix):
        has_ptf = self._calib_has_ptf(calib_data)
        n_rows  = 2 if has_ptf else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=self.figsize, dpi=self.dpi, squeeze=False)
        fig.suptitle(f"{title_prefix} Calibration — Scatter ante vs post",
                     fontsize=13, fontweight="bold", color="#1E3A5F", y=0.99)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07, hspace=0.45)

        self._calib_scatter(axes[0, 0], calib_data,
                            x_col="te_ante_ann", y_col="te_post_ann",
                            xlabel="TE ex-ante annualisee", ylabel="TE ex-post annualisee",
                            title="TEST 1 — TE active : ex-ante vs ex-post")
        if has_ptf:
            self._calib_scatter(axes[1, 0], calib_data,
                                x_col="vol_ptf_ante_ann", y_col="vol_ptf_post_ann",
                                xlabel="Vol portefeuille ex-ante annualisee",
                                ylabel="Vol portefeuille ex-post annualisee",
                                title="TEST 2 — Vol portefeuille pur : ex-ante vs ex-post")
        self._save(pdf, fig, tight=False)

    # ── SLIDE 2 : ratio TE (ligne 1) + ratio Vol (ligne 2)

    def _page_calib_ratio(self, pdf, calib_data, *, title_prefix):
        has_ptf = self._calib_has_ptf(calib_data)
        n_rows  = 2 if has_ptf else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=self.figsize, dpi=self.dpi, squeeze=False)
        fig.suptitle(f"{title_prefix} Calibration — Ratio sigma2_ante / sigma2_post",
                     fontsize=13, fontweight="bold", color="#1E3A5F", y=0.99)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07, hspace=0.45)

        self._calib_ratio_line(axes[0, 0], calib_data,
                               ratio_col="ratio_active",
                               title="TEST 1 — Ratio TE active : sigma2_ante / sigma2_post")
        if has_ptf:
            self._calib_ratio_line(axes[1, 0], calib_data,
                                   ratio_col="ratio_ptf",
                                   title="TEST 2 — Ratio Vol portefeuille : sigma2_ante / sigma2_post")
        self._save(pdf, fig, tight=False)

    # ── SLIDE 3 : tableau recap TE (ligne 1) + tableau recap Vol (ligne 2)

    def _page_calib_tables(self, pdf, calib_data, *, title_prefix):
        has_ptf = self._calib_has_ptf(calib_data)
        n_rows  = 2 if has_ptf else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=self.figsize, dpi=self.dpi, squeeze=False)
        fig.suptitle(f"{title_prefix} Calibration — Tableaux recapitulatifs",
                     fontsize=13, fontweight="bold", color="#1E3A5F", y=0.99)
        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.04, hspace=0.50)

        self._calib_table(axes[0, 0], calib_data,
                          ratio_col="ratio_active", bias_col="bias_active_ann",
                          note="TEST 1 — Ratio = sigma2_ante_active / sigma2_post_active  |  Ideal = 1\n"
                               "sigma2_ante = te_ex_ante²  |  sigma2_post = Var(r_ptf - r_bench) inter-rebal\n"
                               "Sous-est. : ratio < 1  |  Sur-est. : ratio > 1")
        if has_ptf:
            self._calib_table(axes[1, 0], calib_data,
                              ratio_col="ratio_ptf", bias_col="bias_ptf_ann",
                              note="TEST 2 — Ratio = sigma2_ante_ptf / sigma2_post_ptf  |  Ideal = 1\n"
                                   "sigma2_ante = w_ptf' Sigma w_ptf  |  sigma2_post = Var(r_ptf) inter-rebal\n"
                                   "Test pur qualite de Sigma — independant de l'optimizer")
        self._save(pdf, fig, tight=False)

    # Point d'entree public : appelle les 3 slides dans l'ordre
    def _page_variance_calibration(self, pdf, calib_data, *, title_prefix):
        """Produit 3 slides de calibration : scatter / ratio / tableaux."""
        if not calib_data:
            return
        self._page_calib_scatter(pdf, calib_data, title_prefix=title_prefix)
        self._page_calib_ratio(pdf, calib_data, title_prefix=title_prefix)
        self._page_calib_tables(pdf, calib_data, title_prefix=title_prefix)



    # Page Attribution sectorielle
    def _page_attribution(
        self,
        pdf,
        bt: Dict[str, Any],
        sector_map: Dict[str, str],
        *,
        title_prefix: str,
        year: Optional[int] = None,
        universe_returns: Optional[pd.DataFrame] = None,
    ):
        """
        Pour chaque modèle disponible dans bt, tente l'attribution sectorielle.
        universe_returns : DataFrame (T x N) des rendements par actif (all_returns).Utilisé comme source de ra si non disponible dans res.            
        """
        for i, (model, res) in enumerate(bt.items()):
            if isinstance(res, dict) and "error" in res:
                continue

            _, _, ret_p, ret_b = _extract(res)
            wp = _get(res, "weights_in_force","portfolio_weights", "weights", "w_port")
            wb = _get(res, "bench_weights_in_force", "benchmark_weights", "bench_weights", "w_bench")

            # ra : cherche d'abord dans res, sinon utilise universe_returns passé en argument
            ra = _get(res, "asset_returns", "returns_assets")
            if ra is None and universe_returns is not None:
                ra = universe_returns

            # Toutes les sources doivent être des DataFrames
            if not all(isinstance(x, pd.DataFrame) for x in [wp, wb, ra]):
                continue
            if ret_p is None or ret_b is None:
                continue

            # Filtrage par année
            if year is not None:
                common = (
                    ret_p.index.intersection(ret_b.index)
                    .intersection(wp.index).intersection(wb.index).intersection(ra.index)
                )
                mask = common[common.year == year]
                if len(mask) < 5:
                    continue
                ret_p = ret_p.loc[mask]
                ret_b = ret_b.loc[mask]
                wp = wp.loc[mask]
                wb = wb.loc[mask]
                ra = ra.loc[mask]

            attr_df = _sector_attribution(ret_p, ret_b, wp, wb, ra, sector_map)
            if attr_df.empty:
                continue

            # Noms de colonnes abrégés pour tenir dans le tableau
            COL_RENAME = {
                "w_port":             "w_ptf",
                "w_bench":            "w_bench",
                "w_active":           "w_act",
                "ret_port":           "ret_ptf",
                "ret_bench":          "ret_bench",
                "ret_active":         "ret_act",
                "contrib_port":       "ctr_ptf",
                "contrib_bench":      "ctr_bench",
                "contrib_active":     "ctr_act",
            }
            
            """"allocation_effect":  "alloc",
                "selection_effect":   "select",
                "interaction_effect": "interact",
                "total_active_effect":"tot_act",
            """

            disp_cols = [c for c in COL_RENAME if c in attr_df.columns]
            df_disp = attr_df[disp_cols].copy()

            # Format pourcentage
            for c in disp_cols:
                df_disp[c] = df_disp[c].map(lambda v: f"{v:.2%}" if pd.notna(v) else "")

            # Headers abrégés
            col_labels = ["Secteur"] + [COL_RENAME[c] for c in disp_cols]

            # Noms de secteur : retour à la ligne automatique (max 18 chars par ligne)
            def _wrap_sector(s: str, width: int = 18) -> str:
                return "\n".join(textwrap.wrap(str(s), width=width))

            cell_text = []
            for idx, row in df_disp.iterrows():
                cell_text.append([_wrap_sector(idx)] + list(row.values))

            # Figure : 70% tableau, 30% barplot — figsize agrandi en hauteur pour les secteurs
            n_rows = len(df_disp) + 1  # +1 header
            fig_h = max(self.figsize[1], n_rows * 0.38 + 1.5)
            fig = plt.figure(figsize=(self.figsize[0], fig_h), dpi=self.dpi)
            fig.suptitle(
                f"{title_prefix} Attribution sectorielle — {model}",
                fontsize=13, fontweight="bold", color="#1E3A5F", y=0.99,
            )
            gs_attr = gridspec.GridSpec(
                1, 2, width_ratios=[3, 1], figure=fig,
                left=0.02, right=0.98, top=0.93, bottom=0.03, wspace=0.06,
            )
            ax_tab = fig.add_subplot(gs_attr[0])
            ax_bar = fig.add_subplot(gs_attr[1])

            # Tableau
            ax_tab.axis("off")

            # Hauteur de ligne adaptée au nombre de lignes de texte max dans la colonne Secteur
            max_lines = max(len(_wrap_sector(str(idx)).split("\n")) for idx in df_disp.index)
            row_height = max(1.3, max_lines * 0.85)

            tab = ax_tab.table(
                cellText=cell_text, colLabels=col_labels,
                loc="center", cellLoc="center",
                bbox=[0.0, 0.0, 1.0, 1.0],
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(7)
            tab.scale(1.0, row_height)

            for (r, c), cell in tab.get_celld().items():
                cell.set_linewidth(0.3)
                if r == 0:
                    cell.set_facecolor("#1E3A5F")
                    cell.set_text_props(color="white", weight="bold", fontsize=6.5)
                elif r <= len(df_disp) and str(df_disp.index[r - 1]) == "TOTAL":
                    cell.set_facecolor("#D0E4FF")
                    cell.set_text_props(weight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#F5F5F5")
                # Colonne secteur (c==0) : alignement gauche et wrap activé
                if c == 0 and r > 0:
                    cell.set_text_props(ha="left", fontsize=6.5)

            # Barplot effet allocation (sans TOTAL)
            attr_no_total = attr_df.drop("TOTAL", errors="ignore")
            sectors_plot  = list(attr_no_total.index)
            alloc_vals    = attr_no_total["w_active"].values

            colors_bar = ["#2563EB" if v >= 0 else "#DC2626" for v in alloc_vals]
            ax_bar.barh(sectors_plot, alloc_vals * 100, color=colors_bar,
                        edgecolor="white", height=0.65)
            ax_bar.axvline(0, color="black", linewidth=0.8)
            ax_bar.set_xlabel("Active Weights (%)", fontsize=8)
            ax_bar.set_title("Poids actif", fontsize=9, fontweight="bold")
            ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
            ax_bar.tick_params(axis="y", labelsize=6.5)
            ax_bar.tick_params(axis="x", labelsize=7)
            ax_bar.grid(True, axis="x", alpha=0.3)
            # Masquer les labels y du barplot (déjà lisibles dans le tableau)
            ax_bar.set_yticklabels([])

            self._save(pdf, fig, tight=False)



    @staticmethod
    def _filter_year(  s: Optional[pd.Series],year: int,rebase: bool = True,) -> Optional[pd.Series]:
        """ Utilitaire : filtre + rebase NAV/TE par année """
        
        if s is None or not hasattr(s.index, "year"):
            return None
        
        mask = s.index.year == year
        s_yr = s.loc[mask].dropna()
        
        if len(s_yr) < 2:
            return None
        
        if rebase:
            s_yr = s_yr / s_yr.iloc[0]
            
        return s_yr