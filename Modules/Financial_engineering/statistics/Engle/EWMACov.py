from __future__ import annotations
import sys
from typing import Any, Dict, Tuple, Iterable
import numpy as np
import pandas as pd
import math
import warnings
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import uuid


# Roots & imports projet
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from Modules.Financial_engineering.statistics.multivariate_vol_estimation import MultiVolModel, CovariancePath, _safe_symmetrize, _chol_or_nearest_psd, _as_dt_df
from Modules.Financial_engineering.statistics.Engle.ewma_qmv_numba import tune_lambda_qmv_numba


@dataclass
class EWMARollState:
    """
    Classe contenant l'état intermédiaire de la récurrence EWMA.

    Attributes
    ----------
    path : CovariancePath
        Trajectoire de matrices de covariance en cours de calcul.
    """

    path: "CovariancePath"


class EWMACov(MultiVolModel):
    """
    Classe contenant les méthodes d'estimation de covariance par lissage exponentiel EWMA.

    Attributes
    ----------
    lmbda : float
        Facteur de lissage dans (0, 1). Valeur standard RiskMetrics : 0.94.
    init : str
        Méthode d'initialisation : 'scov' (covariance empirique) ou 'diag' (variances seules).
    window : int
        Longueur de la fenêtre utilisée pour l'initialisation de Σ_0.
    tune_lambda : bool
        Si True, recalibre périodiquement lambda via quasi-maximum de vraisemblance (QMV).
    lambda_grid : list[float] or None
        Grille de valeurs de λ testées lors du recalibrage. Si None, grille par défaut.
    qmv_ridge : float
        Terme de régularisation ridge pour le calcul du score QMV.
    tune_min_obs : int
        Nombre minimum d'observations actives requis pour déclencher le recalibrage.
    tune_every : int
        Fréquence de recalibrage en nombre de pas de temps.
    tune_lookback : int
        Longueur de la fenêtre de données utilisée pour le recalibrage.
    """

    def __init__(self, lmbda: float = 0.94,init: str = "scov", window: int = 252,  tune_lambda: bool = False,
        lambda_grid: Iterable[float] | None = None, qmv_ridge: float = 1e-8, tune_min_obs: int = 30,
        tune_every: int = 121,tune_lookback: int = 756,):

        if not (0.0 < lmbda < 1.0):
            raise ValueError("λ doit être dans (0,1).")

        # Facteur de lissage exponentiel
        self.lmbda = float(lmbda)
        self.init = str(init)
        self._names: Tuple[str, ...] = tuple()

        # Taille de la fenêtre pour l'initialisation de covariance par covariance empirique
        self.window = int(window)

        # Indique si le recalibrage périodique de lambda via QMV est activé
        self.tune_lambda = bool(tune_lambda)

        # Grille de lambda à tester lors du recalibrage
        self.lambda_grid = list(lambda_grid) if lambda_grid is not None else None

        # Ridge pour la stabilité numérique du score QMV
        self.qmv_ridge = float(qmv_ridge)

        # Nombre minimum d'observations actives pour déclencher le recalibrage
        self.tune_min_obs = int(tune_min_obs)

        # Fréquence de recalibrage en nombre de pas de temps
        self.tune_every = int(tune_every)

        # Longueur de la fenêtre historique utilisée pour le recalibrage
        self.tune_lookback = int(tune_lookback)

        # Lambda effectif après recalibrage (None si jamais recalibré)
        self._fitted_lambda: float | None = None

        # Historique optionnel des lambda recalibrés par date (diagnostic)
        self._lambda_by_date: dict[pd.Timestamp, float] = {}

    def _default_lambda_grid(self) -> list[float]:
        """
        Construit la grille de lambda par défaut pour le recalibrage QMV.
        """

        # Grille uniforme entre 0.80 et 0.99 par pas de 0.005
        return [round(x, 3) for x in np.arange(0.80, 0.995, 0.005)]

    def fit(self, R: pd.DataFrame, **_) -> "EWMACov":
        """
        Prépare le modèle sur les rendements R en mémorisant les noms des actifs.

        """

        X = _as_dt_df(R)

        # Mémorisation des noms de colonnes pour les appels ultérieurs
        self._names = tuple(X.columns)

        return self

    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:
        """
        Calcule la trajectoire de covariance EWMA RiskMetrics pure sur l'ensemble des rendements.
        """

        X = _as_dt_df(R)
        T, N = X.shape
        w = self.window
        lmb = self.lmbda
        Xv = X.values.astype(float)

        if T < w + 1:
            raise ValueError(f"Pas assez d'observations : T={T}, window={w}.")

        # Fenêtre d'initialisation : premières w observations
        W0 = Xv[:w]
        W0_clean = np.nan_to_num(W0, nan=0.0)

        if self.init == "diag":
            # Initialisation diagonale : variances empiriques, hors-diagonale à zéro
            var0 = np.var(W0_clean, axis=0, ddof=1)
            S_prev = np.diag(var0)
        else:
            # Initialisation par covariance empirique complète (ddof=1)
            S_prev = _safe_symmetrize(pd.DataFrame(W0_clean).cov(ddof=1).fillna(0.0).to_numpy())

        # Nombre de matrices produites : une par jour t=w..T-1
        n_out = T - w

        # Création du fichier memmap dans le répertoire de cache du projet
        cache_dir = Path(__file__).resolve().parents[3] / "memmap_cache"
        cache_dir.mkdir(exist_ok=True)
        filename = cache_dir / f"ewma_{uuid.uuid4().hex}.dat"

        H = np.memmap(filename, dtype="float64", mode="w+", shape=(n_out, N, N))

        # Enregistrement du chemin memmap pour nettoyage ultérieur
        self._memmap_paths = getattr(self, "_memmap_paths", [])
        self._memmap_paths.append(str(filename))

        out_idx = []

        # Boucle RiskMetrics : une mise à jour par jour de trading
        pbar = tqdm(range(w, T), desc="EWMA RiskMetrics", total=n_out)
        print(f"EWMA: T={T}, N={N}, window={w}, output matrices={n_out}")

        for j, t in enumerate(pbar):

            # Recalibrage périodique de lambda par QMV (optionnel)
            if self.tune_lambda and (j > 0) and (j % self.tune_every == 0):

                # Fenêtre de recalibrage : tune_lookback jours jusqu'à t-1
                start_cal = max(0, t - self.tune_lookback)
                X_cal = X.iloc[start_cal:t]
                X_cal_clean = X_cal.fillna(0.0)

                # Restriction aux colonnes actives pour éviter le bruit des zéros
                active_cols = (X_cal_clean != 0).any(axis=0)
                X_cal_active = X_cal_clean.loc[:, active_cols]

                if len(X_cal_active) >= self.tune_min_obs:
                    lmb = tune_lambda_qmv_numba(self, X_cal_active)
                    self._lambda_by_date[pd.Timestamp(X.index[t])] = float(lmb)

            pbar.set_postfix(date=str(X.index[t].date()), lmb=f"{lmb:.4f}")

            # Rendement du jour t : vecteur (N,), peut contenir des NaN
            r_t = Xv[t]

            # Actifs présents à t : ni NaN ni zéro exact
            active = np.isfinite(r_t) & (r_t != 0.0)

            if active.sum() >= 2:

                # Extraction du sous-vecteur et de la sous-matrice actifs
                r_act = r_t[active]
                outer = np.outer(r_act, r_act)
                S_prev_act = S_prev[np.ix_(active, active)]

                # Mise à jour RiskMetrics sur le bloc actif uniquement
                S_act = lmb * S_prev_act + (1.0 - lmb) * outer
                S_act = _safe_symmetrize(S_act)

                # Reconstruction de la matrice complète
                S_full = lmb * S_prev

                # Remplacement du bloc actif par la mise à jour RiskMetrics
                idx_act = np.where(active)[0]
                S_full[np.ix_(idx_act, idx_act)] = S_act

            else:
                # Aucun actif actif ce jour : décroissance pure sur toute la matrice
                S_full = lmb * S_prev

            S_full = _safe_symmetrize(S_full)

            # Stockage de la matrice du jour t dans le memmap
            H[j] = S_full
            out_idx.append(X.index[t])

            # Mise à jour de Σ_{t-1} pour la prochaine itération
            S_prev = S_full

        H.flush()

        return CovariancePath(H=H, index=pd.DatetimeIndex(out_idx), names=tuple(X.columns),_memmap_path=filename,)

    def conditional_cov_last(self, R: pd.DataFrame) -> CovariancePath:
        """
        Calcule la trajectoire de covariance EWMA alignée sur l'index de R.
        """

        X = _as_dt_df(R)
        T, N = X.shape
        p = X.shape[1]
        w = self.window

        # Nombre de matrices produites
        n = len(X) - w + 1
        lmb = self.lmbda
        Xv = X.values

        if T < w:
            raise ValueError(f"Pas assez d'observations: T={T} <= window={w}.")

        # Initialisation de cov sur la fenêtre initiale [0..w-1]
        W0 = X.iloc[:w]

        if self.init == "diag":
            # Initialisation diagonale : variances empiriques seulement
            S0 = np.diag(np.diag(S0))
        else:
            # Initialisation par covariance empirique complète
            S0 = W0.cov(ddof=1).fillna(0.0).to_numpy()

        # Symétrisation numérique de cov
        S_prev = _safe_symmetrize(S0)

        mats = []
        idx = X.index
        out_idx = []
        lmb = self.lmbda

        # Création du fichier memmap dans le répertoire de cache du projet
        cache_dir = Path(__file__).resolve().parents[3] / "memmap_cache"
        cache_dir.mkdir(exist_ok=True)

        # Nom de fichier unique pour éviter les collisions entre instances parallèles
        filename = cache_dir / f"memmap_{uuid.uuid4().hex}.dat"

        H = np.memmap(filename, dtype="float64", mode="w+", shape=(n, p, p))

        # Enregistrement du chemin memmap pour nettoyage ultérieur
        self._memmap_paths = getattr(self, "_memmap_paths", [])
        self._memmap_paths.append(str(filename))

        # Ré-initialisation de Σ_0 sur la fenêtre initiale (ddof=1)
        S_prev = _safe_symmetrize(X.iloc[:w].cov(ddof=1).fillna(0.0).to_numpy())

        # Boucle EWMA : une mise à jour par jour de trading à partir de t=w+1
        pbar = tqdm(range(w + 1, T), desc="EWMA")
        for j, t in enumerate(pbar, start=1):

            # Recalibrage périodique de lambda par QMV (optionnel)
            if self.tune_lambda and (j % self.tune_every == 0):

                # Fenêtre de recalibrage : tune_lookback jours jusqu'à t-1
                start = max(0, t - self.tune_lookback)
                X_cal = X.iloc[start:t]
                X_tmp = X_cal.fillna(0.0)

                # Restriction aux colonnes actives pour éviter le bruit des zéros
                active_mask = (X_tmp != 0).any(axis=0)
                X_cal = X_cal.loc[:, active_mask]

                lmb = tune_lambda_qmv_numba(self, X_cal)
                self._lambda_by_date[pd.Timestamp(X.index[t])] = float(lmb)

            pbar.set_postfix(date=str(X.index[t].date()), lamb=f"{lmb:.4f}")

            # Fenêtre glissante [t-w..t-1] pour la covariance empirique du choc
            W = X.iloc[t - w : t]
            W = np.nan_to_num(W, nan=0.0)

            # Masque des colonnes actives : au moins une observation non nulle
            active_mask = np.any(W != 0, axis=0)

            # Covariance empirique sur les colonnes actives uniquement
            W_clean = pd.DataFrame(W[:, active_mask])
            S_emp_reduced = W_clean.cov(ddof=1).fillna(0.0).to_numpy()
            S_emp_reduced = _safe_symmetrize(S_emp_reduced)

            # Extraction du bloc actif de Σ_{t-1}
            S_prev_reduced = S_prev[np.ix_(active_mask, active_mask)]

            # Mise à jour EWMA dans l'espace réduit aux actifs actifs
            S_reduced = lmb * S_prev_reduced + (1.0 - lmb) * S_emp_reduced
            S_reduced = _safe_symmetrize(S_reduced)

            # Reconstruction de la matrice complète (actifs inactifs à zéro)
            S_full = np.zeros((X.shape[1], X.shape[1]))
            S_full[np.ix_(active_mask, active_mask)] = S_reduced

            # Stockage de la matrice du jour t dans le memmap
            H[j] = S_full
            out_idx.append(idx[t - 1])

            # Mise à jour de cov_{t-1} pour la prochaine itération
            S_prev = S_full

        H.flush()

        return CovariancePath( H=H,index=pd.DatetimeIndex(out_idx), names=tuple(X.columns), _memmap_path=filename,)

    def update_next_from_path(self, prev_path: "CovariancePath", r_next: pd.Series, dt_next: pd.Timestamp,) -> Tuple["CovariancePath", Dict[str, Any]]:
        """
        Étend la trajectoire d'un pas en calculant cov_t depuis cov_{t-1} et r_t.
        """

        if prev_path.H.ndim != 3:
            raise ValueError("prev_path.H doit être (T,N,N).")

        names = list(prev_path.names)

        # Alignement de r_t sur les actifs connus, NaN remplacés par 0
        x = pd.to_numeric(r_next.reindex(names), errors="coerce").fillna(0.0).values.astype(float)

        # Dernière matrice de covariance connue cov_{t-1}
        S_prev = prev_path.H[-1]

        # Mise à jour RiskMetrics 
        S_new = self.lmbda * S_prev + (1.0 - self.lmbda) * np.outer(x, x)
        S_new = _safe_symmetrize(S_new)

        # Ajout de cov_t à la trajectoire existante
        H_new = np.concatenate([prev_path.H, S_new[None, :, :]], axis=0)
        idx_new = prev_path.index.append(pd.DatetimeIndex([pd.Timestamp(dt_next)]))

        # Construction du path étendu
        out = CovariancePath(H=H_new, index=idx_new, names=prev_path.names)

        # Diagnostics : lambda utilisé pour cette mise à jour
        diag = {"lambda": float(self.lmbda)}

        return out, diag