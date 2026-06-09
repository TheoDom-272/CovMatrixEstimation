"""
Estimateurs de covariance multivariée et classes de base.

Ce fichier contient les briques fondamentales utilisées par tous les estimateurs
de covariance. Il est importé directement par EWMACov, DCC, ledoit_wolf,
covariance_provider, et les fichiers de study.

Classes principales
-------------------
DataFrequency :
    Gère les fréquences de données (daily, weekly) et les constantes associées.
CovariancePath :
    Stocke une trajectoire de matrices de covariance avec support memmap.
MultiVolModel :
    Classe de base abstraite pour tous les estimateurs de covariance.
RollingSampleCov :
    Estimateur de covariance échantillon sur fenêtre glissante.

Singletons
----------
DAILY  : DataFrequency("daily")
WEEKLY : DataFrequency("weekly")

"""

from __future__ import annotations

import gc
import os
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, List

import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from Modules.portfolio_management.backtesting.engine_types import BacktestResult
    from Modules.portfolio_management.backtesting.covariance_provider import CovarianceProviderConfig




def _as_dt_df(R: pd.DataFrame) -> pd.DataFrame:
    """
    Assure un DatetimeIndex trié et un cast numérique permissif sur le DataFrame.

    Copie le DataFrame, tente la conversion de l'index en datetime,
    trie par date, convertit chaque colonne en numérique et supprime
    les lignes et colonnes entièrement NaN.
    """

    # Copie pour ne pas modifier l'original
    X = R.copy()

    # Tente la conversion de l'index en DatetimeIndex
    if not isinstance(X.index, pd.DatetimeIndex):
        try:
            X.index = pd.to_datetime(X.index)
        except Exception:
            pass

    # Trie par ordre chronologique
    X = X.sort_index()

    # Conversion permissive en numérique (NaN si valeur non convertible)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Supprime les lignes et colonnes entièrement NaN
    X = X.dropna(how="all")
    X = X.dropna(axis=1, how="all")
    return X


def _safe_symmetrize(A: np.ndarray) -> np.ndarray:
    """
    Symétrise une matrice carrée via (A + A.T) / 2 pour la stabilité numérique.
    """
    return 0.5 * (A + A.T)


def _chol_or_nearest_psd(S: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """
    Retourne la décomposition de Cholesky de S.

    Si S n'est pas SPD, ajoute un jitter diagonal croissant (jusqu'à 1e-4)
    jusqu'à ce que la décomposition réussisse.
    """
    try:
        # Tente la décomposition directe
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Ajoute un jitter diagonal croissant jusqu'à ce que SPD soit satisfait
        d = S.shape[0]
        k = 0
        while True:
            try:
                return np.linalg.cholesky(_safe_symmetrize(S + (jitter * (10.0 ** k)) * np.eye(d)))
            except np.linalg.LinAlgError:
                k += 1
                if k > 6:
                    raise


@dataclass(frozen=True)
class DataFrequency:
    """
    Classe contenant les constantes dérivées de la fréquence des données.

    Immuable (frozen=True). Singletons DAILY et WEEKLY définis après la classe.

    Attributes
    ----------
    freq : str
        Fréquence des données : 'daily' ou 'weekly'.

    Properties
    ----------
    ann_factor : int
        Facteur d'annualisation : 252 (daily) ou 52 (weekly).
    default_window : int
        Fenêtre de lookback standard : 1 an de données.
    te_rolling_window : int
        Fenêtre rolling pour la TE ex-post : 1 an de données.
    """

    freq: str = "daily"  # "daily" ou "weekly"

    @property
    def ann_factor(self) -> int:
        """Facteur d'annualisation : 252 (daily) ou 52 (weekly)."""
        return 252 if self.freq == "daily" else 52

    @property
    def default_window(self) -> int:
        """Fenêtre de lookback standard : 1 an de données."""
        return self.ann_factor

    @property
    def te_rolling_window(self) -> int:
        """Fenêtre rolling pour TE ex-post : 1 an de données."""
        return self.ann_factor


# Singletons pratiques pour éviter d'instancier DataFrequency à chaque appel
DAILY  = DataFrequency("daily")
WEEKLY = DataFrequency("weekly")


def purge_memmap_cache(cache_dir: str | os.PathLike) -> None:
    """
    Supprime définitivement les fichiers .dat du cache memmap (bypass corbeille Windows).

    Crée le répertoire s'il n'existe pas encore.
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file():
                try:
                    # Suppression directe sans passer par la corbeille
                    f.unlink()
                except Exception as e:
                    warnings.warn(f"[Cleanup] Impossible de supprimer {f}: {e}")
    else:
        # Crée le répertoire s'il est absent pour les prochains appels
        cache_dir.mkdir(parents=True, exist_ok=True)




@dataclass
class CovariancePath:
    """
    Classe contenant une trajectoire de matrices de covariance avec support memmap.

    Stocke un tableau 3D (T, N, N) de matrices de covariance, potentiellement
    sur disque via numpy memmap pour les grandes dimensions. Gère la fermeture
    et la suppression du fichier memmap lors de la libération.

    Attributes
    ----------
    H : np.ndarray
        Tableau (T, N, N) contenant la trajectoire des matrices de covariance.
    index : pd.DatetimeIndex
        Index temporel de longueur T aligné sur la trajectoire.
    names : tuple of str
        Noms des actifs correspondant aux dimensions de la matrice.
    _memmap_path : Path or None
        Chemin vers le fichier memmap sur disque, pour suppression lors de close().

    Methods
    -------
    close() -> None :
        Ferme le handle memmap et supprime le fichier sur disque.
    at(i) -> pd.DataFrame :
        Retourne la matrice de covariance à la position i sous forme de DataFrame.
    stack() -> pd.DataFrame :
        Empile (t, i, j) en DataFrame multi-index (date, row, col).
    diag_series() -> dict :
        Retourne les variances diagonales sous forme de séries par actif.
    """


    H: np.ndarray                          # tableau (T, N, N) de covariances
    index: pd.DatetimeIndex
    names: Tuple[str, ...]
    _memmap_path: Optional[Path] = field(default=None, repr=False)

    def close(self) -> None:
        """Ferme le handle memmap et supprime le fichier sur disque si présent."""

        if isinstance(self.H, np.memmap):
            self.H.flush()
            del self.H
            gc.collect()
        if self._memmap_path is not None and Path(self._memmap_path).exists():
            try:
                Path(self._memmap_path).unlink()
            except Exception as e:
                warnings.warn(f"[CovariancePath.close] Impossible de supprimer {self._memmap_path}: {e}")

    def at(self, i: int) -> pd.DataFrame:
        """
        Retourne la matrice de covariance à la position i sous forme de DataFrame.
        """
        return pd.DataFrame(self.H[i], index=self.names, columns=self.names)

    def stack(self) -> pd.DataFrame:
        """
        Empile la trajectoire (t, i, j) en DataFrame multi-index (date, row, col).
        """

        T, N, _ = self.H.shape

        # Construit le multi-index (date, row, col) pour toutes les combinaisons
        idx = pd.MultiIndex.from_product([self.index, self.names, self.names], names=["date", "row", "col"],)

        return pd.DataFrame(self.H.reshape(T * N * N), index=idx, columns=["value"])

    def diag_series(self) -> Dict[str, pd.Series]:
        """
        Retourne les variances diagonales (sigma^2_t) sous forme de séries par actif.
        """

        out = {}
        for k, name in enumerate(self.names):
            out[name] = pd.Series(self.H[:, k, k], index=self.index, name=name)
        return out


class MultiVolModel:
    """
    Classe de base commune pour tous les estimateurs de matrice de covariance.

    Définit l'interface que chaque estimateur doit implémenter (fit, conditional_cov,
    snapshot_cov, step_cov). Les sous-classes surchargent les méthodes pertinentes.

    Methods
    -------
    fit(R, **kwargs) -> MultiVolModel :
        Ajuste le modèle sur les rendements R.
    conditional_cov(R) -> CovariancePath :
        Retourne la trajectoire de covariance alignée sur l'index de R.
    snapshot_cov(window, prev_state) -> tuple :
        Calcule la covariance sur une fenêtre (snapshot).
    step_cov(r_t, prev_state) -> tuple :
        Mise à jour online à partir du rendement du jour.
    """

    def fit(self, R: pd.DataFrame, **kwargs) -> "MultiVolModel":
        """
        Ajuste le modèle sur les rendements R.

        Doit être surchargée par chaque sous-classe.
        """
        raise NotImplementedError

    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:
        """
        Retourne la trajectoire de covariance alignée sur l'index de R après fit().

        Doit être surchargée par chaque sous-classe.
        """
        raise NotImplementedError

    def snapshot_cov(self, window: pd.DataFrame, prev_state: Any = None,) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
        """
        Calcule la covariance sur une fenêtre (snapshot).

        Implémentation par défaut : calcule le path complet et prend la dernière valeur.
        Fallback lent mais safe si la sous-classe ne surcharge pas cette méthode.

        Parameters
        ----------
        window : pd.DataFrame
            Fenêtre de rendements sur laquelle estimer la covariance.
        prev_state : any
            État précédent du modèle (ignoré dans l'implémentation par défaut).

        Returns
        -------
        tuple
            (Sigma_t, prev_state, dict_info).
        """

        # Calcule le path complet et prend la dernière matrice
        path    = self.fit(window).conditional_cov(window)
        Sigma_t = path.H[-1]
        return Sigma_t, prev_state, {}

    def step_cov(self, r_t: pd.Series, prev_state: Any = None,) -> Tuple[np.ndarray, Any, Dict[str, Any]]:

        """
        Mise à jour online à partir du rendement du jour r_t.
        """
        raise NotImplementedError("This model does not support online step updates.")



class RollingSampleCov(MultiVolModel):
    """
    Classe contenant les méthodes d'estimation de covariance sur fenêtre glissante.

    Calcule cov_t = covariance échantillon sur les 'window' dernières observations.
    Utilise numpy memmap pour stocker la trajectoire sans saturer la RAM.

    Attributes
    ----------
    window : int
        Taille de la fenêtre glissante en nombre de périodes.
    ddof : int
        Degrés de liberté pour le calcul de covariance (1 = non-biaisé).

    Methods
    -------
    fit(R) -> RollingSampleCov :
        Prépare le modèle (noms, index).
    conditional_cov(R) -> CovariancePath :
        Calcule la trajectoire complète via memmap.
    compute_cov_at_rebal(returns_window, all_cols) -> np.ndarray :
        Calcule la covariance à un instant de rebalancement avec repadding.
    """

    def __init__(self, window: int = 60, ddof: int = 1):
        """
        Initialise le modèle Rolling avec la taille de fenêtre et le ddof.
        """

        self.window = int(window)
        self.ddof   = int(ddof)

        # Initialisation des attributs remplis lors du fit
        self._names: Tuple[str, ...] = tuple()
        self._index: Optional[pd.DatetimeIndex] = None

    def fit(self, R: pd.DataFrame, **_) -> "RollingSampleCov":
        """
        Prépare le modèle sur les rendements R : enregistre noms et index.
        """

        X = _as_dt_df(R)
        if X.shape[0] < self.window:
            raise ValueError("Pas assez d'observations pour la fenêtre choisie.")
        
        # Enregistre les noms d'actifs et l'index temporel pour conditional_cov
        self._names = tuple(X.columns)
        self._index = X.index
        return self

    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:
        """
        Calcule la trajectoire de covariance sur fenêtre glissante, stockée via memmap.

        Pour chaque position t (de window à T), calcule la covariance sur les
        window dernières observations. Les actifs inactifs (colonnes nulles) sont
        filtrés puis repadés à zéro dans la matrice complète.
        """

        # Prépare les données et détermine les dimensions
        X   = _as_dt_df(R)
        N   = X.shape[1]
        n_eff = len(X) - self.window + 1

        # Crée le répertoire de cache memmap si absent
        cache_dir = Path(__file__).resolve().parents[2] / "memmap_cache"
        cache_dir.mkdir(exist_ok=True)

        # Chemin unique pour ce memmap
        filename = cache_dir / f"rolling_cov_{uuid.uuid4().hex}.dat"

        # Alloue le memmap (n_eff, N, N) en écriture
        H = np.memmap(filename, dtype="float64", mode="w+", shape=(n_eff, N, N))

        out_idx = []
        i = 0

        # Boucle rolling sur toutes les fenêtres
        for t in tqdm(range(self.window, len(X) + 1), total=n_eff, desc="Rolling covariance"):
            W = X[t - self.window:t]

            # Remplace les NaN par 0 pour le filtrage des colonnes inactives
            W = np.nan_to_num(W, nan=0.0)

            # Garde uniquement les colonnes avec au moins une observation non nulle
            col_keep = np.any(W != 0, axis=0)
            W_clean  = pd.DataFrame(W[:, col_keep])

            # Calcule la covariance échantillon sur les colonnes actives
            S = W_clean.cov(ddof=self.ddof).values
            S = _safe_symmetrize(S)

            # Repadde à la dimension complète N x N avec zéros pour les inactifs
            Sigma = np.zeros((N, N))
            idx_clean = np.where(col_keep)[0]
            Sigma[np.ix_(idx_clean, idx_clean)] = S

            # Écrit la matrice dans le memmap à la position i
            H[i] = Sigma
            out_idx.append(X.index[t - 1])
            i += 1

        # Force l'écriture sur disque
        H.flush()

        return CovariancePath(H=H, index=pd.DatetimeIndex(out_idx), names=tuple(X.columns), _memmap_path=filename,)

    def compute_cov_at_rebal(self, returns_window: pd.DataFrame, all_cols: list[str],) -> np.ndarray:
        """
        Calcule la covariance sur la fenêtre fournie à un instant de rebalancement.

        Retourne une matrice (len(all_cols) x len(all_cols)) avec repadding à zéro
        pour les colonnes inactives.
        """
        
        p_full = len(all_cols)

        # Réindexe sur all_cols et remplace les NaN par 0 (actifs absents)
        X = np.nan_to_num(returns_window.reindex(columns=all_cols).values.copy(), nan=0.0,)

        # Filtre les colonnes actives : au moins une observation non nulle
        col_mask = np.any(X != 0.0, axis=0)
        n_active = int(col_mask.sum())

        # Matrice nulle si pas assez d'actifs actifs
        if n_active < 2:
            return np.zeros((p_full, p_full))

        # Fenêtre nettoyée sur les colonnes actives uniquement
        W_clean = X[:, col_mask]

        # Tronque à self.window observations si la fenêtre est plus longue
        if len(W_clean) > self.window:
            W_clean = W_clean[-self.window:]

        # Covariance échantillon via pandas (gestion des NaN résiduels pairwise)
        S_clean = _safe_symmetrize(pd.DataFrame(W_clean).cov(ddof=self.ddof).values)

        # Repadde à la dimension complète avec zéros pour les inactifs
        Sigma_full = np.zeros((p_full, p_full))
        idx_active = np.where(col_mask)[0]
        Sigma_full[np.ix_(idx_active, idx_active)] = S_clean

        return Sigma_full



from Modules.study.covariance_study.stat_study import (StatSimConfig, StatSimResult, StatEvalResult,)
from Modules.study.covariance_study.eco_study import EcoResult
from Modules.study.covariance_study.pipeline import ModelEvaluator

__all__ = [
    # Utilitaires
    "_as_dt_df", "_safe_symmetrize", "_chol_or_nearest_psd",

    # Classes principales
    "DataFrequency", "DAILY", "WEEKLY", "CovariancePath", "MultiVolModel", "RollingSampleCov", "purge_memmap_cache",

    # Réexports study
    "StatSimConfig", "StatSimResult", "StatEvalResult", "EcoResult", "ModelEvaluator",
]