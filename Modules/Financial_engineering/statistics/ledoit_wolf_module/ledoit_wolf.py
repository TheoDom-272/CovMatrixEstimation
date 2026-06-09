"""
Estimateurs de covariance Ledoit-Wolf : shrinkage linéaire, OAS, ANLS et QIS.
 
Ce fichier implémente quatre variantes de l'estimateur de covariance avec shrinkage
de Ledoit & Wolf, toutes héritant de MultiVolModel et compatibles avec CovarianceProvider.
Chaque estimateur corrige le biais de la covariance échantillon en rétrécissant
les valeurs propres selon une cible ou une formule analytique.
 
Classes
-------
LedoitWolfLinearShrinkage :
    Ledoit & Wolf (2004) - shrinkage linéaire vers la cible sphérique (identité rescalée).
LedoitWolfOAS :
    Oracle Approximating Shrinkage (Chen, Wiesel, Eldar & Hero, 2010) — via sklearn.
    Meilleure approximation de l'oracle dans le cas gaussien que LW2004.
LedoitWolfANLS :
    Ledoit & Wolf (2020) — shrinkage non-linéaire analytique via KDE Epanechnikov.
    Corrige le biais de chaque valeur propre individuellement.
LedoitWolfQIS :
    Ledoit & Wolf (2022) — Quadratic-Inverse Shrinkage via noyau de Cauchy.
    Travaille dans l'espace des précisions avec rescaling pour préserver la trace de S.
 

 
Les méthodes compute_cov_at_rebal() et snapshot_cov() permettent l'intégration
dans le pipeline CovarianceProvider (mode rebal).
"""



from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import tempfile
import os

from sklearn.covariance import LedoitWolf,OAS
import uuid



# Roots & imports projet
ROOT = Path(__file__).resolve().parents[3] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Imports 
from Modules.Financial_engineering.statistics.multivariate_vol_estimation import (MultiVolModel,CovariancePath, _as_dt_df, _safe_symmetrize,)


# Helpers génériques
def _demean(X: np.ndarray) -> np.ndarray:
    """Centre chaque colonne."""
    return X - np.mean(X, axis=0, keepdims=True)


def _sample_cov(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Covariance échantillon : (1/(n-ddof)) X^T X, X supposé déjà centré si besoin."""
    n = X.shape[0]
    denom = max(n - int(ddof), 1)
    S = X.T @ X
    S *= 1.0 / denom     # in-place, évite allocation
    return S

def _sample_cov_pairwise(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Covariance pairwise via pandas — ignore les zéros comme des NaN."""
    X_masked = X.copy().astype(float)
    X_masked[X_masked == 0.0] = np.nan
    return pd.DataFrame(X_masked).cov(ddof=ddof).fillna(0.0).to_numpy()


# Ledoit & Wolf (2004) — Linear Shrinkage
class LedoitWolfLinearShrinkage(MultiVolModel):
    """
    Classe contenant les méthodes d'estimation de covariance par shrinkage linéaire
    selon Ledoit & Wolf (2004).
 
    Rétrécit la covariance échantillon S vers la cible sphériquet optimal rho, calculé analytiquement sans
    validation croisée. Fonctionne aussi via sklearn (use_package=True).
    Formule : Sigma = rho * T + (1 - rho) * S.
 
    Attributes
    ----------
    window : int or None
        Taille de la fenêtre glissante. Si None, estimation statique sur tout l'historique.
    demean : bool
        Si True, centre les rendements avant estimation.
    ddof : int
        Degrés de liberté pour la covariance échantillon (0 = MLE, 1 = non-biaisé).
    eps : float
        Plancher numérique pour éviter les divisions par zéro.
    use_package : bool
        Si True, délègue à sklearn.covariance.LedoitWolf au lieu du calcul from-scratch.
 
    Methods
    -------
    fit(R) -> LedoitWolfLinearShrinkage :
        Ajuste le modèle sur les rendements R.
    conditional_cov(R) -> CovariancePath :
        Calcule la trajectoire de covariance (statique ou rolling via memmap).
    compute_cov_at_rebal(returns_window, all_cols) -> np.ndarray :
        Calcule la covariance à un instant de rebalancement avec repadding.
    snapshot_cov(window, prev_state) -> tuple :
        Estime la covariance sur une fenêtre unique (interface CovarianceProvider).
    """
    

    def __init__(self,window: Optional[int] = None,demean: bool = True,ddof: int = 0,eps: float = 1e-12,use_package: bool = False):
        self.window = None if window is None else int(window)
        self.demean = bool(demean)
        self.ddof = int(ddof)
        self.eps = float(eps)

        self._names: Tuple[str, ...] = tuple()
        self._fit_info: Dict[str, Any] = {}
        self._Sigma_full: Optional[np.ndarray] = None
        
        self.use_package = bool(use_package)
    
    @staticmethod
    def _b2_fast(X, S):
        """
        Calcule b^2 la variance d'échantillonnage de S : représente l'erreur quadratique moyenne
            entre les matrices (issues des observations) et leur moyenne Sx.
        """
        n, p = X.shape
        XX = np.sum(X * X, axis=1)     # Somme sur les colonne du carré des éléments : variance non normalisé de chaque actif
        q1 = np.sum(XX * XX)           # 

        XS = X @ S                     # GEMM
        q2 = np.sum(np.sum(XS * X, axis=1))  # O(n)

        normS2 = np.sum(S * S)         # O(p^2)

        b2 = (q1 + n * normS2 - 2*q2) / (n*n)
        return max(b2, 0.0)


    def _estimate_once(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Estime Sigma sur une matrice de rendements X (n x p)."""

        #Si souhaité, centre les données avant estimation
        if self.demean: 
            X = _demean(X)

        n, p = X.shape
        if n < 2:
            raise ValueError("Pas assez d'observations pour estimer une covariance.")
        
        
        if self.use_package:
                lw = LedoitWolf().fit(X)
                Sigma = lw.covariance_
                info = {"rho": lw.shrinkage_}
                return _safe_symmetrize(Sigma), info
        else:

            #Estimation de la covariance échantillon 
            S = _safe_symmetrize(_sample_cov(X, ddof=self.ddof))

            #Calcul de s_bar = (1/p) tr(S), 
            s_bar = float(np.trace(S) / p) 

            #Cible de shrinkage : matrice sphérique
            T = s_bar * np.eye(p) 

            # Calcul de d^2 (distance au sens de frobenius entre S et T) : Dispersion total des valeur propres autour de leur moyenne
            d2 = float(np.sum((S - T) ** 2)) 

            # Estimation de b^2 (dispersion des x xᵀ autour de S) Variance d'échantillonnage de S
            b2 = self._b2_fast(X, S) 

            # Si d2 proche de 0, S est déjà sphérique -> l'estimateur est identique quelle que soit rho.
            if d2 <= self.eps:
                rho = 1.0
            else:

                # Calcul de rho = b^2 / d^2, borné entre 0 et 1
                rho = float(np.clip(b2 / d2, 0.0, 1.0)) 

            # Shrinkage linéaire : convex combination de S et T
            Sigma = _safe_symmetrize(rho * T + (1.0 - rho) * S) 

            # Diagnostics pour analyse post-fit
            info = {"rho": rho, "s_bar": s_bar, "d2": d2, "b2": b2}

            return Sigma, info

    @property
    def last_fit_info(self) -> Dict[str, Any]:
        """Derniers diagnostics calculés (rho, d2, b2, etc.)."""
        return dict(self._fit_info)


    def fit(self, R: pd.DataFrame, **_) -> "LedoitWolfLinearShrinkage":
        """
        Ajuste le modèle sur les rendements R.
        """

        Xdf = _as_dt_df(R)
        self._names = tuple(Xdf.columns)

        # Pré-calcul si non-rolling
        if self.window is None:
            Sigma, info = self._estimate_once(Xdf.values)
            self._Sigma_full = Sigma
            self._fit_info = info
        else:
            if len(Xdf) < self.window:
                raise ValueError("Pas assez d'observations pour la fenêtre choisie.")
            self._Sigma_full = None
            self._fit_info = {}

        return self

    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:
        """
        Calcule la trajectoire de covariance sur fenêtre glissante ou statique.
        """

        Xdf = _as_dt_df(R)
        #X = Xdf.values
        X = np.nan_to_num(Xdf.values.copy(), nan=0.0)
        idx = Xdf.index
        p = X.shape[1]

        # Cas constant
        if self.window is None:
            if self._Sigma_full is None:
                Sigma, info = self._estimate_once(X)
                self._Sigma_full = Sigma
                self._fit_info = info
            H = np.repeat(self._Sigma_full[None, :, :], repeats=len(Xdf), axis=0)
            return CovariancePath(H=H, index=idx, names=tuple(Xdf.columns))

        # Cas rolling
        w = int(self.window)
        n = len(Xdf) - w + 1     # nombre total de matrices
        mats = []
        out_idx = []
        self._fit_info = {}  # reset (info rolling dépend de t)

        # Chemin contrôlé dans le projet
        cache_dir = Path(__file__).resolve().parents[3] / "memmap_cache"
        cache_dir.mkdir(exist_ok=True)

        # Crée un nom unique
        filename = cache_dir / f"memmap_{uuid.uuid4().hex}.dat"

        H = np.memmap(filename, dtype="float64", mode="w+", shape=(n, p, p))

        # On stocke le chemin pour nettoyage futur
        self._memmap_paths = getattr(self, "_memmap_paths", [])
        self._memmap_paths.append(str(filename))


        for i,t in enumerate(tqdm(range(w, len(Xdf) + 1), total=n, desc="Linear Shrinkage cov rolling")):
            W = X[t - w : t]
            
            W = np.nan_to_num(W, nan=0.0)
            col_keep = np.any(W != 0, axis=0)
            W_clean = W[:, col_keep]
            row_keep = np.any(W_clean != 0, axis=1)
            W_clean = W_clean[row_keep, :]

            Sigma_clean, info_clean = self._estimate_once(W_clean)
            Sigma2 =np.zeros((p,p))
            idx_clean= np.where(col_keep)[0]
            Sigma2[np.ix_(idx_clean,idx_clean)]= Sigma_clean
            #Sigma, info = self._estimate_once(W)
            self._fit_info = info_clean
            
            #mats.append(Sigma)
            H[i] = Sigma2 
            out_idx.append(idx[t - 1])

        return CovariancePath(H=H, index=pd.DatetimeIndex(out_idx), names=tuple(Xdf.columns),_memmap_path=filename)
    
    def compute_cov_at_rebal(self,returns_window: pd.DataFrame,all_cols: list[str],) -> np.ndarray:
        """
        Calcule la covariance à un instant de rebal sur la fenêtre fournie.
        """

        p_full = len(all_cols)
        X = np.nan_to_num(returns_window.reindex(columns=all_cols).values.copy(), nan=0.0)

        # Filtrage colonnes actives 
        col_mask = np.any(X != 0.0, axis=0)
        n_active  = col_mask.sum()

        if n_active < 3:
            return np.zeros((p_full, p_full))

        W_clean = X[:, col_mask]

        # Estimation sur la fenêtre nettoyée
        Sigma_clean, _ = self._estimate_once(W_clean)

        # Repadding à la taille de l'univers complet
        Sigma_full = np.zeros((p_full, p_full))
        idx_active = np.where(col_mask)[0]
        Sigma_full[np.ix_(idx_active, idx_active)] = Sigma_clean

        return Sigma_full

    
    def snapshot_cov(self,window: pd.DataFrame,prev_state=None) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        """ Estime la covariance à un instant donné sur une fenêtre de rendements."""

        #Recupère les données sur la fenêtre
        Xdf = _as_dt_df(window).dropna() #A corriger 

        # Vérifications
        if self.window is not None and len(Xdf) < int(self.window):
            raise ValueError("Pas assez d'observations pour la fenêtre choisie.")
        
        #Estimation
        X = Xdf.values if self.window is None else Xdf.values[-int(self.window):]
        Sigma, info = self._estimate_once(X)

        return Sigma, None, info


    @property
    def fit_info(self) -> Dict[str, Any]:
        """Informations du dernier fit (rho, etc.)."""
        return dict(self._fit_info)



class LedoitWolfOAS(MultiVolModel):
    """
    Classe contenant les méthodes d'estimation de covariance par Oracle Approximating
    Shrinkage selon Chen, Wiesel, Eldar & Hero (2010), via scikit-learn.
 
    Attributes
    ----------
    window : int or None
        Taille de la fenêtre glissante. Si None, estimation statique.
    demean : bool
        Si True, centre les rendements avant estimation.
    ddof : int
        Degrés de liberté (ignoré par OAS sklearn, conservé pour cohérence d'interface).
    use_package : bool
        Doit être True (OAS n'existe pas en version from-scratch).
 
    Methods
    -------
    fit(R) -> LedoitWolfOAS :
        Ajuste le modèle sur les rendements R.
    conditional_cov(R) -> CovariancePath :
        Calcule la trajectoire de covariance (statique ou rolling via memmap).
    compute_cov_at_rebal(returns_window, all_cols) -> np.ndarray :
        Calcule la covariance à un instant de rebalancement avec repadding.
    snapshot_cov(window, prev_state) -> tuple :
        Estime la covariance sur une fenêtre unique.
    """

    def __init__(self,window: Optional[int] = None,demean: bool = True, ddof: int = 0,use_package: bool = True):

        self.window = None if window is None else int(window)
        self.demean = bool(demean)
        self.ddof = int(ddof)
        self.use_package = bool(use_package)

        self._names: Tuple[str, ...] = tuple()
        self._Sigma_full: Optional[np.ndarray] = None
        self._fit_info: Dict[str, Any] = {}


    # Estimation pour une fenêtre X (n×p)
    def _estimate_once(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.demean:
            X = _demean(X)

        if not self.use_package:
            raise NotImplementedError("OAS n'existe pas en version 'from scratch'. " "Utilisez use_package=True.")

        oas = OAS().fit(X)
        Sigma = _safe_symmetrize(oas.covariance_)
        info = {"rho": float(oas.shrinkage_)}
        return Sigma, info


    def fit(self, R: pd.DataFrame, **_) -> "LedoitWolfOAS": 
        Xdf = _as_dt_df(R)
        self._names = tuple(Xdf.columns)

        if self.window is None:
            Sigma, info = self._estimate_once(Xdf.values)
            self._Sigma_full = Sigma
            self._fit_info = info
        else:
            if len(Xdf) < self.window:
                raise ValueError("Pas assez d'observations pour la fenêtre choisie.")
            self._Sigma_full = None
            self._fit_info = {}

        return self


    
    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:
        """Rolling ou constant covariance, retournée sous forme CovariancePath"""
        
        Xdf = _as_dt_df(R)
        X = Xdf.values
        idx = Xdf.index
        p = X.shape[1]

        # Cas constant
        if self.window is None:
            if self._Sigma_full is None:
                Sigma, info = self._estimate_once(X)
                self._Sigma_full = Sigma
                self._fit_info = info

            H = np.repeat(self._Sigma_full[None, :, :], repeats=len(Xdf), axis=0)
            return CovariancePath(H=H, index=idx, names=self._names)

        # Cas rolling
        w = int(self.window)
        n = len(Xdf) - w + 1
        out_idx = []

        # Chemin contrôlé dans le projet
        cache_dir = Path(__file__).resolve().parents[3] / "memmap_cache"
        cache_dir.mkdir(exist_ok=True)

        # Crée un nom unique
        filename = cache_dir / f"memmap_{uuid.uuid4().hex}.dat"

        H = np.memmap(filename, dtype="float64", mode="w+", shape=(n, p, p))

        # On stocke le chemin pour nettoyage futur
        self._memmap_paths = getattr(self, "_memmap_paths", [])
        self._memmap_paths.append(str(filename))

        for i, t in enumerate(tqdm(range(w, len(Xdf) + 1),total=n,desc="OAS cov rolling")):                                       
            W = X[t - w : t]
            W = np.nan_to_num(W, nan=0.0)
            col_mask = np.any(W != 0.0, axis=0)
            n_active = col_mask.sum()

            if n_active < 2:
                H[i] = np.zeros((p, p))
                out_idx.append(idx[t - 1])
                continue

            W_clean = W[:, col_mask]

        

            #Sigma, info = self._estimate_once(W)
            Sigma_clean, info = self._estimate_once(W_clean)
            self._fit_info = info
            
            # Repadding à taille p (zéros pour les actifs inexistants)
            Sigma_full = np.zeros((p, p))
            idx_active = np.where(col_mask)[0]
            Sigma_full[np.ix_(idx_active, idx_active)] = Sigma_clean

            
            H[i] = Sigma_full
            out_idx.append(idx[t - 1])

        H.flush()
        return CovariancePath(H=H, index=pd.DatetimeIndex(out_idx), names=self._names,_memmap_path=filename)
    

    def compute_cov_at_rebal(self,returns_window: pd.DataFrame,all_cols: list[str],) -> np.ndarray:
        """
        Calcule la covariance à un instant de rebal sur la fenêtre fournie.
        Retourne une matrice (len(all_cols) x len(all_cols)) avec repadding
        pour les colonnes inactives (identique au rolling).
        """
        p_full = len(all_cols)
        X = np.nan_to_num(returns_window.reindex(columns=all_cols).values.copy(), nan=0.0)

        # Filtrage colonnes actives 
        col_mask = np.any(X != 0.0, axis=0)
        n_active  = col_mask.sum()

        if n_active < 3:
            return np.zeros((p_full, p_full))

        W_clean = X[:, col_mask]

        # Estimation sur la fenêtre nettoyée
        Sigma_clean, _ = self._estimate_once(W_clean)

        # Repadding à la taille de l'univers complet
        Sigma_full = np.zeros((p_full, p_full))
        idx_active = np.where(col_mask)[0]
        Sigma_full[np.ix_(idx_active, idx_active)] = Sigma_clean

        return Sigma_full


    def snapshot_cov(self,window: pd.DataFrame,prev_state=None) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        """Snapshot pour une seule fenêtre"""
        
        Xdf = _as_dt_df(window).dropna()

        if self.window is not None and len(Xdf) < self.window:
            raise ValueError("Pas assez d'observations pour la fenêtre choisie.")

        X = Xdf.values if self.window is None else Xdf.values[-self.window:]
        Sigma, info = self._estimate_once(X)
        return Sigma, None, info

    @property
    def last_fit_info(self) -> Dict[str, Any]:
        return dict(self._fit_info)







class LedoitWolfANLS(MultiVolModel):
    """
    Classe contenant les méthodes d'estimation de covariance par shrinkage non-linéaire
    analytique selon Ledoit & Wolf (2020).

 
    Attributes
    ----------
    window : int or None
        Taille de la fenêtre glissante. Si None, estimation statique.
    demean : bool
        Si True, centre les rendements avant estimation.
    ddof : int
        Degrés de liberté pour la covariance pairwise.
    chunk_size : int
        Taille des blocs pour le calcul de la KDE (limite la mémoire sur grand p).
    eps : float
        Plancher numérique pour les valeurs propres et les dénominateurs.
 
    Methods
    -------
    fit(R) -> LedoitWolfANLS :
        Ajuste le modèle sur les rendements R.
    conditional_cov(R) -> CovariancePath :
        Calcule la trajectoire de covariance (statique ou rolling via memmap).
    compute_cov_at_rebal(returns_window, all_cols) -> np.ndarray :
        Calcule la covariance à un instant de rebalancement avec repadding.
    snapshot_cov(window, prev_state) -> tuple :
        Estime la covariance sur une fenêtre unique.
    """
    

    def __init__(self,window: Optional[int] = None,demean: bool = True,ddof: int = 0, chunk_size: int = 1024,eps: float = 1e-12,use_package=False):
        
        self.window = None if window is None else int(window)
        self.demean = bool(demean)
        self.ddof = int(ddof)
        self.chunk_size = int(chunk_size)
        self.eps = float(eps)

        self._names: Tuple[str, ...] = tuple()
        self._Sigma_full: Optional[np.ndarray] = None
        self._fit_info: Dict[str, Any] = {}
        
        self.use_package = bool(use_package)



    @staticmethod
    def _k_epanechnikov(u: np.ndarray) -> np.ndarray:
        """Noyau Epanechnikov"""
        sqrt5 = np.sqrt(5)
        out = np.zeros_like(u, dtype=float)
        m = np.abs(u) <= sqrt5
        out[m] = (3.0/ ( 4.0 * sqrt5 )) * (1.0 - (u[m] ** 2) / 5.0)
        return out
    
    @staticmethod
    def _Hk_epanechnikov(u: np.ndarray) -> np.ndarray:
        """Transformée de Hilbert du noyau d'Epanechnikov."""
        sqrt5 = np.sqrt(5)
        inside = np.abs(u) < sqrt5
        num = np.maximum(np.abs(sqrt5 - u), 1e-300)
        den = np.maximum(np.abs(sqrt5 + u), 1e-300)
        result = -3.0*  u / (10.0 * np.pi)
        result[inside] += (3.0/ (4.0*sqrt5*np.pi)) * (1.0 - (u[inside]**2)/5.0) * np.log(num[inside]/den[inside])
        return result


    def _estimate_once(self, X: np.ndarray,n_nominal: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Estime ANLS sur une fenêtre X (n x p)."""
        
        #Si souhaité, centre les données avant estimation
        if self.demean:
            X = _demean(X)

        # recupère les dimensions et vérifie qu'il y a assez d'observations pour estimer une covariance
        n, p = X.shape
        if n < 2:
            raise ValueError("Pas assez d'observations pour estimer une covariance.")

        # Estimation de la covariance échantillon sur la fenêtre 
        S = _safe_symmetrize(_sample_cov_pairwise(X, ddof=self.ddof))

        # Décomposition spectrale 
        lam_raw, U = np.linalg.eigh(S)   # valeurs propres triées croissant, U orthogonale

        #Ratio n/p représente la complexité du problème d'estimation de la covariance
        n_for_c = n_nominal if n_nominal is not None else n

        # Ratio de concentration
        c = float(p / n_for_c)

        # Fenêtre de lissage du noyau pour l'estimation de la densité des valeurs propres 
        h = float(n ** (-1.0 / 3.0))

        # Si p>n split pour les valeurs propres nulles
        if c > 1.0:
            
            # calcule un seuil pour identifier les valeurs propres nulles, basé sur la plus grande valeur propre empirique et un plancher numérique
            threshold = self.eps * max(float(lam_raw.max()), 1.0)

            # masque pour les valeurs propres non nulles
            mask_nz = lam_raw > threshold # les ~n valeurs propres positives

            # recupère les valeurs propres non nulles
            lam = np.maximum(lam_raw[ mask_nz], self.eps)

            # recupère les vecteurs propres associés aux valeurs propres non nulles et nulles
            U_nz = U[:,  mask_nz]

            # vecteurs propres associés aux valeurs propres nulles (directions de variance nulle, bruit d'estimation maximal)
            U_null  = U[:, ~mask_nz]

        # Sinon, on garde toutes les valeurs propres et vecteurs propres
        else:
            # recupére les valeurs propres, avec un plancher numérique pour éviter les problèmes de zéro ou de négatif
            lam  = np.maximum(lam_raw, self.eps)
            U_nz = U

        # nombre de valeurs propres qui entrent dans la KDE
        p_kde = len(lam)   

        # Fenetre de lissage
        ph = float(p_kde) * h

        #----------------------------------------------------------------------------------
        # Estimation pour chaque valeur propre empirique lam_i la densité locale et son asymétrie local permettant de detemriner 
        # à quel point il faut shrinker lam_i pour corriger le biais d'échantillonnage de la covariance échantillon S, 
        # en particulier dans les directions de faible variance (petites valeurs propres) où le bruit d'estimation est plus fort.
        #----------------------------------------------------------------------------------

        # Estimation de la densité de probabilité des distributions des valeurs propres (f_n) et de sa transformée de Hilbert (Hf_n) par KDE ---

        # Initialisation des tableaux de résultats
        f_vals = np.zeros(p_kde, dtype=float) #f_vals = np.empty(p, dtype=float)
        H_vals = np.zeros(p_kde, dtype=float) #H_vals = np.empty(p, dtype=float)

        # Traitement par blocs pour limiter la mémoire utilisée (utile si p est grand)
        cs = max(self.chunk_size, 1)

        #Itération par blocs sur les valeurs propres pour calculer f_n et Hf_n
        for i0 in range(0, p_kde, cs):

            # borne supérieure du bloc (exclus)
            i1 = min(p_kde, i0 + cs)
            lam_i = lam[i0:i1]  # shape (b,)
            
            # Argument du noyau : distance entre chaques valeurs propres, normalisée par h
            u = (lam_i[:, None] - lam[None, :]) / h
            
            # Évaluation du noyau et de sa transformée de Hilbert sur la grille u
            k_u  = self._k_epanechnikov(u)        # Determine si la valeur propre  est isolée ou entourée d'autres valeurs proches.
            Hk_u = self._Hk_epanechnikov(u)       # Mesure le déséquilibre entre les valeurs propres a gauche et à droite de lam_i.
            
            # Somme sur j, puis normalisation par (ph) :Mesure de l'asymetrie locale de la distribution des valeurs propres autour de lam_i
            f_vals[i0:i1] = k_u.sum(axis=1)  / ph
            H_vals[i0:i1] = Hk_u.sum(axis=1) / ph
            
            
        # Coefficients de shrinkage
        pi_c_lam_f = np.pi * c * lam * f_vals # partie imaginaire de la fonction de Stieltjes associée à la distribution des valeurs propres.
        pi_c_lam_H = np.pi * c * lam * H_vals # Contribution de la partie réelle de la fonction de Stieltjes, qui capture l'asymétrie locale de la distribution des valeurs propres autour de lam_i.

        # Plancher numérique pour éviter les divisions par zéro ou les shrinkages extrêmes
        denom = pi_c_lam_f ** 2 + (1.0 - c - pi_c_lam_H) ** 2
        denom = np.maximum(denom, self.eps)   # plancher numérique

        # shrinkage non linéaire de chaque valeur propre empirique lam_i, ajusté en fonction de la densité locale et de l'asymétrie de la distribution des valeurs propres autour de lam_i. 
        d = lam / denom 
        d = np.maximum(d, self.eps)

        #Si p>n, il y a des valeurs propres nulles à traiter : on leur attribue une valeur de shrinkage basée sur la moyenne harmonique des d_i pour les valeurs propres non nulles, ajustée par le ratio de concentration c.
        if c > 1.0:

            # valeurs propres nulles 
            lam_harm  = float(p_kde) / float(np.sum(1.0 / lam))
            d_null  = max(lam_harm / (c * (c - 1.0)), self.eps)
            d_null_arr = np.full(int((~mask_nz).sum()), d_null)

            #Reconstruction de la matrice de covariance corrigée 
            Sigma = _safe_symmetrize((U_nz   * d[None, :]) @ U_nz.T +(U_null * d_null_arr[None, :]) @ U_null.T)
        else:
            Sigma = _safe_symmetrize((U_nz * d[None, :]) @ U_nz.T) 

        # Diagnostics
        trace_S = float(np.trace(S))
        trace_Sigma = float(np.trace(Sigma))
        info: Dict[str, Any] = {
            "c": c,
            "h_n": h,
            "n": n,
            "p": p,
            "lambda_min": float(lam.min()),
            "lambda_max": float(lam.max()),
            "f_min": float(f_vals.min()),
            "f_max": float(f_vals.max()),
            "d_min": float(d.min()),
            "d_max": float(d.max()),
            "trace_ratio": trace_Sigma / trace_S if trace_S > 0 else float("nan"),
        }
        return Sigma, info


    # Interface MultiVolModel
    @property
    def last_fit_info(self) -> Dict[str, Any]:
        """Derniers diagnostics calculés (c, h_n, f_min, f_max, d_min, d_max, trace_ratio…)."""
        return dict(self._fit_info)

    def fit(self, R: pd.DataFrame, **_) -> "LedoitWolfANLS":
        Xdf = _as_dt_df(R)
        self._names = tuple(Xdf.columns)

        if self.window is None:
            Sigma, info = self._estimate_once(Xdf.values)
            self._Sigma_full = Sigma
            self._fit_info = info
        else:
            if len(Xdf) < int(self.window):
                raise ValueError("Pas assez d'observations pour la fenêtre choisie.")
            self._Sigma_full = None
            self._fit_info = {}

        return self

    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:

        # Preparation des données
        Xdf = _as_dt_df(R)
        X = np.nan_to_num(Xdf.values.copy(), nan=0.0)
        idx = Xdf.index
        T, p = X.shape
        

        # Cas constant (window=None) : estimation sur tout l'historique
        if self.window is None:
            if self._Sigma_full is None:
                Sigma, info = self._estimate_once(X)
                self._Sigma_full = Sigma
                self._fit_info = info
            H = np.repeat(self._Sigma_full[None, :, :], repeats=len(Xdf), axis=0)
            return CovariancePath(H=H, index=idx, names=tuple(Xdf.columns),_memmap_path=None)

        # Cas rolling - ANLS 
        w = int(self.window)
        mats = []
        out_idx = []
        n = len(Xdf) - w + 1  # nombre total de matrices
        
        
        # Dossier de cache contrôlé
        cache_dir = Path(__file__).resolve().parents[3] / "memmap_cache"
        cache_dir.mkdir(exist_ok=True)

        # Nom de fichier unique
        filename = cache_dir / f"memmap_anls_{uuid.uuid4().hex}.dat"

        # Allocation memmap sur disque
        H = np.memmap(filename, dtype="float64", mode="w+", shape=(n, p, p))

        # Stocker le chemin pour nettoyage ultérieur
        self._memmap_paths = getattr(self, "_memmap_paths", [])
        self._memmap_paths.append(str(filename))


        #Itération sur les fenêtres glissantes de la matrice de rendements X 
        for i, t in enumerate(tqdm(range(w, len(Xdf) + 1), total=len(Xdf) - w + 1, desc="ANLS cov rolling")):
            W = X[t - w : t]

            # Filtrage des colonnes actives, on considère qu'une colonne est inactive si tous les rendements de la fenêtre sont nuls
            W = np.nan_to_num(W, nan=0.0)
            col_mask = np.any(W != 0.0, axis=0)
            n_active = col_mask.sum()

            if n_active < 2:
                H[i] = np.zeros((p, p))
                out_idx.append(idx[t - 1])
                continue
            
            # Extraction de la sous-matrice W_clean contenant uniquement les colonnes actives pour l'estimation
            W_clean = W[:, col_mask]

            #Estimation de la covariance sur la fenêtre nettoyée (W_clean)
            Sigma_clean, info = self._estimate_once(W_clean, n_nominal=w)
            self._fit_info = info   # on garde le dernier pour inspection

            # Repadding à taille p (zéros pour les actifs inexistants)
            Sigma_full = np.zeros((p, p))
            idx_active = np.where(col_mask)[0]
            Sigma_full[np.ix_(idx_active, idx_active)] = Sigma_clean
            
            H[i] = Sigma_full
            out_idx.append(idx[t - 1])

        H.flush()
        return CovariancePath(H=H, index=pd.DatetimeIndex(out_idx), names=tuple(Xdf.columns), _memmap_path=str(filename))
    
    def compute_cov_at_rebal(self,returns_window: pd.DataFrame,all_cols: list[str],) -> np.ndarray:
        """
        Calcule la covariance à un instant de rebal sur la fenêtre fournie.
        Retourne une matrice (len(all_cols) x len(all_cols)) avec repadding
        pour les colonnes inactives (identique au rolling).
        """
        p_full = len(all_cols)
        X = np.nan_to_num(returns_window.reindex(columns=all_cols).values.copy(), nan=0.0)

        # Filtrage colonnes actives 
        col_mask = np.any(X != 0.0, axis=0)
        n_active  = col_mask.sum()

        if n_active < 3:
            return np.zeros((p_full, p_full))

        W_clean = X[:, col_mask]

        # Estimation sur la fenêtre nettoyée
        Sigma_clean, _ = self._estimate_once(W_clean, n_nominal=len(returns_window))

        # Repadding à la taille de l'univers complet
        Sigma_full = np.zeros((p_full, p_full))
        idx_active = np.where(col_mask)[0]
        Sigma_full[np.ix_(idx_active, idx_active)] = Sigma_clean

        return Sigma_full

    def snapshot_cov(self, window: pd.DataFrame, prev_state=None) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        """Estime la covariance à un instant donné sur une fenêtre de rendements."""
        Xdf = _as_dt_df(window)

        X_raw = np.nan_to_num(Xdf.values.copy(), nan=0.0)   # ← nan_to_num, pas dropna
        X = X_raw if self.window is None else X_raw[-int(self.window):]
        Sigma, info = self._estimate_once(X)
        return Sigma, None, info

    @property
    def fit_info(self) -> Dict[str, Any]:
        return dict(self._fit_info)





# Ledoit & Wolf (2022) - QIS (Quadratic-Inverse Shrinkage)
class LedoitWolfQIS(MultiVolModel):
    """
    Classe contenant les méthodes d'estimation de covariance par Quadratic-Inverse
    Shrinkage selon Ledoit & Wolf (2022).
 
    Attributes
    ----------
    window : int or None
        Taille de la fenêtre glissante. Si None, estimation statique.
    demean : bool
        Si True, centre les rendements avant estimation.
    chunk_size : int
        Conservé pour cohérence d'interface avec ANLS.
    eps : float
        Plancher numérique pour les valeurs propres et les dénominateurs.
 
    Methods
    -------
    fit(R) -> LedoitWolfQIS :
        Ajuste le modèle sur les rendements R.
    conditional_cov(R) -> CovariancePath :
        Calcule la trajectoire de covariance (statique ou rolling via memmap).
    compute_cov_at_rebal(returns_window, all_cols) -> np.ndarray :
        Calcule la covariance à un instant de rebalancement avec repadding.
    snapshot_cov(window, prev_state) -> tuple :
        Estime la covariance sur une fenêtre unique.
    """

    def __init__(self, window: Optional[int] = None, demean: bool = True, chunk_size: int = 1024, eps: float = 1e-10,):
        self.window     = None if window is None else int(window)
        self.demean     = bool(demean)
        self.chunk_size = int(chunk_size)
        self.eps        = float(eps)

        self._names:      Tuple[str, ...]      = tuple()
        self._Sigma_full: Optional[np.ndarray] = None
        self._fit_info:   Dict[str, Any]       = {}


    def _estimate_once(self, X:np.ndarray, n_nominal: Optional[int] = None,) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Estime QIS sur une fenêtre X (N x p).
        """

        # Centrage des données 
        # Après demeaning, la taille effective est réduite de 1 (k=1).
        if self.demean:
            X = _demean(X)

        # Dimensions et vérification du nombre d'observations
        N, p = X.shape
        if N < 3:
            raise ValueError("Pas assez d'observations pour QIS (N < 3).")

        # Taille effective n et ratio de concentration c 
        # n_nominal permet de surcharger la taille de fenêtre (cas rolling avec colonnes inactives filtrées : on préserve le vrai ratio c = p_active/w).
        n = (n_nominal - 1) if n_nominal is not None else (N - 1)
        c = float(p) / float(n)

        # Covariance échantillon et décomposition spectrale 
        S = _safe_symmetrize(_sample_cov_pairwise(X, ddof=1))
        lam_raw, U = np.linalg.eigh(S)
        lam_raw = np.real(lam_raw)
        U = np.real(U)

        # Bandwidth du noyau de Cauchy 
        # - Le facteur min(c**2, 1.0 / c**2) adapte h à la sévérité de la malédiction dimensionnelle : il est maximal (=1) quand c=1, et décroît symétriquement quand c s'éloigne de 1 vers 0 ou l'infini.
        # - Le facteur p^{-0.35} assure la convergence asymptotique.
        h = (min(c**2, 1.0 / c**2) ** 0.35) / (p ** 0.35)

        # Sélection des valeurs propres non-nulles 
        # Quand p > n, S est singulière et a exactement p-n valeurs propres nulles. Les min(p,n) plus grandes valeurs propres sont non-nulles.
        idx_start = max(0, p - n)
        invlambda = 1.0 / np.maximum(lam_raw[idx_start:], self.eps)  # shape (n_nz,)
        n_nz = len(invlambda)

        # Stein shrinker lissé et son conjugué  
        # QIS travaille dans l'espace des inverses des valeurs propres (1/λ).
        Lj  = invlambda[None, :]   # inverses broadcastés sur les lignes, shape (1, n_nz)
        Lj_i = Lj - invlambda[:, None]  #  différences des inverses, shape (n_nz, n_nz)
        denom = np.maximum(Lj_i**2 + h**2 * Lj**2, self.eps)  #  dénominateur Cauchy

        # Stein shrinker lissé : mesure l'attraction vers ses voisins.
        theta   = (Lj * Lj_i  / denom).mean(axis=1)  # Le facteur (1/λⱼ) donne proportionnellement plus de poids aux grandes précisions (petites variances), cohérent avec la loi de Marchenko-Pastur.
        
        # Conjugué de θ̂ au sens du signal analytique.
        Htheta  = (Lj * (h * Lj)/ denom).mean(axis=1)  # C'est la transformée de Hilbert analytique du noyau de Cauchy. Contrairement à ANLS (Epanechnikov), aucune approximation numérique.

        # L'amplitude au carré du signal analytique associé au Stein shrinker. C'est la deuxième cible de shrinkage QIS, elle domine quand c tends vers 1.
        Atheta2 = theta**2 + Htheta**2  #  Quand c est modéré (~0.5), c'est stein shrinker qui domine ; quand c → 1, c'est Htheta^2.

        # Valeurs propres shrunken 
        # La formule est quadratique en c, avec trois termes dont les poids somment à 1 pour tout c, comme un trinôme carré.
        # Cas p ≤ n (c ≤ 1) :
        if p <= n:
            #   Les trois termes pondèrent : la valeur propre brute (dominante pour c→0), le Stein shrinker (dominante pour c~0.5), l'amplitude carrée (pour c→1).
            delta = 1.0 / ((1 - c)**2 * invlambda + 2 * c * (1 - c) * invlambda * theta + c**2 * invlambda * Atheta2)        # shape (p,)

        # Cas p > n (c > 1) :
        else:
            # Pour les p-n valeurs propres nulles : C'est une valeur commune à toutes les directions indiscernables.
            delta0   = 1.0 / ((c - 1.0) * invlambda.mean()) # scalaire

             # Pour les n valeurs propres non-nulles :  la formule complète dégénère vers ce terme.
            delta_nz = 1.0 / np.maximum(invlambda * Atheta2, self.eps)

            # Concaténation : (p-n) entrées delta0, puis n entrées delta_nz
            delta = np.concatenate([np.full(p - n, delta0), delta_nz])  # shape (p,)


        # Rescaling pour préserver la trace de S 
        # Les valeurs propres shrunken ne préservent pas trace(S) en général. LW appliquent un rescaling global multiplicatif.
        scale = lam_raw[idx_start:].sum() / delta.sum()
        delta_QIS = delta * scale   # shape (p,) si p>n, (n_nz,)=( p,) si p<=n

        # Reconstruction de Sigma 
        # U est la matrice COMPLÈTE des vecteurs propres (p x p). delta_QIS est de taille p dans les deux cas (p<=n et p>n). => U @ diag(delta_QIS) @ U.T reconstruit tout en une opération.
        # Pour p>n, les (p-n) premières entrées de delta_QIS correspondent aux vecteurs propres nuls de U (les premières colonnes triées par valeur propre).
        Sigma = _safe_symmetrize(U @ np.diag(delta_QIS) @ U.T)

        # 10. Diagnostics 
        trace_S     = float(np.trace(S))
        trace_Sigma = float(np.trace(Sigma))
        info: Dict[str, Any] = {
            "c":           c,
            "h_n":         h,
            "n":           n,
            "p":           p,
            "n_nz":        n_nz,
            "p_gt_n":      p > n,
            "lambda_min":  float(lam_raw[idx_start:].min()),
            "lambda_max":  float(lam_raw[idx_start:].max()),
            "theta_min":   float(theta.min()),
            "theta_max":   float(theta.max()),
            "Htheta_min":  float(Htheta.min()),
            "Htheta_max":  float(Htheta.max()),
            "delta_min":   float(delta_QIS.min()),
            "delta_max":   float(delta_QIS.max()),
            "trace_ratio": trace_Sigma / trace_S if trace_S > 0 else float("nan"),
        }
        return Sigma, info



    # Interface MultiVolModel
    @property
    def last_fit_info(self) -> Dict[str, Any]:
        """Derniers diagnostics : c, h_n, n_nz, theta, Htheta, delta, trace_ratio."""
        return dict(self._fit_info)

    def fit(self, R: pd.DataFrame, **_) -> "LedoitWolfQIS":
        Xdf = _as_dt_df(R)
        self._names = tuple(Xdf.columns)

        if self.window is None:
            Sigma, info      = self._estimate_once(Xdf.values)
            self._Sigma_full = Sigma
            self._fit_info   = info
        else:
            if len(Xdf) < int(self.window):
                raise ValueError("Pas assez d'observations pour la fenêtre choisie.")
            self._Sigma_full = None
            self._fit_info   = {}

        return self

    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:

        Xdf = _as_dt_df(R)
        X   = np.nan_to_num(Xdf.values.copy(), nan=0.0)
        idx = Xdf.index
        T, p = X.shape

        # Cas constant (window=None) 
        if self.window is None:
            if self._Sigma_full is None:
                Sigma, info      = self._estimate_once(X)
                self._Sigma_full = Sigma
                self._fit_info   = info
            H = np.repeat(self._Sigma_full[None, :, :], repeats=len(Xdf), axis=0)
            return CovariancePath(H=H, index=idx, names=tuple(Xdf.columns), _memmap_path=None)

        # Cas rolling 
        w     = int(self.window)
        n_out = len(Xdf) - w + 1
        out_idx = []

        cache_dir = Path(__file__).resolve().parents[3] / "memmap_cache"
        cache_dir.mkdir(exist_ok=True)
        filename = cache_dir / f"memmap_qis_{uuid.uuid4().hex}.dat"

        H = np.memmap(filename, dtype="float64", mode="w+", shape=(n_out, p, p))
        self._memmap_paths = getattr(self, "_memmap_paths", [])
        self._memmap_paths.append(str(filename))

        for i, t in enumerate(tqdm(range(w, len(Xdf) + 1), total=n_out, desc="QIS cov rolling")):
            W        = X[t - w : t]
            W        = np.nan_to_num(W, nan=0.0)
            col_mask = np.any(W != 0.0, axis=0)
            n_active = col_mask.sum()

            if n_active < 3:
                H[i] = np.zeros((p, p))
                out_idx.append(idx[t - 1])
                continue

            W_clean = W[:, col_mask]

            # n_nominal=w : préserve le vrai ratio c = p_active/w
            # même si des colonnes ont été filtrées.
            Sigma_clean, info = self._estimate_once(W_clean, n_nominal=w)
            self._fit_info    = info

            # Repadding : colonnes inactives restent à zéro
            Sigma_full = np.zeros((p, p))
            idx_active = np.where(col_mask)[0]
            Sigma_full[np.ix_(idx_active, idx_active)] = Sigma_clean

            H[i] = Sigma_full
            out_idx.append(idx[t - 1])

        H.flush()
        return CovariancePath(H=H, index=pd.DatetimeIndex(out_idx),names=tuple(Xdf.columns),_memmap_path=str(filename),)



    def compute_cov_at_rebal(self,returns_window: pd.DataFrame,all_cols: list[str],) -> np.ndarray:
        """
        Calcule la covariance à un instant de rebal sur la fenêtre fournie.
        Retourne une matrice (len(all_cols) x len(all_cols)) avec repadding
        pour les colonnes inactives (identique au rolling).
        """
        p_full = len(all_cols)
        X = np.nan_to_num(returns_window.reindex(columns=all_cols).values.copy(), nan=0.0)

        # Filtrage colonnes actives 
        col_mask = np.any(X != 0.0, axis=0)
        n_active  = col_mask.sum()

        if n_active < 3:
            return np.zeros((p_full, p_full))

        W_clean = X[:, col_mask]

        # Estimation sur la fenêtre nettoyée
        Sigma_clean, _ = self._estimate_once(W_clean, n_nominal=len(returns_window))

        # Repadding à la taille de l'univers complet
        Sigma_full = np.zeros((p_full, p_full))
        idx_active = np.where(col_mask)[0]
        Sigma_full[np.ix_(idx_active, idx_active)] = Sigma_clean

        return Sigma_full

    def snapshot_cov(self, window: pd.DataFrame, prev_state=None) -> Tuple[np.ndarray, None, Dict[str, Any]]:
        """Estime la covariance à un instant donné sur une fenêtre de rendements."""
        Xdf   = _as_dt_df(window)
        X_raw = np.nan_to_num(Xdf.values.copy(), nan=0.0)
        X     = X_raw if self.window is None else X_raw[-int(self.window):]
        Sigma, info = self._estimate_once(X)
        return Sigma, None, info

    @property
    def fit_info(self) -> Dict[str, Any]:
        return dict(self._fit_info)

