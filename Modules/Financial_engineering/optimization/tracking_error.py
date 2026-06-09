"""
Objectifs d'optimisation et utilitaires pour la minimisation de la tracking error.
 
Ce fichier définit les fonctions de calcul de TE et les classes d'objectif
branchées sur le moteur d'optimisation (SLSQPOptimizer ou ClarabelOptimizer).
 
Classes
-------
TrackingErrorObjective :
    Minimise (w - b)' Sigma (w - b) sur l'espace réduit K. Gradient analytique disponible pour SLSQP.
TrackingErrorToBenchmarkReturnObjective :
    Minimise TE^2 vs rendements du benchmark, estimée via COV_II et COV_Ib.
TrackingErrorFullUniverseObjective :
    Minimise TE^2 sur l'univers complet N, en optimisant uniquement sur les K kept. Gradient analytique disponible pour SLSQP. Log des évaluations optionnel.
 
Fonctions
---------
active_weights : Calcule w - b avec validation des inputs.
tracking_error_variance : Calcule (w - b)' Sigma (w - b).
tracking_error : Calcule sqrt(TE^2).
estimate_cov_ii_and_cov_ib : Estime COV_II et COV_Ib sur une fenêtre de rendements.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .base import Objective

import numpy as np
import pandas as pd




# Utilitaires de validation et de calcul
def _validate_weights(x: np.ndarray, name: str) -> None:
    """
    Valide qu'un vecteur de poids est 1D et ne contient que des valeurs finies.
 
    Parameters
    ----------
    x : np.ndarray
        Vecteur à valider.
    name : str
        Nom du vecteur (utilisé dans le message d'erreur).
 
    Raises
    ------
    ValueError
        Si x n'est pas 1D ou contient des NaN/inf.
    """

    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")


def _validate_cov(cov: np.ndarray, n: int) -> None:
    """
    Valide qu'une matrice de covariance est carrée (n x n), finie et symétrique.
 
    Parameters
    ----------
    cov : np.ndarray
        Matrice à valider.
    n : int
        Dimension attendue.
 
    Raises
    ------
    ValueError
        Si la matrice n'est pas (n, n), contient des NaN/inf, ou n'est pas symétrique.
    """

    if cov.ndim != 2 or cov.shape != (n, n):
        raise ValueError(f"cov must be shape ({n}, {n}).")
    if not np.all(np.isfinite(cov)):
        raise ValueError("cov contains non-finite values.")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("cov must be symmetric (within numerical tolerance).")


def active_weights(w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calcule les active weights : a = w - b.
 
    Parameters
    ----------
    w : np.ndarray
        Poids du portefeuille (K,).
    b : np.ndarray
        Poids du benchmark (K,).
 
    Returns
    -------
    np.ndarray
        Vecteur d'active weights (K,).
    """

    # Utilise la fonction de validation
    _validate_weights(w, "w")
    _validate_weights(b, "b")


    if w.shape != b.shape:
        raise ValueError("w and b must have the same shape.")
    
    # différence entre poids portefeuille et benchmark
    return w - b


def tracking_error_variance(w: np.ndarray, b: np.ndarray, cov: np.ndarray) -> float:
    """
    Calcule la variance de tracking error : TE^2 = (w - b)' Sigma (w - b).
 
    Parameters
    ----------
    w : np.ndarray
        Poids du portefeuille (K,).
    b : np.ndarray
        Poids du benchmark (K,).
    cov : np.ndarray
        Matrice de covariance (K, K).
 
    Returns
    -------
    float
        Variance de TE (non annualisée).
    """

    # a = w - b
    a = active_weights(w, b)

    # vérifie la cohérence dimensionnelle
    _validate_cov(cov, n=a.shape[0])

    # forme quadratique
    return float(a.T @ cov @ a)


def tracking_error(w: np.ndarray, b: np.ndarray, cov: np.ndarray) -> float:
    """
    Calcule la tracking error : TE = sqrt(TE^2).
 
    Parameters
    ----------
    w : np.ndarray
        Poids du portefeuille (K,).
    b : np.ndarray
        Poids du benchmark (K,).
    cov : np.ndarray
        Matrice de covariance (K, K).
 
    Returns
    -------
    float
        Tracking error (non annualisée).
    """

    #Utilise la fonction de TE
    tev = tracking_error_variance(w, b, cov)

    # max(., 0) pour éviter sqrt de négatif par erreur numérique
    return float(np.sqrt(max(tev, 0.0)))


@dataclass(frozen=True)
class TrackingErrorObjective(Objective):
    """
    Minimise la variance de tracking error vs poids benchmark sur l'espace réduit K.
 
    Objectif : f(w) = (w - b)' Sigma (w - b)
 
    Utilisé en mode 'weights' sans redistribution, où le benchmark est projeté
    sur les K actifs investissables avant optimisation.
 
    Attributes
    ----------
    benchmark_weights : np.ndarray
        Poids du benchmark projetés sur les K actifs investissables.
    cov : np.ndarray
        Matrice de covariance sur les K actifs.
    debug : bool
        Si True, valide les inputs à l'initialisation.
 
    Methods
    -------
    value(w) -> float :
        Évalue (w - b)' Sigma (w - b).
    gradient(w) -> np.ndarray :
        Retourne le gradient analytique 2 * Sigma * (w - b).
    """

    # b : poids benchmark sur les K actifs
    benchmark_weights: np.ndarray  

    # Sigma : matrice de covariance (K, K)
    cov: np.ndarray

    debug: bool = False

    def __post_init__(self) -> None:
        b = np.asarray(self.benchmark_weights, dtype=float)
        cov = np.asarray(self.cov, dtype=float)

        # Validation
        if self.debug:
            _validate_weights(b, "benchmark_weights")
            _validate_cov(cov, n=b.shape[0])

        # Précalcule Sigma @ b une fois pour accélérer les évaluations du gradient
        object.__setattr__(self, "benchmark_weights", b)
        object.__setattr__(self, "cov", cov)
        object.__setattr__(self, "_cov_b", cov @ b)


    def value(self, w: np.ndarray) -> float:
        """
        Évalue (w - b)' Sigma (w - b).
 
        Parameters
        ----------
        w : np.ndarray
            Poids courants du portefeuille (K,).
 
        Returns
        -------
        float
            Variance de TE au point w.
        """
        w = np.asarray(w, dtype=float)
        return tracking_error_variance(w, self.benchmark_weights, self.cov)


    def gradient(self, w: np.ndarray) -> Optional[np.ndarray]:
        """
        Retourne le gradient analytique
        Utilise la forme précalculée : 2 * (Sigma @ w - Sigma @ b)
        pour éviter de recalculer Sigma @ b à chaque itération.
 
        Parameters
        ----------
        w : np.ndarray
            Poids courants (K,).
 
        Returns
        -------
        np.ndarray
            Gradient de shape (K,).
        """
        w = np.asarray(w, dtype=float)
        return 2.0 * (self.cov @ w - self._cov_b)




# Utilitaires pour l'objectif vs rendements benchmark
def _clean_joint_sample(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Supprime les lignes où X ou y contient des NaN ou des valeurs infinies.
 
    Parameters
    ----------
    X : np.ndarray
        Matrice de rendements (T, N).
    y : np.ndarray
        Vecteur de rendements du benchmark (T,).
 
    Returns
    -------
    X_clean, y_clean : np.ndarray
        Matrices filtrées sur les lignes valides.
    """

    # True si y est fini à cette date
    mask = np.isfinite(y)

    # True si toutes les colonnes de X sont finies
    mask &= np.isfinite(X).all(axis=1)

    return X[mask], y[mask]


def estimate_cov_ii_and_cov_ib(returns_window: pd.DataFrame,benchmark_returns_window: pd.Series,) -> tuple[np.ndarray, np.ndarray]:
    """
    Estime COV_II (covariance entre actifs) et COV_Ib (covariance actifs-benchmark)
    sur une fenêtre de rendements.
 
    COV_II = (1/(T-1)) * Xc' Xc   avec Xc = rendements centrés
    COV_Ib = (1/(T-1)) * Xc' yc   avec yc = rendements benchmark centrés
 
    Ces deux matrices sont utilisées par TrackingErrorToBenchmarkReturnObjective
    pour construire l'objectif TE-min vs rendements du benchmark.
 
    Parameters
    ----------
    returns_window : pd.DataFrame
        Matrice de rendements (T x N) sur l'univers investissable.
    benchmark_returns_window : pd.Series
        Rendements du benchmark officiel complet (T,).
 
    Returns
    -------
    cov_ii : np.ndarray
        Matrice de covariance (N, N) entre actifs investissables.
    cov_ib : np.ndarray
        Vecteur de covariance (N,) entre actifs et benchmark.
 
    Raises
    ------
    ValueError
        Si moins de 10 observations valides après nettoyage.
    """

    #Recupère les rendements actifs et benchmark, reindex et float
    X = returns_window.values.astype(float)
    y = benchmark_returns_window.reindex(returns_window.index).values.astype(float)

    # Supprime les lignes avec NaN ou inf dans X ou y
    X2, y2 = _clean_joint_sample(X, y)
    if X2.shape[0] < 10:
        raise ValueError("Not enough clean observations to estimate covariances.")

    # Centre les rendements
    Xc = X2 - X2.mean(axis=0, keepdims=True)
    yc = y2 - y2.mean()

    # normalisation sans biais
    T = X2.shape[0]
    denom = float(T - 1)

    # covariance entre actifs (N, N)
    cov_ii = (Xc.T @ Xc) / denom 

    # covariance actifs-benchmark (N, 1)
    cov_ib = (Xc.T @ yc.reshape(-1, 1)) / denom 

    # aplatit en (N,)
    cov_ib = cov_ib.reshape(-1)                 

    # Symétrise COV_II pour éviter les erreurs numériques
    cov_ii = 0.5 * (cov_ii + cov_ii.T)
    return cov_ii, cov_ib



# Objectif 2 : TE-min vs rendements benchmark
class TrackingErrorToBenchmarkReturnObjective(Objective):
    """
    Minimise la TE^2 vs les rendements du benchmark officiel complet.
 
    Objectif : f(w) = w' COV_II w - 2 * w' COV_Ib
 
    Où COV_II est la matrice de covariance entre actifs investissables et
    COV_Ib est le vecteur de covariance entre actifs et rendements du benchmark.
    Cette formulation permet de répliquer le rendement du benchmark complet
    même quand seul un sous-ensemble d'actifs est investissable.
 
    Attributes
    ----------
    cov_ii : np.ndarray
        Matrice de covariance (N, N) entre actifs investissables.
    cov_ib : np.ndarray
        Vecteur de covariance (N,) entre actifs et rendements du benchmark.
 
    Methods
    -------
    value(w) -> float :
        Évalue w' COV_II w - 2 * w' COV_Ib.
    gradient(w) -> np.ndarray :
        Retourne le gradient analytique 2 * (COV_II @ w - COV_Ib).
    """

    def __init__(self, cov_ii: np.ndarray, cov_ib: np.ndarray) -> None:
        """Initialise l'objectif avec les covariances données et valide les entrées."""

        # Conversion en float
        cov_ii = np.asarray(cov_ii, dtype=float)
        cov_ib = np.asarray(cov_ib, dtype=float)

        # Validation de la forme de COV_II (doit être carrée)
        if cov_ii.ndim != 2 or cov_ii.shape[0] != cov_ii.shape[1]:
            raise ValueError("cov_ii must be square (N,N).")

        n = cov_ii.shape[0]

        # Validation de la cohérence entre COV_II et COV_Ib
        if cov_ib.shape != (n,):
            raise ValueError("cov_ib must be shape (N,).")

        # Validation des valeurs finies
        if not np.all(np.isfinite(cov_ii)) or not np.all(np.isfinite(cov_ib)):
            raise ValueError("Non-finite values in covariances.")

        # covariance entre actifs investissables
        self.cov_ii = cov_ii

        # covariance actifs-benchmark
        self.cov_ib = cov_ib


    def value(self, w: np.ndarray) -> float:
        """
        Évalue f(w) = w' COV_II w - 2 * w' COV_Ib.
 
        Parameters
        ----------
        w : np.ndarray
            Poids courants du portefeuille (N,).
 
        Returns
        -------
        float
            Valeur de l'objectif TE^2 vs rendements benchmark.
        """
        
        w = np.asarray(w, dtype=float)
        return float(w.T @ self.cov_ii @ w - 2.0 * w.T @ self.cov_ib) 

    def gradient(self, w: np.ndarray) -> Optional[np.ndarray]:
        """
        Retourne le gradient analytique : ∇_w f(w) = 2 * (COV_II @ w - COV_Ib).
 
        Parameters
        ----------
        w : np.ndarray
            Poids courants (N,).
 
        Returns
        -------
        np.ndarray
            Gradient de shape (N,).
        """

        w = np.asarray(w, dtype=float)

        # gradient analytique
        return 2.0 * (self.cov_ii @ w - self.cov_ib)


# Objectif 3 : TE-min full universe (N actifs, K investissables)
@dataclass(frozen=True)
class TrackingErrorFullUniverseObjective(Objective):
    """
    Minimise la TE^2 sur l'univers complet N, en optimisant sur les K actifs investissables.
 
    Objectif : f(w) = (w_full - b_full)' Sigma_full (w_full - b_full)
 
    Où w_full est un vecteur N avec des zéros partout sauf aux positions kept_idx
    où w_full[kept_idx] = w (les K poids optimisés). Cet objectif capture les termes
    croisés dans la covariance entre actifs investissables et actifs exclus.
 
    Attributes
    ----------
    cov_full : np.ndarray
        Matrice de covariance complète (N, N) sur l'univers benchmark.
    benchmark_weights_full : np.ndarray
        Poids du benchmark complet (N,).
    kept_idx : np.ndarray
        Indices (int) des K actifs investissables dans l'espace N.
    debug : bool
        Si True, valide kept_idx à l'initialisation.
    log_evaluations : bool
        Si True, enregistre chaque appel à value() dans eval_log
 
    Methods
    -------
    value(w) -> float :
        Évalue la TE^2 complète en plongeant w dans l'espace N.
    gradient(w) -> np.ndarray :
        Gradient analytique en dimension K (via précalcul de Sigma_KK et Sigma_K).
    """

    cov_full: np.ndarray           
    benchmark_weights_full: np.ndarray  
    kept_idx: np.ndarray          
    debug :bool = False
    log_evaluations : bool = False


    def __post_init__(self) -> None:

        # Récupération de la matrice de covaraince, poids du benchmark et actifs investissable
        cov = np.asarray(self.cov_full, dtype=float)
        b = np.asarray(self.benchmark_weights_full, dtype=float)
        kept = np.asarray(self.kept_idx, dtype=int)
        
         # Validation des inputs
        _validate_weights(b, "benchmark_weights_full")
        _validate_cov(cov, n=b.shape[0])

        # Validation plus stricte de kept_idx en mode debug
        if self.debug : 
            if kept.ndim != 1:
                raise ValueError("kept_idx must be a 1D integer array.")
            if kept.size < 1:
                raise ValueError("kept_idx must be non-empty.")
            if kept.min() < 0 or kept.max() >= b.shape[0]:
                raise ValueError("kept_idx contains out-of-range indices.")
            if len(np.unique(kept)) != kept.size:
                raise ValueError("kept_idx must not contain duplicates.")

        # Précalcule les matrices nécessaires pour le gradient et l'évaluation efficace
        object.__setattr__(self, "cov_full", cov)
        object.__setattr__(self, "benchmark_weights_full", b)
        object.__setattr__(self, "kept_idx", kept)
        object.__setattr__(self, "_cov_b", cov @ b)
        object.__setattr__(self, "_cov_kept", cov[:, kept])   
        object.__setattr__(self, "_cov_kk",   cov[np.ix_(kept, kept)])
        object.__setattr__(self,"eval_log",[] if self.log_evaluations else None)


    def _embed_w_full(self, w: np.ndarray) -> np.ndarray:
        """
        Plonge le vecteur K dans l'espace N : w_full = 0 partout sauf aux positions kept_idx.
 
        Parameters
        ----------
        w : np.ndarray
            Poids du portefeuille sur les K actifs investissables.
 
        Returns
        -------
        np.ndarray
            Vecteur N avec w aux positions kept_idx et 0 ailleurs.
        """
       
        w = np.asarray(w, dtype=float)
        k = self.kept_idx.size
        if w.ndim != 1 or w.shape[0] != k:
            raise ValueError(f"w must be shape ({k},).")

        # Initialise un vecteur N de zéros, puis place les poids aux positions investissables
        w_full = np.zeros_like(self.benchmark_weights_full, dtype=float)
        w_full[self.kept_idx] = w 

        return w_full

    def value(self, w: np.ndarray) -> float:
        """
        Évalue la TE^2 complète : f(w) = (w_full - b_full)' Sigma_full (w_full - b_full).
 
        Plonge w dans l'espace N, calcule l'écart complet vs benchmark,
        et évalue la forme quadratique avec la covariance complète
        (incluant les termes croisés avec les actifs exclus).
 
        Si log_evaluations=True, enregistre (w, TE annualisée) dans eval_log.
 
        Parameters
        ----------
        w : np.ndarray
            Poids sur les K actifs investissables.
 
        Returns
        -------
        float
            Variance de TE complète.
        """

        # Vecteur N complet avec les poids optimisés aux positions kept_idx et zéro ailleurs
        w_full = self._embed_w_full(w) 

        # Ecart complet vs benchmark complet
        a_full = w_full - self.benchmark_weights_full 

        #Calcul de la TE^2 complète , donc corrélation entre actifs inclus et exclus ici (termes croisés dans la matrice de covariance complète)
        val = float(a_full.T @ self.cov_full @ a_full) 
        
        # Log si activé : enregistre la TE annualisée à chaque évaluation
        if self.log_evaluations and self.eval_log is not None:
            te_ann = float(np.sqrt(max(val, 0.0)) * np.sqrt(252.0))
            self.eval_log.append((w.copy(),te_ann))
        
        return val 
    
    def get_eval_log(self):
        """
        Retourne le log des évaluations sous forme de DataFrame.
 
        Returns
        -------
        pd.DataFrame or None
            DataFrame avec colonnes 'iteration' et 'te_ann', ou None si log vide.
        """

        if not self.eval_log:
            return None
        
        # Chaque entrée est un tuple (w, te_ann), on ne garde que te_ann pour le DataFrame
        return pd.DataFrame([{"iteration": i, "te_ann": e["te_ann"]}for i, e in enumerate(self._eval_log)])
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """
        Gradient analytique efficace via précalcul
        Utilise les matrices précalculées _cov_kk et _cov_b pour éviter
        de reconstruire w_full à chaque itération du solveur.
 
        Parameters
        ----------
        w : np.ndarray
            Poids courants sur les K actifs.
 
        Returns
        -------
        np.ndarray
            Gradient de shape (K,).
        """
        g_K = 2.0 * (self._cov_kk @ w - self._cov_b[self.kept_idx])
        return g_K


