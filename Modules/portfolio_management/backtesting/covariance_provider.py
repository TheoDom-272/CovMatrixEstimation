"""
Fourniture des matrices de covariance pour le moteur de backtesting.
 
Ce fichier centralise la configuration et l'accès aux estimateurs de covariance
utilisés pendant le backtest. Il supporte deux modes de calcul : le mode 'path'
(pré-calcul sur toute la série en une fois, stocké en memmap) et le mode 'rebal'
(calcul à la demande à chaque date de rebalancement).
 
Classes
-------
BaseCovConfig :
    Dataclass de base pour la configuration d'un estimateur de covariance, commune à toutes les méthodes.
RollingCovConfig :
    Configuration pour la covariance rolling (fenêtre glissante).
EWMACovConfig :
    Configuration pour l'estimateur EWMA (RiskMetrics), avec option de tuning du lambda.
LedoitWolfConfig :
    Configuration pour les estimateurs Ledoit-Wolf et variantes (LW2004, ANLS2020, OAS, QIS).
DCCConfig :
    Configuration pour le modèle DCC-GARCH gaussien, avec options de re-fit périodique.
CovarianceProvider :
    Classe principale qui instancie le bon estimateur selon la config, calcule et expose
    les matrices de covariance au moteur de backtest (via path précomputé ou calcul on-demand).
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple,Union

import numpy as np
import pandas as pd



# Base utils 
from Modules.Financial_engineering.statistics.multivariate_vol_estimation import ( _safe_symmetrize, RollingSampleCov,)

# Types
CovEstimator = Callable[[pd.DataFrame], np.ndarray]

Method = Literal["sample", "Rolling", "EWMA", "DCC", "ledoit_wolf"]
ComputeMode = Literal["path", "rebal"]
PathScope = Literal["all_dates", "decision_dates"]
LWVariant = Literal["lw_2004", "anls_2020"]



def _ensure_cholesky_compatible(S: np.ndarray, jitter: float = 1e-10, max_pow: int = 8) -> np.ndarray:
    """
    Force une matrice de covariance à être symétrique et Cholesky-compatible.
 
    Si la matrice n'est pas définie positive, on ajoute un jitter diagonal
    progressivement croissant jusqu'à obtenir une décomposition de Cholesky valide.
 
    Parameters
    ----------
    S : np.ndarray
        Matrice de covariance candidate (N x N).
    jitter : float
        Valeur de base du jitter diagonal ajouté à chaque tentative.
    max_pow : int
        Nombre maximal de tentatives (le jitter est multiplié par 10^k à chaque essai).
 
    Returns
    -------
    np.ndarray
        Matrice symétrique et définie positive.
    """

    #Symétrisation via la fonction utilitaire de base
    S = _safe_symmetrize(np.asarray(S, dtype=float))

    #Nombre de variables
    n = S.shape[0]

    #Itération sur chaque puissance de jitter
    for k in range(max_pow + 1):

        #Check de Cholesky
        try:
            np.linalg.cholesky(S) 
            return S #Si réussi, on retourne la matrice
        
        #Si échec, on ajoute du jitter diagonal et réessaye
        except np.linalg.LinAlgError:
            S = _safe_symmetrize(S + (jitter * (10.0 ** k)) * np.eye(n))
    return S



@dataclass
class BaseCovConfig:
    """
    Configuration de base commune à tous les estimateurs de covariance.
 
    Attributes
    ----------
    method : str
        Nom de la méthode d'estimation ('Rolling', 'EWMA', 'ledoit_wolf', 'DCC', 'sample').
    compute_mode : str
        Mode de calcul : 'path' pour un pré-calcul complet, 'rebal' pour un calcul à la demande.
    path_scope : str
        Granularité du path précomputé : 'decision_dates' ou 'all_dates'.
    cov_dataçfreq : str
        Fréquence des returns passés à l'estimateur
    """

    method: Literal["sample","Rolling","EWMA","DCC","ledoit_wolf"]
    compute_mode: Literal["rebal", "daily"] = "rebal"
    path_scope: Literal["decision_dates", "all_dates"] = "decision_dates"
    cov_data_freq: str = "daily" 


@dataclass
class RollingCovConfig(BaseCovConfig):
    """
    Configuration pour la covariance rolling sur fenêtre glissante.
 
    Attributes
    ----------
    rolling_window : int
        Taille de la fenêtre glissante en jours de bourse.
    rolling_ddof : int
        Degré de liberté pour le calcul de la variance sur chaque fenêtre.
    """

    rolling_window: int = 252
    rolling_ddof: int = 1

@dataclass
class EWMACovConfig(BaseCovConfig):
    """
    Configuration pour l'estimateur EWMA (RiskMetrics) de la covariance.
 
    Attributes
    ----------
    ewma_lambda : float
        Facteur de décroissance exponentielle (0 < lambda < 1). Plus lambda est proche de 1, plus les observations anciennes ont de poids.
    ewma_init : str
        Mode d'initialisation de la matrice de covariance au démarrage : 'scov' (covariance empirique sur la fenêtre initiale) ou 'diag' (diagonale des variances).
    tune_lambda : bool
        Si True, le lambda optimal est estimé par quasi-maximum de vraisemblance sur les données.
    rolling_window : int
        Taille de la fenêtre d'initialisation (utilisée pour 'scov' ou le tuning).
    """

    ewma_lambda: float = 0.94
    ewma_init: Literal["scov", "diag"] = "scov"
    tune_lambda: bool = False
    rolling_window: int = 252


@dataclass
class LedoitWolfConfig(BaseCovConfig):
    """
    Configuration pour les estimateurs Ledoit-Wolf et leurs variantes non-linéaires.
 
    Attributes
    ----------
    lw_variant : str
        Variante de l'estimateur :
        - 'lw_2004' : shrinkage linéaire vers la matrice identité (Ledoit-Wolf 2004).
        - 'oas_2012' : Oracle Approximating Shrinkage (Ledoit-Wolf 2012).
        - 'anls_2020' : shrinkage non-linéaire analytique (Ledoit-Wolf 2020).
        - 'QIS_2022' : Quadratic Inverse Shrinkage (Ledoit-Wolf 2022).
    lw_window : int or None
        Taille de la fenêtre glissante. None = utilise toutes les données disponibles.
    lw_demean : bool
        Si True, les rendements sont centrés avant estimation.
    lw_ddof : int
        Degré de liberté pour le calcul de la covariance empirique sous-jacente.
    chunk_size : int
        Taille des blocs de calcul pour ANLS et QIS (optimisation mémoire sur grands univers).
    use_package : bool
        Si True, utilise l'implémentation du package sklearn pour LW linéaire.
    """

    lw_variant: Literal["lw_2004", "lw_schaefer"] = "lw_2004"
    lw_window: Optional[int] = None
    lw_demean: bool = True
    lw_ddof: int = 0
    chunk_size: int = 1024
    use_package: bool = False
    
    
@dataclass
class DCCConfig(BaseCovConfig):
    """
    Configuration pour le modèle DCC-GARCH gaussien (Dynamic Conditional Correlation).
 
    Attributes
    ----------
    dcc_use_package : bool
        Si True, utilise une implémentation externe (arch). Si False, estimation MLE from scratch.
    dcc_lambda_std : float
        Lambda EWMA pour la standardisation des volatilités individuelles.
    dcc_vol_model : str
        Modèle de volatilité univarié utilisé pour standardiser les rendements : 'ewma' ou 'garch'.
    dcc_refit_enabled : bool
        Si True, le modèle est re-fitté périodiquement pendant le backtest.
    dcc_refit_lookback : int
        Nombre de jours de données utilisés pour chaque re-fit.
    dcc_refit_mode : str
        Mode de fenêtre pour le re-fit : 'rolling' (fenêtre fixe) ou 'expanding' (fenêtre croissante).
    dcc_refit_every : int or None
        Fréquence du re-fit en jours de bourse. None = utilise dcc_refit_dates.
    dcc_refit_dates : pd.DatetimeIndex or None
        Dates explicites de re-fit (par exemple, alignées sur les dates de rebalancement).
    dcc_refit_min_obs : int
        Nombre minimal d'observations requis pour autoriser un re-fit.
    dcc_include_refit_date_in_forecast : bool
        Si True, la date de re-fit est incluse dans la prévision de covariance.
    """
        
    dcc_use_package: bool = True
    dcc_lambda_std: float = 0.94
    dcc_vol_model: str = "garch"  # "ewma" ou "garch"

    dcc_refit_enabled: bool = False
    dcc_refit_lookback: int = 252
    dcc_refit_mode: str = "rolling"          # "rolling" ou "expanding"
    dcc_refit_every: Optional[int] = None    # ex: 21 (proche du mensuel en jours de bourse)
    dcc_refit_dates: Optional[pd.DatetimeIndex] = None  # pour aligner sur rebal dates
    dcc_refit_min_obs: int = 60
    dcc_include_refit_date_in_forecast: bool = True


# Union pour l’IDE et le typage
CovConfig = Union[
    RollingCovConfig,
    EWMACovConfig,
    LedoitWolfConfig,
    DCCConfig
]


def make_cov_config(method: Method, **kwargs) -> CovConfig:
    """
    Fonction utilitaire pour instancier la bonne config de covariance selon la méthode choisie.
 
    Parameters
    ----------
    method : str
        Nom de la méthode ('sample', 'rolling', 'ewma', 'ledoit_wolf', 'DCC').
    **kwargs :
        Paramètres additionnels passés au dataclass de config correspondant.
 
    Returns
    -------
    CovConfig
        Instance de la config adaptée à la méthode.
    """

    kwargs["method"] = method
    if method == "rolling":
        return RollingCovConfig(**kwargs)
    if method == "ewma":
        return EWMACovConfig(**kwargs)
    if method == "ledoit_wolf":
        return LedoitWolfConfig(**kwargs)
    if method == "DCC":
        return DCCConfig(**kwargs)
    raise ValueError(f"Unknown method: {method}")



# Provider
class CovarianceProvider:
    """
    Classe principale qui fournit les matrices de covariance au moteur de backtesting.
 
    Selon le mode de calcul configuré, elle pré-calcule le path complet de covariances
    sur toute la série (mode 'path'), ou calcule la covariance à la demande à chaque
    rebalancement sur la fenêtre courante (mode 'rebal').
 
    En mode 'path', les matrices sont stockées dans un fichier memmap pour permettre
    le partage entre workers parallèles et eviter la surchage de mémoire.
 
    Attributes
    ----------
    cfg : CovConfig
        Configuration de l'estimateur (méthode, mode, paramètres spécifiques).
 
    Methods
    -------
    from_memmap(cfg, memmap_path, memmap_shape, path_index, path_names) -> CovarianceProvider :
        Reconstruit un provider en lecture seule depuis un fichier memmap existant.
    close_readonly() -> None :
        Libère le handle memmap local sans supprimer le fichier sur disque.
    precompute_path(returns) -> None :
        Fit le modèle et calcule le path complet de covariances sur toute la série de rendements.
    get_cov_at(asof) -> np.ndarray :
        Retourne la matrice de covariance à une date donnée depuis le path précomputé. Prend la dernière date disponible <= asof.
    get_cov_subset(asof, sub_cols) -> np.ndarray :
        Extrait la sous-matrice de covariance pour un sous-ensemble de colonnes depuis le path.
    get_cov_rebal(returns_window, sub_cols) -> np.ndarray :
        Calcule la covariance à la demande sur la fenêtre fournie (mode 'rebal'). Appelle compute_cov_at_rebal() sur le modèle si disponible.
    """

    def __init__(self, cfg: CovConfig) -> None:
        self.cfg = cfg
        self._online_cache: Dict[Tuple[int, Tuple[str, ...]], Dict[str, Any]] = {}  

        # cache on-demand (rebal)
        self._cache: Dict[Tuple[pd.Timestamp, Tuple[str, ...]], np.ndarray] = {}

        # path store (path)
        self._path: Dict[Tuple[pd.Timestamp, Tuple[str, ...]], np.ndarray] = {}
        self._path_cache: dict[tuple[str, ...], dict[pd.Timestamp, np.ndarray]] = {}
        self._full_cols: Optional[Tuple[str, ...]] = None
        self._path_index: Optional[pd.DatetimeIndex] = None
        self._path_H: Optional[np.ndarray] = None
        self._covariance_path: Optional["CovariancePath"] = None

        # builder, instancié à partir de la config dans le constructeur
        self._estimator = self._build_multivol_model()

        
    # Keys
    @staticmethod
    def _key(asof: pd.Timestamp, cols: list[str]) -> Tuple[pd.Timestamp, Tuple[str, ...]]:
        return (pd.Timestamp(asof), tuple(map(str, cols)))
    
    def _online_key(self, cols):
        return (id(self._model), tuple(cols))
    
    @staticmethod
    def _ensure_df(X: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et normalise un DataFrame de rendements avant estimation.
        Convertit l'index en DatetimeIndex, trie les dates, force les colonnes
        en numérique et supprime les lignes/colonnes entièrement nulles.
        """

        Y = X.copy()
        if not isinstance(Y.index, pd.DatetimeIndex):
            Y.index = pd.to_datetime(Y.index)
        Y = Y.sort_index()
        for c in Y.columns:
            Y[c] = pd.to_numeric(Y[c], errors="coerce")
        Y = Y.dropna(how="all")
        Y = Y.dropna(axis=1, how="all")
        return Y


    
    @classmethod
    def from_memmap(cls,cfg: "CovConfig",memmap_path: Path,memmap_shape: tuple,path_index: pd.DatetimeIndex,path_names: tuple,) -> "CovarianceProvider":
        """
        Reconstruit un CovarianceProvider en lecture seule depuis un fichier memmap existant.
 
        Utilisé par les workers parallèles pour accéder au path de covariances
        précomputé par le processus principal, sans recalcul ni duplication mémoire.
        Le fichier memmap n'est pas possédé par ce provider (pas de suppression à la fermeture).
 
        Parameters
        ----------
        cfg : CovConfig
            Configuration de l'estimateur (doit correspondre à celle utilisée lors du pré-calcul).
        memmap_path : Path
            Chemin vers le fichier .dat contenant les matrices de covariance.
        memmap_shape : tuple
            Dimensions du tableau memmap (T, N, N).
        path_index : pd.DatetimeIndex
            Index temporel correspondant aux T matrices stockées.
        path_names : tuple
            Noms des N actifs (colonnes) dans l'ordre du memmap.
 
        Returns
        -------
        CovarianceProvider
            Provider configuré en mode lecture seule sur le memmap.
        """

        from Modules.Financial_engineering.statistics.multivariate_vol_estimation import CovariancePath

        #Recupère la configuration et construit le provider
        provider = cls(cfg=cfg)
        provider.cfg.compute_mode = "path"

        # Ouvre le fichier en lecture seule
        H_readonly = np.memmap(memmap_path, dtype="float64",mode="r",shape=memmap_shape,)

        # Stocke les références au path dans le provider
        provider._full_cols       = path_names
        provider._path_index      = pd.DatetimeIndex(path_index)
        provider._path_H          = H_readonly
        provider._covariance_path = CovariancePath(H=H_readonly,index=pd.DatetimeIndex(path_index),names=path_names,_memmap_path=None,) 
        return provider

    def close_readonly(self) -> None:
        """
        Libère le handle memmap local sans supprimer le fichier sur disque.
        """

        import gc
        if self._path_H is not None and isinstance(self._path_H, np.memmap):
            del self._path_H
            gc.collect()
        self._path_H          = None
        self._covariance_path = None


    def _build_multivol_model(self):
        """
        Instancie le bon estimateur de covariance selon la configuration fournie.
 
        Lit le champ 'method' de la config et retourne l'objet modèle correspondant
        (RollingSampleCov, EWMACov, DCCGaussian, ou une variante Ledoit-Wolf).
        Toutes les importations sont internes pour éviter les dépendances inutiles au démarrage.
        """

        #Recupère la méthode
        method = self.cfg.method.lower()

        # Instancie le modèle Rolling avec les paramètres de la config
        if method == "rolling":
            from Modules.Financial_engineering.statistics.multivariate_vol_estimation import RollingSampleCov
            window = int(getattr(self.cfg, "rolling_window", 252))
            ddof = int(getattr(self.cfg, "rolling_ddof", 1))
            return RollingSampleCov(window=window, ddof=ddof)

        # Instancie le modèle EWMA avec les paramètres de la config
        if method == "ewma":
            from Modules.Financial_engineering.statistics.Engle.EWMACov import EWMACov
            lmbda = float(getattr(self.cfg, "ewma_lambda", 0.94))
            init = str(getattr(self.cfg, "ewma_init", "scov"))
            window = int(getattr(self.cfg, "rolling_window", 252))
            tune_lambda = bool(getattr(self.cfg, "tune_lambda", False))
            return EWMACov(lmbda=lmbda, init=init,tune_lambda=tune_lambda, window=window)

        # Instancie le modèle DCC-GARCH avec les paramètres de la config
        if method == "dcc":
            from Modules.Financial_engineering.statistics.Engle.DCC import DCCGaussian
            lambda_std = float(getattr(self.cfg, "dcc_lambda_std", 0.94))
            vol_model = str(getattr(self.cfg, "dcc_vol_model", "garch"))
            use_pkg = bool(getattr(self.cfg, "dcc_use_package", True))
            return DCCGaussian(lambda_std=lambda_std, vol_model=vol_model, use_package=use_pkg)

        # Instancie le modèle Ledoit-Wolf ou ses variantes avec les paramètres de la config
        if method == "ledoit_wolf":
            from Modules.Financial_engineering.statistics.ledoit_wolf_module.ledoit_wolf import (LedoitWolfLinearShrinkage,LedoitWolfANLS,LedoitWolfOAS,LedoitWolfQIS)
            
            variant = str(getattr(self.cfg, "lw_variant", "lw_2004")).lower()
            window = getattr(self.cfg, "lw_window", None)  # None = full window
            demean = bool(getattr(self.cfg, "lw_demean", True))
            ddof = int(getattr(self.cfg, "lw_ddof", 0))
            use_package = bool(getattr(self.cfg, "use_package", True))

            if variant in {"lw_2004", "lw2004", "linear"}:
                return LedoitWolfLinearShrinkage(window=window, demean=demean, ddof=ddof,use_package=use_package)

            if variant in {"anls_2020", "anls2020", "nonlinear"}:
                chunk = int(getattr(self.cfg, "chunk_size", 1024))
                return LedoitWolfANLS(window=window, demean=demean, ddof=ddof, chunk_size=chunk,use_package=use_package)
            
            if variant in {"OAS", "oas", "oas_2012"}:
                return LedoitWolfOAS(window=window, demean=demean, ddof=ddof,use_package=use_package)
            
            if variant in {"QIS", "qis", "QIS_2022"}:
                chunk = int(getattr(self.cfg, "chunk_size", 1024))
                return LedoitWolfQIS(window=window, demean=demean, chunk_size=chunk)

            raise ValueError(f"Unknown lw_variant={variant!r}")

        raise ValueError(f"Unsupported covariance method={method!r}")


    def precompute_path(self, returns: pd.DataFrame) -> None:
        """
        Pré-calcule le path complet de covariances sur toute la série de rendements.
 
        Fit le modèle une fois sur l'univers complet, puis stocke le résultat
        (tableau H de shape (T, N, N), index temporel, noms des colonnes).
        Doit être appelé avant get_cov_at() ou get_cov_subset() en mode 'path'.
 
        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame de rendements (index = dates, colonnes = tickers), univers complet.
        """

        # Validation de la configuration
        if str(getattr(self.cfg, "compute_mode", "path")).lower() != "path":
            raise ValueError("This provider is configured for path mode only.")

        #Recupère les rendements nettoyés et normalisés
        R = self._ensure_df(returns)
        if R.empty:
            raise ValueError("returns is empty after cleaning.")
        
        # Resample si estimation hebdo demandée
        cov_data_freq = str(getattr(self.cfg, "cov_data_freq", "daily")).lower()
        if cov_data_freq == "weekly":
            R = (1 + R).resample("W-FRI").prod() - 1
            R = R.dropna(how="all")

        #Instancie le modèle et calcule le path complet de covariances
        model = self._build_multivol_model()
        path = model.fit(R).conditional_cov(R)

        # Stocke les références au path dans le provider
        self._full_cols = tuple(map(str, path.names))
        self._path_index = pd.DatetimeIndex(path.index)
        self._path_H = path.H
        self._covariance_path = path

    
    def get_cov_at(self, asof: pd.Timestamp) -> np.ndarray:
        """
        Retourne la matrice de covariance complète (N x N) à une date donnée.
 
        Cherche dans le path précomputé la dernière date disponible <= asof.
        Nécessite que precompute_path() ait été appelé au préalable.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date cible pour la covariance.
 
        Returns
        -------
        np.ndarray
            Matrice de covariance (N x N) correspondant à la date la plus récente <= asof.
        """

        # Validation de la configuration et des données du path
        if self._path_index is None or self._path_H is None:
            raise RuntimeError("Covariance path not precomputed. Call precompute_path() first.")

        # Convertit asof en Timestamp
        asof = pd.Timestamp(asof)

        #Recupère l'index temporel du path
        idx = self._path_index

        # Cherche la position de la dernière date <= asof
        pos = idx.searchsorted(asof, side="right") - 1
        if pos < 0:
            raise KeyError(f"No covariance available at or before {asof}. Path starts at {idx[0]}.")

        # Retourne la matrice de covariance correspondante à la position trouvée
        return self._path_H[pos]
    
    
    def get_cov_subset(self, asof: pd.Timestamp, sub_cols: list[str]) -> np.ndarray:
        """
        Extrait la sous-matrice de covariance pour un sous-ensemble d'actifs depuis le path.
 
        Récupère la covariance complète à la date asof puis slicke sur sub_cols.
        Garantit que le résultat est Cholesky-compatible avant de le retourner.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date cible pour la covariance.
        sub_cols : list[str]
            Liste des tickers à conserver (doit être un sous-ensemble de l'univers complet).
 
        Returns
        -------
        np.ndarray
            Sous-matrice de covariance (M x M) pour les M actifs de sub_cols.
        """

        # Récupère la matrice de covariance complète à la date asof
        S_full = self.get_cov_at(asof)

        # Validation de l'existence des colonnes demandées dans le path
        if self._full_cols is None:
            raise RuntimeError("full_cols not set.")

        #Intersection entre sub_cols demandées et full_cols du path, avec préservation de l'ordre de full_cols
        full_cols = list(self._full_cols)
        sub_cols = [str(c) for c in sub_cols]

        # Filtre sub_cols pour ne garder que ceux présents dans full_cols, en préservant l'ordre de full_cols
        sub_cols = [c for c in sub_cols if c in set(full_cols)]
        if len(sub_cols) < 2:
            raise ValueError("subset too small after intersection with full_cols.")

        # Trouve les indices des colonnes de sub_cols dans full_cols
        idx = [full_cols.index(c) for c in sub_cols]

        # Slicing de la matrice complète pour n'avoir que les lignes et colonnes correspondant à sub_cols
        S_sub = S_full[np.ix_(idx, idx)]

        # Garantit que la sous-matrice est symétrique et définie positive avant de la retourner
        return _ensure_cholesky_compatible(S_sub)
    
    
    def get_cov_rebal(self,returns_window: pd.DataFrame,sub_cols: list[str],) -> np.ndarray:
        """
        Calcule la covariance à la demande sur la fenêtre de rendements fournie (mode 'rebal').
 
        Appelle compute_cov_at_rebal() sur le modèle sous-jacent si disponible,
        puis extrait la sous-matrice pour sub_cols et garantit la compatibilité Cholesky.
 
        Parameters
        ----------
        returns_window : pd.DataFrame
            Fenêtre glissante de rendements disponibles à la date de rebalancement.
        sub_cols : list[str]
            Sous-ensemble de tickers investissables à cette date.
 
        Returns
        -------
        np.ndarray
            Sous-matrice de covariance (M x M) pour les M actifs de sub_cols.
        """

        # Récupère le paramètre de fréquence des données a fournir aux modèles
        cov_data_freq = str(getattr(self.cfg, "cov_data_freq", "daily")).lower()

        # Adapte la fréquence des rendements en fonction du paramètre
        rw = returns_window
        if cov_data_freq == "weekly":
            rw = (1 + rw).resample("W-FRI").prod() - 1
            rw = rw.dropna(how="all")

        # Validation de la présence de la méthode compute_cov_at_rebal dans le modèle
        all_cols = [str(c) for c in returns_window.columns]
        sub_cols_str = [str(c) for c in sub_cols]

        # Si le modèle supporte compute_cov_at_rebal
        if hasattr(self._estimator, "compute_cov_at_rebal"):

            # Appelle la méthode spécifique du modèle pour calculer la covariance à la date de rebalancement
            Sigma_full = self._estimator.compute_cov_at_rebal(returns_window=rw,all_cols=all_cols,)

            # Subset
            idx = [all_cols.index(c) for c in sub_cols_str if c in all_cols]
            if len(idx) < 2:
                raise ValueError("subset too small.")
            
            # Reconstruit la sous-matrice de covariance pour sub_cols
            S_sub = Sigma_full[np.ix_(idx, idx)]

            # Garantit que la sous-matrice est symétrique et définie positive avant de la retourner
            return _ensure_cholesky_compatible(S_sub)







