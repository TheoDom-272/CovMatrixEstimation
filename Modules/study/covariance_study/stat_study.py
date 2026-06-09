"""
Évaluation statistique des estimateurs de covariance sur données simulées.

Ce fichier implémente les deux DGP (Data Generating Processes) utilisés pour
comparer les estimateurs de covariance en dehors de tout contexte de backtest réel.
Les rendements sont entièrement synthétiques, les estimateurs sont évalués sur
des données dont on connaît la vraie matrice de covariance.

Deux DGP disponibles :
- static_oracle : covariance vraie fixe, T observations i.i.d.
- factor_shock  : loadings et variances idio suivent un AR(1), covariance vraie variable.

Classes
-------
StatSimConfig :
    Dataclass de configuration du Monte Carlo statistique (DGP, métriques, seeds, etc.).
StatSimResult :
    Résultat d'un scénario unique (un seed) : pertes matricielles par modèle.
StatEvalResult :
    Résultat agrégé sur n_scenarios seeds : moyenne et écart-type des pertes.
StatStudy :
    Classe principale contenant les méthodes de simulation et d'évaluation statistique.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from Modules.Financial_engineering.statistics.multivariate_vol_estimation import (MultiVolModel, RollingSampleCov, _safe_symmetrize, _chol_or_nearest_psd,)

if TYPE_CHECKING:
    pass


@dataclass
class StatSimConfig:
    """
    Classe contenant les paramètres de configuration de l'évaluation statistique
    des estimateurs de covariance sur données simulées.

    Attributes
    ----------
    dgp_type : str
        Type de DGP : 'static_oracle' (covariance fixe) ou 'factor_shock' (covariance variable).
    n_factors : int
        Nombre de facteurs dans la structure de covariance simulée.
    idio_strength : float
        Poids du risque idiosyncratique dans la covariance (static_oracle uniquement).
    target_daily_vol : float
        Volatilité journalière cible pour le rescaling de la covariance simulée.
    p_ratio : float or None
        Ratio N/T effectif pour le DGP static_oracle. Si None, T est dérivé de T_sim ou R_ref.
    rho_B : float
        Persistance AR(1) des loadings factoriels (factor_shock uniquement).
    sigma_B : float
        Amplitude des chocs sur les loadings (factor_shock uniquement).
    rho_d : float
        Persistance AR(1) des log-variances idiosyncratiques (factor_shock uniquement).
    sigma_d : float
        Amplitude des chocs sur les log-variances (factor_shock uniquement).
    rho : float
        Paramètre legacy conservé pour rétrocompatibilité.
    noise_scale : float
        Paramètre legacy conservé pour rétrocompatibilité.
    innovation : str
        Distribution des innovations : 'gaussian' ou 'student'.
    student_df : float
        Degrés de liberté de la loi de Student (si innovation='student').
    add_drift : bool
        Si True, ajoute une dérive non nulle aux rendements simulés.
    metrics : tuple
        Métriques de perte à calculer : 'frobenius', 'spectral', 'stein', 'precision'.
    n_scenarios : int
        Nombre de scénarios Monte Carlo (seeds indépendants).
    random_state : int
        Graine de base ; le seed du scénario i vaut random_state + i.
    T_sim : int or None
        Longueur de la série simulée. Si None, utilise len(R_ref).
    N_sim : int or None
        Nombre d'actifs simulés. Si None, utilise len(R_ref.columns).
    burn_in : int
        Nombre de périodes de chauffe exclues du calcul des pertes.
    """

    # Choix du DGP
    dgp_type: str = "factor_shock"   # "static_oracle" ou "factor_shock"

    # Structure factorielle (commune aux deux DGP)
    n_factors:        int   = 10
    idio_strength:    float = 0.30    # utilisé par static_oracle uniquement
    target_daily_vol: float = 0.01

    # Paramètres static_oracle
    p_ratio: Optional[float] = None   # N/T effectif

    # Paramètres factor_shock
    rho_B:   float = 0.95    # persistance AR(1) des loadings
    sigma_B: float = 0.05    # amplitude des chocs loadings
    rho_d:   float = 0.97    # persistance AR(1) des log-variances idio
    sigma_d: float = 0.10    # amplitude des chocs log-variances idio

    # Paramètres legacy (conservés pour rétrocompatibilité)
    rho:         float = 0.995
    noise_scale: float = 0.02

    # Innovation
    innovation: str   = "gaussian"   # "gaussian" ou "student"
    student_df: float = 5.0
    add_drift:  bool  = False

    # Métriques
    metrics: tuple = ("frobenius", "spectral", "stein")

    # Monte Carlo
    n_scenarios:  int           = 1
    random_state: int           = 42
    T_sim:        Optional[int] = None
    N_sim:        Optional[int] = None
    burn_in:      int           = 60


@dataclass
class StatSimResult:
    """
    Classe contenant le résultat d'un scénario de simulation unique (un seed).

    Attributes
    ----------
    loss_summary : pd.DataFrame
        DataFrame (index=modèle, colonnes=métriques) avec les pertes moyennées sur T.
    dgp_params : dict
        Paramètres du DGP utilisés pour ce scénario (N, T, innovation, etc.).
    seed : int
        Graine aléatoire utilisée pour ce scénario.
    """

    loss_summary: pd.DataFrame    # index=model, cols=métriques — moyenne sur T
    dgp_params:   Dict[str, Any]  # paramètres du DGP utilisés
    seed:         int


@dataclass
class StatEvalResult:
    """
    Classe contenant les résultats agrégés sur n_scenarios seeds indépendants.

    Attributes
    ----------
    loss_mean : pd.DataFrame
        Moyenne des pertes inter-scénarios (index=modèle, colonnes=métriques).
    loss_std : pd.DataFrame
        Écart-type des pertes inter-scénarios (index=modèle, colonnes=métriques).
    loss_all : list of StatSimResult
        Liste des résultats individuels par scénario pour analyse détaillée.
    cfg : StatSimConfig
        Configuration utilisée pour la simulation.
    """

    loss_mean: pd.DataFrame
    loss_std:  pd.DataFrame
    loss_all:  List[StatSimResult]
    cfg:       StatSimConfig



class StatStudy:
    """
    Classe contenant les méthodes de simulation statistique et d'évaluation
    des estimateurs de covariance sur données synthétiques.

    Les rendements sont entièrement simulés depuis un DGP connu, ce qui permet
    de mesurer l'erreur d'estimation par rapport à la vraie covariance.

    Methods
    -------
    run_stat_evaluation(R_ref, model_specs, cfg, exporter) -> StatEvalResult :
        Lance cfg.n_scenarios scénarios indépendants et agrège les résultats.
    """

    @staticmethod
    def _build_stats_model_from_cov_cfg(cov_cfg) -> MultiVolModel:
        """
        Construit une instance de MultiVolModel depuis une configuration cov_cfg.

        Dispatche vers la bonne classe d'estimateur selon cov_cfg.method.
        Les imports sont locaux pour éviter les imports circulaires.

        Parameters
        ----------
        cov_cfg : CovarianceProviderConfig
            Configuration de l'estimateur de covariance.

        Returns
        -------
        MultiVolModel
            Instance de l'estimateur correspondant à la méthode configurée.

        Raises
        ------
        ValueError
            Si la méthode ou la variante Ledoit-Wolf est inconnue.
        """

        # recupere la methode en minuscule pour la comparaison
        method = str(getattr(cov_cfg, "method", "")).lower()

        # Rolling Sample Covariance : estimateur classique à fenêtre glissante
        if method == "rolling":

            # Récupère les paramètres de la configuration rolling
            window = int(getattr(cov_cfg, "rolling_window", 252))
            ddof   = int(getattr(cov_cfg, "rolling_ddof", 1))
            return RollingSampleCov(window=window, ddof=ddof)

        # EWMA de RiskMetrics/Engle : modèle à mémoire exponentielle, adapté aux données financières
        if method == "ewma":
            
            # Import local pour éviter un import circulaire au chargement du module
            from Modules.Financial_engineering.statistics.Engle.EWMACov import EWMACov
            window      = int(getattr(cov_cfg, "rolling_window", 252))
            tune_lambda = bool(getattr(cov_cfg, "tune_lambda", False))
            lmbda       = float(getattr(cov_cfg, "ewma_lambda", 0.94))
            init        = str(getattr(cov_cfg, "ewma_init", "scov"))
            return EWMACov(lmbda=lmbda, init=init, tune_lambda=tune_lambda, window=window)

        # DCC de Engle (Dynamic Conditional Correlation) : modèle à structure factorielle dynamique
        if method == "dcc":
            from Modules.Financial_engineering.statistics.Engle.DCC import DCCGaussian
            lambda_std         = float(getattr(cov_cfg, "dcc_lambda_std", 0.94))
            vol_model          = str(getattr(cov_cfg, "dcc_vol_model", "garch"))
            refit_enabled      = bool(getattr(cov_cfg, "dcc_refit_enabled", False))
            refit_lookback     = int(getattr(cov_cfg, "dcc_refit_lookback", 252))
            refit_mode         = str(getattr(cov_cfg, "dcc_refit_mode", "rolling"))
            refit_every        = getattr(cov_cfg, "dcc_refit_every", None)
            refit_min_obs      = int(getattr(cov_cfg, "dcc_refit_min_obs", 60))
            include_refit_date = bool(getattr(cov_cfg, "dcc_include_refit_date_in_forecast", True))
            refit_dates        = getattr(cov_cfg, "dcc_refit_dates", None)
            return DCCGaussian(
                lambda_std=lambda_std, vol_model=vol_model,
                refit_enabled=refit_enabled, refit_lookback=refit_lookback,
                refit_mode=refit_mode, refit_every=refit_every,
                refit_dates=refit_dates, refit_min_obs=refit_min_obs,
                include_refit_date_in_forecast=include_refit_date,
            )

        # Ledoit-Wolf : famille d'estimateurs à shrinkage linéaire ou non linéaire vers une matrice cible, avec différentes variantes proposées dans la littérature
        if method == "ledoit_wolf":
            from Modules.Financial_engineering.statistics.ledoit_wolf_module.ledoit_wolf import (LedoitWolfLinearShrinkage, LedoitWolfANLS, LedoitWolfOAS, LedoitWolfQIS,)

            variant     = str(getattr(cov_cfg, "lw_variant", "lw_2004")).lower()
            window      = getattr(cov_cfg, "lw_window", None)
            demean      = bool(getattr(cov_cfg, "lw_demean", True))
            ddof        = int(getattr(cov_cfg, "lw_ddof", 0))
            use_package = bool(getattr(cov_cfg, "use_package", True))

            # Dispatch vers la bonne variante de Ledoit-Wolf selon la configuration
            if variant in {"lw_2004", "lw2004", "linear"}:
                return LedoitWolfLinearShrinkage(window=window, demean=demean, ddof=ddof, use_package=use_package)
            if variant in {"anls_2020", "anls2020", "nonlinear"}:
                chunk = int(getattr(cov_cfg, "chunk_size", 1024))
                return LedoitWolfANLS(window=window, demean=demean, ddof=ddof, chunk_size=chunk, use_package=use_package)
            if variant in {"OAS", "oas", "oas_2012"}:
                return LedoitWolfOAS(window=window, demean=demean, ddof=ddof, use_package=use_package)
            if variant in {"QIS", "qis", "QIS_2022"}:
                chunk = int(getattr(cov_cfg, "chunk_size", 1024))
                return LedoitWolfQIS(window=window, demean=demean, chunk_size=chunk)
            raise ValueError(f"Unknown lw_variant={variant!r}")

        raise ValueError(f"Unsupported cov_cfg.method={method!r}")

    @staticmethod
    def _make_random_spd_cov(N: int, rng: np.random.Generator, target_daily_vol: float = 0.01, n_factors: int = 3, idio_strength: float = 0.30,) -> np.ndarray:
        """
        Construit une matrice de covariance SPD aléatoire à structure factorielle.

        Parameters
        ----------
        N : int
            Nombre d'actifs.
        rng : np.random.Generator
            Générateur de nombres aléatoires.
        target_daily_vol : float
            Volatilité journalière cible pour le rescaling.
        n_factors : int
            Nombre de facteurs dans la structure de covariance.
        idio_strength : float
            Poids du risque idiosyncratique (0 = purement factoriel).

        Returns
        -------
        np.ndarray
            Matrice de covariance SPD de forme (N, N).
        """

        # Validation des paramètres
        if N <= 0:
            raise ValueError("N doit être > 0.")

        # Nombre de facteurs effectif (au moins 1)
        n_factors = max(1, int(n_factors))

        # Loadings aléatoires des facteurs : chaque actif est exposé à K facteurs
        B = rng.normal(loc=0.0, scale=1.0, size=(N, n_factors))

        # Variances idiosyncratiques positives tirées d'une lognormale
        idio_var = rng.lognormal(mean=0.0, sigma=0.5, size=N)

        # Structure factorielle : B * B' donne la composante commune
        str_corr = (B @ B.T)

        # Mélange factoriel + idiosyncratique selon idio_strength
        Sigma = str_corr + (idio_strength * np.diag(idio_var))

        # Symétrise et stabilise pour garantir SPD
        Sigma = _safe_symmetrize(Sigma)
        Sigma = Sigma + 1e-8 * np.eye(N)

        # Rescale pour viser target_daily_vol via la moyenne des variances diagonales
        current_var = float(np.mean(np.diag(Sigma)))
        target_var  = float(target_daily_vol ** 2)
        if current_var <= 0:
            raise ValueError("current_var <= 0: Sigma mal construite.")
        scale = target_var / current_var
        Sigma = Sigma * scale

        return _safe_symmetrize(Sigma)

    @staticmethod
    def _burnin_mean_from_sigma0(Sigma0: np.ndarray, rng: np.random.Generator, burn_in: int = 60,) -> np.ndarray:
        """
        Génère un échantillon de chauffe depuis Sigma0 et retourne la moyenne empirique.

        Utilisé pour initialiser la dérive mu quand add_drift=True.

        Parameters
        ----------
        Sigma0 : np.ndarray
            Matrice de covariance initiale de forme (N, N).
        rng : np.random.Generator
            Générateur de nombres aléatoires.
        burn_in : int
            Nombre d'observations de chauffe.

        Returns
        -------
        np.ndarray
            Vecteur de dérive de forme (N,).
        """
        N  = Sigma0.shape[0]
        L0 = _chol_or_nearest_psd(_safe_symmetrize(Sigma0))
        Z  = rng.standard_normal((burn_in, N))

        # Génère les rendements de chauffe par transformation de Cholesky
        R0 = Z @ L0.T
        mu = R0.mean(axis=0)
        return mu

    @staticmethod
    def _sample_innovation(rng: np.random.Generator, N: int, cfg: StatSimConfig,) -> np.ndarray:

        """
        Tire un vecteur d'innovation de taille N selon la distribution configurée.

        Deux distributions disponibles :
        - gaussian : hypothèse classique, queues fines.
        - student  : via mélange gaussien/chi2, queues épaisses réalistes pour données financières.

        Parameters
        ----------
        rng : np.random.Generator
            Générateur de nombres aléatoires.
        N : int
            Dimension du vecteur d'innovation.
        cfg : StatSimConfig
            Configuration contenant le type d'innovation et les degrés de liberté.

        Returns
        -------
        np.ndarray
            Vecteur d'innovation de forme (N,).
        """
        #Si l'innovation est de type Student, on utilise un mélange gaussien/chi2 pour simuler une Student multivariée
        if cfg.innovation == "student":
            nu = float(cfg.student_df)
            if nu <= 2.0:
                raise ValueError("student_df doit être > 2 pour avoir une variance finie.")
            
            # Mélange gaussien/chi2 pour simuler une Student multivariée
            chi2_draw = rng.chisquare(nu) / nu
            z = rng.standard_normal(N) / np.sqrt(chi2_draw)

        else:
            # Innovation gaussienne standard
            z = rng.standard_normal(N)
        return z

    @staticmethod
    def _stein_loss(S_hat: np.ndarray, S_true: np.ndarray) -> float:
        """
        Calcule la perte de Stein (entropique) entre S_hat et S_true.

        C'est le critère théorique d'optimalité de Ledoit-Wolf.
        Invariant aux transformations linéaires, pénalise les erreurs sur tout le spectre,
        y compris les petites valeurs propres (contrairement à Frobenius).

        Parameters
        ----------
        S_hat : np.ndarray
            Matrice de covariance estimée.
        S_true : np.ndarray
            Matrice de covariance vraie.

        Returns
        -------
        float
            Valeur de la perte de Stein. NaN si l'une des matrices est singulière.
        """
        
        N  = S_true.shape[0]
        eps = 1e-10 * np.eye(N)

        try:
            # Symétrisation
            S_true_s = _safe_symmetrize(S_true) + eps
            S_hat_s  = _safe_symmetrize(S_hat)  + eps

            # Cholesky
            L_true = np.linalg.cholesky(S_true_s)
            L_hat  = np.linalg.cholesky(S_hat_s)

            # Calcul de la matrice de transformation
            M = np.linalg.solve(L_true, L_hat)
            trace_term = float(np.sum(M * M))

            # Log-déterminants via la diagonale de Cholesky (numériquement stable)
            logdet_true = 2.0 * float(np.sum(np.log(np.diag(L_true))))
            logdet_hat = 2.0 * float(np.sum(np.log(np.diag(L_hat))))
            logdet_ratio = logdet_hat - logdet_true

            return float(trace_term - logdet_ratio - N)
        except np.linalg.LinAlgError:
            return float("nan")

    @staticmethod
    def _precision_loss(S_hat: np.ndarray, S_true: np.ndarray) -> float:
        """
        Calcule la perte de Frobenius sur les matrices de précision.

        Pertinent pour l'optimisation de portefeuille car les poids optimaux
        dépendent directement de la matrice de précision (inverse de la covariance).

        Parameters
        ----------
        S_hat : np.ndarray
            Matrice de covariance estimée.
        S_true : np.ndarray
            Matrice de covariance vraie.

        Returns
        -------
        float
            Norme de Frobenius de (P_hat - P_true). NaN si l'une des matrices est singulière.
        """

        N = S_true.shape[0]
        eps = 1e-10 * np.eye(N)

        try:
            # Inversion des matrices symétrisées
            P_hat  = np.linalg.inv(_safe_symmetrize(S_hat)  + eps)
            P_true = np.linalg.inv(_safe_symmetrize(S_true) + eps)

            # Calcul de la différence de précision
            diff   = P_hat - P_true
            return float(np.sqrt(np.sum(diff * diff)))
        
        except np.linalg.LinAlgError:
            return float("nan")

    @classmethod
    def _compute_losses_on_aligned_paths(cls, H_true: np.ndarray, H_est: np.ndarray, cfg: StatSimConfig, burn_in: int = 0,) -> Dict[str, float]:
        """
        Calcule la moyenne des métriques de perte sur la fenêtre [burn_in:T].

        Parameters
        ----------
        H_true : np.ndarray
            Tableau de covariances vraies de forme (T, N, N).
        H_est : np.ndarray
            Tableau de covariances estimées de forme (T, N, N).
        cfg : StatSimConfig
            Configuration contenant la liste des métriques à calculer.
        burn_in : int
            Nombre de périodes initiales à exclure du calcul.

        Returns
        -------
        dict
            Dictionnaire métrique -> valeur moyenne sur [burn_in:T].
        """

        #Recupere le nombre de dates T à partir de H_true, et définit la date de début en fonction du burn_in
        T = H_true.shape[0]
        start = min(burn_in, max(T - 1, 0))

        # Initialise un accumulateur pour chaque métrique demandée
        accum: Dict[str, List[float]] = {m: [] for m in cfg.metrics}

        #Itére sur les dates à partir de start, calcule les métriques pour chaque date et accumulateur
        for t in range(start, T):

            # Symétrisation sécurisée des matrices de covariance
            S_true = _safe_symmetrize(H_true[t])
            S_est  = _safe_symmetrize(H_est[t])

            # Si frobenius est demandé, calcule la norme de Frobenius de la différence S_est - S_true
            if "frobenius" in cfg.metrics:
                diff = S_est - S_true
                accum["frobenius"].append(float(np.sqrt(np.sum(diff * diff))))

            # Si spectral est demandé, calcule la plus grande valeur propre en valeur absolue de la différence S_est - S_true
            if "spectral" in cfg.metrics:
                diff = S_est - S_true
                eigv = np.linalg.eigvalsh(diff)
                accum["spectral"].append(float(np.max(np.abs(eigv))))

            # Si stein est demandé, calcule la perte de Stein entre S_est et S_true
            if "stein" in cfg.metrics:
                accum["stein"].append(cls._stein_loss(S_est, S_true))

            #Si precision est demandé, calcule la perte de précision entre S_est et S_true
            if "precision" in cfg.metrics:
                accum["precision"].append(cls._precision_loss(S_est, S_true))

        # Moyenne des pertes sur toutes les dates (NaN ignorés)
        return {m: float(np.nanmean(v)) if v else float("nan") for m, v in accum.items()}

    @classmethod
    def _run_one_scenario(cls, R_ref: pd.DataFrame, model_specs: List[Any], cfg: StatSimConfig, seed: int,) -> StatSimResult:
        """
        Dispatcher : choisit le DGP selon cfg.dgp_type et lance le scénario.

        R_ref sert uniquement à fournir N, les noms d'actifs et l'index de dates.

        Parameters
        ----------
        R_ref : pd.DataFrame
            DataFrame de référence pour les dimensions et l'index de dates.
        model_specs : list
            Liste de ModelSpec à évaluer.
        cfg : StatSimConfig
            Configuration de la simulation.
        seed : int
            Graine aléatoire pour ce scénario.

        Returns
        -------
        StatSimResult
            Résultat du scénario avec les pertes par modèle.
        """

        #Si le DGP est de type static_oracle, on lance le scénario correspondant, sinon on lance le scénario factor_shock
        if cfg.dgp_type == "static_oracle":
            return cls._run_scenario_static_oracle(R_ref, model_specs, cfg, seed)
        else:
            return cls._run_scenario_factor_shock(R_ref, model_specs, cfg, seed)

    @classmethod
    def _run_scenario_static_oracle(cls, R_ref: pd.DataFrame, model_specs: List[Any], cfg: StatSimConfig, seed: int,) -> StatSimResult:
        """
        DGP Static Oracle : covariance vraie fixe, T observations i.i.d.

        on tire une cov_true unique,on simule T observations i.i.d. depuis elle, chaque estimateur
        produit une cov estimate, et on mesure l'erreur.

        Parameters
        ----------
        R_ref : pd.DataFrame
            DataFrame de référence pour N, noms d'actifs et index de dates.
        model_specs : list
            Liste de ModelSpec à évaluer.
        cfg : StatSimConfig
            Configuration de la simulation.
        seed : int
            Graine aléatoire pour ce scénario.

        Returns
        -------
        StatSimResult
            Pertes par modèle et paramètres du DGP utilisés.
        """

        rng = np.random.default_rng(seed)

        # Détermine les dimensions N et T
        idx_ref = pd.DatetimeIndex(pd.to_datetime(R_ref.index)).sort_values()
        
        # Si N_sim est fourni, on utilise N_sim et des noms d'actifs génériques, sinon on dérive N et les noms d'actifs de R_ref
        if cfg.N_sim is not None:
            N     = int(cfg.N_sim)
            names = tuple(f"A{i:04d}" for i in range(N))
        else:
            N     = len(R_ref.columns)
            names = tuple(map(str, R_ref.columns))

        # T piloté par p_ratio si fourni, sinon par T_sim, sinon par R_ref
        if cfg.p_ratio is not None:

            # Le burn_in permet de tester les modèles fonctionnant avec des paths (EWMA, DCC, etc.)
            T = max(N + cfg.burn_in + 10, int(round(N / cfg.p_ratio)))
        
        #Si p_ratio n'est pas fourni, on utilise T_sim si il est fourni, sinon on utilise len(idx_ref)
        elif cfg.T_sim is not None:
            T = int(cfg.T_sim)

        #Si ni p_ratio ni T_sim ne sont fournis, on utilise len(idx_ref) pour T
        else:
            T = len(idx_ref)

        # Construit l'index de dates synthétique de longueur T
        if T <= len(idx_ref):
            idx = idx_ref[:T]
        else:
            # Étend l'index avec des dates ouvrables si T dépasse R_ref
            extra = pd.bdate_range(start=idx_ref[-1] + pd.Timedelta(days=1), periods=T - len(idx_ref))
            idx = pd.DatetimeIndex(list(idx_ref) + list(extra))

        # Tire une covariance vraie unique pour tout le scénario
        Sigma_true = cls._make_random_spd_cov(N=N, rng=rng, target_daily_vol=cfg.target_daily_vol, n_factors=cfg.n_factors, idio_strength=cfg.idio_strength,)
        L_true = _chol_or_nearest_psd(Sigma_true)

        # Dérive : zéro par défaut, sinon estimée depuis un burn-in
        if cfg.add_drift:

            # Si add_drift=True, on génère un burn-in de rendements i.i.d. depuis Sigma_true et on utilise la moyenne empirique comme dérive
            mu = cls._burnin_mean_from_sigma0(Sigma0=Sigma_true, rng=rng, burn_in=cfg.burn_in)
        else:

            #Sinon, on utilise une dérive nulle
            mu = np.zeros(N, dtype=float)

        # Initialise un tableau pour stocker les rendements simulés de forme (T, N)
        R_arr = np.empty((T, N), dtype=float)

        #Itére sur les dates, tire un vecteur d'innovation selon la distribution configurée, et génère les rendements par transformation de Cholesky
        for t in range(T):

            # Innovation de taille N tirée selon la distribution configurée (gaussian ou student)
            z = cls._sample_innovation(rng, N, cfg)

            #Génère les rendements de la date t par transformation de Cholesky : R_t = mu + L_true @ z
            R_arr[t] = mu + L_true @ z

        # Convertit le tableau de rendements en DataFrame avec l'index de dates et les noms d'actifs
        R_sim = pd.DataFrame(R_arr, index=idx, columns=list(names))

        # Fonctions de détection du type de modèle pour le dispatch d'estimation
        def _is_ewma_model(mdl: MultiVolModel) -> bool:
            return mdl.__class__.__name__.lower().startswith("ewma")

        def _is_rolling_model(mdl: MultiVolModel) -> bool:
            return isinstance(mdl, RollingSampleCov)

        # Évalue chaque estimateur sur les données simulées
        loss_rows: Dict[str, Dict[str, float]] = {}

        #Itére sur les modèles à évaluer
        for spec in model_specs:

            #Récupère le nom du modèle depuis spec.name ou utilise str(spec) par défaut
            name = getattr(spec, "name", str(spec))

            # Récupère la configuration de covariance cov_cfg depuis spec.cov_cfg, ou None par défaut
            cov_cfg = getattr(spec, "cov_cfg", None)

            # Si cov_cfg est None, on remplit les pertes avec NaN et on continue au modèle suivant
            if cov_cfg is None:
                loss_rows[name] = {m: float("nan") for m in cfg.metrics}
                continue
            
            
            try:

                # Construit le modèle d'estimation de covariance depuis la configuration cov_cfg
                mdl = cls._build_stats_model_from_cov_cfg(cov_cfg)

                # si le modèle a un attribut _Sigma_full, on le réinitialise à None pour éviter les fuites de mémoire entre les modèles
                if hasattr(mdl, "_Sigma_full"):
                    mdl._Sigma_full = None

                # Si le modèle est de type EWMA, on utilise le path complet avec burn-in pour l'estimation
                if _is_ewma_model(mdl):
                    
                    #Fit du modèle sur le path complet pour obtenir l'estimation de la covariance à chaque date
                    mdl_fit  = mdl.fit(R_sim)

                    #Récupère les covariances estimées à chaque date depuis le path du modèle, en alignant avec les covariances vraies
                    path_est = mdl_fit.conditional_cov(R_sim)

                    # Le burn_in est appliqué pour aligner les paths d'estimation et de vérité terrain, en excluant les périodes de chauffe initiales du calcul des pertes
                    burn   = min(cfg.burn_in, path_est.H.shape[0] - 1)

                    # Récupère les covariances estimées à partir du path du modèle, en alignant avec les covariances vraies et en appliquant le burn_in
                    H_est  = np.array(path_est.H[burn:], copy=True)
                    H_true = np.repeat(Sigma_true[None, :, :], H_est.shape[0], axis=0)

                    # Ferme le path du modèle pour libérer la mémoire, et supprime la référence à path_est pour éviter les fuites de mémoire
                    path_est.close()
                    del path_est

                # Si le modèle est de type Rolling, on utilise la fenêtre glissante pour estimer une covariance statique à chaque date de rebalancement
                elif _is_rolling_model(mdl):
                    # Rolling : estimation statique unique via compute_cov_at_rebal
                    T_eff  = int(round(N / cfg.p_ratio))
                    R_eff = R_sim.iloc[-T_eff:]
                    Sigma_hat = mdl.compute_cov_at_rebal(returns_window=R_eff, all_cols=list(R_sim.columns))
                    H_est = Sigma_hat[None, :, :]
                    H_true = Sigma_true[None, :, :]

                # Autres modèles statiques : estimation sur la fenêtre complète
                else:
                    # Modèles statiques (LW, ANLS, QIS, OAS) : estimation sur la fenêtre effective
                    T_eff = int(round(N / cfg.p_ratio))
                    R_eff = R_sim.iloc[-T_eff:]

                    mdl.window = None
                    mdl_fit = mdl.fit(R_eff)
                    Sigma_hat = mdl_fit._Sigma_full
                    H_est = Sigma_hat[None, :, :]
                    H_true = Sigma_true[None, :, :]

                # Calcule les pertes sur le path aligné
                loss_rows[name] = cls._compute_losses_on_aligned_paths(H_true, H_est, cfg, burn_in=0)

            except Exception as e:
                print(f"  [StatSim/static_oracle] Erreur {name} seed={seed}: {e}")
                loss_rows[name] = {m: float("nan") for m in cfg.metrics}

        gc.collect()

        # Convertit les pertes en DataFrame avec les modèles en index et les métriques en colonnes
        loss_summary = pd.DataFrame(loss_rows).T
        loss_summary.index.name = "model"

        return StatSimResult(
            loss_summary=loss_summary,
            dgp_params={"dgp_type":"static_oracle", "N": N,"T": T, "p_ratio": round(N / T, 4), "n_factors": cfg.n_factors,"innovation": cfg.innovation,"add_drift":  cfg.add_drift,},
            seed=seed,)

    @classmethod
    def _run_scenario_factor_shock(cls, R_ref: pd.DataFrame,  model_specs: List[Any],cfg: StatSimConfig, seed: int,) -> StatSimResult:
        """
        DGP Factor-Shock : loadings et variances idio suivent un AR(1).

        La covariance vraie varie dans le temps selon des processus AR(1).
        La perte est la moyenne temporelle des erreurs d'estimation.

        Parameters
        ----------
        R_ref : pd.DataFrame
            DataFrame de référence pour N, noms d'actifs et index de dates.
        model_specs : list
            Liste de ModelSpec à évaluer.
        cfg : StatSimConfig
            Configuration de la simulation.
        seed : int
            Graine aléatoire pour ce scénario.

        Returns
        -------
        StatSimResult
            Pertes par modèle et paramètres du DGP utilisés.
        """

        # Générateur de nombres aléatoires avec la graine du scénario
        rng = np.random.default_rng(seed)

        # Détermine T effectif depuis T_sim ou R_ref
        idx_ref   = pd.DatetimeIndex(pd.to_datetime(R_ref.index)).sort_values()
        T_sim_eff = cfg.T_sim if cfg.T_sim is not None else len(idx_ref)
        T_sim_eff = min(T_sim_eff, len(idx_ref))

        # Construit l'index de dates synthétique de longueur T_sim_eff, en étendant avec des dates ouvrables si nécessaire
        if cfg.T_sim is not None and cfg.T_sim > len(idx_ref):

            # Étend l'index avec des dates ouvrables si T_sim dépasse R_ref
            extra = pd.bdate_range(start=idx_ref[-1] + pd.Timedelta(days=1), periods=cfg.T_sim - len(idx_ref),)
            idx = pd.DatetimeIndex(list(idx_ref) + list(extra))
            T_sim_eff = cfg.T_sim
        else:
            idx = idx_ref[:T_sim_eff]

        # Détermine N et les noms d'actifs
        if cfg.N_sim is not None:
            N     = int(cfg.N_sim)
            names = tuple(f"A{i:04d}" for i in range(N))
        else:
            N     = len(R_ref.columns)
            names = tuple(map(str, R_ref.columns))

        # Nombre de facteurs effectif
        K = max(1, int(cfg.n_factors))

        # Initialisation des états AR(1) : loadings et log-variances idio
        B_prev  = rng.normal(0.0, 1.0 / np.sqrt(K), size=(N, K))
        ld_prev = rng.normal(0.0, 1.0, size=N)

        # Coefficients des innovations AR(1) pour garantir la stationnarité
        coef_B = np.sqrt(max(0.0, 1.0 - cfg.rho_B ** 2)) * cfg.sigma_B
        coef_d = np.sqrt(max(0.0, 1.0 - cfg.rho_d ** 2)) * cfg.sigma_d

        # Tableaux pour stocker les rendements simulés et les covariances vraies à chaque date
        R_arr = np.empty((T_sim_eff, N), dtype=float)
        H_arr = np.empty((T_sim_eff, N, N), dtype=float)

        # Initialise la dérive mu
        if cfg.add_drift:
            # La dérive est estimée depuis un burn-in de rendements simulés avec la covariance initiale Sigma_burn, qui est construite à partir des loadings et log-variances initiaux.
            Sigma_burn = _safe_symmetrize(B_prev @ B_prev.T + np.diag(np.exp(ld_prev))) + 1e-8 * np.eye(N)
            mu = cls._burnin_mean_from_sigma0(Sigma0=Sigma_burn, rng=rng, burn_in=cfg.burn_in)
        else:
            mu = np.zeros(N, dtype=float)

        # Boucle temporelle : met à jour la covariance vraie et simule les rendements
        for t in range(T_sim_eff):

            # Mise à jour des loadings : AR(1) avec innovation gaussienne
            eps_B = rng.normal(0.0, 1.0, size=(N, K))
            B_t   = cfg.rho_B * B_prev + coef_B * eps_B

            # Mise à jour des log-variances idio : AR(1) avec innovation gaussienne
            eps_d = rng.normal(0.0, 1.0, size=N)
            ld_t  = cfg.rho_d * ld_prev + coef_d * eps_d

            # Covariance vraie au temps t : structure factorielle + risque idio
            Sigma_t  = _safe_symmetrize(B_t @ B_t.T + np.diag(np.exp(ld_t)))
            H_arr[t] = Sigma_t

            # Simulation des rendements depuis la vraie covariance au temps t
            z_t = cls._sample_innovation(rng, N, cfg)
            L_t = _chol_or_nearest_psd(Sigma_t)
            R_arr[t] = mu + L_t @ z_t

            B_prev  = B_t
            ld_prev = ld_t

        # Rescale global pour respecter target_daily_vol
        mean_var = float(np.mean(H_arr[:, np.arange(N), np.arange(N)]))

        if mean_var > 0:
            scale  = (cfg.target_daily_vol ** 2) / mean_var
            H_arr *= scale
            R_arr *= np.sqrt(scale)

        # Convertit les rendements simulés en DataFrame avec l'index de dates et les noms d'actifs
        R_sim = pd.DataFrame(R_arr, index=idx, columns=list(names))

        # Évalue chaque estimateur sur les données simulées
        loss_rows: Dict[str, Dict[str, float]] = {}

        #Itére sur les modèles à évaluer
        for spec in model_specs:

            #Récupère le nom du modèle depuis spec.name ou utilise str(spec) par défaut
            name = getattr(spec, "name", str(spec))
            cov_cfg = getattr(spec, "cov_cfg", None)
            if cov_cfg is None:
                loss_rows[name] = {m: float("nan") for m in cfg.metrics}
                continue

            try:
                # Calcule le path de covariance estimé sur les rendements simulés
                mdl  = cls._build_stats_model_from_cov_cfg(cov_cfg)
                mdl_fit  = mdl.fit(R_sim)
                path_est = mdl_fit.conditional_cov(R_sim)

                # Aligne les indices entre le path vrai et le path estimé
                common_idx = idx.intersection(path_est.index)
                if len(common_idx) == 0:
                    loss_rows[name] = {m: float("nan") for m in cfg.metrics}
                    path_est.close()
                    continue

                # Récupère les positions des dates communes dans les deux paths pour aligner les covariances vraies et estimées
                pos_t = np.array([idx.get_loc(d) for d in common_idx])
                pos_e = path_est.index.get_indexer(common_idx)

                # Le burn_in est appliqué pour aligner les paths d'estimation et de vérité terrain, en excluant les périodes de chauffe initiales du calcul des pertes
                burn = min(cfg.burn_in, len(common_idx) - 1)
                loss_rows[name] = cls._compute_losses_on_aligned_paths(H_arr[pos_t], path_est.H[pos_e], cfg, burn_in=burn)
                path_est.close()

            except Exception as e:
                print(f"  [StatSim/factor_shock] Erreur {name} seed={seed}: {e}")
                loss_rows[name] = {m: float("nan") for m in cfg.metrics}

        gc.collect()

        # Convertit les pertes en DataFrame avec les modèles en index et les métriques en colonnes
        loss_summary = pd.DataFrame(loss_rows).T
        loss_summary.index.name = "model"

        return StatSimResult(
            loss_summary=loss_summary,
            dgp_params={"dgp_type": "factor_shock","N":  N, "K": K,"T_sim": T_sim_eff,"rho_B": cfg.rho_B,"sigma_B": cfg.sigma_B,"rho_d": cfg.rho_d,
                "sigma_d":    cfg.sigma_d,"innovation": cfg.innovation,  "add_drift":  cfg.add_drift,}, seed=seed, )

    @classmethod
    def run_stat_evaluation(cls, R_ref: pd.DataFrame, model_specs: List[Any], cfg: StatSimConfig, exporter=None,) -> StatEvalResult:
        """
        Lance cfg.n_scenarios scénarios indépendants et agrège les résultats.

        Supporte un exporter optionnel pour le checkpoint et l'écriture au fil de l'eau.
        Si tous les modèles d'un seed sont déjà dans le checkpoint, le seed est skippé.

        Parameters
        ----------
        R_ref : pd.DataFrame
            DataFrame de référence pour N, noms d'actifs et index de dates.
        model_specs : list
            Liste de ModelSpec à évaluer.
        cfg : StatSimConfig
            Configuration de la simulation.
        exporter : StatMonteCarloExporter or None
            Exporter pour le checkpoint et l'écriture des résultats au fil de l'eau.

        Returns
        -------
        StatEvalResult
            Résultats agrégés (moyenne et écart-type des pertes sur tous les scénarios).
        """
        import time

        print(
            f"[StatSim] {cfg.n_scenarios} scénario(s) | "
            f"DGP={cfg.dgp_type} | N_factors={cfg.n_factors} | "
            f"innovation={cfg.innovation} | "
            f"métriques={list(cfg.metrics)}"
        )

        all_results: List[StatSimResult] = []

        for i in range(cfg.n_scenarios):
            seed = cfg.random_state + i

            # Vérifie le checkpoint si un exporter est fourni
            if exporter is not None:
                from Modules.Financial_engineering.Export.Montecarlo_cov_export import StatScenarioKey

                def _make_key(spec, seed_val):
                    """Construit la clé de checkpoint pour un modèle et un seed donné."""

                    # Retourne une clé unique pour ce modèle et ce seed, en incluant les paramètres du DGP pour différencier les scénarios
                    return StatScenarioKey(
                        model_name = getattr(spec, "name", str(spec)),
                        dgp_type   = cfg.dgp_type,
                        N_sim  = cfg.N_sim if cfg.N_sim is not None else len(R_ref.columns),
                        n_factors  = cfg.n_factors,
                        innovation = cfg.innovation,
                        p_ratio  = getattr(cfg, "p_ratio", None),
                        rho_B  = getattr(cfg, "rho_B",   None),
                        sigma_B  = getattr(cfg, "sigma_B", None),
                        seed   = seed_val,
                    )

                # Si tous les modèles de ce seed sont déjà calculés, skippe le seed entier
                all_keys = [_make_key(spec, seed) for spec in model_specs]
                if all(exporter.already_done(k) for k in all_keys):
                    print(f"  [StatSim] Scénario {i+1}/{cfg.n_scenarios} (seed={seed}) — SKIP (checkpoint)")
                    continue
            
            # Lance le scénario et mesure la durée
            print(f"  [StatSim] Scénario {i+1}/{cfg.n_scenarios} (seed={seed})")
            t0  = time.perf_counter()
            res = cls._run_one_scenario(R_ref=R_ref, model_specs=model_specs, cfg=cfg, seed=seed)
            duration = time.perf_counter() - t0
            all_results.append(res)

            # Écriture au fil de l'eau si exporter fourni
            if exporter is not None:
                from Modules.Financial_engineering.Export.Montecarlo_cov_export import StatScenarioKey

                # Construit la clé de checkpoint pour ce scénario
                loss_df = res.loss_summary

                # Itére sur les modèles de ce scénario pour construire les résultats individuels et les écrire via l'exporter
                for spec in model_specs:
                    name = getattr(spec, "name", str(spec))
                    key  = _make_key(spec, seed)

                    if exporter.already_done(key):
                        continue
                    
                    # Construit le résultat à partir de la ligne de pertes du modèle, ou construit une erreur si une exception est levée, puis écrit le résultat via l'exporter
                    try:
                        loss_row = (loss_df.loc[name].to_dict() if name in loss_df.index else {})
                        result   = exporter.build_result(key, loss_row, duration=duration)
                    except Exception as exc:
                        result = exporter.build_error(key, exc, duration=duration)

                    exporter.write(result)

        # Flush final après tous les scénarios
        if exporter is not None:
            exporter.flush()

        # Si tous les scénarios étaient dans le checkpoint, retourne un résultat vide valide
        if not all_results:
            empty = pd.DataFrame(index=[getattr(s, "name", str(s)) for s in model_specs], columns=list(cfg.metrics), dtype=float,)
            empty.index.name = "model"
            return StatEvalResult(loss_mean=empty, loss_std=empty, loss_all=[], cfg=cfg)

        # Agrège les pertes sur tous les scénarios en un tableau 3D
        all_dfs   = [r.loss_summary for r in all_results]
        all_models = sorted({m for df in all_dfs for m in df.index})
        all_metrics = sorted({c for df in all_dfs for c in df.columns})

        # Empile les pertes dans un tableau 3D de forme (n_scenarios, n_models, n_metrics), en alignant les modèles et les métriques, et en remplissant de NaN les valeurs manquantes
        stacked = np.full((len(all_results), len(all_models), len(all_metrics)), np.nan, dtype=float)
        for s, df in enumerate(all_dfs):
            for mi, m in enumerate(all_models):
                if m in df.index:
                    for ci, c in enumerate(all_metrics):
                        if c in df.columns:
                            stacked[s, mi, ci] = float(df.loc[m, c])

        # Calcule la moyenne
        loss_mean = pd.DataFrame(np.nanmean(stacked, axis=0), index=all_models, columns=all_metrics)
        loss_mean.index.name = "model"

        # Calcule l'écart-type
        loss_std = pd.DataFrame(np.nanstd(stacked, axis=0, ddof=1) if len(all_results) > 1 else np.zeros_like(stacked[0]), index=all_models, columns=all_metrics, )
        loss_std.index.name = "model"

        print(f"[StatSim] Terminé — résultats moyennés sur {len(all_results)} scénario(s).")

        return StatEvalResult(loss_mean=loss_mean, loss_std=loss_std, loss_all=all_results, cfg=cfg)