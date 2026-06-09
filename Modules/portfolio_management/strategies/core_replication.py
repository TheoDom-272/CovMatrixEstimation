"""
Stratégie de réplication d'indice par minimisation de la tracking error (TE-min).
 
Ce fichier contient un moteur d'allocation core utilisé dans le backtest.
Il supporte deux modes d'allocation :
- Mode 'returns'  : minimisation de la TE vs rendements du benchmark via un solveur QP
  (Clarabel ou SLSQP), en utilisant la matrice de covariance estimée sur la fenêtre glissante.
- Mode 'weights'  : soit redistribution benchmark-like (sans solveur), soit TE-min classique
  vs poids du benchmark (avec solveur).
 
Classes
-------
CoreRedistributionMethod :
    Enumération des méthodes de redistribution des poids benchmark quand l'univers investissable est un sous-ensemble du benchmark.
CoreReplicationConfig :
    Configuration de la redistribution du benchmark (exclusions, freeze, méthode).
CoreWeightRedistributor :
    Implémente la réplication benchmark sans solveur : exclusions, freeze des core tickers, redistribution du poids manquant (globale ou sectorielle).
CoreTEMinConfig :
    Configuration complète du moteur TE-min (mode, solveur, contraintes, filtres data).
CoreTEMinAllocator :
    Moteur d'allocation principal : prépare l'univers, estime la covariance via le provider, et appelle le solveur QP pour minimiser la TE ex-ante.
 
Fonctions
---------
_remove_zero_cov_assets :
    Utilitaire interne qui purge les actifs sans historique (variance nulle dans la cov) avant de passer au solveur.
"""



from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

import time as _time

from Modules.Financial_engineering.optimization.base import (OptimizationProblem, LinearEqualityConstraint, Bounds,)
from Modules.Financial_engineering.optimization.tracking_error import (TrackingErrorObjective, TrackingErrorFullUniverseObjective)

from Modules.portfolio_management.backtesting.covariance_provider import CovarianceProvider

from Modules.Financial_engineering.optimization.optimizers import (SLSQPOptimizer, ClarabelOptimizer, make_optimizer,)


# Type alias pour un estimateur de covariance callable
CovEstimator = Callable[[pd.DataFrame], np.ndarray]

# Utilitaire 
def _remove_zero_cov_assets(cov: np.ndarray,kept_tickers: list[str],b_kept: pd.Series,tol: float = 1e-12,) -> tuple[np.ndarray, list[str], pd.Series, list[int]]:
    """
    Retire les actifs dont toute la ligne et colonne de covariance sont nulles.
 
    Parameters
    ----------
    cov : np.ndarray
        Matrice de covariance (N x N) avant purge.
    kept_tickers : list[str]
        Noms des N actifs correspondant aux lignes/colonnes de cov.
    b_kept : pd.Series
        Poids benchmark projetés sur les N actifs, indexés par ticker.
    tol : float
        Seuil de tolérance pour considérer une variance comme nulle.
 
    Returns
    -------
    cov_active : np.ndarray
        Sous-matrice de covariance sur les actifs actifs (K x K, K <= N).
    active_tickers : list[str]
        Noms des K actifs conservés.
    b_active : pd.Series
        Poids benchmark renormalisés sur les K actifs.
    active_local_idx : list[int]
        Indices des actifs conservés dans le vecteur original (pour remapping final).
    """

    # variance de chaque actif
    diag = np.diag(cov)     

    # True si l'actif a un historique à cette date
    active_mask = diag > tol                  

    # indices des actifs actifs
    active_local_idx = np.where(active_mask)[0].tolist()

    # Si tous les actifs sont actifs, rien à purger
    if len(active_local_idx) == len(kept_tickers):
        return cov, kept_tickers, b_kept, active_local_idx  # rien à purger

    # Sous-ensemble des tickers et de la matrice après purge
    active_tickers = [kept_tickers[i] for i in active_local_idx]
    cov_active = cov[np.ix_(active_local_idx, active_local_idx)]

    # Renormalise les poids benchmark sur l'espace réduit
    b_active = b_kept.reindex(active_tickers).fillna(0.0)
    sb = float(b_active.sum())

    # renormalisation classique
    if sb > tol:
        b_active = b_active / sb

    # fallback uniforme si benchmark vide
    else:
        b_active[:] = 1.0 / len(active_tickers)

    return cov_active, active_tickers, b_active, active_local_idx



# Enum : méthodes de redistribution

class CoreRedistributionMethod(str, Enum):
    """
    Méthodes de redistribution des poids benchmark quand l'univers investissable
    est un sous-ensemble du benchmark (exclusions, filtres small caps, etc.).
 
    Attributes
    ----------
    GLOBAL_PROPORTIONAL :
        Redistribue le poids manquant proportionnellement aux poids benchmark des candidats.
    GLOBAL_INVERSE :
        Redistribue inversement aux poids benchmark (favorise les petites capitalisations).
    SECTOR_MATCH_PROPORTIONAL :
        Redistribution proportionnelle secteur par secteur.
    SECTOR_MATCH_INVERSE :
        Redistribution inverse secteur par secteur.
    """
    
    GLOBAL_PROPORTIONAL = "GLOBAL_PROPORTIONAL"
    GLOBAL_INVERSE = "GLOBAL_INVERSE"
    SECTOR_MATCH_PROPORTIONAL = "SECTOR_MATCH_PROPORTIONAL"
    SECTOR_MATCH_INVERSE = "SECTOR_MATCH_INVERSE"


@dataclass(frozen=True)
class CoreReplicationConfig:

    """
    Configuration de la redistribution du benchmark (sans solveur).
 
    Utilisée par CoreWeightRedistributor pour construire des poids qui répliquent
    le benchmark tout en respectant des contraintes d'exclusion, de freeze et de
    redistribution sectorielle.
 
    Attributes
    ----------
    core_budget : float
        Budget total de poids à allouer (1.0 = fully invested, 0.95 = core 95%).
    method : CoreRedistributionMethod
        Règle de redistribution des poids des actifs exclus ou non gelés.
    explicit_exclusions : list[str] or None
        Tickers explicitement exclus de l'allocation.
    mcap_exclude_pct : float or None
        Pourcentage des plus petites capitalisations du benchmark à exclure.
    market_caps : dict or pd.DataFrame or None
        Capitalisations boursières par ticker (dict = constant, DataFrame = time-series).
    core_ticker_threshold : float or None
        Seuil de poids benchmark au-delà duquel un actif est gelé à son poids benchmark.
    sector_map : dict or None
        Mapping ticker - secteur. Requis pour les méthodes SECTOR_MATCH_*.
    eps : float
        Tolérance numérique globale.
    """
    
    core_budget: float = 1.0
    method: CoreRedistributionMethod = CoreRedistributionMethod.SECTOR_MATCH_INVERSE
    explicit_exclusions: Optional[List[str]] = None
    mcap_exclude_pct: Optional[float] = None
    market_caps: Optional[Union[Mapping[str, float], pd.DataFrame]] = None 
    core_ticker_threshold: Optional[float] = 0.01
    sector_map: Optional[Mapping[str, str]] = None
    eps: float = 1e-12


# redistribution benchmark (sans solveur)
class CoreWeightRedistributor:
    """
    Implémente la réplication benchmark-like sans solveur.
 
    Construit des poids qui répliquent le benchmark en respectant trois règles :
    1. Les actifs exclus (explicitement ou par filtre small caps) reçoivent un poids nul.
    2. Les 'core tickers' (poids benchmark >= seuil) sont gelés à leur poids benchmark.
    3. Le budget restant est redistribué aux candidats selon la méthode configurée.
 
    Attributes
    ----------
    cfg : CoreReplicationConfig
        Configuration de la redistribution.
 
    Methods
    -------
    build_core_weights_from_benchmark(asof, w_bench, universe_tickers, bench_tickers_all) -> pd.Series :
        Construit les poids core en appliquant exclusions, freeze et redistribution.
    """

    def __init__(self, cfg: CoreReplicationConfig) -> None:
        self.cfg = cfg

    # Helpers internes
    @staticmethod
    def _clean_tickers(x: List[str]) -> List[str]:
        """Nettoie une liste de tickers : supprime les chaînes vides et strip les espaces."""
        return [str(t).strip() for t in x if str(t).strip()]

    


    def _get_mcap_snapshot(self,market_caps,asof: pd.Timestamp,tickers: List[str],) -> pd.Series:
        """
        Retourne une Series de capitalisations boursières indexée par tickers à la date asof.
 
        Supporte dict constant ou DataFrame time-series (searchsorted sur l'index).
 
        Parameters
        ----------
        market_caps : dict or pd.DataFrame or None
            Source de capitalisations boursières.
        asof : pd.Timestamp
            Date à laquelle récupérer le snapshot.
        tickers : list[str]
            Tickers pour lesquels extraire les capitalisations.
 
        Returns
        -------
        pd.Series
            Capitalisations indexées par ticker (NaN si absent).
        """

        #Verification que la variable des market cap existe
        if market_caps is None:
            return pd.Series(dtype=float)

        # Cas dict : valeurs constantes dans le temps
        if isinstance(market_caps, dict):
            return pd.Series({t: market_caps.get(t, np.nan) for t in tickers}, dtype=float)

        # Cas DataFrame time-series : prend la dernière ligne disponible <= asof
        df = market_caps
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()

            #For la conversion de l'index en DateTimeIndex
            df.index = pd.to_datetime(df.index)

        # Trie le df
        df = df.sort_index()

        # dernière date <= asof
        pos = df.index.searchsorted(pd.Timestamp(asof), side="right") - 1

        # Si aucune donnée disponible avant asof
        if pos < 0:
            return pd.Series(dtype=float)

        return df.iloc[pos].reindex(tickers).astype(float)


    def _compute_exclusions(self,asof: pd.Timestamp,universe_tickers: List[str],bench_tickers_all: List[str],) -> set[str]:
        """
        Construit l'ensemble d'exclusions finales appliquées à l'univers investissable.
 
        Applique : exclusions explicites, filtre small caps sur benchmark complet,
        puis intersection avec l'univers investissable.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date courante (pour le snapshot de capitalisations).
        universe_tickers : list[str]
            Univers investissable.
        bench_tickers_all : list[str]
            Benchmark complet (pour le filtre small caps).
 
        Returns
        -------
        set[str]
            Ensemble des tickers à exclure, intersection avec l'univers investissable.
        """

        # Formate les tickers
        excl: set[str] = set()
        universe_tickers = self._clean_tickers(universe_tickers)
        bench_tickers_all = self._clean_tickers(bench_tickers_all)

        # exclusions explicites définies dans la config
        if self.cfg.explicit_exclusions:
            excl |= {str(t).strip() for t in self.cfg.explicit_exclusions if str(t).strip()}

        # filtre small caps calculé sur le benchmark complet (avant intersection investissable)
        if self.cfg.mcap_exclude_pct is not None and float(self.cfg.mcap_exclude_pct) > 0:

            # clamp entre 0 et 1
            p = min(max(float(self.cfg.mcap_exclude_pct), 0.0), 1.0)
 
            #Recupère les capitalisations boursières à la date asof pour les tickers du benchmark complet
            caps = self._get_mcap_snapshot(self.cfg.market_caps, asof=asof, tickers=bench_tickers_all)

            # supprime les valeurs invalides
            caps = caps.replace([np.inf, -np.inf], np.nan).dropna()

            # garde uniquement les caps > 0
            caps = caps[caps > 0]

            #Si on a des caps >0
            if len(caps) > 0:
                # trie du plus petit au plus grand
                caps = caps.sort_values(ascending=True)  

                # nombre d'actifs à exclure
                k = int(np.ceil(p * len(caps)))
                k = min(max(k, 0), len(caps))

                # les k plus petites caps
                small_caps = set(caps.iloc[:k].index.astype(str))
                excl |= small_caps

        # on ne garde que les exclusions qui sont dans l'univers investissable
        return set(universe_tickers) & excl


    def _allocate_by_scores(self,candidates: List[str],bench_weights: pd.Series,total_budget: float, inverse: bool,) -> pd.Series:
        """
        Alloue total_budget sur les candidats selon leur score benchmark.
 
        Mode proportionnel (inverse=False) : score = poids benchmark.
        Mode inverse (inverse=True) : score = 1 / poids benchmark.
 
        Parameters
        ----------
        candidates : list[str]
            Tickers candidats à la redistribution (non exclus et non gelés).
        bench_weights : pd.Series
            Poids benchmark des candidats.
        total_budget : float
            Budget total à redistribuer.
        inverse : bool
            Si True, utilise 1/w_bench comme score.
 
        Returns
        -------
        pd.Series
            Poids alloués aux candidats, sommant à total_budget.
        """

        # Récupération des paramètres
        eps = float(self.cfg.eps)
        total_budget = float(total_budget)

        # Rien à allouer si budget nul ou liste vide
        if total_budget <= eps or len(candidates) == 0:
            return pd.Series(dtype=float)

        # Poids benchmark des candidats
        bw = bench_weights.reindex(candidates).fillna(0.0).astype(float) #Recupere les poids benchmark des candidats

        # Score selon le mode choisi
        if inverse:
            # Calcule les scores inverses
            scores = 1.0 / (bw.clip(lower=eps)) 
        else:
            scores = bw.clip(lower=0.0)

        s = float(scores.sum())

        # Si les scores sont tous nuls, allocation uniforme sur les candidats
        if s <= eps:
            return pd.Series(total_budget / len(candidates), index=candidates, dtype=float)

        # Alloue le budget proportionnellement aux scores normalisés
        return ((scores / s) * total_budget).astype(float) 


    # Méthode principale
    def build_core_weights_from_benchmark(self,asof: pd.Timestamp,w_bench: pd.Series, universe_tickers: List[str], bench_tickers_all: List[str],) -> pd.Series:
        """
        Construit les poids core (somme = cfg.core_budget) en répliquant le benchmark.
 
        Applique dans l'ordre : exclusions, freeze des core tickers, redistribution
        du budget restant selon la méthode configurée.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date courante.
        w_bench : pd.Series
            Poids benchmark projetés sur l'univers investissable.
        universe_tickers : list[str]
            Tickers de l'univers investissable.
        bench_tickers_all : list[str]
            Tickers du benchmark complet (pour filtre small caps).
 
        Returns
        -------
        pd.Series
            Poids core indexés par universe_tickers, sommant à cfg.core_budget.
        """
        
        # Recupere le paramètre de tolérance numérique mis en configuration
        eps = float(self.cfg.eps) 
        
        # Nettoyage des tickers
        universe_tickers = self._clean_tickers(universe_tickers)
        bench_tickers_all = self._clean_tickers(bench_tickers_all)

        # vérification
        if len(universe_tickers) == 0:
            return pd.Series(dtype=float)

        # Aligne et sécurise les poids benchmark sur l'univers investissable
        w_bench = w_bench.reindex(universe_tickers).fillna(0.0).astype(float)
        s0 = float(w_bench.sum())

        # Si le Benchmark est vide, poids uniformes comme base
        if s0 <= eps:
            w_bench = pd.Series(1.0 / len(universe_tickers), index=universe_tickers, dtype=float)

        # Clamp le budget entre 0 et 1
        core_budget = min(max(float(self.cfg.core_budget), 0.0), 1.0)

        # Calcule les exclusions (explicites + small caps)
        excl = self._compute_exclusions(asof=asof,universe_tickers=universe_tickers,bench_tickers_all=bench_tickers_all,)

        # Identifie les core tickers à geler (poids benchmark >= seuil et non exclus)
        frozen: List[str] = []
        thr = self.cfg.core_ticker_threshold
        if thr is not None and float(thr) > 0:
            tthr = float(thr)
            frozen = [t for t in universe_tickers if (float(w_bench.get(t, 0.0)) >= tthr) and (t not in excl)] #determine les tickers à freezer

        # Poids cibles de départ : budget × poids benchmark
        w_target = (w_bench * core_budget).astype(float)

        # Force les poids des exclus à 0
        if excl:
            w_target.loc[list(excl)] = 0.0

        # Récupère les poids des frozen et calcule le budget qu'ils consomment
        frozen_w = w_target.reindex(frozen).fillna(0.0).astype(float)
        frozen_sum = float(frozen_w.sum())

        # Budget restant à redistribuer après freeze
        remaining_budget = float(core_budget - frozen_sum)
        
         # Si le budget restant à redistribuer est trop faible
        if remaining_budget < -1e-10:

            # Si frozen trop gros -> renormalise frozen pour faire core_budget
            if frozen_sum > eps:
                w_target.loc[frozen] = (frozen_w / frozen_sum) * core_budget
            
            #Sinon poids à 0
            else:
                w_target.loc[:] = 0.0

            return w_target.astype(float)
        
        # Recupère les tickers qui ne sont pas freeze et pas exclus
        candidates = [t for t in universe_tickers if (t not in excl) and (t not in frozen)] 

        # Si rien à redistribuer
        if remaining_budget <= eps or len(candidates) == 0:
            return w_target.astype(float)

        # Recupere la méthode de redistribution choisie
        method = self.cfg.method 

        # redistribution globale
        if method in (CoreRedistributionMethod.GLOBAL_PROPORTIONAL, CoreRedistributionMethod.GLOBAL_INVERSE):
            inv = (method == CoreRedistributionMethod.GLOBAL_INVERSE)
            alloc = self._allocate_by_scores(candidates,bench_weights=w_bench,total_budget=remaining_budget,inverse=inv,)
            w_target.loc[candidates] = alloc.reindex(candidates).fillna(0.0).values
            return w_target.astype(float)

        # redistribution sectorielle
        sector_map = self.cfg.sector_map or {}

        def sec_of(t: str) -> str:
            # Retourne le secteur d'un ticker, 'UNKNOWN' s'il n'est pas dans le mapping
            return str(sector_map.get(t, "UNKNOWN"))
        
        # Recupère le mapping secteur pour tout l'univers investissable
        sec_series_all = pd.Series({t: sec_of(t) for t in universe_tickers}) 

        # Calcule les poids benchmark par secteur
        bench_by_sector = w_bench.groupby(sec_series_all).sum() 

        # Calcule le budget cible par secteur
        target_sector_budget = (bench_by_sector * core_budget).astype(float) 

        # Soustrait les poids consommés par les frozen, secteur par secteur
        if frozen:

            # Recupère les secteurs des titres gelés
            sec_series_f = pd.Series({t: sec_of(t) for t in frozen}) 

            # Somme des poids par secteurs des titres gelés
            frozen_by_sector = frozen_w.groupby(sec_series_f).sum() 

        else:
            frozen_by_sector = pd.Series(dtype=float)

        # Calcule le budget restant par secteur après freeze
        rem_by_sector = (target_sector_budget - frozen_by_sector).fillna(target_sector_budget).astype(float) 

        # Regroupe les candidats par secteur
        candidates_by_sector: Dict[str, List[str]] = {}
        for t in candidates:
            candidates_by_sector.setdefault(sec_of(t), []).append(t)

        # Spill : si un secteur n'a pas de candidats, son budget est reversé aux secteurs actifs
        spill = 0.0
        for sec, b in rem_by_sector.items():
            if float(b) <= eps:
                continue
            if sec not in candidates_by_sector or len(candidates_by_sector[sec]) == 0:
                spill += float(b)

                # ce secteur ne reçoit plus rien
                rem_by_sector.loc[sec] = 0.0

        if spill > eps:

            # Redistribue le spill proportionnellement aux budgets des secteurs actifs restants
            active_secs = [s for s in rem_by_sector.index if float(rem_by_sector.get(s, 0.0)) > eps and s in candidates_by_sector]
            denom = float(rem_by_sector.reindex(active_secs).sum())
            if denom > eps and active_secs:
                rem_by_sector.loc[active_secs] = rem_by_sector.loc[active_secs] + spill * (rem_by_sector.loc[active_secs] / denom)

            # Fallback global si tous les secteurs actifs ont un budget nul
            else:
                inv = (method == CoreRedistributionMethod.SECTOR_MATCH_INVERSE)
                alloc = self._allocate_by_scores(candidates,bench_weights=w_bench,total_budget=remaining_budget,inverse=inv,)
                w_target.loc[candidates] = alloc.reindex(candidates).fillna(0.0).values
                return w_target.astype(float)

        inv = (method == CoreRedistributionMethod.SECTOR_MATCH_INVERSE)  # Determine si on utilise la méthode inverse
        parts: List[pd.Series] = []

        # Alloue le budget de chaque secteur à ses candidats
        # Itere sur les secteurs
        for sec, names in candidates_by_sector.items(): 

            # Recupère le budget restant pour le secteur
            b = float(rem_by_sector.get(sec, 0.0)) 

            if b <= eps or not names:
                continue

            # Alloue le budget du secteur aux titres du secteur avec la méthode choisie
            parts.append(self._allocate_by_scores(names, bench_weights=w_bench, total_budget=b, inverse=inv))

        if parts:

            # Concatène toutes les poids des titres pour reformer le portefeuille
            alloc = pd.concat(parts) 

            # Applique les poids alloués aux titres dans le portefeuille cible
            w_target.loc[alloc.index] = alloc.values 

        # Renormalisation numérique défensive sans toucher aux frozen
        s = float(w_target.sum())
        if abs(s - core_budget) > 1e-8 and s > eps:
            non_frozen = [t for t in universe_tickers if t not in frozen]
            s_nf = float(w_target.reindex(non_frozen).sum())

            if s_nf > eps:
                # Facteur de correction pour que la somme des non-frozen == budget restant
                factor = float((core_budget - frozen_sum) / s_nf) if (core_budget - frozen_sum) > -eps else 0.0
                w_target.loc[non_frozen] *= factor

                if frozen:
                    # remet les poids frozen intacts
                    w_target.loc[frozen] = frozen_w.values

        return w_target.astype(float)



# Dataclass : configuration du moteur TE-min
@dataclass(frozen=True)
class CoreTEMinConfig:
    """
    Configuration complète du moteur d'allocation TE-min.
 
    Attributes
    ----------
    mode : str
        Mode d'allocation : 'returns' (TE-min vs rendements) ou 'weights' (vs poids benchmark).
    long_only : bool
        Si True, contraint tous les poids à être >= 0.
    strict_benchmark_weights : bool
        Si True, lève une erreur si le benchmark contient des actifs hors univers investissable.
    enable_weight_redistribution : bool
        Si True en mode 'weights', utilise CoreWeightRedistributor au lieu du solveur.
    redistribution_method : CoreRedistributionMethod
        Méthode de redistribution (si enable_weight_redistribution=True).
    sector_map : dict or None
        Mapping ticker - secteur pour les méthodes sectorielles.
    explicit_exclusions : tuple[str] or None
        Tickers explicitement exclus de l'allocation.
    core_ticker_threshold : float or None
        Seuil de poids benchmark pour geler un actif à son poids benchmark.
    mcap_exclude_pct : float or None
        Pourcentage des plus petites capitalisations à exclure.
    market_caps : dict or pd.DataFrame or None
        Capitalisations boursières pour le filtre small caps.
    min_obs : int
        Nombre minimal d'observations requis dans la fenêtre pour déclencher l'allocation.
    min_coverage : float
        Taux minimal de données non-NaN requis par actif pour le garder dans l'univers.
    eps : float
        Tolérance numérique globale.
    restrict_to_benchmark : bool
        Si True, filtre kept_tickers sur les actifs présents dans le benchmark (poids > 0).
    optimizer_name : str
        Solveur : 'slsqp' ou 'clarabel'.
    """

    mode: str = "returns"
    long_only: bool = True
    strict_benchmark_weights: bool = True
    enable_weight_redistribution: bool = False
    redistribution_method: CoreRedistributionMethod = CoreRedistributionMethod.SECTOR_MATCH_INVERSE
    sector_map: Optional[Mapping[str, str]] = None
    explicit_exclusions: Optional[tuple[str, ...]] = None
    core_ticker_threshold: Optional[float] = 0.01
    mcap_exclude_pct: Optional[float] = None
    market_caps: Optional[Union[Mapping[str, float], pd.DataFrame]] = None
    min_obs: int = 30
    min_coverage: float = 0.90
    eps: float = 1e-12
    restrict_to_benchmark: bool = True
    optimizer_name: str = "slsqp"  


# Classe principale : moteur d'allocation TE-min
class CoreTEMinAllocator:
    """
    Moteur d'allocation pour la minimisation de la tracking error (TE-min).
 
    Supporte deux modes :
    - 'returns' : cov via CovarianceProvider, objectif TrackingErrorFullUniverseObjective, solveur QP.
    - 'weights' : redistribution benchmark-like (CoreWeightRedistributor) ou TE-min vs poids.
 
    Attributes
    ----------
    cfg : CoreTEMinConfig
        Configuration de l'allocateur.
    cov_estimator : callable
        Estimateur de covariance de fallback (covariance empirique par défaut).
    optimizer : SLSQPOptimizer or ClarabelOptimizer
        Solveur d'optimisation instancié selon cfg.optimizer_name.
 
    Methods
    -------
    set_covariance_provider(provider) -> None :
        Injecte le CovarianceProvider fourni par le moteur de backtest.
    allocate(...) -> tuple :
        Calcule et retourne les poids optimaux à la date asof.
    """

    def __init__(self,cfg: CoreTEMinConfig,cov_estimator: Optional[CovEstimator] = None,optimizer: Optional[SLSQPOptimizer] = None, cov_provider: Optional[CovarianceProvider] = None,) -> None:
        self.cfg = cfg
        self.cov_estimator = cov_estimator or self.sample_covariance # fallback si pas de provider
        self.optimizer = optimizer or SLSQPOptimizer()
        self._cov_provider = cov_provider
        self._last_rebal_dt : Optional[pd.Timestamp] = None  # mémorise la dernière date de rebal
        self._solver_eval_log: Optional[list] = None    # log des évaluations du solveur

        # Instancie le solveur selon le nom configuré (SLSQP ou Clarabel)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = make_optimizer(cfg.optimizer_name)

    # statics 
    @staticmethod
    def sample_covariance(df: pd.DataFrame) -> np.ndarray:
        """Covariance empirique symétrisée - fallback si pas de provider."""
        cov = df.cov().values
        return 0.5 * (cov + cov.T)

    @staticmethod
    def _normalize_vec(w: np.ndarray, eps: float) -> np.ndarray:
        """Renormalise un vecteur de poids pour sommer à 1. Retourne w inchangé si somme <= eps."""
        s = float(np.sum(w))
        return w if s <= eps else (w / s)

    @staticmethod
    def _finite_coverage_mask(x: pd.DataFrame, min_coverage: float, min_obs: int) -> pd.Series:
        """
        Retourne un masque booléen indiquant les actifs avec suffisamment de données.
        """
         # masque booléen : True si la valeur est finie
        finite = np.isfinite(x.values)

        # taux de couverture par actif
        coverage = finite.mean(axis=0)

        # nombre d'observations valides par actif
        n_obs = finite.sum(axis=0)

        return pd.Series((coverage >= min_coverage) & (n_obs >= min_obs), index=x.columns)
    
    def set_covariance_provider(self, provider: CovarianceProvider) -> None:
        """Injecte le CovarianceProvider fourni par le moteur de backtest."""
        self._cov_provider = provider

    def _cov_subset(self,asof: pd.Timestamp, cols: list[str],returns_window: Optional[pd.DataFrame] = None,) -> np.ndarray:
        """
        Retourne la sous-matrice de covariance pour les actifs cols à la date asof.
 
        Mode 'rebal' : calcul à la demande via get_cov_rebal() (fenêtre requise).
        Mode 'path'  : accès au memmap précomputé via get_cov_subset().
        """

        if self._cov_provider is not None:
            mode = getattr(self._cov_provider.cfg, "compute_mode", "path")

            # Mode rebal
            if mode == "rebal":
                # Vérification que la fenêtre de rendements est passé
                if returns_window is None:
                    raise RuntimeError("returns_window requis en mode rebal.")
                
                # Utilisation du provider de covariance pour calculer la covariance à la date de rebal
                return np.asarray(self._cov_provider.get_cov_rebal(returns_window=returns_window,sub_cols=cols,),dtype=float,)
            
            # Mode path : accès direct au memmap précomputé
            else:
                return np.asarray(self._cov_provider.get_cov_subset(asof=asof, sub_cols=cols),dtype=float,)

        raise RuntimeError("No covariance provider available.")



    def allocate(self,asof: pd.Timestamp,returns_window: pd.DataFrame, benchmark_returns_window: pd.Series, 
        kept_assets: list[str], current_weights: pd.Series,benchmark_weights: Optional[pd.Series] = None) -> Tuple[pd.Series, Dict[str, Any]]:                        
        """
        Calcule et retourne les poids optimaux à la date asof.
 
        Dispatcher entre mode 'returns' et mode 'weights'. Filtre l'univers,
        vérifie les conditions minimales, puis délègue au bon sous-mode.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date de rebalancement courante.
        returns_window : pd.DataFrame
            Fenêtre glissante de rendements.
        benchmark_returns_window : pd.Series
            Rendements du benchmark sur la même fenêtre.
        kept_assets : list[str]
            Actifs investissables imposés par le moteur.
        current_weights : pd.Series
            Poids du portefeuille au rebalancement précédent (warm start).
        benchmark_weights : pd.Series or None
            Poids du benchmark à la date asof.
 
        Returns
        -------
        w_full : pd.Series, diag : dict
        """    
        
        mode = str(self.cfg.mode).lower().strip()

        # full universe (benchmark universe) = colonnes de returns_window
        tickers_all = list(returns_window.columns.astype(str))

        # Intersection entre kept_assets (imposé par l'engine) et les tickers disponibles
        kept_tickers = [str(t) for t in kept_assets if str(t) in set(tickers_all)]
        
        # Filtrage pour ne garder que les titres qui sont dans l'indice a cette date
        if self.cfg.restrict_to_benchmark and benchmark_weights is not None:
            bench_active= set(benchmark_weights[benchmark_weights>0].index.astype(str))
            kept_tickers = [t for t in kept_tickers if t in bench_active]
            tickers_all = [t for t in tickers_all if t in bench_active]

        # Fallback si trop peu d'actifs investissables
        if len(kept_tickers) < 2:
            w_fb = self._fallback_weights(current_weights, list(current_weights.index))
            return w_fb, {"fallback": True, "reason": "too_few_kept_assets", "mode": mode}

        # Fenêtre de rendements sur l'univers complet (sans filtrage)
        rw_full = returns_window.reindex(columns=tickers_all).copy()

        # Vérifie la taille minimale de la fenêtre
        if len(rw_full) < max(self.cfg.min_obs, 10):
            w_fb = self._fallback_weights(current_weights, list(current_weights.index))
            return w_fb, {"fallback": True, "reason": "window_too_short", "mode": mode}

        # Dispatch vers le bon mode
        if mode == "returns":
            return self._allocate_returns(asof=asof, tickers_all=tickers_all, kept_tickers=kept_tickers, benchmark_weights=benchmark_weights,  
                                          rw=rw_full, benchmark_returns_window=benchmark_returns_window,current_weights=current_weights,)
        
        if mode == "weights":
            return self._allocate_weights(asof=asof, tickers_all=tickers_all, kept_tickers=kept_tickers, returns_window=returns_window,
                                          rw=returns_window[kept_tickers].copy(), current_weights=current_weights, benchmark_weights=benchmark_weights,)

        raise ValueError(f"[CoreTEMinAllocator] Unknown mode: {self.cfg.mode}")



    def _prepare_universe(self, returns_window: pd.DataFrame) -> Tuple[list[str], list[str], pd.DataFrame]:
        """Prépare l'univers investissable en appliquant les filtres de couverture."""

        tickers_all = list(returns_window.columns)

        # Determine les assets à garder en fonction des paramètres de couverture et d'observations minimales fournis
        mask_assets = self._finite_coverage_mask(returns_window, self.cfg.min_coverage, self.cfg.min_obs)

        # Filtre les tickers à garder
        kept_tickers = list(mask_assets[mask_assets].index)  # garde uniquement les True

        rw = returns_window[kept_tickers].copy()

        return tickers_all, kept_tickers, rw

    def _fallback_weights(self, current_weights: pd.Series, tickers_all: list[str]) -> pd.Series:
        """ Poids fallback: renormalisation des poids courants sur tout l'univers."""

        eps = float(self.cfg.eps)
        n_all = len(tickers_all)
        w = current_weights.reindex(tickers_all).fillna(0.0).astype(float)
        s = float(w.sum())
        if s > eps:
            return (w / s).astype(float)
        
        return pd.Series(1.0 / n_all, index=tickers_all, dtype=float)


    def _build_constraints(self, n: int) -> Tuple[LinearEqualityConstraint, Optional[Bounds]]:
        """
        Construit la contrainte sum(w) = 1 et les bornes long-only si configurées.
 
        Parameters
        ----------
        n : int
            Nombre d'actifs dans l'espace d'optimisation.
 
        Returns
        -------
        eq : LinearEqualityConstraint
        bounds : Bounds or None
        """

        # Contrainte d'égalité : somme des poids = 1
        A = np.ones((1, n), dtype=float)
        b = np.array([1.0], dtype=float)
        eq = LinearEqualityConstraint(A=A, b=b)

        # Bornes [0, 1] par actif si long-only
        bounds = None
        if bool(self.cfg.long_only):
            lb = np.zeros(n, dtype=float)
            ub = np.ones(n, dtype=float)
            bounds = Bounds(lb=lb, ub=ub)

        return eq, bounds

    def _initial_guess(self, current_weights: pd.Series, kept_tickers: list[str]) -> np.ndarray:
        """
        Construit le point de départ du solveur à partir des poids courants.
        """

        eps = float(self.cfg.eps)

        # Projette les poids courants sur l'espace investissable et normalise
        w0 = current_weights.reindex(kept_tickers).fillna(0.0).astype(float).values

        return self._normalize_vec(w0, eps)


    def _solve(self,objective,n_assets: int,eq: LinearEqualityConstraint,bounds: Optional[Bounds],w0: np.ndarray,
               metadata: Optional[Dict[str, Any]],) -> Tuple[np.ndarray, bool,float]:
        """
        Résout le problème d'optimisation et retourne les poids optimaux.
        En cas d'échec du solveur, retourne les poids initiaux renormalisés (fallback défensif).
 
        Returns
        -------
        w : np.ndarray, success : bool, solve_ms : float
        """

        # Tolérance numérique
        eps = float(self.cfg.eps)

        # Construction du problème
        problem = OptimizationProblem(objective=objective,n_assets=int(n_assets),eq=eq,bounds=bounds,metadata=metadata or {},)
        
        # Résolution
        _t0 = _time.perf_counter()
        res = self.optimizer.solve(problem, w0=w0)
        solve_ms = (_time.perf_counter() - _t0) * 1000.0   # en millisecondes

        # Résultat
        if not res.success:
            return self._normalize_vec(w0, eps), False, solve_ms

        # Renormalise la solution pour garantir sum(w) = 1 malgré les erreurs numériques
        return self._normalize_vec(np.asarray(res.w, dtype=float), eps), True, solve_ms


    # Remapping et fallback
    def _remap_to_full_universe(self, tickers_all: list[str], kept_tickers: list[str], w_kept: np.ndarray) -> pd.Series:
        """
        Remappe les poids optimisés (espace K) vers l'univers complet (espace N).
        Les actifs hors kept_tickers reçoivent un poids nul puis on renormalise.
        """

        # Tolérance numérique
        eps = float(self.cfg.eps)

        # Initialise tous les poids à 0, puis assigne les poids optimaux aux kept
        w_full = pd.Series(0.0, index=tickers_all, dtype=float)
        w_full.loc[kept_tickers] = w_kept

        s = float(w_full.sum())

        # Cas dégénéré, poids uniformes sur l'univers complet
        if s <= eps:
            w_full[:] = 1.0 / len(tickers_all)

        #Sinon, renormalise pour sommer à 1
        else:
            w_full = w_full / s

        return w_full.astype(float)

    def _allocate_returns(self, asof, tickers_all, kept_tickers, benchmark_weights,rw, benchmark_returns_window, current_weights) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Allocation via minimisation de la TE sur l'univers complet (mode 'returns').
        Estime la cov full (N x N), construit TrackingErrorFullUniverseObjective
        et résout le QP sur les K actifs investissables.
        """

        # Garde-fous : benchmark et fenêtre doivent être présents et valides
        if benchmark_weights is None or len(benchmark_weights) == 0:
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "missing_benchmark_weights", "mode": "returns"}

        if rw is None or rw.empty:
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "missing_returns_window", "mode": "returns"}

        cols_rw = list(rw.columns.astype(str))
        tickers_all_str = [str(t) for t in tickers_all]

        # Vérifie que tous les tickers de l'univers complet sont dans la fenêtre
        missing_in_rw = [t for t in tickers_all_str if t not in cols_rw]
        if len(missing_in_rw) > 0:
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "rw_missing_full_tickers", "n_missing": float(len(missing_in_rw)), "mode": "returns"}

        # Sélectionne les colonnes dans l'ordre pour la cov full
        rw_full = rw.loc[:, tickers_all_str].copy()

        if len(rw_full) < max(self.cfg.min_obs, 10):
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "window_too_short", "mode": "returns"}

        # Estime la matrice de covariance complète (N x N) via le provider
        cov_full = self._cov_subset(asof=asof, cols=tickers_all, returns_window=rw_full)
        cols_full = list(rw_full.columns.astype(str))
        n_full = len(cols_full)

        kept_tickers_str = [str(t) for t in kept_tickers]

        # Vérifie que tous les kept_tickers sont présents dans la cov full
        missing_kept = [t for t in kept_tickers_str if t not in cols_full]
        if len(missing_kept) > 0:
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "kept_not_in_full", "n_missing": float(len(missing_kept)), "mode": "returns"}

        # Indices des kept_tickers dans l'espace full (N)
        kept_idx_all = np.array([cols_full.index(t) for t in kept_tickers_str], dtype=int)
        if len(kept_idx_all) == 0:
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "no_kept_assets", "mode": "returns"}

        # Purge les actifs sans historique à cette date (variance nulle dans cov_full)
        cov_diag = np.diag(cov_full)

        # True = actif existe
        active_mask = cov_diag[kept_idx_all] > 1e-12          

        # indices dans kept_idx_all
        active_local = np.where(active_mask)[0]                

        # indices dans cov_full (N)
        kept_idx = kept_idx_all[active_local]                  

        # noms correspondants
        kept_tickers_active = [kept_tickers_str[i] for i in active_local]

        # nombre d'actifs passés au solveur
        k = len(kept_idx)

        # nombre d'actifs purgés
        n_purged = len(kept_idx_all) - k

        if k == 0:
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "no_active_kept_assets", "mode": "returns"}

        # Aligne les poids benchmark sur l'espace full (N)
        b_full = (benchmark_weights.reindex(cols_full).astype(float).fillna(0.0).values)

        # Vérifie la validité du vecteur benchmark (au moins 90% de valeurs finies, somme non nulle)
        if np.isfinite(b_full).mean() < 0.9 or float(np.abs(b_full).sum()) <= 0.0:
            w_fb = self._fallback_weights(current_weights, tickers_all)
            return w_fb, {"fallback": True, "reason": "benchmark_weights_invalid", "mode": "returns"}

        # Paramètre pour l'analyse des optimum sur la dernière date de rebal
        is_last = (self._last_rebal_dt is not None and pd.Timestamp(asof) >= self._last_rebal_dt)
        is_slsqp = isinstance(self.optimizer, SLSQPOptimizer)
        
        # Objectif TE pleine? kept_idx est maintenant purgé
        objective = TrackingErrorFullUniverseObjective(cov_full=cov_full, benchmark_weights_full=b_full, kept_idx=kept_idx, log_evaluations = is_last and is_slsqp,)

        # Prépare les contraintes et le point de départ
        eq, bounds = self._build_constraints(n=k)
        w0 = self._initial_guess(current_weights=current_weights, kept_tickers=kept_tickers_active)

        # Résout le QP
        w_active, success,solve_ms = self._solve(objective=objective,n_assets=k,eq=eq,bounds=bounds,w0=w0,
                                                metadata={"asof": str(asof), "mode": "returns", "objective": "TE_full_vs_benchmark_weights"},)
        

        # Log solveur sur la dernière date de rebal uniquement
        if is_last:
            if is_slsqp:
                # SLSQP : le log est déjà rempli par value() On ajoute juste le point de départ en premier si absent
                raw_log = list(objective.eval_log) if objective.eval_log else []
                self._solver_eval_log = raw_log

            else:
                # On calcule manuellement les 2 points : départ et arrivée
                te_w0  = float(np.sqrt(max(objective.value(w0), 0.0)) * np.sqrt(252.0))
                te_wst = float(np.sqrt(max(objective.value(np.asarray(w_active, dtype=float)), 0.0)) * np.sqrt(252.0))
                self._solver_eval_log = [(w0.copy(),te_w0), (np.asarray(w_active, dtype=float),  te_wst),]

        # Reconstruit le vecteur full (N) avec les poids optimaux sur les kept actifs
        w_full_vec = np.zeros(n_full, dtype=float)
        w_full_vec[kept_idx_all] = np.asarray(w_active, dtype=float)   # kept_idx_all

         # TE ex-ante : sqrt(a' Sigma a) avec a = w - b sur l'univers complet
        a_full = w_full_vec - b_full
        var = float(a_full.T @ np.asarray(cov_full, dtype=float) @ a_full)
        te_ex_ante = float(np.sqrt(max(var, 0.0)))

        # variance ex-ante du portefeuille 
        var_ptf_ante = float(w_full_vec.T @ cov_full @ w_full_vec)

        # Remapping final sur univers complet
        w_full_series = self._remap_to_full_universe(tickers_all_str, kept_tickers_active, w_active)

        diag = {
            "mode": "returns",
            "opt_success": bool(success),
            "n_full": float(n_full),
            "n_kept": float(len(kept_idx_all)),    # original avant purge
            "n_active": float(k),                  # actifs réellement dans le solveur
            "n_purged": float(n_purged),            # actifs retirés
            "te_ex_ante": te_ex_ante if np.isfinite(te_ex_ante) else np.nan,
            "var_ptf_ante":  var_ptf_ante if np.isfinite(var_ptf_ante) else np.nan,
            "objective": "TE_full_vs_benchmark_weights",
            "solve_ms": round(solve_ms, 3),
        }

        return w_full_series, diag

    # mode weights 
    def _allocate_weights(self,asof: pd.Timestamp,tickers_all: list[str],kept_tickers: list[str],returns_window: pd.DataFrame,rw: pd.DataFrame,
        current_weights: pd.Series, benchmark_weights: Optional[pd.Series],) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Allocation en mode 'weights' : redistribution benchmark-like ou TE-min vs poids.
 
        Si enable_weight_redistribution=True, CoreWeightRedistributor (sans solveur).
        Sinon, _optimize_vs_benchmark_weights (avec solveur).
        """

        # Recupere le paramètre de tolérance numérique mis en configuration
        eps = float(self.cfg.eps)

        if benchmark_weights is None:
            raise ValueError("[CoreTEMinAllocator/weights] benchmark_weights is required in mode='weights'.")

        # Poids benchmark complet avant projection sur l'investissable
        b_full = benchmark_weights.astype(float)
        bench_tickers_all = list(b_full.index.astype(str))

        # Vérification stricte : le benchmark ne doit pas contenir d'actifs hors univers
        if bool(self.cfg.strict_benchmark_weights) and (not bool(self.cfg.enable_weight_redistribution)):
            outside = set(b_full.index) - set(tickers_all)
            if outside:
                raise ValueError(
                    "[CoreTEMinAllocator/weights] benchmark_weights contains assets outside investable universe. "
                    "Provide aligned benchmark weights or set strict_benchmark_weights=False or enable_weight_redistribution=True."
                )

        # Projette et renormalise le benchmark sur l'univers investissable
        b_kept = b_full.reindex(kept_tickers).fillna(0.0).astype(float)
        sb = float(b_kept.sum())
        if sb <= eps:
            b_kept = pd.Series(1.0 / len(kept_tickers), index=kept_tickers, dtype=float)
        else:
            b_kept = b_kept / sb

        # Redistribution benchmark-like 
        if bool(self.cfg.enable_weight_redistribution):
            rep_cfg = CoreReplicationConfig(
                core_budget=1.0,
                method=self.cfg.redistribution_method,
                explicit_exclusions=list(self.cfg.explicit_exclusions) if self.cfg.explicit_exclusions else None,
                mcap_exclude_pct=self.cfg.mcap_exclude_pct,
                market_caps=self.cfg.market_caps,
                core_ticker_threshold=self.cfg.core_ticker_threshold,
                sector_map=self.cfg.sector_map,
                eps=self.cfg.eps,
            )

            # Instanciation du redistributeur
            redistributor = CoreWeightRedistributor(rep_cfg)

            # Construction des poids core répliquant le benchmark avec exclusions + freeze + redistribution
            w_kept_s = redistributor.build_core_weights_from_benchmark(
                asof=asof,                      
                w_bench=b_kept,                  # bench projeté sur investissable
                universe_tickers=kept_tickers,   # investissable kept
                bench_tickers_all=bench_tickers_all,  # bench complet pour filtre small caps
            ).reindex(kept_tickers).fillna(0.0).astype(float)

            # Verification long only 
            if bool(self.cfg.long_only):
                w_kept_s = w_kept_s.clip(lower=0.0)

            # Renormalise pour sommer à 1
            s = float(w_kept_s.sum())
            if s <= eps:
                w_kept_s[:] = 1.0 / len(kept_tickers)
            else:
                w_kept_s = w_kept_s / s

            # Calcul du TE ex-ante pour diagnostics 
            te_ex_ante = self._te_ex_ante_weights_mode(asof=asof,tickers_all=tickers_all,kept_tickers=kept_tickers,returns_window=returns_window,  w_kept=w_kept_s,b_kept=b_kept,)

            w_full = self._remap_to_full_universe(tickers_all, kept_tickers, w_kept_s.values)
            diag = {
                "mode": "weights",
                "redistribution": True,
                "method": str(self.cfg.redistribution_method),
                "n_kept": float(len(kept_tickers)),
                "te_ex_ante": te_ex_ante if np.isfinite(te_ex_ante) else np.nan,
            }
            
            return w_full, diag

        # Optimisation TE-min classique vs b
        return self._optimize_vs_benchmark_weights(asof=asof,tickers_all=tickers_all,kept_tickers=kept_tickers, returns_window=returns_window,rw=rw,current_weights=current_weights,b_kept=b_kept,)


    def _optimize_vs_benchmark_weights(self, asof, tickers_all, kept_tickers, returns_window, rw,current_weights, b_kept) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        TE-min classique vs poids benchmark (mode weights sans redistribution).
 
        Estime la cov sur kept_tickers, purge les actifs sans historique,
        construit TrackingErrorObjective et résout le QP.
        """

        # nombre d'actifs gardés (original, avant purge)
        n_original = len(kept_tickers)
        original_kept_tickers = list(kept_tickers)  # copie défensive

        # Calcul de la covariance
        cov_full = self._cov_full(asof=asof, df=rw)
        cols = list(rw.columns.astype(str))
        kept_idx = [cols.index(t) for t in kept_tickers]
        cov = cov_full[np.ix_(kept_idx, kept_idx)]

        if cov.shape != (n_original, n_original):
            raise ValueError( f"[CoreTEMinAllocator/weights_opt] cov_estimator must return shape {(n_original, n_original)}; got {cov.shape}.")

        # purge des actifs inexistants à cette date (ligne/colonne de zéros) 
        cov, kept_tickers, b_kept, active_local_idx = _remove_zero_cov_assets(cov, kept_tickers, b_kept)
        n = len(kept_tickers)  # espace réduit pour le solveur

        # Construction du problème d'optimisation
        objective = TrackingErrorObjective(benchmark_weights=b_kept.values, cov=cov)
        eq, bounds = self._build_constraints(n=n)
        w0 = self._initial_guess(current_weights=current_weights, kept_tickers=kept_tickers)

        # Résout le QP
        w_active, success,solve_ms = self._solve(objective=objective,n_assets=n,eq=eq,bounds=bounds,w0=w0,metadata={"asof": str(asof), "mode": "weights_opt"},)

        # Calcul TE ex-ante sur espace réduit 
        a = (w_active - b_kept.values)
        var = float(a.T @ cov @ a)
        te_ex_ante = float(np.sqrt(max(var, 0.0)))

        # remapping en deux niveaux : actif -> kept_original -> full 
        w_kept = np.zeros(n_original, dtype=float)
        for local_i, orig_i in enumerate(active_local_idx):
            w_kept[orig_i] = w_active[local_i]

        w_full = self._remap_to_full_universe(tickers_all, original_kept_tickers, w_kept)
        diag = {
            "mode": "weights_opt",
            "opt_success": bool(success),
            "n_kept": float(n_original),       # original pour la lisibilité
            "n_active": float(n),              # actifs réellement dans le solveur
            "te_ex_ante": te_ex_ante if np.isfinite(te_ex_ante) else np.nan,
            "solve_ms": round(solve_ms, 3),
        }
        return w_full, diag


    def _te_ex_ante_weights_mode(self,asof: pd.Timestamp,tickers_all: list[str],kept_tickers: list[str],returns_window: pd.DataFrame,w_kept: pd.Series,b_kept: pd.Series,) -> float:
        """
        Calcule la TE ex-ante pour le mode redistribution (diagnostics uniquement).
 
        Estime la cov full via le provider, extrait la sous-matrice sur kept_tickers,
        et calcule sqrt(a'Ca) avec a = w_kept - b_kept.
 
        Returns
        -------
        float
            TE ex-ante (non annualisée), ou NaN si le calcul échoue.
        """

        try:
            # Estime la cov full puis extrait la sous-matrice sur kept_tickers
            cov_full = self._cov_full(asof=asof, df=returns_window)
            cols = list(returns_window.columns.astype(str))
            kept_idx = [cols.index(t) for t in kept_tickers]
            cov = cov_full[np.ix_(kept_idx, kept_idx)]

            # Active weights : différence entre poids portefeuille et benchmark
            a = (w_kept.reindex(kept_tickers).values - b_kept.reindex(kept_tickers).values)
            var = float(a.T @ cov @ a)
            return float(np.sqrt(max(var, 0.0)))
        except Exception:
            return float("nan")


