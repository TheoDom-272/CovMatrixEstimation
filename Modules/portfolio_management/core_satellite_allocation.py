"""
Stratégies d'allocation core-satellite et façades pour le moteur de backtest.
 
Ce fichier expose les classes de stratégie branchées directement sur le moteur
de backtest via l'interface AllocationStrategy. Il agit comme couche de façade
au-dessus du moteur d'allocation core_replication.py.
 
Classes
-------
CoreTEMinStrategy :
    Façade principale pour la stratégie TE-min. Délègue toute la logique d'allocation à CoreTEMinAllocator (core_replication.py) et formate le résultat en AllocationDecision.
HierarchicalSectorInverseVolStrategy :
    Stratégie satellite standalone : allocation sectorielle hiérarchique par inverse-volatilité, sans optimisation.
CoreSatelliteAllocation :
    Orchestrateur qui combine une stratégie core et une stratégie satellite en un portefeuille blendé (core_weight * w_core + (1 - core_weight) * w_sat).
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Any, Dict

import numpy as np
import pandas as pd

from Modules.portfolio_management.backtesting.engine_types import AllocationDecision
from Modules.portfolio_management.strategies.core_replication import (CoreTEMinAllocator, CoreTEMinConfig)




@dataclass(frozen=True)
class CoreTEMinStrategy:
    """
    Façade stratégie pour la minimisation de la tracking error.
 
    Implémente l'interface AllocationStrategy du moteur de backtest et délègue
    entièrement la logique d'allocation à CoreTEMinAllocator (core_replication.py).
    Formate le résultat en AllocationDecision (poids + diagnostics).
 
    Attributes
    ----------
    cfg : CoreTEMinConfig
        Configuration complète de l'allocateur TE-min.
 
    Methods
    -------
    allocate(asof, returns_window, benchmark_returns_window, kept_assets, current_weights, benchmark_weights) -> AllocationDecision :
        Délègue l'allocation à CoreTEMinAllocator et retourne les poids et diagnostics.
    set_covariance_provider(provider) -> None :
        Injecte le CovarianceProvider du moteur de backtest dans l'allocateur interne.
    """

    cfg: CoreTEMinConfig = field(default_factory=CoreTEMinConfig)
    _impl: CoreTEMinAllocator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_impl", CoreTEMinAllocator(cfg=self.cfg))

    def allocate(self,asof: pd.Timestamp,returns_window: pd.DataFrame,benchmark_returns_window: pd.Series,
                kept_assets: list[str],current_weights: pd.Series,benchmark_weights: Optional[pd.Series] = None,) -> AllocationDecision:

        """
        Délègue l'allocation au moteur CoreTEMinAllocator et retourne un AllocationDecision.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date de rebalancement courante.
        returns_window : pd.DataFrame
            Fenêtre glissante de rendements (index = dates, colonnes = tickers).
        benchmark_returns_window : pd.Series
            Rendements du benchmark sur la même fenêtre.
        kept_assets : list[str]
            Sous-ensemble d'actifs investissables imposé par le moteur.
        current_weights : pd.Series
            Poids du portefeuille au rebalancement précédent.
        benchmark_weights : pd.Series or None
            Poids du benchmark à la date asof.
 
        Returns
        -------
        AllocationDecision
            Poids optimaux et diagnostics de l'allocation.
        """

        #Appel la méthode allocate du Moteur CoreTEMinAllocator
        w, diag = self._impl.allocate(asof=asof, returns_window=returns_window, benchmark_returns_window=benchmark_returns_window,
                                      kept_assets=kept_assets, current_weights=current_weights, benchmark_weights=benchmark_weights,)

        # renvoie AllocationDecision
        return AllocationDecision(weights=w, diagnostics=diag) 
    

    def set_covariance_provider(self, provider) -> None:
        """
        Injecte le CovarianceProvider du moteur de backtest dans l'allocateur interne.
 
        Appelé automatiquement par BacktestEngine.run() si la stratégie expose
        cette méthode. Permet au provider précomputé d'être partagé
        entre la stratégie et le moteur sans recalcul.
 
        Parameters
        ----------
        provider : CovarianceProvider
            Provider de covariance précomputé ou on-demand.
        """

        if hasattr(self, "_impl") and hasattr(self._impl, "set_covariance_provider"):
            self._impl.set_covariance_provider(provider)



@dataclass(frozen=True)
class HierarchicalSectorInverseVolStrategy:
    """
    Stratégie satellite standalone : allocation sectorielle hiérarchique par inverse-volatilité.
 
    Alloue en deux étapes sans solveur :
    1. Poids par secteur = 1 / vol(secteur) ^ sector_power (la vol du secteur est approximée par la vol moyenne de ses membres).
    2. Poids intra-secteur = 1 / vol(actif) ^ asset_power
 
    Attributes
    ----------
    sector_map : Mapping or pd.Series
        Mapping ticker - secteur. Tous les actifs hors mapping sont assignés à 'UNKNOWN'.
    lookback : int
        Nombre de jours utilisés pour calculer les volatilités (fenêtre glissante).
    sector_power : float
        Exposant appliqué à l'inverse-vol sectorielle (1.0 = proportionnel à 1/vol).
    asset_power : float
        Exposant appliqué à l'inverse-vol intra-sectorielle.
    vol_floor : float
        Plancher de volatilité pour éviter les divisions par zéro.
    long_only : bool
        Si True, tous les poids sont clippés à 0 avant renormalisation.
    max_weight : float or None
        Poids maximum par actif (ex: 0.05 = max 5%). None = pas de contrainte.
 
    Methods
    -------
    allocate(asof, returns_window, benchmark_returns_window, current_weights, benchmark_weights) -> pd.Series :
        Calcule et retourne les poids par inverse-vol sectorielle.
    """

    sector_map: Mapping[str, str] | pd.Series
    lookback: int = 60
    sector_power: float = 1.0
    asset_power: float = 1.0
    vol_floor: float = 1e-6
    long_only: bool = True
    max_weight: Optional[float] = None 


    def allocate(self, asof: pd.Timestamp, returns_window: pd.DataFrame,  benchmark_returns_window: pd.Series, 
                 current_weights: pd.Series, benchmark_weights: Optional[pd.Series] = None,) -> pd.Series:
        
        """
        Calcule les poids par inverse-volatilité sectorielle hiérarchique.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date de rebalancement (non utilisée directement, passée pour compatibilité).
        returns_window : pd.DataFrame
            Fenêtre de rendements (les lookback dernières lignes sont utilisées).
        benchmark_returns_window : pd.Series
            Rendements du benchmark (non utilisés ici).
        current_weights : pd.Series
            Poids courants (non utilisés, présents pour compatibilité d'interface).
        benchmark_weights : pd.Series or None
            Poids du benchmark (non utilisés ici).
 
        Returns
        -------
        pd.Series
            Poids par actif, indexés par les colonnes de returns_window, sommant à 1.
        """

        # Récupère la liste des tickers
        tickers = list(returns_window.columns)

        # Nombre de ticker
        n = len(tickers)

        # Vérification qu'on est des tickers
        if n == 0:
            raise ValueError("Empty universe in returns_window.")

        # Récupération des rendements sur la fenetre de lookback
        rw = returns_window.tail(int(self.lookback))

        #Vérification qu'on est au moins 10 rendements
        if len(rw) < 10:
            return pd.Series(1.0 / n, index=tickers, dtype=float)

        # Calcule de la std par actif
        vol_i = rw.std(ddof=1).astype(float).replace(0.0, np.nan)

        #Inversement des variances avec un floor pour éviter les divisions par 0
        inv_i = 1.0 / np.maximum(vol_i.values, float(self.vol_floor))

        # Application de l'exposant par actif
        inv_i = inv_i ** float(self.asset_power)

        # Récupération des inverses vol positif, si négatif fixé à 0
        inv_i = np.where(np.isfinite(inv_i), inv_i, 0.0)
        inv_i_s = pd.Series(inv_i, index=tickers, dtype=float)

        # Mapping secteurs
        if isinstance(self.sector_map, pd.Series):
            sec = self.sector_map.reindex(tickers).fillna("UNKNOWN").astype(str)
        else:
            sec = pd.Series({t: self.sector_map.get(t, "UNKNOWN") for t in tickers})

        #Récupère la liste des secteurs
        sector_list = sorted(sec.unique())
        sector_inv = {}

        # Pour chaque secteurs, classes les actifs avec leur vol et calcul l'inverse volatilité du secteur
        for sname in sector_list:
            members = sec[sec == sname].index
            v = vol_i.reindex(members).values.astype(float)
            v = np.where(np.isfinite(v), v, np.nan)
            if np.all(np.isnan(v)):
                sector_inv[sname] = 0.0
            else:
                v_mean = float(np.nanmean(v))
                sector_inv[sname] = (1.0 / max(v_mean, float(self.vol_floor))) ** float(self.sector_power)


        sector_inv = pd.Series(sector_inv, dtype=float)
        if float(sector_inv.sum()) <= 0.0:
            return pd.Series(1.0 / n, index=tickers, dtype=float)

        w_sector = sector_inv / float(sector_inv.sum())

        # intra-secteur
        w = pd.Series(0.0, index=tickers, dtype=float)
        for sname, ws in w_sector.items():
            members = sec[sec == sname].index
            if len(members) == 0:
                continue
            inv_m = inv_i_s.reindex(members).fillna(0.0)
            if float(inv_m.sum()) <= 0.0:
                w.loc[members] = ws * (1.0 / len(members))
            else:
                w.loc[members] = ws * (inv_m / float(inv_m.sum()))

        # cap optionnel
        if self.max_weight is not None:
            cap = float(self.max_weight)
            w = w.clip(lower=0.0, upper=cap)
            s = float(w.sum())
            w = (w / s) if s > 1e-12 else pd.Series(1.0 / n, index=tickers, dtype=float)

        if self.long_only:
            w = w.clip(lower=0.0)
        s = float(w.sum())
        return (w / s) if s > 1e-12 else pd.Series(1.0 / n, index=tickers, dtype=float)


# 3) Orchestrateur Core + Satellite (blend)
@dataclass(frozen=True)
class CoreSatelliteAllocation:
    """
    Orchestrateur qui combine une stratégie core et une stratégie satellite.
 
    Calcule séparément les poids core et satellite, puis les blende selon
    core_weight. Si satellite=None ou core_weight >= 1, retourne directement
    le core fully invested.
 
    Attributes
    ----------
    core : Any
        Stratégie core (doit implémenter allocate()). Typiquement CoreTEMinStrategy.
    satellite : Any or None
        Stratégie satellite (doit implémenter allocate()). None = core only.
    core_weight : float
        Part du portefeuille allouée au core (entre 0 et 1). Le satellite reçoit (1 - core_weight).
 
    Methods
    -------
    allocate(asof, returns_window, benchmark_returns_window, current_weights, benchmark_weights) -> AllocationDecision :
        Calcule les poids blendés core + satellite et retourne un AllocationDecision.
    """

    core: Any
    satellite: Optional[Any] = None
    core_weight: float = 1.0

    @staticmethod
    def _unwrap_weights(x: Any, tickers: list[str]) -> pd.Series:

        """
        Extrait les poids depuis un AllocationDecision ou une Series.
 
        Parameters
        ----------
        x : AllocationDecision, pd.Series, or array-like
            Résultat d'une stratégie (AllocationDecision ou poids directs).
        tickers : list[str]
            Index cible pour le résultat (reindex + fillna(0)).
 
        Returns
        -------
        pd.Series
            Poids indexés par tickers (valeurs manquantes mises à 0).
        """

        # AllocationDecision
        if hasattr(x, "weights"):
            w = getattr(x, "weights")
        else:
            w = x

        # Series
        if isinstance(w, pd.Series):
            return w.reindex(tickers).fillna(0.0).astype(float)

        # array-like
        try:
            arr = np.asarray(w, dtype=float)
            if arr.ndim != 1 or arr.shape[0] != len(tickers):
                raise ValueError("Invalid weights shape.")
            return pd.Series(arr, index=tickers, dtype=float)
        except Exception:
            return pd.Series(0.0, index=tickers, dtype=float)

    @staticmethod
    def _unwrap_diag(x: Any) -> Dict[str, Any]:
        """
        Extrait le dictionnaire de diagnostics depuis un AllocationDecision ou autre.
 
        Parameters
        ----------
        x : AllocationDecision or other
            Résultat d'une stratégie.
 
        Returns
        -------
        dict
            Dictionnaire de diagnostics (vide si absent).
        """

        if hasattr(x, "diagnostics"):
            d = getattr(x, "diagnostics")
            return dict(d) if isinstance(d, dict) else {}
        return {}
    

    def allocate(self,asof: pd.Timestamp,returns_window: pd.DataFrame,benchmark_returns_window: pd.Series,current_weights: pd.Series,
        benchmark_weights: Optional[pd.Series] = None,) -> AllocationDecision:
        """
        Calcule les poids blendés core + satellite.
 
        Appelle core.allocate() et, si un satellite est configuré,
        satellite.allocate(). Blende les deux selon core_weight.
        Si satellite=None ou core_weight >= 1, retourne le core seul.
 
        Parameters
        ----------
        asof : pd.Timestamp
            Date de rebalancement courante.
        returns_window : pd.DataFrame
            Fenêtre glissante de rendements.
        benchmark_returns_window : pd.Series
            Rendements du benchmark sur la même fenêtre.
        current_weights : pd.Series
            Poids du portefeuille au rebalancement précédent.
        benchmark_weights : pd.Series or None
            Poids du benchmark à la date asof.
 
        Returns
        -------
        AllocationDecision
            Poids blendés normalisés et diagnostics combinés (core + satellite).
        """

        # Liste des tickers
        tickers = list(returns_window.columns)

        #Nombre de tickers
        n = len(tickers)

        #Vérification du nombre de ticker
        if n == 0:
            raise ValueError("Empty universe in returns_window.")
        
        # Appel de la stratégie core
        core_dec = self.core.allocate(asof=asof, returns_window=returns_window, benchmark_returns_window=benchmark_returns_window,
                                      current_weights=current_weights, benchmark_weights=benchmark_weights,)
        
        #Récupération des poids et diagnostic de la stratégie core
        w_core = self._unwrap_weights(core_dec, tickers)
        d_core = self._unwrap_diag(core_dec)

        # Si pas de stratégie satallite
        if self.satellite is None or float(self.core_weight) >= 0.999999:
            # Somme des poids de la partie core
            s = float(w_core.sum())

            # Normalisation défensive
            w = (w_core / s) if s > 1e-12 else pd.Series(1.0 / n, index=tickers, dtype=float)

            # Retourne la décision d'allocation avec 100% des poids Core
            return AllocationDecision(weights=w.astype(float), diagnostics={"blend": "core_only", **d_core})

        # Appel de la stratégie Satellite
        sat_dec = self.satellite.allocate(asof=asof, returns_window=returns_window, benchmark_returns_window=benchmark_returns_window,
                                          current_weights=current_weights, benchmark_weights=benchmark_weights,)

        # Récupération des poids et diagnostics de la stratégie Satellite
        w_sat = self._unwrap_weights(sat_dec, tickers)
        d_sat = self._unwrap_diag(sat_dec)

        #Calcul du poids attribué à la partie Satelitte (1 - w_core)
        cw = min(max(float(self.core_weight), 0.0), 1.0)
        w = cw * w_core + (1.0 - cw) * w_sat

        #Somme des poids finaux
        s = float(w.sum())

        #Normalisation
        if s <= 1e-12:
            w = pd.Series(1.0 / n, index=tickers, dtype=float)
        else:
            w = w / s

        # Diagnostics final
        diagnostics = {
            "blend": "core_satellite",
            "core_weight": cw,
            "sat_weight": float(1.0 - cw),
            "core": d_core,
            "satellite": d_sat,
        }

        #Renvoie l'allocatioDecision avec les poids finaux et le diagnostique finale
        return AllocationDecision(weights=w.astype(float), diagnostics=diagnostics)
