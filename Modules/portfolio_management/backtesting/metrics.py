"""
Définition des métriques calculées pendant le backtest.
 
Chaque métrique suit le Protocol MetricCalculator : elle est réinitialisée en début
de backtest puis mise à jour à chaque pas de temps par le moteur.
 
Classes
-------
MetricCalculator :
    Interface (Protocol) que toute métrique doit respecter pour être branchée sur le moteur de backtest.
ExPostTrackingError :
    Calcule la tracking error ex-post annualisée sur une fenêtre glissante d'active returns.
DecisionMetricForwardFill :
    Propage en forward-fill une métrique ex-ante produite par la stratégie aux dates de rebalancement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import numpy as np
import pandas as pd


class MetricCalculator(Protocol):
    """
    Interface que toute métrique de backtest doit implémenter pour être branchée sur le moteur.
 
    Le moteur appelle reset() en début de run, puis update() à chaque pas de temps.
    La valeur retournée par update() est stockée dans la série de résultats correspondante.
 
    Attributes
    ----------
    name : str
        Nom de la métrique, utilisé comme clé dans les résultats du backtest.
 
    Methods
    -------
    reset(dates) -> None :
        Réinitialise l'état interne de la métrique avant le début du backtest.
    update(dt, i, dates, nav, portfolio_returns, benchmark_returns, weights, decision_diagnostics) -> float or None :
        Calcule et retourne la valeur de la métrique à la date dt.
        Retourne None si la métrique n'est pas définie à cette date.
    """

    name: str

    #fonctions à implémenter par les métriques concrètes
    def reset(self, dates: pd.DatetimeIndex) -> None: ...

    def update(self, dt: pd.Timestamp, i: int, dates: pd.DatetimeIndex, nav: pd.Series, portfolio_returns: pd.Series,
        benchmark_returns: pd.Series, weights: pd.DataFrame, decision_diagnostics: Optional[Dict[str, float]],) -> Optional[float]:
        """
        Calcule et retourne la valeur de la métrique au pas de temps dt.
 
        Parameters
        ----------
        dt : pd.Timestamp
            Date courante dans la boucle de backtest.
        i : int
            Index entier de dt dans l'index de dates du backtest.
        dates : pd.DatetimeIndex
            Index complet des dates du backtest.
        nav : pd.Series
            Série de NAV du portefeuille (partiellement remplie jusqu'à i).
        portfolio_returns : pd.Series
            Série des rendements quotidiens du portefeuille (partiellement remplie).
        benchmark_returns : pd.Series
            Série des rendements quotidiens du benchmark (partiellement remplie).
        weights : pd.DataFrame
            Historique des poids du portefeuille (partiellement rempli).
        decision_diagnostics : dict or None
            Dictionnaire de métriques ex-ante produites par la stratégie lors du dernier rebalancement (ex: TE ex-ante, nombre d'actifs). None si pas de rebalancement ce jour.
 
        Returns
        -------
        float or None
            Valeur de la métrique à dt, ou None si non définie.
        """


@dataclass
class ExPostTrackingError:
    """
    Tracking error ex-post annualisée calculée sur une fenêtre glissante d'active returns.
 
    La TE est calculée à chaque pas de temps comme l'écart-type annualisé des active returns
    (rendements portefeuille - rendements benchmark) sur les `window` derniers jours.
 
    Attributes
    ----------
    window : int
        Taille de la fenêtre glissante en jours de bourse (défaut : 252 = 1 an).
    annualization : float
        Facteur d'annualisation appliqué à l'écart-type (défaut : 252.0).
    name : str
        Nom de la métrique dans les résultats du backtest.
 
    Methods
    -------
    reset(dates) -> None :
        Pas d'état interne à réinitialiser pour cette métrique.
    update(dt, i, dates, nav, portfolio_returns, benchmark_returns, weights, decision_diagnostics) -> float or None :
        Calcule la TE ex-post annualisée sur la fenêtre courante. Retourne NaN si moins de 5 observations sont disponibles.
    """

    window: int = 252
    annualization: float = 252.0
    name: str = "te_ex_post"

    def reset(self, dates: pd.DatetimeIndex) -> None:
        pass

    def update(self,dt: pd.Timestamp,i: int,
        dates: pd.DatetimeIndex,nav: pd.Series,portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,weights: pd.DataFrame,decision_diagnostics: Optional[Dict[str, float]],) -> Optional[float]:

        """
        Calcule la tracking error ex-post annualisée à la date dt.
 
        Parameters
        ----------
        dt : pd.Timestamp
            Date courante.
        i : int
            Index entier de dt.
        dates : pd.DatetimeIndex
            Index complet des dates.
        nav : pd.Series
            NAV du portefeuille (non utilisée ici).
        portfolio_returns : pd.Series
            Rendements quotidiens du portefeuille.
        benchmark_returns : pd.Series
            Rendements quotidiens du benchmark.
        weights : pd.DataFrame
            Historique des poids (non utilisé ici).
        decision_diagnostics : dict or None
            Diagnostics de la stratégie (non utilisés ici).
 
        Returns
        -------
        float
            TE ex-post annualisée sur la fenêtre, ou NaN si insuffisamment d'observations.
        """

        if i < 1:
            return np.nan
        
        # Défini la fenêtre glissante
        start = max(0, i - self.window + 1)

        # calcule les active returns sur la fenêtre
        active = (portfolio_returns.iloc[start : i + 1] - benchmark_returns.iloc[start : i + 1]).dropna()

        if len(active) < 5:
            return np.nan
        
        # Retourne le TE annualisé
        return float(active.std(ddof=1) * np.sqrt(self.annualization))


@dataclass
class DecisionMetricForwardFill:
    """
    Métrique ex-ante produite par la stratégie aux dates de rebalancement, propagée en forward-fill.
 
    À chaque rebalancement, la stratégie produit un dictionnaire de diagnostics (ex: TE ex-ante,
    nombre d'actifs actifs). Cette métrique extrait la valeur associée à une clé donnée et la
    forward-fill entre les dates de rebalancement pour obtenir une série continue.
 
    Attributes
    ----------
    key : str
        Clé à extraire du dictionnaire decision_diagnostics produit par la stratégie.
    name : str
        Nom de la métrique dans les résultats du backtest. Si vide, prend la valeur de `key` par défaut.
 
    Methods
    -------
    reset(dates) -> None :
        Réinitialise la dernière valeur mémorisée à NaN avant le début du backtest.
    update(dt, i, dates, nav, portfolio_returns, benchmark_returns, weights, decision_diagnostics) -> float :
        Met à jour la valeur si un nouveau diagnostic est disponible, sinon forward-fill.
    """

    key: str
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.key

    def reset(self, dates: pd.DatetimeIndex) -> None:
        """Réinitialise la dernière valeur mémorisée à NaN."""
        self._last = np.nan

    def update(self,dt: pd.Timestamp,i: int,dates: pd.DatetimeIndex,nav: pd.Series,
        portfolio_returns: pd.Series,benchmark_returns: pd.Series,weights: pd.DataFrame,
        decision_diagnostics: Optional[Dict[str, float]],) -> Optional[float]:
        """
        Retourne la valeur courante de la métrique ex-ante, avec forward-fill entre rebalancements.
 
        Si decision_diagnostics contient la clé recherchée, la valeur est mise à jour.
        Sinon, la dernière valeur connue est renvoyée (forward-fill).
 
        Parameters
        ----------
        dt : pd.Timestamp
            Date courante.
        i : int
            Index entier de dt (non utilisé ici).
        dates : pd.DatetimeIndex
            Index complet des dates (non utilisé ici).
        nav : pd.Series
            NAV du portefeuille (non utilisée ici).
        portfolio_returns : pd.Series
            Rendements du portefeuille (non utilisés ici).
        benchmark_returns : pd.Series
            Rendements du benchmark (non utilisés ici).
        weights : pd.DataFrame
            Historique des poids (non utilisé ici).
        decision_diagnostics : dict or None
            Diagnostics produits par la stratégie lors du dernier rebalancement.
 
        Returns
        -------
        float
            Valeur courante de la métrique (mise à jour ou forward-fill).
        """
        
        # Met à jour la valeur si un nouveau diagnostic est disponible, sinon forward-fill
        if decision_diagnostics is not None and self.key in decision_diagnostics:
            self._last = float(decision_diagnostics[self.key])
        return float(self._last)
