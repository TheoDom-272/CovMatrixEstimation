"""
Règles de rebalancement conditionnel déclenchées pendant le backtest.
 
En complément du calendrier fixe défini par RebalanceSchedule, ces règles permettent
de déclencher un rebalancement supplémentaire si une condition de marché est remplie
(ex : la tracking error ex-post dépasse un seuil critique).
 
Classes
-------
RebalanceRule :
    Interface (Protocol) que toute règle de rebalancement doit respecter pour être branchée sur le moteur de backtest.
ExPostTEThresholdRule :
    Déclenche un rebalancement si la tracking error ex-post sur fenêtre glissante dépasse un seuil défini, avec gestion d'un cooldown entre deux triggers.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional

import numpy as np
import pandas as pd


class RebalanceRule(Protocol):
    """
    Interface qu'une règle de rebalancement conditionnel doit implémenter.
 
    Le moteur de backtest appelle should_rebalance() à chaque pas de temps,
    en plus du calendrier fixe. Si la méthode retourne True, un rebalancement
    est déclenché ce jour-là indépendamment du calendrier.
 
    Methods
    -------
    should_rebalance(dt, i, dates, portfolio_returns, benchmark_returns, current_weights, nav, weights_history) -> bool :
        Retourne True si un rebalancement doit être déclenché à la date dt.
    """

    def should_rebalance(
        self,
        dt: pd.Timestamp,
        i: int,
        dates: pd.DatetimeIndex,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        current_weights: pd.Series,
        nav: pd.Series,
        weights_history: pd.DataFrame,
    ) -> bool:
        ...


@dataclass(frozen=True)
class ExPostTEThresholdRule:
    """
    Règle de rebalancement déclenchée quand la tracking error ex-post dépasse un seuil.
 
    La TE ex-post est calculée comme l'écart-type annualisé des active returns
    (portefeuille - benchmark) sur une fenêtre glissante. Si elle dépasse `threshold`,
    un rebalancement est forcé. Un cooldown optionnel empêche des triggers consécutifs
    trop rapprochés.
 
    Attributes
    ----------
    threshold : float
        Seuil de TE annualisée au-delà duquel le rebalancement est déclenché (ex : 0.03 pour 3%).
    window : int
        Taille de la fenêtre glissante de calcul de la TE en jours de bourse (défaut : 252).
    annualization : float
        Facteur d'annualisation de la TE (défaut : sqrt(252)).
    cooldown : int
        Nombre minimal de jours de bourse entre deux triggers successifs.
        0 = pas de cooldown (un trigger peut se produire chaque jour).
    _last_trigger_index : int or None
        Index du dernier trigger (état interne, géré automatiquement).
 
    Methods
    -------
    should_rebalance(dt, i, dates, portfolio_returns, benchmark_returns, current_weights, nav, weights_history) -> bool :
        Retourne True si la TE ex-post sur la fenêtre courante dépasse le seuil, sous réserve que le cooldown soit respecté.
    """
 
    threshold: float
    window: int = 252
    annualization: float = float(np.sqrt(252.0))
    cooldown: int = 0

    # État minimal pour gérer le cooldown 
    _last_trigger_index: Optional[int] = None

    def should_rebalance(self,dt: pd.Timestamp,i: int,dates: pd.DatetimeIndex,portfolio_returns: pd.Series,benchmark_returns: pd.Series,
        current_weights: pd.Series,nav: pd.Series,weights_history: pd.DataFrame,) -> bool:

        """
        Evalue si un rebalancement conditionnel doit être déclenché à la date dt.
 
        Calcule la TE ex-post sur la fenêtre glissante et compare au seuil.
        Respecte le cooldown si configuré : si le dernier trigger est trop récent,
        retourne False même si le seuil est dépassé.
 
        Parameters
        ----------
        dt : pd.Timestamp
            Date courante dans la boucle de backtest.
        i : int
            Index entier de dt dans l'index de dates.
        dates : pd.DatetimeIndex
            Index complet des dates du backtest.
        portfolio_returns : pd.Series
            Série des rendements quotidiens du portefeuille.
        benchmark_returns : pd.Series
            Série des rendements quotidiens du benchmark.
        current_weights : pd.Series
            Poids courants du portefeuille (non utilisés ici).
        nav : pd.Series
            NAV du portefeuille (non utilisée ici).
        weights_history : pd.DataFrame
            Historique des poids (non utilisé ici).
 
        Returns
        -------
        bool
            True si la TE dépasse le seuil et que le cooldown est respecté.
        """
        
        # Conditions préalables : on doit avoir au moins 2 jours de données et une fenêtre d'au moins 2 jours pour calculer une TE significative
        if i <= 1:
            return False
        if self.window <= 1:
            return False

        # Cooldown
        if self.cooldown > 0 and self._last_trigger_index is not None:
            if (i - self._last_trigger_index) < self.cooldown:
                return False

        # Si on n'a pas assez de données pour remplir la fenêtre, on attend
        start = i - self.window + 1
        if start < 0:
            return False

        # Calcul de la serie de rendements actifs sur la fenêtre glissante
        active = (portfolio_returns.iloc[start : i + 1] - benchmark_returns.iloc[start : i + 1]).dropna()

        # Si la fenêtre est trop petite (ex: moins de 10 jours de données), on considère que la TE n'est pas fiable et on n'active pas le trigger
        if len(active) < max(10, self.window // 3):
            return False

        # Calcul de la TE annualisée
        te = float(active.std(ddof=1) * self.annualization)

        #si la TE dépasse le seuil, on mémorise l'index du trigger et on retourne True
        if te > self.threshold:
            object.__setattr__(self, "_last_trigger_index", i)
            return True
        
        # Sinon, pas de trigger
        return False
