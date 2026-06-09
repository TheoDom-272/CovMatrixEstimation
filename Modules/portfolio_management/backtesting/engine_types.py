# Modules/portfolio_management/backtesting/engine_types.py

"""
Contrats d'interface et structures de données du moteur de backtest.

Ce fichier regroupe tous les dataclasses et Protocols utilisés par le moteur.
Il ne contient aucune logique de calcul, uniquement les types qui circulent
entre les composants du backtest (config, décisions, résultats).

Classes
-------
AllocationStrategy :
    Interface (Protocol) que toute stratégie d'allocation doit implémenter pour être branchée sur le moteur de backtest.
AllocationDecision :
    Conteneur retourné par une stratégie lors d'un rebalancement, regroupant les poids cibles et les diagnostics associés.
BacktestConfig :
    Configuration complète du moteur de backtest : schedule, lags, coûts, règles de rebalancement, logging, et fournisseur de covariance.
ExecutionConfig :
    Paramètres d'exécution pour la conversion poids en quantités (lot size, notionnel minimum, arrondi).
CashConfig :
    Paramètres de gestion de la poche cash structurelle et de son rendement.
FlowConfig :
    Paramètres de gestion des flux de souscription/rachat.
BacktestResult :
    Objet de sortie du backtest, contenant NAV, rendements, poids, diagnostics, métriques et journaux de trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, Literal

import numpy as np
import pandas as pd

from .cost import TransactionCostModel
from .rebalancing import RebalanceSchedule
from .rebalance_rules import RebalanceRule
from .metrics import MetricCalculator
from .covariance_provider import CovConfig


class AllocationStrategy(Protocol):
    """
    Interface que toute stratégie d'allocation doit respecter pour être utilisée par le moteur.

    Le moteur appelle allocate() à chaque date de rebalancement en lui fournissant
    la fenêtre de rendements et les poids courants. La stratégie retourne les poids
    cibles, éventuellement accompagnés de diagnostics.

    Methods
    -------
    allocate(asof, returns_window, benchmark_returns_window, kept_assets, current_weights, benchmark_weights) -> pd.Series or AllocationDecision :
        Calcule et retourne les poids cibles à la date asof. Peut retourner une Series simple ou un AllocationDecision avec diagnostics.
    """

    def allocate(
        self,
        asof: pd.Timestamp,
        returns_window: pd.DataFrame,
        benchmark_returns_window: pd.Series,
        kept_assets: list[str],
        current_weights: pd.Series,
        benchmark_weights: Optional[pd.Series] = None,
    ) -> "pd.Series | AllocationDecision":
        ...


@dataclass(frozen=True)
class AllocationDecision:
    """
    Conteneur de décision d'allocation retourné par une stratégie lors d'un rebalancement.

    Regroupe les poids cibles et un dictionnaire de diagnostics optionnels
    (ex : TE ex-ante, nombre d'actifs actifs, statut du solveur).

    Attributes
    ----------
    weights : pd.Series
        Poids cibles du portefeuille, indexés par ticker, sommant à 1.
    diagnostics : dict
        Métriques produites par la stratégie lors de la décision
    """

    weights: pd.Series
    diagnostics: Dict[str, float]


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration complète du moteur de backtest.

    Regroupe tous les paramètres nécessaires au déroulement du backtest

    Attributes
    ----------
    schedule : RebalanceSchedule
        Calendrier de rebalancement (fréquence et timing).
    lookback : int
        Taille de la fenêtre historique utilisée pour le calcul du signal (en jours).
    cost_model : TransactionCostModel
        Modèle de coûts de transaction appliqué à chaque rebalancement.
    drop_na_assets : bool
        Si True, les actifs avec rendement manquant sont mis à 0 plutôt que de lever une erreur.
    allow_bench_outside_universe : bool
        Si True, le benchmark peut contenir des actifs absents de l'univers investissable.
    data_lag : int
        Nombre de jours de décalage entre la date de décision et la dernière donnée utilisée (évite le look-ahead bias).
    apply_lag : int
        Nombre de jours entre la décision et l'application effective des nouveaux poids.
    bench_lag : int or None
        Lag appliqué aux poids du benchmark. Si None, prend la valeur de data_lag.
    bench_rebalance_offset : int
        Décalage en jours de trading entre les dates de rebalancement du portefeuille et celles du benchmark (négatif = le benchmark se rebalance avant).
    rebalance_rules : tuple of RebalanceRule
        Règles de rebalancement conditionnel déclenchées en plus du calendrier fixe.
    rebalance_metrics : tuple of MetricCalculator
        Métriques plugables calculées à chaque pas de temps pendant le backtest.
    verbose : bool
        Active le logging console à chaque rebalancement si True.
    log_on_rebalance_only : bool
        Si True, le log ne s'affiche qu'aux dates de rebalancement.
    log_every_n_rebalances : int
        Fréquence du log : 1 = tous les rebalancements, 2 = un sur deux, etc.
    execution : ExecutionConfig or None
        Paramètres d'exécution pour le mode positions (lot size, notionnel minimum). None = mode poids uniquement.
    cash : CashConfig or None
        Paramètres de gestion de la poche cash. None = pas de cash explicite.
    flows : FlowConfig or None
        Paramètres de gestion des flux de souscription/rachat. None = pas de flux.
    cov : CovConfig or None
        Configuration du fournisseur de covariance injecté dans la stratégie.
    """

    schedule: RebalanceSchedule
    lookback: int = 121
    cost_model: TransactionCostModel = field(default_factory=lambda: TransactionCostModel(0.0, 0.0))
    drop_na_assets: bool = True
    allow_bench_outside_universe: bool = False

    data_lag: int = 1
    apply_lag: int = 1
    bench_lag: Optional[int] = None
    bench_rebalance_offset: int = -2

    rebalance_rules: Tuple[RebalanceRule, ...] = ()
    rebalance_metrics: tuple[MetricCalculator, ...] = ()

    verbose: bool = False
    log_on_rebalance_only: bool = True
    log_every_n_rebalances: int = 1

    execution: Optional["ExecutionConfig"] = None
    cash: Optional["CashConfig"] = None
    flows: Optional["FlowConfig"] = None

    cov: Optional[CovConfig] = None


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Paramètres d'exécution pour la conversion poids cibles en quantités (mode positions).

    Utilisé uniquement quand des prix sont fournis au moteur (universe_prices != None).

    Attributes
    ----------
    lot_size : int
        Taille minimale d'un ordre en nombre d'actions (1 = pas de contrainte de lot).
    min_trade_notional : float
        Notionnel minimum en-dessous duquel un ordre est annulé (en unités monétaires).
    min_position_notional : float
        Notionnel minimum d'une position maintenue. Les positions plus petites sont coupées à 0.
    min_weight : float
        Poids minimum en-dessous duquel un actif est exclu de l'allocation (ex: 0.0001 = 1 bp).
    rounding : str
        Méthode d'arrondi des quantités : 'floor' (conservateur) ou 'round' (standard).
    allow_fractional : bool
        Si True, les quantités fractionnaires sont autorisées.
    """

    lot_size: int = 1
    min_trade_notional: float = 100.0
    min_position_notional: float = 200.0
    min_weight: float = 0.0001
    rounding: Literal["floor", "round"] = "floor"
    allow_fractional: bool = False


@dataclass(frozen=True)
class CashConfig:
    """
    Paramètres de gestion de la poche cash et de son rendement.

    Attributes
    ----------
    target_cash_weight : float
        Part cible du portefeuille maintenue en cash.
    cash_return : float
        Rendement journalier constant du cash.
    """

    target_cash_weight: float = 0.01 # 1%
    cash_return: float = 0.0 # Non rémunéré par défaut


@dataclass(frozen=True)
class FlowConfig:
    """
    Paramètres de gestion des flux de souscription et de rachat.

    Attributes
    ----------
    timing : str
        Moment d'application du flux dans la journée. 'close' = appliqué au close.
    invest_immediately : bool
        Si True, le flux entrant est immédiatement investi pro-rata lors du même rebalancement. Si False, il reste en cash jusqu'au prochain rebalancement.
    """

    timing: Literal["close"] = "close"
    invest_immediately: bool = False


@dataclass(frozen=True)
class BacktestResult:
    """
    Objet de sortie du backtest, produit par BacktestEngine.run().

    Contient l'ensemble des séries temporelles et diagnostics produits pendant le backtest,
    depuis le premier rebalancement réussi jusqu'à la fin de la période.

    Attributes
    ----------
    nav : pd.Series
        Valeur liquidative du portefeuille, normalisée à 1.0 au premier rebalancement.
    portfolio_returns : pd.Series
        Rendements journaliers du portefeuille (net de coûts).
    benchmark_returns : pd.Series
        Rendements journaliers du benchmark (driftés entre rebalancements).
    weights : pd.DataFrame
        Poids du portefeuille au close (copie de weights_close, plus prudent).
    weights_in_force : pd.DataFrame
        Poids effectifs en séance (valorisés au close t-1, avant drift du jour).
    weights_close : pd.DataFrame
        Poids au close après drift journalier (et après exécution si mode positions).
    diagnostics : dict
        Métriques globales du backtest (nombre de jours, NAV finale, TE ex-post annualisée).
    metrics : pd.DataFrame
        Séries temporelles des métriques plugables.
    cash : pd.Series or None
        Valeur de la poche cash quotidienne (mode positions uniquement).
    holdings : pd.DataFrame or None
        Quantités par actif à chaque date (mode positions uniquement).
    trades : pd.DataFrame or None
        Journal de tous les trades exécutés (mode positions uniquement).
    bench_weights_in_force : pd.DataFrame or None
        Poids du benchmark en séance (avant drift).
    bench_weights_close : pd.DataFrame or None
        Poids du benchmark au close (après drift et rebalancement benchmark).
    rebal_dates : pd.DatetimeIndex or None
        Dates de décision de rebalancement du portefeuille.
    first_rebal_date : pd.Timestamp or None
        Date du premier rebalancement réussi (point de départ des séries de sortie).
    rebal_diagnostics : pd.DataFrame or None
        Diagnostics produits par la stratégie à chaque rebalancement (indexés par date).
    solver_eval_log : list or None
        Log d'évaluation du solveur (convergence, statut) si disponible.
    """

    nav: pd.Series
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    weights: pd.DataFrame
    weights_in_force: pd.DataFrame
    weights_close: pd.DataFrame
    diagnostics: Dict[str, float]
    metrics: pd.DataFrame
    cash: Optional[pd.Series] = None
    holdings: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    bench_weights_in_force: Optional[pd.DataFrame] = None
    bench_weights_close: Optional[pd.DataFrame] = None
    rebal_dates: Optional[pd.DatetimeIndex] = None
    first_rebal_date: Optional[pd.Timestamp] = None
    rebal_diagnostics: Optional[pd.DataFrame] = None
    solver_eval_log: Optional[list] = None