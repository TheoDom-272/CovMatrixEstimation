"""
Évaluation économique des estimateurs de covariance par backtest TE-min.

Ce fichier implémente le backtest économique des stratégies de minimisation
de tracking error ex-ante. Pour chaque estimateur de covariance, un backtest
complet est lancé sur données réelles via le moteur BacktestEngine.

Classes
-------
EcoResult :
    Dataclass contenant les résultats agrégés d'un backtest économique multi-modèles.
EcoStudy :
    Classe principale contenant les méthodes de backtest économique.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from Modules.Financial_engineering.statistics.multivariate_vol_estimation import (
    _safe_symmetrize,
    purge_memmap_cache,
)

if TYPE_CHECKING:
    from Modules.portfolio_management.backtesting.engine_types import BacktestResult




@dataclass
class EcoResult:
    """
    Classe contenant les résultats d'une évaluation économique multi-modèles.

    Attributes
    ----------
    te_series : dict
        Dictionnaire modèle -> série temporelle des active returns (TE ex-post journaliers).
    te_summary : pd.DataFrame
        DataFrame (index=modèle) avec les métriques TE agrégées (te_ann, te_vol, te_mse, te_mae).
    bt_results : dict
        Dictionnaire modèle -> BacktestResult complet produit par BacktestEngine.
    best_model : str
        Nom du modèle avec la TE annualisée la plus faible.
    """

    te_series:  Dict[str, pd.Series]
    te_summary: pd.DataFrame
    bt_results: Dict[str, Any]
    best_model: str


class EcoStudy:
    """
    Classe contenant les méthodes de backtest économique des estimateurs de covariance.

    Lance un backtest de minimisation de tracking error ex-ante pour chaque modèle
    spécifié, en utilisant le moteur BacktestEngine. Les résultats sont comparés
    via les métriques de TE ex-post.

    Methods
    -------
    backtest_expost_metrics(result, ann_factor) -> dict :
        Calcule les métriques standardisées depuis un BacktestResult.
    economic_backtests_min_te(...) -> EcoResult :
        Lance les backtests pour tous les modèles et agrège les résultats.
    """

    @staticmethod
    def _safe_float(x) -> float:
        """Convertit x en float, retourne NaN si la conversion échoue."""
        try:
            return float(x)
        except Exception:
            return float("nan")

    @classmethod
    def backtest_expost_metrics(cls, result: "BacktestResult", ann_factor: int = 252,) -> Dict[str, float]:
        """
        Calcule des métriques standardisées à partir d'un BacktestResult du moteur.

        Calcule la TE journalière, annualisée, le ratio d'information, le MSE/MAE,
        et les performances cumulées du portefeuille et du benchmark.

        Parameters
        ----------
        result : BacktestResult
            Résultat de backtest produit par BacktestEngine.run().
        ann_factor : int
            Facteur d'annualisation (252 pour données journalières).

        Returns
        -------
        dict
            Dictionnaire des métriques : te_daily, te_ann, information_ratio,
            te_mse, te_mae, bench_cum, port_cum, rel_cum_vs_bench.

        Raises
        ------
        ValueError
            Si portfolio_returns ou benchmark_returns est absent du BacktestResult.
        """

        # Récupère les séries de rendements portefeuille et benchmark
        port = getattr(result, "portfolio_returns", None)
        bench = getattr(result, "benchmark_returns", None)

        if port is None or bench is None:
            raise ValueError("BacktestResult must expose portfolio_returns and benchmark_returns.")

        # Nettoie et aligne les deux séries sur leur index commun
        port  = pd.Series(port).dropna()
        bench = pd.Series(bench).dropna()
        idx   = port.index.intersection(bench.index)
        port  = port.reindex(idx)
        bench = bench.reindex(idx)

        # Active returns : différence entre rendements portefeuille et benchmark
        active = port - bench

        # TE journalière = écart-type des active returns
        te_daily = active.std(ddof=1)
        te_ann   = te_daily * math.sqrt(ann_factor)

        # Rendement actif annualisé
        active_mean_daily = active.mean()
        active_mean_ann   = active_mean_daily * ann_factor

        # IR : rendement actif annualisé divisé par la TE annualisée
        ir = (active_mean_ann / te_ann) if (te_ann and te_ann > 0) else float("nan")

        # MSE et MAE sur les active returns
        te_mse = float(np.mean(np.square(active.values))) if len(active) else float("nan")
        te_mae = float(np.mean(np.abs(active.values)))    if len(active) else float("nan")

        # Performances cumulées sur la période complète
        bench_cum = float((1.0 + bench).cumprod().iloc[-1]) if len(bench) else float("nan")
        port_cum  = float((1.0 + port).cumprod().iloc[-1])  if len(port)  else float("nan")

        # Performance relative du portefeuille par rapport au benchmark
        rel_cum = (port_cum / bench_cum - 1.0) if (bench_cum and bench_cum > 0) else float("nan")

        return {
            "T":                 float(len(idx)),
            "te_daily":          cls._safe_float(te_daily),
            "te_ann":            cls._safe_float(te_ann),
            "active_mean_ann":   cls._safe_float(active_mean_ann),
            "information_ratio": cls._safe_float(ir),
            "te_mse":            cls._safe_float(te_mse),
            "te_mae":            cls._safe_float(te_mae),
            "bench_cum":         cls._safe_float(bench_cum),
            "port_cum":          cls._safe_float(port_cum),
            "rel_cum_vs_bench":  cls._safe_float(rel_cum),
        }

    @staticmethod
    def _te_metrics(te: pd.Series, ann_factor: int = 252) -> Dict[str, float]:
        """
        Calcule les métriques de tracking error depuis une série d'active returns.

        Parameters
        ----------
        te : pd.Series
            Série d'active returns journaliers.
        ann_factor : int
            Facteur d'annualisation.

        Returns
        -------
        dict
            Dictionnaire des métriques : te_vol, te_ann, te_mse, te_mae.
        """

        # Nettoie la série des active returns et convertit en numpy array
        x = te.dropna().values
        if len(x) == 0:
            return {"te_vol": np.nan, "te_ann": np.nan, "te_mse": np.nan, "te_mae": np.nan}

        # TE MSE et MAE sur les active returns
        te_mse = float(np.mean(x**2))
        te_mae = float(np.mean(np.abs(x)))

        # TE vol = RMS des active returns (racine du MSE)
        te_vol = float(np.sqrt(te_mse))
        te_ann = float(te_vol * np.sqrt(ann_factor))
        return {"te_vol": te_vol, "te_ann": te_ann, "te_mse": te_mse, "te_mae": te_mae}

    @classmethod
    def economic_backtests_min_te(
        cls,
        all_returns: pd.DataFrame,
        universe_returns: pd.DataFrame,
        universe_prices: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        rebal_dates_port: pd.DatetimeIndex,
        rebal_dates_bench: pd.DatetimeIndex,
        model_specs: List[Any],
        engine_kwargs: Optional[Dict[str, Any]] = None,
        port_root=None,
        enabled_export: bool = False,
        ann_factor: int = 252,
        precomputed_cov_provider=None,
        indice_name=None,
    ) -> EcoResult:
        """
        Lance les backtests de minimisation de TE ex-ante pour chaque modèle.

        Pour chaque ModelSpec, configure et lance un BacktestEngine avec la stratégie
        CoreTEMinStrategy. Les résultats sont agrégés dans un EcoResult.

        Parameters
        ----------
        all_returns : pd.DataFrame
            Rendements de tous les actifs sur la période complète.
        universe_returns : pd.DataFrame
            Rendements de l'univers investissable.
        universe_prices : pd.DataFrame
            Prix de l'univers investissable (pour l'inventaire portefeuille).
        benchmark_weights : pd.DataFrame
            Poids du benchmark à chaque date.
        rebal_dates_port : pd.DatetimeIndex
            Dates de rebalancement du portefeuille.
        rebal_dates_bench : pd.DatetimeIndex
            Dates de rebalancement du benchmark.
        model_specs : list
            Liste de ModelSpec (name, cov_cfg, optimizer_name) à backtester.
        engine_kwargs : dict or None
            Paramètres additionnels du moteur (rebal_freq, verbose).
        port_root : Path or None
            Répertoire de sortie pour les inventaires portefeuille.
        enabled_export : bool
            Si True, active l'export des inventaires portefeuille.
        ann_factor : int
            Facteur d'annualisation.
        precomputed_cov_provider : CovarianceProvider or None
            Fournisseur de covariance précomputé (mode path).
        indice_name : str or None
            Nom de l'indice (utilisé pour les fichiers d'export et la config restrict_to_benchmark).

        Returns
        -------
        EcoResult
            Résultats agrégés : TE series, TE summary, BacktestResults, meilleur modèle.
        """
        # Imports locaux pour éviter les imports circulaires au chargement du module
        from Modules.portfolio_management.backtesting.cost import TransactionCostModel
        from Modules.portfolio_management.backtesting.covariance_provider import CovarianceProvider
        from Modules.portfolio_management.core_satellite_allocation import CoreTEMinStrategy
        from Modules.portfolio_management.backtesting.rebalancing import RebalanceSchedule
        from Modules.portfolio_management.backtesting.engine_types import BacktestConfig, ExecutionConfig, CashConfig
        from Modules.portfolio_management.backtesting.engine import BacktestEngine
        from Modules.portfolio_management.strategies.core_replication import CoreTEMinConfig
        from Modules.portfolio_management.backtesting.metrics import ExPostTrackingError, DecisionMetricForwardFill
        from Modules.portfolio_management.export.port_inventory_exporter import PortInventoryExporter, PortInventoryConfig

        # Extrait et retire les paramètres propres au moteur de engine_kwargs
        engine_kwargs = dict(engine_kwargs or {})

        # Fréquence de rebalancement (par défaut trimestrielle)
        rebal_freq = engine_kwargs.pop("rebal_freq", "Q")

        # Verbose : affiche les logs du moteur (par défaut True)
        verbose = engine_kwargs.pop("verbose", True)

        # Schedule de rebalancement standard selon la fréquence spécifiée
        schedule = RebalanceSchedule(freq=rebal_freq)

        # Coûts de transaction (ici nuls pour isoler l'effet de la covariance)
        costs = TransactionCostModel(proportional_bps=0.00, fixed_cost=0.0)

        # Dictionnaires pour stocker les résultats intermédiaires et finaux
        te_series:  Dict[str, pd.Series] = {}
        bt_results: Dict[str, Any]       = {}
        rows: Dict[str, Dict[str, float]] = {}

        # Itère sur chaque modèle à backtester
        for spec in model_specs:
            print(f"Running backtest for model: {spec.name}...")

            # Récupère la configuration de covariance du modèle
            strat_cov = spec.cov_cfg

            # Configure la stratégie CoreTEMin
            core_cfg = CoreTEMinConfig(
                mode="returns",
                long_only=True,
                min_obs=30,
                min_coverage=0.90,
                optimizer_name=getattr(spec, "optimizer_name", "slsqp"),
                # Restrict_to_benchmark est False uniquement pour ALM Classic (fond de fonds)
                restrict_to_benchmark=(indice_name is None or "ALM CLASSIC" not in str(indice_name).upper()),
            )

            # Instancie la stratégie CoreTEMin avec la configuration de covariance du modèle
            core = CoreTEMinStrategy(cfg=core_cfg)

            # Crée et injecte le fournisseur de covariance dans la stratégie
            cov_provider = CovarianceProvider(cfg=strat_cov)
            core._impl.set_covariance_provider(cov_provider)

            # La stratégie à backtester est la CoreTEMinStrategy avec la covariance spécifique du modèle
            strategy = core

            # Configuration complète du backtest
            config = BacktestConfig(
                schedule=schedule,
                lookback=ann_factor,           # nb jours de lookback pour les stratégies
                cost_model=costs,              # coûts de transaction (ici nuls)
                data_lag=1,
                apply_lag=1,
                bench_lag=1,
                cov=strat_cov,                 # pour info dans les logs uniquement
                rebalance_metrics=(
                    ExPostTrackingError(window=ann_factor, annualization=ann_factor),
                    DecisionMetricForwardFill(key="te_ex_ante", name="te_ex_ante"),
                ),
                verbose=verbose,
                log_every_n_rebalances=1,
                execution=ExecutionConfig(
                    lot_size=1,
                    min_trade_notional=100.0,
                    min_position_notional=200.0,
                    min_weight=0.0001,
                    rounding="floor",
                    allow_fractional=False,
                ),
                cash=CashConfig(
                    target_cash_weight=0.01,
                    cash_return=0.0,
                ),
            )

            # Instancie et lance le moteur de backtest
            engine = BacktestEngine(config=config)
            kept_assets = list(universe_returns.columns)

            # Lancement du backtest pour le modèle courant
            res = engine.run(
                all_returns=all_returns,
                universe_returns=universe_returns,
                benchmark_weights=benchmark_weights,
                strategy=strategy,
                kept_assets=kept_assets,
                rebal_dates=rebal_dates_port,
                rebal_benchmark_dates=rebal_dates_bench,
                precomputed_cov_provider=precomputed_cov_provider,
                **engine_kwargs,
            )

            # Exporte l'inventaire portefeuille si demandé
            if port_root is not None:

                # Construit le chemin de sortie pour l'inventaire portefeuille du modèle courant
                xlsx_path = port_root / f"{indice_name.strip()}_{spec.name}.xlsx"

                # Configure l'exporter d'inventaire portefeuille
                inv_config = PortInventoryConfig(
                    ptf_name=f"{indice_name.strip()}_{spec.name}",
                    output_path=xlsx_path,
                    enabled=enabled_export,
                    rebalance_date_mode="weights_in_force_change",
                    weight_source="weights_in_force",
                    drop_zeros=True,
                    tol=1e-6,
                    apply_lag=1,
                )

                # Instancie et lance l'exporter d'inventaire portefeuille
                exporter_inv = PortInventoryExporter(inv_config)
                exporter_inv.run_if_enabled(res)

            # Stocke le résultat complet du backtest pour le modèle courant
            bt_results[spec.name] = res

            # Récupère les séries de rendements portefeuille et benchmark
            if hasattr(res, "portfolio_returns"):
                port_ret = res.portfolio_returns
            elif hasattr(res, "returns"):
                port_ret = res.returns
            else:
                raise AttributeError(
                    "BacktestResult ne contient pas de série de rendements portefeuille " "(portfolio_returns/returns).")

            # Récupère les rendements du benchmark
            port_ret  = pd.Series(getattr(res, "portfolio_returns")).dropna()
            bench_ret = pd.Series(getattr(res, "benchmark_returns")).dropna()

            # Aligne les deux séries sur leur index commun
            idx = port_ret.index.intersection(bench_ret.index)
            port_ret = port_ret.reindex(idx)
            bench_ret = bench_ret.reindex(idx)

            # Calcule les active returns et les métriques TE
            te = port_ret - bench_ret
            te_series[spec.name] = te
            rows[spec.name] = cls._te_metrics(te, ann_factor=ann_factor)

            gc.collect()

            # Purge le cache memmap après chaque modèle
            cache_dir = Path(__file__).resolve().parents[3] / "memmap_cache"
            purge_memmap_cache(cache_dir)
            print("[Cleanup] memmap_cache purgé.")

        # Construit le DataFrame de résumé TE (une ligne par modèle)
        te_summary = pd.DataFrame(rows).T
        te_summary.index.name = "model"

        # Identifie le meilleur modèle = TE annualisée la plus faible
        best_model = te_summary["te_ann"].astype(float).idxmin()

        return EcoResult(te_series=te_series, te_summary=te_summary,  bt_results=bt_results, best_model=best_model,)