"""
Pipeline d'évaluation des estimateurs de covariance.

Ce fichier orchestre les évaluations statistique et économique via ModelEvaluator,
qui est le point d'entrée unique utilisé depuis l'application et les scripts.
Il délègue à StatStudy (stat_study.py) et EcoStudy (eco_study.py).

Classes
-------
ModelEvaluator :
    Classe principale du pipeline. Expose full_evaluation() comme interface publique,
    et run_stat_evaluation() / economic_backtests_min_te() comme délégateurs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from Modules.study.covariance_study.stat_study import StatStudy, StatSimConfig, StatSimResult, StatEvalResult
from Modules.study.covariance_study.eco_study  import EcoStudy,  EcoResult

if TYPE_CHECKING:
    pass


class ModelEvaluator:
    """
    Classe principale du pipeline d'évaluation des estimateurs de covariance.

    Orchestre les deux branches d'évaluation :
    - Statistique (StatStudy) : simulation DGP, calcul des pertes matricielles.
    - Économique (EcoStudy)   : backtest TE-min sur données réelles.

    Les deux branches sont indépendantes et peuvent être activées séparément
    via les flags make_stats et make_eco dans full_evaluation().

    Methods
    -------
    run_stat_evaluation(R_ref, model_specs, cfg, exporter) -> StatEvalResult :
        Délègue à StatStudy.run_stat_evaluation().
    economic_backtests_min_te(...) -> EcoResult :
        Délègue à EcoStudy.economic_backtests_min_te().
    backtest_expost_metrics(result, ann_factor) -> dict :
        Délègue à EcoStudy.backtest_expost_metrics().
    full_evaluation(...) -> dict :
        Lance les deux branches selon les flags et retourne un dict unifié.
    """

    @classmethod
    def run_stat_evaluation(cls, R_ref: pd.DataFrame, model_specs: List[Any], cfg: StatSimConfig, exporter=None,) -> StatEvalResult:
        """
        Délègue l'évaluation statistique à StatStudy.run_stat_evaluation().

        Parameters
        ----------
        R_ref : pd.DataFrame
            DataFrame de référence pour N, noms d'actifs et index de dates.
        model_specs : list
            Liste de ModelSpec à évaluer.
        cfg : StatSimConfig
            Configuration de la simulation Monte Carlo.
        exporter : StatMonteCarloExporter or None
            Exporter pour le checkpoint et l'écriture au fil de l'eau.

        Returns
        -------
        StatEvalResult
            Résultats agrégés des simulations statistiques.
        """
        return StatStudy.run_stat_evaluation(R_ref=R_ref, model_specs=model_specs, cfg=cfg, exporter=exporter)

    @classmethod
    def economic_backtests_min_te(cls, all_returns: pd.DataFrame, universe_returns: pd.DataFrame, universe_prices: pd.DataFrame,
                                  benchmark_weights: pd.DataFrame, rebal_dates_port: pd.DatetimeIndex, rebal_dates_bench: pd.DatetimeIndex,
                                  model_specs: List[Any], engine_kwargs: Optional[Dict[str, Any]] = None, port_root=None, enabled_export: bool = False,
                                  ann_factor: int = 252, precomputed_cov_provider=None, indice_name=None,) -> EcoResult:
        """
        Délègue le backtest économique à EcoStudy.economic_backtests_min_te().

        Parameters
        ----------
        all_returns : pd.DataFrame
            Rendements de tous les actifs sur la période complète.
        universe_returns : pd.DataFrame
            Rendements de l'univers investissable.
        universe_prices : pd.DataFrame
            Prix de l'univers investissable.
        benchmark_weights : pd.DataFrame
            Poids du benchmark à chaque date.
        rebal_dates_port : pd.DatetimeIndex
            Dates de rebalancement du portefeuille.
        rebal_dates_bench : pd.DatetimeIndex
            Dates de rebalancement du benchmark.
        model_specs : list
            Liste de ModelSpec à backtester.
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
            Nom de l'indice pour les fichiers d'export.

        Returns
        -------
        EcoResult
            Résultats agrégés des backtests économiques.
        """
        return EcoStudy.economic_backtests_min_te(all_returns=all_returns, universe_returns=universe_returns, universe_prices=universe_prices,
            benchmark_weights=benchmark_weights, rebal_dates_port=rebal_dates_port, rebal_dates_bench=rebal_dates_bench, model_specs=model_specs,
            engine_kwargs=engine_kwargs, port_root=port_root, enabled_export=enabled_export, ann_factor=ann_factor, precomputed_cov_provider=precomputed_cov_provider,
            indice_name=indice_name,
        )

    @classmethod
    def backtest_expost_metrics(cls, result, ann_factor: int = 252,) -> Dict[str, float]:
        """
        Délègue le calcul des métriques ex-post à EcoStudy.backtest_expost_metrics().

        Parameters
        ----------
        result : BacktestResult
            Résultat de backtest produit par BacktestEngine.run().
        ann_factor : int
            Facteur d'annualisation.

        Returns
        -------
        dict
            Métriques standardisées : te_daily, te_ann, information_ratio, etc.
        """
        return EcoStudy.backtest_expost_metrics(result=result, ann_factor=ann_factor)

    @classmethod
    def full_evaluation(cls, model_specs: List[Any],
        all_returns: Optional[pd.DataFrame] = None,
        universe_returns: Optional[pd.DataFrame] = None,
        prices_inv: Optional[Any] = None,
        benchmark_weights: Optional[pd.DataFrame] = None,
        rebal_dates_port: Optional[pd.DatetimeIndex] = None,
        rebal_dates_bench: Optional[pd.DatetimeIndex] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        port_root: Optional[Any] = None,
        enabled_export: bool = False,
        ann_factor: int = 252,
        indice_name: Optional[str] = None,
        precomputed_cov_provider: Optional[Any] = None,
        # Stats
        make_stats: bool = False,
        stat_sim_cfg: Optional[StatSimConfig] = None,
        R_ref_stats: Optional[pd.DataFrame] = None,
        # Éco
        make_eco: bool = True,
    ) -> Dict[str, Any]:
        
        """
        Point d'entrée principal du pipeline d'évaluation.

        Lance les branches statistique et/ou économique selon les flags,
        et retourne un dictionnaire unifié avec les résultats.

        Parameters
        ----------
        model_specs : list
            Liste de ModelSpec (name, cov_cfg, optimizer_name) à évaluer.
        all_returns : pd.DataFrame or None
            Rendements de tous les actifs (requis si make_eco=True).
        universe_returns : pd.DataFrame or None
            Rendements de l'univers investissable (requis si make_eco=True).
        prices_inv : pd.DataFrame or None
            Prix de l'univers investissable.
        benchmark_weights : pd.DataFrame or None
            Poids du benchmark (requis si make_eco=True).
        rebal_dates_port : pd.DatetimeIndex or None
            Dates de rebalancement du portefeuille (requis si make_eco=True).
        rebal_dates_bench : pd.DatetimeIndex or None
            Dates de rebalancement du benchmark (requis si make_eco=True).
        engine_kwargs : dict or None
            Paramètres additionnels du moteur de backtest.
        port_root : Path or None
            Répertoire de sortie pour les inventaires portefeuille.
        enabled_export : bool
            Si True, active l'export des inventaires portefeuille.
        ann_factor : int
            Facteur d'annualisation.
        indice_name : str or None
            Nom de l'indice.
        precomputed_cov_provider : CovarianceProvider or None
            Fournisseur de covariance précomputé (mode path).
        make_stats : bool
            Si True, lance l'évaluation statistique.
        stat_sim_cfg : StatSimConfig or None
            Configuration de la simulation statistique (requis si make_stats=True).
        R_ref_stats : pd.DataFrame or None
            DataFrame de référence pour les stats. Si None, utilise universe_returns.
        make_eco : bool
            Si True, lance l'évaluation économique.

        Returns
        -------
        dict
            Dictionnaire avec clés 'simulation' et 'economic', chacune contenant
            les résultats de la branche correspondante ou None si désactivée.
        """

        result: Dict[str, Any] = {}

        # Branche statistique : simulation DGP + calcul des pertes matricielles
        if make_stats and stat_sim_cfg is not None:
            print("Evaluation statistique en cours")

            # R_ref_stats permet de découpler les dimensions stats de universe_returns
            r_ref = R_ref_stats if R_ref_stats is not None else universe_returns

            # Validation de la présence des données nécessaires pour la branche statistique
            if r_ref is None:
                raise ValueError("make_stats=True mais ni R_ref_stats ni universe_returns fourni.")
            
            # Lancement de l'évaluation statistique
            sim_res = cls.run_stat_evaluation(R_ref=r_ref, model_specs=model_specs, cfg=stat_sim_cfg)

            # Stockage des résultats de la simulation statistique
            result["simulation"] = sim_res
        else:
            result["simulation"] = None

        # Branche économique : backtest TE-min sur données réelles
        if make_eco:

            # Validation de la présence des données nécessaires pour la branche économique
            if any(x is None for x in [all_returns, universe_returns, benchmark_weights, rebal_dates_port, rebal_dates_bench]):
                raise ValueError(
                    "make_eco=True mais des données obligatoires sont manquantes "
                    "(all_returns, universe_returns, benchmark_weights, "
                    "rebal_dates_port, rebal_dates_bench)."
                )

            print("Backtest économique en cours")

            # Lancement du backtest économique
            eco = cls.economic_backtests_min_te(
                all_returns=all_returns,
                universe_returns=universe_returns,
                benchmark_weights=benchmark_weights,
                rebal_dates_port=rebal_dates_port,
                rebal_dates_bench=rebal_dates_bench,
                universe_prices=prices_inv,
                model_specs=model_specs,
                engine_kwargs=engine_kwargs,
                port_root=port_root,
                enabled_export=enabled_export,
                ann_factor=ann_factor,
                precomputed_cov_provider=precomputed_cov_provider,
                indice_name=indice_name,
            )

            # Stockage des résultats du backtest économique
            result["economic"] = {
                "te_summary": eco.te_summary,
                "te_series":  eco.te_series,
                "best_model": eco.best_model,
                "bt_results": eco.bt_results,
            }
        else:
            result["economic"] = None

        return result