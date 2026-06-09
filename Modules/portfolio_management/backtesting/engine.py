# Modules/portfolio_management/backtesting/engine.py

"""
Moteur principal du backtest de portefeuille.

Ce fichier contient uniquement la logique de déroulement du backtest :
initialisation, boucle journalière, drift des poids, rebalancement benchmark,
décisions d'allocation, gestion des coûts et finalisation des résultats.

Les dataclasses et Protocols sont dans engine_types.py.
L'exécution en mode positions (shares) est dans engine_execution.py.
Le logger console est dans engine_logger.py.

Classes
-------
BacktestEngine :
    Moteur principal du backtest. Prend une BacktestConfig et exécute le backtest via la méthode run(), qui retourne un BacktestResult.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .engine_types import (AllocationStrategy, AllocationDecision, BacktestConfig, BacktestResult, ExecutionConfig, CashConfig,)
from .engine_logger import BacktestLogger
from .engine_execution import PositionExecutor
from .covariance_provider import CovarianceProvider


class BacktestEngine:
    """
    Moteur principal du backtest de portefeuille.

    Exécute un backtest complet à partir d'une configuration et d'une stratégie d'allocation. Supporte deux modes :
    - Mode poids-only : le portefeuille est modélisé en poids (fractions, pas de prix requis).
    - Mode positions : conversion des poids en quantités avec prix réels, journal de trades.

    La boucle journalière gère dans l'ordre : le rendement du jour (portefeuille et
    benchmark), le drift des poids, l'exécution des ordres en attente, la décision
    de rebalancement, et la mise à jour des métriques plugables.

    Attributes
    ----------
    config : BacktestConfig
        Configuration complète du backtest.
    _logger : BacktestLogger
        Logger console pour le suivi du backtest à chaque rebalancement.

    Methods
    -------
    run(all_returns, universe_returns, benchmark_weights, strategy, kept_assets, ...) -> BacktestResult :
        Exécute le backtest complet et retourne les résultats.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._logger = BacktestLogger(verbose=config.verbose,log_every_n_rebalances=config.log_every_n_rebalances,)


    # Helpers : inputs
    @staticmethod
    def _to_series_benchmark(benchmark_returns: "pd.DataFrame | pd.Series") -> pd.Series:
        """Convertit les rendements du benchmark en Series si fournis en DataFrame."""

        # Si benchmark_returns est déjà une Series, on la retourne telle quelle
        if isinstance(benchmark_returns, pd.Series):
            return benchmark_returns
        
        #Si benchmark_returns est un DataFrame, on vérifie qu'il n'a qu'une colonne et on retourne cette colonne comme Series
        if isinstance(benchmark_returns, pd.DataFrame):
            if benchmark_returns.shape[1] != 1:
                raise ValueError("benchmark_returns DataFrame must have exactly 1 column.")
            return benchmark_returns.iloc[:, 0]
        raise TypeError("benchmark_returns must be a Series or a single-column DataFrame.")


    def _validate_and_align_inputs(self, universe_returns: pd.DataFrame, benchmark_weights: pd.DataFrame, all_returns: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        """
        Harmonise les index des inputs et vérifie leur validité.

        Calcule l'index commun entre les trois DataFrames, vérifie qu'il y a
        suffisamment de dates communes par rapport au lookback configuré,
        et réindexe tous les inputs sur cet index commun.
        """

        #Verification des types d'index
        if not isinstance(universe_returns.index, pd.DatetimeIndex):
            raise TypeError("universe_returns index must be DatetimeIndex.")
        if not isinstance(benchmark_weights.index, pd.DatetimeIndex):
            raise TypeError("benchmark_weights index must be DatetimeIndex.")

        #Détermine l'index commun entre les trois DataFrames
        common_index = (universe_returns.index.intersection(benchmark_weights.index).intersection(all_returns.index))

        # Si l'index commun est vide ou trop court par rapport au lookback, on lève une erreur
        if len(common_index) < max(self.config.lookback + 2, 10):
            raise ValueError("Not enough common dates between inputs for the chosen lookback.")

        # Réindexe les DataFrames sur l'index commun et les trie par date croissante
        universe_returns  = universe_returns.loc[common_index].sort_index()
        benchmark_weights = benchmark_weights.loc[common_index].sort_index()
        all_returns       = all_returns.loc[common_index].sort_index()

        return universe_returns, benchmark_weights, all_returns, common_index


    # Helpers : dates et lags
    @staticmethod
    def _shift_trading_date(dates: pd.DatetimeIndex,  dt: pd.Timestamp,  offset: int,) -> Optional[pd.Timestamp]:
        """
        Décale dt de offset jours de bourse dans l'index dates.
        Retourne None si le résultat est hors bornes.
        """

        # Tente de trouver la position de dt dans dates. Si dt n'est pas dans dates, retourne None.
        try:
            loc = dates.get_loc(dt)
        except KeyError:
            return None
        
        # Calcule la nouvelle position en ajoutant l'offset.
        new_loc = int(loc) + int(offset)

        #Vérifie que la nouvelle position est dans les bornes de dates. Si non, retourne None.
        if new_loc < 0 or new_loc >= len(dates):
            return None
        
        # Si tout est ok, retourne la date correspondante à la nouvelle position.
        return dates[new_loc]

    def _get_benchmark_rebalance_dates(self, dates: pd.DatetimeIndex, port_reb_dates: pd.DatetimeIndex,) -> pd.DatetimeIndex:
        """Calcule les dates de rebalancement du benchmark en appliquant un offset aux dates de rebalancement du portefeuille."""

        # Récupération de l'offset configuré pour le rebalancement du benchmark par rapport au portefeuille
        off = int(self.config.bench_rebalance_offset)

        # Initialisation de la liste de dates de rebalancement du benchmark
        out = []

        # Itére sur les dates de rebalancement du portefeuille
        for d in port_reb_dates:

            # Décale la date de rebalancement du portefeuille de l'offset configuré pour obtenir la date de rebalancement du benchmark correspondante
            dd = self._shift_trading_date(dates, pd.Timestamp(d), off)

            #Si la date décalée est valide (pas hors bornes), on l'ajoute à la liste de dates de rebalancement du benchmark
            if dd is not None:
                out.append(dd)
        #Si aucune date valide n'a été trouvée, on retourne un DatetimeIndex vide
        if not out:
            return pd.DatetimeIndex([], tz=dates.tz)
        
        #Si toutes les dates sont valides, on retourne un DatetimeIndex construit à partir de la liste de dates de rebalancement
        return pd.DatetimeIndex(out).unique().sort_values()

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Retourne les dates de rebalancement selon le calendrier configuré."""
        return self.config.schedule.rebalance_dates(dates)

    def _get_lags(self) -> Tuple[int, int, int]:
        """Retourne les lags configurés pour le signal, l'application des poids et le benchmark."""

        # Récupère les lags configurés pour le signal (data_lag), l'application des poids (apply_lag) et le benchmark (bench_lag).
        data_lag  = int(self.config.data_lag)
        apply_lag = int(self.config.apply_lag)
        bench_lag = int(self.config.data_lag if self.config.bench_lag is None else self.config.bench_lag)

        # Vérifie que les lags sont des entiers positifs ou nuls. Si un lag est négatif, lève une erreur.
        if data_lag < 0 or apply_lag < 0 or bench_lag < 0:
            raise ValueError("data_lag/apply_lag/bench_lag must be >= 0.")
        return data_lag, apply_lag, bench_lag


    # Helpers : initialisation
    def _init_pending(self) -> Dict[pd.Timestamp, pd.Series]:
        """Initialise le dictionnaire des poids en attente d'application (apply_lag)."""
        return {}

    def _init_initial_weights(self, initial_weights: Optional[pd.Series], benchmark_weights: pd.DataFrame, tickers: list[str],) -> pd.Series:
        """
        Initialise les poids de départ du portefeuille.

        Si initial_weights est None, utilise les poids du benchmark au jour 0
        (normalisés à 1). Sinon utilise les poids fournis après vérification.
        """
        #Si initial_weights n'est pas fourni
        if initial_weights is None:
            
            # Initialise les poids à 0 pour tous les tickers
            w0 = pd.Series(0.0, index=tickers, dtype=float)

            # Récupère les poids du benchmark au jour 0, les réindexe sur les tickers et remplace les NaN par 0
            b0 = benchmark_weights.iloc[0].reindex(tickers).fillna(0.0)

            #Somme des poids du benchmark au jour 0.
            s  = float(b0.sum())

            # Renormalise les poids du benchmark si leur somme est significativement différente de 0
            if not (b0 == 0).all():
                w0 = b0.copy() / s
            # sinon répartit les poids uniformément entre les tickers
            else:
                w0[:] = 1.0 / len(tickers)

        #Si initial_weights est fourni, on vérifie qu'il est valide et on l'utilise
        else:
            #Récupère les poids initiaux, les réindexe sur les tickers et remplace les NaN par 0
            w0 = initial_weights.reindex(tickers).fillna(0.0).astype(float)

            # Somme des poids initiaux
            s  = float(w0.sum())

            #Contrôle que la somme des poids initiaux est proche de 1. Si ce n'est pas le cas, lève une erreur.
            if abs(s - 1.0) > 1e-8:
                raise ValueError("initial_weights must sum to 1.")
            
        return w0

    def _init_outputs(self, dates: pd.DatetimeIndex, tickers: list[str], initial_nav: float, w0: pd.Series,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Initialise les DataFrames et Series de sortie du backtest."""

        port_ret         = pd.Series(index=dates, dtype=float)
        nav              = pd.Series(index=dates, dtype=float)
        weights_in_force = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        weights_close    = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        current_w        = w0.copy()
        nav.iloc[0]      = float(initial_nav)

        return weights_in_force, weights_close, port_ret, nav, current_w


    # Helpers : boucle journalière
    def _apply_pending_if_any(self, dt: pd.Timestamp, pending: Dict[pd.Timestamp, pd.Series], current_w: pd.Series,) -> pd.Series:
        """Applique les poids en attente si dt est leur date d'effet."""

        #Si dt est dans le pending
        if dt in pending:

            # on applique les poids correspondants et on les retire du pending
            return pending.pop(dt)
        
        # Sinon, on retourne les poids courants inchangés.
        return current_w

    def _compute_day_return(self, dt: pd.Timestamp, current_w: pd.Series, universe_returns: pd.DataFrame,) -> float:
        """Calcule le rendement du portefeuille au jour dt avec les poids effectifs."""

        # Récupère les rendements de l'univers au jour dt
        r_t = universe_returns.loc[dt]

        #Si drop_na_assets est True, on remplace les NaN par 0 (actif non tradable = pas de rendement).
        if self.config.drop_na_assets:
            r_vec = r_t.fillna(0.0)

        #Sinon, si il y a des NaN, on lève une erreur.
        else:
            if r_t.isna().any():
                raise ValueError(f"Missing returns at {dt} while drop_na_assets=False.")
            r_vec = r_t

        # Si pas d'erreur, calcule du rendement du portefeuille comme la somme pondérée des rendements de l'univers avec les poids courants.
        return float((current_w.values * r_vec.values).sum())

    def _get_r_vec(self, dt: pd.Timestamp, universe_returns: pd.DataFrame) -> pd.Series:
        """Retourne les rendements du jour avec gestion des NaN selon la config."""
        r_t = universe_returns.loc[dt]
        if self.config.drop_na_assets:
            return r_t.fillna(0.0)
        if r_t.isna().any():
            raise ValueError(f"Missing returns at {dt} while drop_na_assets=False.")
        return r_t

    def _drift_weights(self, current_w: pd.Series, r_vec: pd.Series) -> pd.Series:
        """
        Dérive les poids selon la performance du jour (buy-and-hold intra-rebal).

        Applique la performance journalière aux poids courants et renormalise
        pour maintenir un portefeuille fully-invested.
        """

        #Calcul du taux de croissance de chaque actif
        g = (1.0 + r_vec.astype(float))

        # Calcul du portefeuille brut avant renormalisation
        port_gross = float((current_w.values * g.values).sum())

        # Si port_gross est non fini ou trop petit, on répartit les poids uniformément.
        if not np.isfinite(port_gross) or port_gross <= 0.0:
            s = float(current_w.sum())
            if s > 1e-12:
                return current_w / s
            return pd.Series(1.0 / len(current_w), index=current_w.index, dtype=float)

        #Application du drift aux poids courants
        w_drift = current_w.values * g.values / port_gross

        # Renormalisation des poids pour maintenir un portefeuille fully-invested
        w = pd.Series(w_drift, index=current_w.index, dtype=float)
        s = float(w.sum())
        return (w / s) if s > 1e-12 else pd.Series(1.0 / len(w), index=w.index, dtype=float)

    def _should_rebalance(self, dt: pd.Timestamp, i: int, reb_dates: pd.DatetimeIndex, dates: pd.DatetimeIndex, port_ret: pd.Series, bench_ret: pd.Series,
                        current_w: pd.Series, nav: pd.Series, weights: pd.DataFrame,) -> bool:
        """ Fonction de décision de rebalancement. """
        
        # Si dt est dans reb_dates, retourne True (rebalancement forcé)
        if dt in reb_dates:
            return True
        
        # Sinon, itère sur les règles de rebalancement configurées et retourne True si l'une d'elles indique qu'il faut rebalancer.
        for rule in self.config.rebalance_rules:
            # Utilise l'objet rule pour decider s'il faut rebalancer en fonction de la règle instanciée.
            if rule.should_rebalance(dt=dt, i=i, dates=dates, portfolio_returns=port_ret, benchmark_returns=bench_ret, current_weights=current_w, nav=nav,weights_history=weights,):
                return True
            
        # Si aucune règle ne déclenche le rebalancement, retourne False.
        return False

    def _compute_target_weights_if_possible(self, i: int, dt: pd.Timestamp, dates: pd.DatetimeIndex, universe_returns: pd.DataFrame,
        bench_ret: pd.Series, kept_assets, tickers: list[str], strategy: AllocationStrategy, current_w: pd.Series,
        data_lag: int, bench_lag: int, benchmark_weights: Optional[pd.DataFrame],) -> Optional[Tuple[pd.Series, Optional[Dict[str, float]]]]:
        """
        Calcule les poids cibles si suffisamment d'historique est disponible.

        Respecte le data_laget le bench_lag (poids benchmark disponibles à dt - bench_lag).
        Retourne None si l'un des lags n'est pas respecté.
        """

        # Vérifie qu'on respecte le data_lag
        end_loc = i - data_lag
        if end_loc < 0:
            return None

        # Recupère la configuration de covariance pet le paramètre de fenêtre pour le calcul de la covariance.
        cov_cfg      = self.config.cov
        model_window = (getattr(cov_cfg, "lw_window", None) or getattr(cov_cfg, "rolling_window", None) or self.config.lookback)
        model_window = int(model_window)

        #Vérifie qu'on a suffisamment de données historiques pour calculer les poids cibles en respectant le bench_lag (+ 1).
        start_loc = end_loc - model_window + 1
        if start_loc < 0:
            return None

        # Vérifie qu'on respecte le bench_lag pour le benchmark
        bench_loc = i - bench_lag
        if bench_loc < 0:
            return None

        # Prépare les fenêtres de données à fournir à la stratégie : rendements de l'univers, rendements du benchmark, poids du benchmark.
        window_full  = universe_returns.iloc[start_loc: end_loc + 1]
        b_w          = benchmark_weights.iloc[bench_loc].astype(float).fillna(0.0)
        window_full  = window_full.reindex(columns=b_w.index.astype(str))
        window_full  = window_full.dropna(axis=1, how="all")
        bench_window = bench_ret.iloc[start_loc: end_loc + 1]

        # Appelle la stratégie pour calculer les poids cibles à partir des données préparées. La stratégie peut retourner soit une AllocationDecision (avec poids et diagnostics), soit directement les poids cibles.
        decision = strategy.allocate(asof=dt, returns_window=window_full, benchmark_returns_window=bench_window, kept_assets=kept_assets,
                                        current_weights=current_w, benchmark_weights=b_w,)

        #Si la stratégie retourne une AllocationDecision
        if isinstance(decision, AllocationDecision):
            # On extrait les poids cibles et les diagnostics de la décision à partir des attributs de l'objet AllocationDecision.
            target_w_raw = decision.weights
            decision_diag = decision.diagnostics

        # Sinon, on considère que la stratégie a retourné directement les poids cibles et on n'a pas de diagnostics.
        else:
            target_w_raw  = decision
            decision_diag = None

        # On réindexe les poids cibles sur les tickers, on remplace les NaN par 0 et on s'assure que ce sont des floats.
        target_w = (target_w_raw.reindex(tickers).fillna(0.0).astype(float))

        # On vérifie que la somme des poids cibles est proche de 1. Si ce n'est pas le cas, on lève une erreur.
        s = float(target_w.sum())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Strategy returned weights not summing to 1 at {dt}: sum={s}.")

        return target_w, decision_diag


    def _apply_costs_and_schedule(self, i: int, dt: pd.Timestamp,  dates: pd.DatetimeIndex, nav: pd.Series,  port_ret: pd.Series,
                                    pending: Dict[pd.Timestamp, pd.Series], current_w: pd.Series, target_w: pd.Series, apply_lag: int,) -> float:
        """ Applique les coûts de transaction et programme l'application des nouveaux poids cibles. """

        # Récupère le coût de transaction en fraction du portefeuille à partir du modèle de coût configuré, en comparant les poids courants et les poids cibles.
        cost_frac = self.config.cost_model.cost_fraction(current_w.values, target_w.values)

        # Si le coût de transaction est supérieur à 0
        if cost_frac > 0.0:
            nav.iloc[i]  = nav.iloc[i] * (1.0 - cost_frac) #Ajustement de la NAV en déduisant les coûts de transaction
            port_ret.iloc[i] = port_ret.iloc[i] - cost_frac #Ajustement du rendement du jour en déduisant les coûts de transaction

        # Determine la date d'effet des nouveaux poids cibles en appliquant le apply_lag,
        eff_loc = i + apply_lag

        # Programme l'application des nouveaux poids cibles à la date d'effet dans le dictionnaire pending.
        if eff_loc < len(dates):
            pending[dates[eff_loc]] = target_w.copy()

        return cost_frac



    # Point d'entrée principal
    def run(self, all_returns: pd.DataFrame, universe_returns: pd.DataFrame, benchmark_weights: pd.DataFrame, strategy: AllocationStrategy, kept_assets: pd.DataFrame,
            universe_prices: Optional[pd.DataFrame] = None, rebal_dates: Optional[pd.DatetimeIndex] = None, rebal_benchmark_dates: Optional[pd.DatetimeIndex] = None,
            flows: Optional[pd.Series] = None, initial_weights: Optional[pd.Series] = None, initial_nav: float = 1.0, precomputed_cov_provider: Optional[CovarianceProvider] = None,
            ) -> BacktestResult:
        
        """
        Exécute le backtest complet et retourne les résultats.

        Orchestre dans l'ordre : validation des inputs, initialisation du fournisseur
        de covariance, initialisation des outputs, boucle journalière (drift, benchmark,
        exécution, décision, métriques), et finalisation (découpe depuis le premier rebal,
        normalisation de la NAV, fermeture des handles memmap).

        Parameters
        ----------
        all_returns : pd.DataFrame
            Rendements de l'univers complet (benchmark inclus), utilisés pour le drift du benchmark et le pré-calcul du path de covariances.
        universe_returns : pd.DataFrame
            Rendements de l'univers investissable uniquement.
        benchmark_weights : pd.DataFrame
            Poids du benchmark à chaque date de rebalancement de l'indice.
        strategy : AllocationStrategy
            Stratégie d'allocation à évaluer.
        kept_assets : list or pd.DataFrame
            Sous-ensemble d'actifs investissables (subset de universe_returns.columns).
        universe_prices : pd.DataFrame or None
            Prix de clôture par actif. Si fourni, active le mode positions (shares).
        rebal_dates : pd.DatetimeIndex or None
            Dates de rebalancement à utiliser. Si None, calculées depuis le schedule.
        rebal_benchmark_dates : pd.DatetimeIndex or None
            Dates de rebalancement du benchmark. Si None, calculées depuis rebal_dates.
        flows : pd.Series or None
            Flux de souscription/rachat journaliers (en unités monétaires).
        initial_weights : pd.Series or None
            Poids initiaux du portefeuille. Si None, utilise les poids benchmark au jour 0.
        initial_nav : float
            Valeur liquidative initiale (défaut : 1.0).
        precomputed_cov_provider : CovarianceProvider or None
            Fournisseur de covariance déjà précomputé. Si None, un nouveau provider est instancié et précomputé selon self.config.cov.

        Returns
        -------
        BacktestResult
            Résultats complets du backtest depuis le premier rebalancement réussi.
        """

        # 1) Validation et alignement des inputs
        universe_returns, benchmark_weights, all_returns, dates = self._validate_and_align_inputs(universe_returns=universe_returns, 
                                                                                                  benchmark_weights=benchmark_weights, 
                                                                                                  all_returns=all_returns,)
       
        # 2) Fournisseur de covariance

        #Si un fournisseur de covariance précomputé est fourni en argument
        if precomputed_cov_provider is not None:
            #Récupère le provider de covariance précomputé fourni en argument
            cov_provider = precomputed_cov_provider
        
        #Si aucun fournisseur de covariance précomputé n'est fourni est que le mode de calcul est "rebal".
        elif self.config.cov is not None and getattr(self.config.cov, "compute_mode", "path") == "rebal":
            # Instancie un nouveau selon la configuration self.config.cov sans précompute le path de covariance.
            cov_provider = CovarianceProvider(cfg=self.config.cov)
        
        # Si aucun fournisseur de covariance précomputé n'est fourni et que le mode de calcul n'est pas "rebal"
        else:
            # Instancie un nouveau fournisseur de covariance selon la configuration self.config.covs.
            cov_provider = CovarianceProvider(cfg=self.config.cov)

            # Précompute le path de covariance sur all_return
            cov_provider.precompute_path(all_returns)

        # Injection du provider dans la stratégie si supportée
        if hasattr(strategy, "set_covariance_provider"):
            strategy.set_covariance_provider(cov_provider)

        #Si la stratégie est un wrapper autour d'une implémentation qui supporte l'injection du provider de covariance, on l'injecte dans l'implémentation.
        elif hasattr(strategy, "_impl") and hasattr(strategy._impl, "set_covariance_provider"):
            strategy._impl.set_covariance_provider(cov_provider)


        # 3) Univers et actifs

        # Récupère la liste des tickers investissables à partir des colonnes de universe_returns, et la liste complète des tickers à partir des colonnes de benchmark_weights.
        tickers_inv = list(universe_returns.columns.astype(str))

        #Récupère la liste complète des tickers à partir des colonnes de benchmark_weights.
        tickers_all = list(benchmark_weights.columns.astype(str))

        # Reindexe les DataFrames d'inputs sur les tickers complets
        all_returns = all_returns.reindex(columns=tickers_all)

        #Si kept_assets n'est pas fourni, on considère que tous les actifs investissables sont gardés.
        if kept_assets is None:
            kept_assets = tickers_inv

        # Assure que kept_assets est une liste de strings correspondant à des tickers
        kept_assets = [str(t) for t in kept_assets]

        # Recupere uniquement les actifs de kept_assets qui sont dans tickers_inv et conserve leur ordre d'origine dans tickers_inv
        kept_assets_ordered = [t for t in tickers_inv if t in set(kept_assets)]

        # Verifie que kept_assets contient au moins 2 actifs investissables. Si ce n'est pas le cas, on lève une erreur.
        if len(kept_assets_ordered) < 2:
            raise ValueError("kept_assets must contain at least 2 investable assets.")


        # 4) Dates et lags

        #Recupère les dates de rebalancement à utiliser : si rebal_dates est fourni en argument, on l'utilise, sinon on les calcule à partir du calendrier configuré.
        reb_dates = rebal_dates if rebal_dates is not None else self._get_rebalance_dates(dates)

        # Recupère les lags configurés pour le signal, l'application des poids et le benchmark.
        data_lag, apply_lag, bench_lag = self._get_lags()

        # Recupere les dates de rebalancement du benchmark si fournies, sinon les calcule à partir des dates de rebalancement du portefeuille et de l'offset configuré.
        bench_reb_dates  = (rebal_benchmark_dates if rebal_benchmark_dates is not None else self._get_benchmark_rebalance_dates(dates, reb_dates))

        # Indique la dernière date de rebal à l'allocateur

        #Si rebal_dates est fourni et non vide
        if rebal_dates is not None and len(rebal_dates) > 0:
            #Instancie l'allocateur pour accéder à son implémentation (si c'est un wrapper)
            alloc = strategy._impl

            #Si l'implémentation de la stratégie d'allocation a un attribut _last_rebal_dt, on le met à jour avec la dernière date de rebalancement du benchmark.
            if hasattr(alloc, "_last_rebal_dt"):

                # met à jour l'attribut avec la dernière date de rebalancement du benchmark.
                alloc._last_rebal_dt = rebal_dates[-1]


        # 5) Initialisation benchmark

        # Initialise le dataframe des poids du benchmark en force (après drift, avant rebalancement)
        bench_weights_in_force = pd.DataFrame(index=dates, columns=tickers_all, dtype=float)

        # Initialise le dataframe des poids du benchmark à la clôture (après rebalancement)
        bench_weights_close = pd.DataFrame(index=dates, columns=tickers_all, dtype=float)

        # Récupère les poids du benchmark au jour 0, les réindexe sur les tickers complets, remplace les NaN par 0 et s'assure que ce sont des floats.
        b0     = benchmark_weights.iloc[0].reindex(tickers_all).fillna(0.0).astype(float)

        # Somme des poids du benchmark au jour 0.
        s0     = float(b0.sum())

        # Renormalise les poids du benchmark au jour 0 s'ils ne sont pas tous nuls, sinon répartit les poids uniformément entre les tickers complets.
        bench_w_close = (b0 / s0) if s0 > 1e-12 else pd.Series(1.0 / len(tickers_all), index=tickers_all, dtype=float)

        # Initialise la série des rendements du benchmark avec les mêmes index que les dates, et le nom "benchmark_returns_drifted".
        bench_ret = pd.Series(index=dates, dtype=float, name="benchmark_returns_drifted")

        # Au jour 0, le rendement du benchmark est de 0 (pas de performance réalisée).
        bench_ret.iloc[0] = 0.0

        # Les poids du benchmark en force et à la clôture au jour 0 sont initialisés avec les poids du benchmark au jour 0 (après renormalisation).
        bench_weights_in_force.iloc[0] = bench_w_close.values
        bench_weights_close.iloc[0]    = bench_w_close.values


        # 6) Mode positions (prix)

        # Si universe_prices est fourni, on active le mode positions
        use_positions = universe_prices is not None

        # Initialisation d'une variable des prix
        prices = None

        # Si le mode position est activé
        if use_positions:

            #Vérification que l'index de universe_prices est un DatetimeIndex. Si ce n'est pas le cas, on lève une erreur.
            if not isinstance(universe_prices.index, pd.DatetimeIndex):
                raise TypeError("universe_prices index must be DatetimeIndex.")
            
            #Recupère les prix de l'univers, les réindexe sur les dates du backtest
            prices = universe_prices.reindex(index=dates)

            #Recupere les tickers présents dans la liste de tickers investissables mais absents des colonnes de universe_prices
            missing_cols = [t for t in tickers_inv if t not in prices.columns]

            #Lève une erreur si des tickers investissables sont absents des colonnes de universe_prices, en affichant les 10 premiers tickers manquants.
            if missing_cols:
                raise ValueError(f"universe_prices missing columns: {missing_cols[:10]}...")
            
            # Reindexe les colonnes de prices sur les tickers investissables pour s'assurer qu'elles sont dans le même ordre
            prices = prices.reindex(columns=tickers_inv)

            # Vérifie que les prix ne contiennent pas de lignes avec tous les éléments NaN. Si c'est le cas, on lève une erreur en indiquant la première date concernée.
            if prices.isna().all(axis=1).any():
                bad = prices.index[prices.isna().all(axis=1)]
                raise ValueError(f"Some dates have all-NaN prices (e.g. {bad[0]}).")

        # Instanciation de l'executor de positions, qui gère la conversion poids positions et le calcul du cash nécessaire en mode position.
        exec_cfg = self.config.execution if self.config.execution is not None else ExecutionConfig()
        cash_cfg = self.config.cash if self.config.cash is not None else CashConfig()
        executor = PositionExecutor(exec_cfg=exec_cfg, cash_cfg=cash_cfg)


        # 7) Initialisation portefeuille

        #Initialisation du dictionnaire des poids en attente d'application (permet l'application du apply_lag)
        pending = self._init_pending()

        # Initialisation des poids de départ du portefeuille : si initial_weights est fourni, on l'utilise après vérification, sinon on utilise les poids du benchmark au jour 0.
        w0 = self._init_initial_weights(initial_weights, benchmark_weights, tickers_inv)

        # Initialisation du DataFrame des poids du portefeuille avec les mêmes index que les dates et les mêmes colonnes que les tickers investissables.
        weights = pd.DataFrame(index=dates, columns=tickers_inv, dtype=float)

         #Initialisation des autres outputs du backtest : poids du portefeuille en force, poids du portefeuille à la clôture, rendement du portefeuille, NAV, et poids courants.
        weights_in_force, weights_close, port_ret, nav, current_w = self._init_outputs(dates, tickers_inv, initial_nav, w0)

        # Initialisation de variables supplémentaires pour le mode positions
        holdings_df  = None
        cash_series  = None
        trades_ledger = []
        shares = None
        cash   = 0.0

        # Si des flux sont fournis, on les réindexe sur les dates du backtest, on remplace les NaN par 0 et on s'assure que ce sont des floats.
        flows_aligned = None
        if flows is not None:
            flows_aligned = flows.reindex(dates).fillna(0.0).astype(float)

        # Variable pour stocker les diagnostics de la dernière décision d'allocation retournés par la stratégie, si elle en fournit.
        last_decision_diag: Optional[Dict[str, float]] = None
        rebal_count     = 0
        _first_rebal_date = None
        _rebal_diags: list = []


        # 8) Initialisation jour 0

        #Si le mode positions est activé
        if use_positions:
            # Innitialisation du DataFrame des positions (nombre de shares) et de la série du cash
            holdings_df  = pd.DataFrame(index=dates, columns=tickers_inv, dtype=float)
            cash_series  = pd.Series(index=dates, dtype=float)

            # Récupération des prix du jour 0, réindexés sur les tickers investissables et convertis en float
            px0 = prices.iloc[0].astype(float)

            # Conversion des poids initiaux en nombre de shares et cash nécessaire pour le jour 0, en utilisant l'executor de positions. Les shares sont initialisés avec ces valeurs.
            shares0, cash0, _ = executor.weights_to_shares(target_weights=w0, prices_t=px0, portfolio_value=float(initial_nav))
            shares = shares0.copy()
            cash   = float(cash0)

            # Enregistre les positions initiales dans le DataFrame des positions et la série du cash
            holdings_df.iloc[0] = shares.values
            cash_series.iloc[0] = cash

            # Calcul de la valeur des actifs au jour 0 et des poids du portefeuille au jour 0 après achat des positions, en utilisant les prix du jour 0. 
            assets0 = float((shares * px0).sum())
            w0_close = ((shares * px0) / assets0).fillna(0.0) if assets0 > 1e-12 else pd.Series(0.0, index=tickers_inv, dtype=float)

            #Initialisation des outputs du backtest au jour 0 : rendement du portefeuille à 0, NAV à initial_nav, poids en force et à la clôture avec les poids calculés après achat des positions.
            port_ret.iloc[0] = 0.0
            nav.iloc[0]  = float(assets0 + cash)
            weights_in_force.iloc[0] = w0_close.values
            weights_close.iloc[0]  = w0_close.values
            weights.iloc[0]  = w0_close.values
        
        # Si mode positions non activé, on initialise les outputs du backtest au jour 0 avec les poids initiaux et un rendement de 0.
        else:
            current_w = w0.copy()
            weights.iloc[0] = current_w.values
            port_ret.iloc[0] = 0.0
            weights_in_force.iloc[0] = current_w.values
            weights_close.iloc[0] = current_w.values
            weights.iloc[0] = current_w.values


        # 9) Métriques plugables
        metrics = {}

        # Itère sur les métriques de rebalancement configurées, réinitialise chacune d'elles avec les dates du backtest.
        for m in self.config.rebalance_metrics:
            m.reset(dates)
            metrics[m.name] = pd.Series(index=dates, dtype=float)

        # Convertit le dictionnaire des métriques en DataFrame pour faciliter l'enregistrement des valeurs au jour le jour.
        metrics_out = pd.DataFrame(metrics)


        # 10) Boucle journalière

        # Itère sur les dates du backtest
        for i, dt in enumerate(dates):

            # Passe l'itération du jour 0 car elle a déjà été initialisée avant la boucle.
            if i == 0:
                continue



            # (A) Rendement du jour

            #Si mode positions activé
            if use_positions:

                #Récupère les prix du jour et prix de la veille réindexés sur les tickers investissables et convertis en float
                px_t = prices.iloc[i].astype(float)
                px_prev = prices.iloc[i - 1].astype(float)

                # Calcul de la valeur du portefeuille à la veille avec le cash
                value_prev = float((shares * px_prev).sum() + cash)

                # Calcul de la valeur des actifs à la veille (sans le cash)
                assets_prev = float((shares * px_prev).sum())

                # Calcul des poids du portefeuille au jour t-1 en utilisant les prix de la veille. Si la valeur des actifs est trop petite, on considère que les poids sont à 0 pour éviter les divisions par zéro.
                w_in_force = ((shares * px_prev) / assets_prev).fillna(0.0).astype(float) if assets_prev > 1e-12 else pd.Series(0.0, index=tickers_inv, dtype=float)
                weights_in_force.iloc[i] = w_in_force.values  # Poids à  l'ouverture du marché, avant le drift intra-journalier

                # Calcul la valeur du cash en appliquant le rendement du cash configuré pour le jour t.
                cash = float(cash * (1.0 + float(cash_cfg.cash_return)))

                # Calcul de la valeur du portefeuille apres drift intra-journalier mais avant l'exécution des ordres en attente, en utilisant les prix du jour t.
                value_pre_flow = float((shares * px_t).sum() + cash)

                # Calcul du rendement du portefeuille pour le jour t.
                r_t  = (value_pre_flow / value_prev - 1.0) if value_prev > 0 else 0.0

                # Enregistre le rendement du portefeuille et la NAV avant l'exécution des ordres en attente.
                port_ret.iloc[i] = float(r_t)
                nav.iloc[i] = float(value_pre_flow)

                # Si des flux sont fournis
                if flows_aligned is not None:
                    # Récupération du flux du jour t
                    f = float(flows_aligned.iloc[i])

                    if f != 0.0:
                        #Application du flux au cash et à la NAV
                        cash = float(cash + f)
                        nav.iloc[i] = float(nav.iloc[i] + f)

                assets_pre_trade = float((shares * px_t).sum())
                w_close_pre_trade = ((shares * px_t) / assets_pre_trade).fillna(0.0).astype(float) if assets_pre_trade > 1e-12 else pd.Series(0.0, index=tickers_inv, dtype=float)
                weights_close.iloc[i] = w_close_pre_trade.values
                weights.iloc[i] = w_close_pre_trade.values

            # Si on est en mode poids-only
            else:
                # Recupère les poids du jours stockés dans le pending s'il y en a, sinon utilise les poids courants.
                current_w = self._apply_pending_if_any(dt, pending, current_w) #Le pending contient des poids si on a rebalancé à la date dt - apply_lag, sinon il est vide pour cette date.

                # Enregistre les poids en force (après application du pending) avant le drift intra-journalier (poids à l'ouverture du marché).
                weights_in_force.iloc[i] = current_w.values

                #Recupère les rendements de l'univers au jour dt, en gérant les NaN selon la configuration.
                r_vec = universe_returns.loc[dt]
                r_day = self._compute_day_return(dt, current_w, universe_returns)

                #Enregistre le rendement du portefeuille du jour dt avant l'exécution des ordres en attente, et la NAV après application du rendement du jour.
                port_ret.iloc[i] = float(r_day)
                nav.iloc[i] = float(nav.iloc[i - 1] * (1.0 + float(r_day)))

                # calcul les poids du portefeuille à la clôture du jour dt en appliquant le drift intra-journalier sur les poids courants, et en gérant les NaN des rendements selon la configuration.
                w_close = self._drift_weights(current_w=current_w, r_vec=r_vec.fillna(0.0) if self.config.drop_na_assets else r_vec)

                # Enregistre les poids à la clôture du jour dt avant l'exécution des ordres en attente.
                weights_close.iloc[i] = w_close.values
                weights.iloc[i] = w_close.values
                current_w = w_close


            # (A-bis) Benchmark : drift + rebalancement

            # Récupère les rendements de l'univers au jour dt pour le benchmark, en gérant les NaN selon la configuration.
            r_vec_b = self._get_r_vec(dt, all_returns)

            #Enregistre les poids du benchmark en force avant drift, à l'ouverture du marché.
            bench_weights_in_force.iloc[i] = bench_w_close.values

            # Calcule le rendement du benchmark pour le jour dt.
            r_bench_day = self._compute_day_return(dt, bench_w_close, all_returns)

            # Enregistre le rendement du benchmark du jour dt
            bench_ret.iloc[i] = float(r_bench_day)

            # Calcule les poids du benchmark à la clôture du jour dt en appliquant le drift intra-journalier sur les poids courants du benchmark, et en gérant les NaN des rendements selon la configuration.
            bench_w_close     = self._drift_weights(current_w=bench_w_close, r_vec=r_vec_b)

            # Si dt est une date de rebalancement du benchmark, on remplace les poids du benchmark à la clôture par les poids cibles du benchmark à cette date (après renormalisation).
            if dt in bench_reb_dates:
                b_target = benchmark_weights.iloc[i].reindex(tickers_all).fillna(0.0).astype(float)
                sbt = float(b_target.sum())
                bench_w_close = b_target / sbt

            # Enregistre les poids du benchmark à la clôture du jour dt.
            bench_weights_close.iloc[i] = bench_w_close.values


  
            # (B) Exécution des ordres en attente (mode positions)
            
            # Initialisation de la variable du coût de transaction en fraction du portefeuille à 0
            cost_frac = 0.0

            # Si on est en mode positions et qu'il y a des poids cibles à appliquer pour la date dt dans le pending (c'est à dire qu'on a rebalancé à la date dt - apply_lag)
            if use_positions and (dt in pending):
                
                # Récupère les poids cibles à appliquer pour la date dt depuis le pending
                target_w = pending.pop(dt).reindex(tickers_inv).fillna(0.0).astype(float)

                # Récupère les prix du jour dt, réindexés sur les tickers investissables et convertis en float
                px_t = prices.iloc[i].astype(float)

                # Calcul de la valeur du portefeuille au jour dt avant l'exécution des ordres en attente, en utilisant les prix du jour dt et le cash disponible.
                value_now = float((shares * px_t).sum() + cash)

                # Calcul de la valeur des actifs au jour dt avant l'exécution des ordres en attente, en utilisant les prix du jour dt.
                assets_now = float((shares * px_t).sum())

                # Calcul des poids du portefeuille au jour dt avant l'exécution des ordres en attente, en utilisant les prix du jour dt.
                w_current_risky = ((shares * px_t) / assets_now).fillna(0.0).astype(float) if assets_now > 1e-12 else pd.Series(0.0, index=tickers_inv, dtype=float)

                # Récupère le coût de transaction en fraction du portefeuille à partir du modèle de coût configuré, en comparant les poids courants et les poids cibles.
                cost_frac = float(self.config.cost_model.cost_fraction(w_current_risky.values, target_w.values))

                # Calcul de la valeur du coût de transaction en multipliant la fraction du coût par la valeur totale du portefeuille avant l'exécution des ordres en attente.
                cost_value = float(cost_frac * value_now)

                # Si le coût est positif, on l'applique en réduisant le cash disponible et la NAV du montant du coût de transaction.
                if cost_value > 0.0:
                    cash = float(cash - cost_value)
                    nav.iloc[i] = float(nav.iloc[i] - cost_value)

                # Calcul de la valeur du portefeuille au jour dt avec la nouvelle valeur de cash
                value_after_cost = float((shares * px_t).sum() + cash)

                # Transformation des poids cibles en nombre de shares à acheter/vendre pour atteindre les poids cibles
                shares_target, cash_target, _ = executor.weights_to_shares(target_weights=target_w, prices_t=px_t, portfolio_value=value_after_cost)

                # Application de l'executor pour calculer les nouvelles positions en nombre de shares et le cash après exécution des ordres nécessaires pour atteindre les poids cibles
                shares, cash, trades_df = executor.execute_rebalance(dt=dt, prices_t=px_t, shares_current=shares, cash_current=cash, shares_target=shares_target,)
                
                # Si des trades ont été exécutés (trades_df n'est pas None et pas vide), on les ajoute au ledger des trades du backtest.
                if not trades_df.empty:
                    trades_ledger.append(trades_df)

                # Enregistre les positions en nombre de shares et le cash après exécution des ordres dans le DataFrame des positions et la série du cash.
                holdings_df.iloc[i] = shares.values
                cash_series.iloc[i] = float(cash)

                # Calcul de la valeur des actifs au jour dt après exécution des ordres, en utilisant les prix du jour dt.
                assets_eod = float((shares * px_t).sum())

                # Calcul des poids du portefeuille à la clôture du jour dt après exécution des ordres, en utilisant les prix du jour dt.
                w_close_post = ((shares * px_t) / assets_eod).fillna(0.0).astype(float) if assets_eod > 1e-12 else pd.Series(0.0, index=tickers_inv, dtype=float)

                # Enregistre les poids à la clôture du jour dt après exécution des ordres.
                weights_close.iloc[i] = w_close_post.values
                weights.iloc[i] = w_close_post.values
 
            #Si on est en mode poids-only et qu'il y a des poids cibles à appliquer pour la date dt dans le pending, on les applique simplement aux poids courants.
            elif use_positions:
                px_t = prices.iloc[i].astype(float)
                holdings_df.iloc[i] = shares.values
                cash_series.iloc[i] = float(cash)



            # (C) Décision de rebalancement

            # Récupère les poids du portefeuille à la clôture du jour dt avant l'exécution des ordres en attente pour les utiliser dans la décision de rebalancement.
            current_w_for_decision = weights_close.iloc[i]
            
            # Détermine si on doit rebalancer à la date dt en appelant la méthode _should_rebalance avec les informations du jour et les poids du portefeuille à la clôture avant exécution des ordres en attente.
            if self._should_rebalance(dt=dt, i=i, reb_dates=reb_dates, dates=dates, port_ret=port_ret, bench_ret=bench_ret, current_w=current_w_for_decision, nav=nav, weights=weights):
                
                # Si on doit rebalancer, on appelle la méthode pour calculer les poids cibles à appliquer, en passant toutes les informations nécessaires à la décision de rebalancement.
                out = self._compute_target_weights_if_possible(i=i, dt=dt, dates=dates, universe_returns=all_returns, bench_ret=bench_ret,
                                                               tickers=tickers_inv, kept_assets=kept_assets_ordered, strategy=strategy,
                                                               current_w=current_w_for_decision, data_lag=data_lag, bench_lag=bench_lag, 
                                                               benchmark_weights=bench_weights_close,)

                # Si la méthode de calcul des poids cibles retourne une sortie non nulle
                if out is not None:

                    # Recupère les poids cibles à appliquer et les diagnostics de la décision de rebalancement depuis la sortie de la méthode.
                    target_w, decision_diag = out
                    last_decision_diag = decision_diag

                    # Si des diagnostics de la décision de rebalancement sont retournés
                    if decision_diag is not None:

                        # initialise une ligne de diagnostic avec la date de rebalancement
                        _diag_row = {"rebal_date": dt}

                        #Itère sur les éléments du diagnostic de la décision de rebalancement, tente de convertir les valeurs en float et de les ajouter à la ligne de diagnostic si elles sont finies.
                        for k, v in decision_diag.items():
                            try:
                                fv = float(v)
                                if np.isfinite(fv):
                                    _diag_row[k] = fv
                            except (TypeError, ValueError):
                                pass
                        _rebal_diags.append(_diag_row)

                    # Si c'est la première fois qu'on rebalance dans le backtest, on met à jour la variable _first_rebal_date avec la date de rebalancement
                    if _first_rebal_date is None:
                        _first_rebal_date = dt
                        _s = slice(_first_rebal_date, None)
                        metrics_out = pd.DataFrame(metrics).loc[_s] if not pd.DataFrame(metrics).empty else pd.DataFrame(metrics)

                    #Calcul de la date d'application effective des poids cibles en ajoutant le lag d'application à l'index de la date de rebalancement dans la liste des dates du backtest.
                    eff_loc = i + apply_lag

                    # Si la date d'application effective est dans les limites des dates du backtest, on enregistre les poids cibles à appliquer dans le pending pour qu'ils soient appliqués à la date d'application effective.
                    if eff_loc < len(dates):
                        pending[dates[eff_loc]] = target_w.copy()

                    # incrémente le compteur de rebalancement
                    rebal_count += 1

                    # Si le mode de log est verbose, on enregistre un log de la décision de rebalancement et on l'affiche.
                    if self.config.verbose:
                        snapshot = {name: float(metrics[name].iloc[i]) for name in metrics.keys()}
                        self._logger.log(rebal_count=rebal_count, dt=dt, i=i, nav=nav, port_ret=port_ret, bench_ret=bench_ret, current_w=current_w_for_decision.copy(),
                                         target_w=target_w, cost_frac=0.0, last_decision_diag=last_decision_diag, metrics_snapshot=snapshot,)


            # (D) Mise à jour des métriques plugables

            #Itère sur les métriques de rebalancement configurées et met à jour chacune d'elles
            for m in self.config.rebalance_metrics:
                val = m.update(dt=dt, i=i, dates=dates, nav=nav, portfolio_returns=port_ret, benchmark_returns=bench_ret, 
                               weights=weights, decision_diagnostics=last_decision_diag,)
                metrics[m.name].iloc[i] = np.nan if val is None else float(val)


        # 11) Diagnostics globaux
        #Recupère les rendements du portefeuille et du benchmark, et calcule le rendement actif
        bench_ret  = bench_ret.astype(float)
        active_ret = port_ret - bench_ret

        # Calcule les diagnostics globaux du backtest : nombre de jours, NAV finale, et tracking error annualisée de l'actif par rapport au benchmark.
        diagnostics: Dict[str, float] = {
            "n_days": float(len(port_ret)),
            "final_nav": float(nav.iloc[-1]),
            "expost_te_annualized":float(active_ret.std(ddof=1) * np.sqrt(252.0)),}

        # 12) Journal des trades
        trades_df = None

        #Si on est en mode positions, on concatène les DataFrames de trades du ledger en un seul DataFrame, ou on crée un DataFrame vide avec les colonnes appropriées si aucun trade n'a été exécuté.
        if use_positions:
            trades_df = (pd.concat(trades_ledger, ignore_index=True) if trades_ledger else pd.DataFrame(columns=["dt", "ticker", "qty_delta", "price", "notional", "side"]))

        # 13) Découpe depuis le premier rebalancement

        #Si une date de premier rebalancement a été enregistrée, on découpe les outputs du backtest depuis cette date pour ne conserver que la période depuis le premier rebalancement réussi.
        if _first_rebal_date is not None:
            _s = slice(_first_rebal_date, None)
            nav_out = nav.loc[_s]
            nav_out = nav_out / nav_out.iloc[0]
            port_ret_out = port_ret.loc[_s]
            bench_ret_out = bench_ret.loc[_s]
            weights_in_force_out = weights_in_force.loc[_s]
            weights_close_out = weights_close.loc[_s]
            bench_wif_out = bench_weights_in_force.loc[_s] if bench_weights_in_force is not None else None
            bench_wc_out = bench_weights_close.loc[_s]    if bench_weights_close    is not None else None

        # Sinon, on conserve l'intégralité des résultats.
        else:
            nav_out = nav
            port_ret_out = port_ret
            bench_ret_out = bench_ret
            weights_in_force_out = weights_in_force
            weights_close_out = weights_close
            bench_wif_out = bench_weights_in_force
            bench_wc_out = bench_weights_close

        # 14) Fermeture du handle memmap
        if precomputed_cov_provider is None:
            if hasattr(cov_provider, "_covariance_path") and cov_provider._covariance_path is not None:
                cov_provider._covariance_path.close()
                cov_provider._covariance_path = None
                cov_provider._path_H = None

        # 15) Diagnostics de rebalancement

        # Si des diagnostics de décision de rebalancement ont été enregistrés, on les convertit en DataFrame et on les indexe sur la date de rebalancement.
        if _rebal_diags:
            _rd = pd.DataFrame(_rebal_diags).set_index("rebal_date")

            # Si une date de premier rebalancement a été enregistrée, on découpe le DataFrame des diagnostics de rebalancement pour ne conserver que la période depuis le premier rebalancement.
            if _first_rebal_date is not None:
                _rd = _rd.loc[_rd.index >= _first_rebal_date]
        else:
            _rd = None

        # 16) Log solveur

        # Initialisation d'une variable pour stocker le log du solveur d'allocation si la stratégie ou son implémentation en fournit un.
        _solver_log = None

        # Si la stratégie est un wrapper autour d'une implémentation qui contient un log du solveur d'allocation, on le récupère pour l'inclure dans les résultats du backtest.
        if hasattr(strategy, "_impl"):
            alloc = strategy._impl

            # Si l'implémentation de la stratégie d'allocation a un attribut _solver_eval_log, on le récupère pour l'inclure dans les résultats du backtest.
            if hasattr(alloc, "_solver_eval_log"):
                _solver_log = alloc._solver_eval_log

        # 17) Résultats du backtest : return de la fonction de backtest avec tous les outputs et diagnostics calculés.
        return BacktestResult(
            nav=nav_out,
            portfolio_returns=port_ret_out,
            benchmark_returns=bench_ret_out,
            weights=weights,
            weights_in_force=weights_in_force_out,
            weights_close=weights_close_out,
            diagnostics=diagnostics,
            metrics=pd.DataFrame(metrics_out),
            cash=cash_series if use_positions else None,
            holdings=holdings_df if use_positions else None,
            trades=trades_df if use_positions else None,
            bench_weights_in_force=bench_wif_out,
            bench_weights_close=bench_wc_out,
            rebal_dates=rebal_dates,
            first_rebal_date=_first_rebal_date,
            rebal_diagnostics=_rd,
            solver_eval_log=_solver_log,
        )