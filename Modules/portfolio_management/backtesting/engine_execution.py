"""
Exécution des ordres en mode positions (shares) dans le moteur de backtest.

Ce fichier gère la conversion des poids cibles en quantités réelles et l'exécution
des ordres au prix de clôture. Il est activé uniquement quand des prix sont fournis
au moteur (universe_prices != None). En mode poids-only, ce module n'est pas utilisé.

Classes
-------
PositionExecutor :
    Convertit des poids cibles en quantités (shares) en appliquant les contraintes d'exécution , puis exécute les ordres au prix de clôture en gérant les cas de cash insuffisant.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .engine_types import ExecutionConfig, CashConfig


class PositionExecutor:
    """
    Gère la conversion des poids en quantités et l'exécution des ordres au close.

    Utilisé par le moteur de backtest en mode positions, quand des prix sont disponibles.
    Applique les contraintes d'exécution définies dans ExecutionConfig et gère la poche cash structurelle définie dans CashConfig.

    Attributes
    ----------
    exec_cfg : ExecutionConfig
        Paramètres d'exécution : lot size, notionnel minimum, arrondi, poids minimum.
    cash_cfg : CashConfig
        Paramètres de gestion de la poche cash structurelle.

    Methods
    -------
    weights_to_shares(target_weights, prices_t, portfolio_value) -> tuple :
        Convertit des poids cibles en quantités, en appliquant la poche cash, les filtres de tradeabilité et l'arrondi par lot.
    execute_rebalance(dt, prices_t, shares_current, cash_current, shares_target) -> tuple :
        Calcule les deltas d'ordres, applique le filtre de notionnel minimum, gère le cas de cash insuffisant, et retourne les nouvelles quantités, le cash résiduel et le journal de trades.
    """

    def __init__(self, exec_cfg: ExecutionConfig, cash_cfg: CashConfig) -> None:
        self.exec_cfg = exec_cfg
        self.cash_cfg = cash_cfg

    def weights_to_shares(self, target_weights: pd.Series, prices_t: pd.Series, portfolio_value: float,) -> Tuple[pd.Series, float, Dict[str, float]]:
        """
        Convertit des poids cibles en quantités (shares).

        Applique dans l'ordre :
        1. La poche cash structurelle.
        2. Le filtre de poids minimum.
        3. Le filtre de notionnel minimum par position.
        4. L'arrondi en lots (floor ou round selon exec_cfg.rounding).

        Parameters
        ----------
        target_weights : pd.Series
            Poids cibles normalisés à 1, indexés par ticker.
        prices_t : pd.Series
            Prix de clôture du jour, indexés par ticker.
        portfolio_value : float
            Valeur totale du portefeuille (NAV) utilisée pour calculer les notionnels.

        Returns
        -------
        qty : pd.Series
            Quantités cibles par actif (en nombre d'actions, >= 0).
        cash_target : float
            Cash résiduel après investissement (portfolio_value - valeur investie).
        diag : dict
            Diagnostics d'exécution : valeur investie, cash effectif, nombre de positions.
        """

        # Prépare les poids : remplace les NaN par 0, assure le type float, et aligne sur les actifs disponibles dans prices_t
        w = target_weights.fillna(0.0).astype(float).copy()
        w = w.reindex(prices_t.index).fillna(0.0)

        # Poche cash structurelle : réserve une fraction de la NAV en cash, et on investit le reste
        cash_w = 0.0 if self.cash_cfg is None else float(self.cash_cfg.target_cash_weight)
        cash_w = min(max(cash_w, 0.0), 1.0)
        risky_budget = (1.0 - cash_w) * float(portfolio_value) # Montant à investir dans les actifs (hors cash)

        # Filtre poids minimum
        if self.exec_cfg.min_weight > 0:
            w[np.abs(w) < float(self.exec_cfg.min_weight)] = 0.0

        # Renormalisation sur les actifs restants
        s = float(w.sum())
        if s > 1e-12:
            w = w / s
        else:
            # Tout a été coupé donc full cash
            return (pd.Series(0.0, index=prices_t.index),float(portfolio_value),{"all_cut_to_cash": 1.0},)

        # Notionnel cible par actif
        target_notional = w * risky_budget

        # Filtre notionnel minimum par position
        if self.exec_cfg.min_position_notional > 0:

            # Les positions dont le notionnel cible est inférieur au minimum sont coupées à zéro
            target_notional[target_notional < float(self.exec_cfg.min_position_notional)] = 0.0
            s2 = float(target_notional.sum())

            if s2 > 1e-12:
                # Renormalisation pour réinvestir le cash libéré par les positions coupées
                target_notional = target_notional * (risky_budget / s2) 
            else:
                return (pd.Series(0.0, index=prices_t.index), float(portfolio_value),{"all_small_positions": 1.0},)

        # Récupération des prix
        px = prices_t.astype(float)
        
        #Calcul des quantités brutes avant arrondi
        raw_qty = target_notional / px.replace(0.0, np.nan)

        # Remplace les quantités infinies ou NaN résultant de prix nuls ou de division par zéro par 0.0
        raw_qty = raw_qty.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        #Si les fractions d'actions sont autorisées, on prend les quantités brutes, sinon on arrondit selon la configuration
        if self.exec_cfg.allow_fractional:
            qty = raw_qty
        else:
            lot = max(int(self.exec_cfg.lot_size), 1)
            if self.exec_cfg.rounding == "round":
                qty = (raw_qty / lot).round() * lot
            else:
                qty = np.floor(raw_qty / lot) * lot

        # Les quantités négatives n'ont pas de sens dans ce contexte (pas de short), on les coupe à zéro
        qty = qty.clip(lower=0.0)

        # Cash résiduel
        invested = float((qty * px).sum())
        cash_target = float(portfolio_value - invested)

        diag = {
            "invested_value": invested,
            "cash_value": cash_target,
            "cash_weight_effective": cash_target / portfolio_value if portfolio_value > 0 else np.nan,
            "n_positions": float((qty > 0).sum()),
        }
        return qty.astype(float), cash_target, diag

    def execute_rebalance(self, dt: pd.Timestamp, prices_t: pd.Series, shares_current: pd.Series, cash_current: float, shares_target: pd.Series,) -> Tuple[pd.Series, float, pd.DataFrame]:
        """
        Exécute les ordres au prix de clôture prices_t.

        Calcule les deltas entre quantités actuelles et cibles, applique le filtre
        de notionnel minimum par trade, et gère le cas de cash insuffisant en
        rescalant proportionnellement les achats.

        Parameters
        ----------
        dt : pd.Timestamp
            Date d'exécution des ordres.
        prices_t : pd.Series
            Prix de clôture du jour, indexés par ticker.
        shares_current : pd.Series
            Quantités actuellement détenues, indexées par ticker.
        cash_current : float
            Cash disponible avant exécution.
        shares_target : pd.Series
            Quantités cibles après rebalancement, indexées par ticker.

        Returns
        -------
        shares_new : pd.Series
            Nouvelles quantités après exécution des ordres.
        cash_new : float
            Cash résiduel après exécution (ventes génèrent du cash, achats en consomment).
        trades_df : pd.DataFrame
            Journal des trades exécutés avec colonnes : dt, ticker, qty_delta, price, notional, side.
        """

        #Recupération des prix et calcul des deltas de quantités
        px = prices_t.astype(float)
        delta = shares_target - shares_current

        #Delta de notional par trade
        trade_notional = delta.abs() * px

        # Filtre notionnel minimum par trade
        if self.exec_cfg.min_trade_notional > 0:
            mask = trade_notional >= float(self.exec_cfg.min_trade_notional)
            delta = delta.where(mask, 0.0)

        # Impact cash : achats consomment, ventes génèrent
        cash_change = -float((delta * px).sum())
        cash_new = float(cash_current + cash_change)

        # Si Cash insuffisant, on rescale les achats proportionnellement
        if cash_new < -1e-6:
            buy = delta.clip(lower=0.0) # Quantités à acheter (delta positif)
            sell = -delta.clip(upper=0.0) # Quantités à vendre (delta négatif, on prend le positif pour calculer la valeur)
            buy_value = float((buy * px).sum()) # Coût total des achats
            sell_value = float((sell * px).sum()) # Cash généré par les ventes
            available = float(cash_current + sell_value) # Cash disponible pour les achats après ventes

            #Si le cash disponible est suffisant pour couvrir les achats, on les exécute normalement.
            if buy_value > 1e-12 and available > 0:
                scale = min(1.0, available / buy_value) # Facteur de rescaling pour les achats
                buy2 = pd.Series(np.floor((buy * scale).values), index=buy.index) # On arrondit à l'entier inférieur pour éviter de dépasser le cash disponible
                delta = buy2 - sell # On conserve les ventes complètes, et on rescale les achats
                cash_change = -float((delta * px).sum()) # Recalcul du cash change après rescaling
                cash_new = float(cash_current + cash_change) # Cash résiduel après exécution

            # Sinon on les rescale proportionnellement au cash disponible
            else:
                # ventes uniquement, aucun achat
                delta = -sell
                cash_change = -float((delta * px).sum())
                cash_new = float(cash_current + cash_change)

        # Nouvelles quantités après exécution
        shares_new = shares_current + delta

        # Journal de trades (uniquement les ordres non nuls)
        trades_df = pd.DataFrame({
            "dt": dt,
            "ticker": delta.index,
            "qty_delta": delta.values,
            "price": px.values,
            "notional": delta.values * px.values,
            "side": np.where(delta.values >= 0, "BUY", "SELL"),
        })

        # On ne garde que les trades avec une quantité non nulle
        trades_df = trades_df.loc[trades_df["qty_delta"].abs() > 0].reset_index(drop=True)

        return shares_new, cash_new, trades_df