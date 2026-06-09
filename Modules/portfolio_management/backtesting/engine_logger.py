"""
Logger console du moteur de backtest.

Gère le formatage et l'affichage des messages de suivi à chaque rebalancement.
Entièrement découplé de la logique du backtest : reçoit les données calculées
par le moteur et les formate en une ligne lisible dans la console.

Classes
-------
BacktestLogger :
    Formate et affiche un message de log à chaque rebalancement, avec les métriques
    de base (NAV, rendements, turnover, coûts) et les diagnostics optionnels
    de la stratégie et des métriques plugables.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class BacktestLogger:
    """
    Logger console pour le moteur de backtest.

    Appelé à chaque rebalancement pour afficher un résumé de l'état du portefeuille.

    Attributes
    ----------
    verbose : bool
        Si False, aucun message n'est affiché (logger désactivé).
    log_every_n_rebalances : int
        Fréquence d'affichage : 1 = chaque rebalancement, 2 = un sur deux, etc.

    Methods
    -------
    log(rebal_count, dt, i, nav, port_ret, bench_ret, current_w, target_w, cost_frac, last_decision_diag, metrics_snapshot) -> None :
        Formate et affiche le message de log pour le rebalancement courant. Ne fait rien si verbose=False ou si le rebalancement ne correspond pas à la fréquence configurée.
    """

    def __init__(self, verbose: bool, log_every_n_rebalances: int) -> None:
        self.verbose = verbose
        self.log_every_n_rebalances = log_every_n_rebalances

    def log(self, rebal_count: int, dt: pd.Timestamp, i: int, nav: pd.Series, port_ret: pd.Series, bench_ret: pd.Series, current_w: pd.Series, target_w: pd.Series,
        cost_frac: float, last_decision_diag: Optional[Dict[str, object]], metrics_snapshot: Optional[Dict[str, object]] = None,) -> None:
        """
        Formate et affiche le message de log pour le rebalancement courant.

        Affiche une ligne de base avec NAV, rendements, turnover et coûts,
        puis ajoute les diagnostics de la stratégie et les métriques plugables
        si disponibles. Les valeurs NaN, None et les chaînes vides sont ignorées.

        Parameters
        ----------
        rebal_count : int
            Numéro séquentiel du rebalancement (commence à 1).
        dt : pd.Timestamp
            Date courante du rebalancement.
        i : int
            Index entier de dt dans l'index de dates du backtest.
        nav : pd.Series
            Série de NAV du portefeuille (partiellement remplie jusqu'à i).
        port_ret : pd.Series
            Série des rendements journaliers du portefeuille.
        bench_ret : pd.Series
            Série des rendements journaliers du benchmark.
        current_w : pd.Series
            Poids du portefeuille avant le rebalancement.
        target_w : pd.Series
            Poids cibles décidés lors du rebalancement.
        cost_frac : float
            Coût de transaction appliqué, en fraction de la NAV.
        last_decision_diag : dict or None
            Diagnostics produits par la stratégie lors de ce rebalancement.
        metrics_snapshot : dict or None
            Snapshot des métriques plugables à la date courante.
        """

        # Vérifie si le log doit être affiché selon la configuration
        if not self.verbose:
            return

        # Contrôle de la fréquence d'affichage
        n_every = int(self.log_every_n_rebalances)
        if n_every <= 0:
            raise ValueError("log_every_n_rebalances must be >= 1.")
        if (rebal_count % n_every) != 0:
            return

        # Récupération de la nav, et des rendements
        nav_t = float(nav.iloc[i])
        r_p = float(port_ret.iloc[i])
        r_b = float(bench_ret.iloc[i]) if np.isfinite(bench_ret.iloc[i]) else np.nan

        # Turnover one-way 
        dw = target_w.values - current_w.values
        turnover = float(np.sum(np.abs(dw)) / 2.0)

        #Base du message de log avec les métriques de base
        base = (f"[Backtest] Rebal #{rebal_count} | dt={dt.date()} | "  f"NAV={nav_t:.6f} | r_p={r_p:+.5%} | r_b={r_b:+.5%} | " f"turnover={turnover:.4f} | cost={float(cost_frac):.6f}")

        #Initialise la liste des parties additionnelles du message (diagnostics et métriques plugables)
        extra_parts: list[str] = []

        # Diagnostics stratégie si disponibles
        if isinstance(last_decision_diag, dict) and len(last_decision_diag) > 0:
            flat = self._flatten_diag(last_decision_diag) # Recupere les diagnostics de la stratégie, en aplatissant les éventuels dictionnaires imbriqués
            for k in sorted(flat.keys()):
                sv = self._fmt_value(flat[k])
                if sv is not None:
                    extra_parts.append(f"{k}={sv}")

        # Snapshot métriques plugables si disponibles
        if isinstance(metrics_snapshot, dict) and len(metrics_snapshot) > 0:
            for k in sorted(metrics_snapshot.keys()):
                sv = self._fmt_value(metrics_snapshot[k])
                if sv is not None:
                    extra_parts.append(f"{k}={sv}")

        # Concatène la ligne de base avec les parties additionnelles
        msg = base + (" | " + " | ".join(extra_parts) if extra_parts else "")

        #Affiche le message formaté dans la console
        print(msg)

    @staticmethod
    def _fmt_value(v: object) -> Optional[str]:
        """
        Formate une valeur en chaîne loggable.

        Retourne None pour les valeurs non affichables (None, NaN, chaîne vide).

        Parameters
        ----------
        v : object
            Valeur à formater (float, int, bool, str, dict, list, ou autre).

        Returns
        -------
        str or None
            Représentation formatée, ou None si la valeur doit être ignorée.
        """

        #Si la valeur est None, ou une chaîne vide, ou un NaN, on retourne None pour l'ignorer dans le log
        if v is None:
            return None

        # NaN numpy
        if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
            return None

        # Nombres, affichés avec 6 chiffres significatifs, sans notation scientifique
        if isinstance(v, (int, float, np.number)):
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    return None
                return f"{fv:.6g}"
            except Exception:
                return None

        # si la valeur est un booléen, on affiche "True" ou "False"
        if isinstance(v, bool):
            return "True" if v else "False"

        # si la valeur est une chaîne, on l'affiche (tronquée si trop longue)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            return (s[:80] + "...") if len(s) > 80 else s

        # Dict / list / tuple : on affiche leur repr, tronquée si trop longue
        if isinstance(v, (dict, list, tuple)):
            s = repr(v)
            return (s[:120] + "...") if len(s) > 120 else s

        # Fallback dans les autres cas : on affiche la repr, tronquée si trop longue
        s = str(v)
        return (s[:80] + "...") if len(s) > 80 else s

    @staticmethod
    def _flatten_diag(d: Dict[str, object], prefix: str = "") -> Dict[str, object]:
        """
        Aplatit un dictionnaire potentiellement imbriqué (un niveau max).

        Parameters
        ----------
        d : dict
            Dictionnaire à aplatir (niveau 1-2 max).
        prefix : str
            Préfixe à ajouter aux clés (utilisé en récursion interne).

        Returns
        -------
        dict
            Dictionnaire aplati avec toutes les valeurs scalaires accessibles.
        """

        #Initilise le dictionnaire de sortie
        out: Dict[str, object] = {}

        #Itere sur les paires clé-valeur du dictionnaire d'entrée
        for k, v in d.items():

            #Recupère la clé complète en ajoutant le préfixe si nécessaire
            kk = f"{prefix}{k}" if prefix else str(k)

            #Si la valeur est un dictionnaire, on itère sur ses paires clé-valeur
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    # La clé finale est la concaténation de la clé du niveau supérieur et de la clé du niveau inférieur
                    out[f"{kk}.{k2}"] = v2
            #Sinon, on ajoute la paire clé-valeur au dictionnaire de sortie
            else:
                out[kk] = v
        return out