"""
Modélisation des coûts de transaction appliqués lors des rebalancements de portefeuille.
 
Classes
-------
TransactionCostModel :
    Calcule le coût total d'un rebalancement à partir du turnover et d'un coût fixe par rebalancement.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TransactionCostModel:
    """
    Modèle simple de coûts de transaction appliqué à chaque rebalancement.
 
    Attributes
    ----------
    proportional_bps : float
        Taux de coût proportionnel en basis points, appliqué sur le turnover one-way.
    fixed_cost : float
        Coût fixe par rebalancement, exprimé en fraction de la NAV.
 
    Methods
    -------
    cost_fraction(w_prev, w_new) -> float :
        Calcule le coût total d'un rebalancement en fraction de la NAV, à partir des poids avant et après rebalancement.
    """

    proportional_bps: float = 0.0
    fixed_cost: float = 0.0

    def cost_fraction(self, w_prev: np.ndarray, w_new: np.ndarray) -> float:
        """
        Calcule le coût total d'un rebalancement en fraction de la NAV.
 
        Le coût proportionnel est calculé sur le turnover one-way (somme des
        variations absolues de poids). Le coût fixe est ajouté systématiquement.
 
        Parameters
        ----------
        w_prev : np.ndarray
            Vecteur de poids avant rebalancement.
        w_new : np.ndarray
            Vecteur de poids cibles après rebalancement.
 
        Returns
        -------
        float
            Coût total en fraction de NAV (toujours >= 0).
        """
        
        w_prev = np.asarray(w_prev, dtype=float)
        w_new = np.asarray(w_new, dtype=float)
        if w_prev.shape != w_new.shape:
            raise ValueError("w_prev and w_new must have same shape.")

        turnover = float(np.sum(np.abs(w_new - w_prev)))
        proportional = (self.proportional_bps / 10000.0) * turnover
        total = proportional + float(self.fixed_cost)

        # Defensive: jamais negatif
        return float(max(total, 0.0))
