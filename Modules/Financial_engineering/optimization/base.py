"""
Contrats d'interface et structures de données pour le moteur d'optimisation.
 
Ce fichier définit les classes abstraites et dataclasses qui constituent
le contrat entre les objectifs d'optimisation, les contraintes et les solveurs.
Aucun calcul numérique ici, uniquement les types qui circulent entre composants.
 
Classes
-------
Objective :
    Interface que tout objectif d'optimisation doit implémenter (méthode value() obligatoire, gradient() optionnel).
LinearEqualityConstraint :
    Dataclass représentant une contrainte linéaire d'égalité de la forme A @ w = b.
Bounds :
    Dataclass représentant des bornes inférieures et supérieures par actif.
OptimizationProblem :
    Conteneur regroupant objectif, nombre d'actifs, contraintes et métadonnées, passé tel quel au solveur.
OptimizationResult :
    Objet de sortie du solveur contenant les poids optimaux, le statut et les diagnostics.
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import numpy as np

# Interface : objectif d'optimisation
class Objective(Protocol):
    """
    Interface que tout objectif d'optimisation doit respecter.
 
    Le solveur appelle value() à chaque évaluation de la fonction objectif.
    Si gradient() est implémenté, le solveur peut l'utiliser pour accélérer
    la convergence (gradient analytique au lieu de différences finies).
 
    Methods
    -------
    value(w) -> float :
        Retourne la valeur scalaire de l'objectif au point w.
    gradient(w) -> np.ndarray or None :
        Retourne le gradient de l'objectif au point w.None si le gradient analytique n'est pas disponible.
    """

    def value(self, w: np.ndarray) -> float:
        """
        Évalue la fonction objectif au point w.
 
        Parameters
        ----------
        w : np.ndarray
            Vecteur de poids courant (K,).
 
        Returns
        -------
        float
            Valeur scalaire de l'objectif (ex: TE^2, variance, etc.).
        """
        ...

    def gradient(self, w: np.ndarray) -> Optional[np.ndarray]:
        """
        Retourne le gradient analytique de l'objectif au point w.
 
        Parameters
        ----------
        w : np.ndarray
            Vecteur de poids courant (K,).
 
        Returns
        -------
        np.ndarray or None
            Gradient de shape (K,), ou None si non disponible.
        """
        return None #Pas obligatoire


# Dataclass : contrainte linéaire d'égalité
@dataclass(frozen=True)
class LinearEqualityConstraint:
    """
    Contrainte linéaire d'égalité de la forme A @ w = b.
 
    Attributes
    ----------
    A : np.ndarray
        Matrice de contrainte de shape (m, n) avec m contraintes et n actifs.
    b : np.ndarray
        Vecteur cible de shape (m,).
 
    Methods
    -------
    check_shapes(n) -> None :
        Vérifie que A et b sont cohérents avec n actifs. Lève ValueError sinon.
    """

    # matrice de contrainte (m, n) : m = nombre de contraintes, n = nombre d'actifs
    A: np.ndarray  

     # vecteur cible (m,)
    b: np.ndarray  

    def check_shapes(self, n: int) -> None:
        """
        Vérifie la cohérence dimensionnelle de la contrainte.
 
        Parameters
        ----------
        n : int
            Nombre d'actifs dans le problème d'optimisation.
 
        Raises
        ------
        ValueError
            Si les dimensions de A ou b ne correspondent pas à n.
        """

        if self.A.ndim != 2:
            raise ValueError("A must be a 2D array.")
        if self.b.ndim != 1:
            raise ValueError("b must be a 1D array.")
        if self.A.shape[1] != n:
            raise ValueError(f"A has {self.A.shape[1]} columns; expected {n}.")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A and b row dimension mismatch.")



# Dataclass : bornes par actif
@dataclass(frozen=True)
class Bounds:
    """
    Bornes inférieures et supérieures par actif.
 
    En mode long-only, lb = 0 et ub = 1 pour chaque actif.
    Des bornes asymétriques permettent de modéliser des contraintes
    de poids minimum ou maximum par actif.
 
    Attributes
    ----------
    lb : np.ndarray
        Bornes inférieures de shape (n,). 
    ub : np.ndarray
        Bornes supérieures de shape (n,). 
 
    Methods
    -------
    check_shapes(n) -> None :
        Vérifie que lb et ub ont bien la bonne dimension. Lève ValueError sinon.
    """

    # bornes inférieures (n,)
    lb: np.ndarray 

    # bornes supérieures (n,)
    ub: np.ndarray 

    def check_shapes(self, n: int) -> None:
        """
        Vérifie la cohérence dimensionnelle des bornes.
 
        Parameters
        ----------
        n : int
            Nombre d'actifs dans le problème d'optimisation.
 
        Raises
        ------
        ValueError
            Si lb ou ub n'ont pas la bonne shape.
        """

        if self.lb.shape != (n,) or self.ub.shape != (n,):
            raise ValueError("Bounds must be vectors of shape (n,).")
        if np.any(self.lb > self.ub):
            raise ValueError("Invalid bounds: some lb > ub.")



# Dataclass : problème d'optimisation

@dataclass(frozen=True)
class OptimizationProblem:
    """
    Conteneur complet d'un problème d'optimisation, passé tel quel au solveur.
 
    Regroupe l'objectif, le nombre d'actifs, les contraintes et les métadonnées
    contextuelles (date, mode, etc.). Le solveur n'a besoin de rien d'autre.
 
    Attributes
    ----------
    objective : Objective
        Fonction objectif à minimiser (implémente value() et optionnellement gradient()).
    n_assets : int
        Nombre d'actifs dans l'espace d'optimisation (dimension du vecteur w).
    eq : LinearEqualityConstraint or None
        Contrainte d'égalité linéaire. None si absente.
    bounds : Bounds or None
        Bornes par actif. None si pas de contrainte de bornes.
    metadata : dict
        Informations contextuelles passées au solveur pour le logging
    """
    objective: Objective 
    n_assets: int
    eq: Optional[LinearEqualityConstraint] = None
    bounds: Optional[Bounds] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class OptimizationResult:
    """
    Résultat produit par un solveur après résolution du problème d'optimisation.
 
    Attributes
    ----------
    w : np.ndarray
        Vecteur de poids optimaux de shape (n,). Toujours retourné, même en cas d'échec.
    success : bool
        True si le solveur a convergé vers un optimum valide. False si fallback.
    message : str
        Message de statut retourné par le solveur.
    objective_value : float
        Valeur de l'objectif évalué en w.
    diagnostics : dict
        Métriques internes du solveur.
    """
    w: np.ndarray
    success: bool
    message: str
    objective_value: float
    diagnostics: Dict[str, float]
