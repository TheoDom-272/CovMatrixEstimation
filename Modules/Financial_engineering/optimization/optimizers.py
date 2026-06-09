"""
Solveurs d'optimisation pour la minimisation de la tracking error.
 
Ce fichier expose deux solveurs interchangeables, tous deux compatibles avec
l'interface OptimizationProblem / OptimizationResult définie dans base.py.
 
- SLSQPOptimizer : solveur scipy (SLSQP), warm-start via w0, gradient analytique
  si disponible. Simple, robuste, légèrement sous-optimal numériquement.
- ClarabelOptimizer : solveur QP C++ (Clarabel), extrait P et q directement depuis
  l'objectif. Trouve le vrai minimum global, numériquement plus précis sur les
  matrices mal conditionnées.
 
Les deux solveurs retournent toujours un OptimizationResult valide, même en cas d'échec.
 
Classes
-------
SLSQPSettings :
    Paramètres de convergence du solveur SLSQP (maxiter, ftol, verbose).
SLSQPOptimizer :
    Wrapper scipy SLSQP avec support des contraintes d'égalité, des bornes et du gradient analytique.
ClarabelSettings :
    Paramètres de convergence du solveur Clarabel (eps_abs, eps_rel, max_iter, verbose).
ClarabelOptimizer :
    Wrapper Clarabel QP avec formulation directe P/q et support long-only via NonnegativeCone.
 
Fonctions
---------
make_optimizer :
    Fabrique le bon solveur par nom de chaîne ('slsqp' ou 'clarabel').
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np

from .base import OptimizationProblem, OptimizationResult
from scipy.optimize import minimize

import scipy.sparse as _sp
import clarabel as _clarabel


@dataclass(frozen=True)
class SLSQPSettings:
    maxiter: int = 10_000
    ftol: float = 1e-12
    disp: bool = False


class SLSQPOptimizer:
    """
    Wrapper scipy SLSQP pour la minimisation sous contraintes.
 
    Supporte les contraintes linéaires d'égalité (A @ w = b), les bornes par actif,
    et le gradient analytique si l'objectif l'expose via objective.gradient().
 
    Attributes
    ----------
    settings : SLSQPSettings
        Paramètres de convergence du solveur.
 
    Methods
    -------
    solve(problem, w0) -> OptimizationResult :
        Résout le problème d'optimisation et retourne les poids optimaux.
    """

    def __init__(self, settings: Optional[SLSQPSettings] = None) -> None:
        # utilise les settings par défaut si non fournis
        self.settings = settings or SLSQPSettings()


    def solve(self, problem: OptimizationProblem, w0: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Résout le problème d'optimisation avec SLSQP.
 
        Construit les contraintes scipy depuis OptimizationProblem, appelle
        scipy.optimize.minimize, et retourne un OptimizationResult normalisé.
 
        Parameters
        ----------
        problem : OptimizationProblem
            Problème à résoudre (objectif, contraintes, bornes, métadonnées).
        w0 : np.ndarray or None
            Point de départ. Si None, utilise des poids uniformes (1/n).
 
        Returns
        -------
        OptimizationResult
            Poids optimaux, statut de convergence et diagnostics.
        """

        if minimize is None:
            raise ImportError("scipy is required: pip install scipy")

        n = problem.n_assets

        # Point de départ : uniforme si non fourni, sinon valide la forme
        if w0 is None:

            # poids uniformes par défaut
            w0 = np.full(n, 1.0 / n, dtype=float)

        else:
            w0 = np.asarray(w0, dtype=float)
            if w0.shape != (n,):
                raise ValueError(f"w0 must be shape ({n},).")


        # Construit les contraintes d'égalité sous forme scipy
        constraints = []
        if problem.eq is not None:
            problem.eq.check_shapes(n)
            A = problem.eq.A
            b = problem.eq.b

            def fun_eq(w: np.ndarray) -> np.ndarray:
                # Résidu de la contrainte d'égalité : A @ w - b (doit être 0)
                return A @ w - b

            def jac_eq(w: np.ndarray) -> np.ndarray: 
                # Contrainte d'égalité linéaire : somme des poids doit être égale à 1 (A = [1, 1, ..., 1], b = [1])
                return A

            constraints.append({"type": "eq", "fun": fun_eq, "jac": jac_eq})

        # Convertit les bornes en format scipy
        bounds_list: Optional[List[Tuple[float, float]]] = None
        if problem.bounds is not None:
            problem.bounds.check_shapes(n)
            bounds_list = list(zip(problem.bounds.lb.tolist(), problem.bounds.ub.tolist()))


        def fun(w: np.ndarray) -> float:
             # Fonction objectif scalaire passée à scipy
            return float(problem.objective.value(w))

         # Utilise le gradient analytique si l'objectif l'expose
        jac = None
        if hasattr(problem.objective, "gradient"):
            def jac_fun(w: np.ndarray) -> np.ndarray:
                g = problem.objective.gradient(w)
                if g is None:
                    raise ValueError("Objective.gradient returned None but was requested.")
                return np.asarray(g, dtype=float)
            
            # gradient analytique, convergence plus rapide
            jac = jac_fun

        # Appel principal SLSQP
        res = minimize(fun=fun, x0=w0, jac=jac, method="SLSQP", bounds=bounds_list, constraints=constraints,
                       options={"maxiter": self.settings.maxiter, "ftol": self.settings.ftol, "disp": self.settings.disp},)

        # poids optimaux bruts
        w_star = np.asarray(res.x, dtype=float)

        # valeur de l'objectif au point optimal
        obj = float(problem.objective.value(w_star))

        #Diagnostique de l'optimisation avec le nombre d'itérations et le code de statut scipy
        diagnostics: Dict[str, float] = {"nit": float(getattr(res, "nit", np.nan)), "status": float(getattr(res, "status", np.nan)),}

        return OptimizationResult(w=w_star, success=bool(res.success), message=str(res.message), objective_value=obj, diagnostics=diagnostics,)




# ClarabelOptimizer

@dataclass(frozen=True)
class ClarabelSettings:
    """
    Paramètres de convergence du solveur Clarabel QP.
 
    Attributes
    ----------
    eps_abs : float
        Tolérance absolue sur la résidu primal/dual.
    eps_rel : float
        Tolérance relative sur la résidu primal/dual.
    max_iter : int
        Nombre maximal d'itérations du solveur intérieur.
    verbose : bool
        Si True, affiche le log de convergence Clarabel dans la console.
    regularization : float
        Régularisation diagonale ajoutée à P (P = P + 2*gamma*I). 0 = pas de régularisation. Utile sur les matrices très mal conditionnées.
    """

    eps_abs: float = 1e-12
    eps_rel: float = 1e-12
    max_iter: int = 10_000
    verbose: bool = False
    regularization: float = 0 #1e-4


class ClarabelOptimizer:
    """
    Wrapper Clarabel QP pour la minimisation de la tracking error.
 
    Attributes
    ----------
    settings : ClarabelSettings
        Paramètres de convergence du solveur.
 
    Methods
    -------
    solve(problem, w0) -> OptimizationResult :
        Résout le QP et retourne les poids optimaux.
    """


    def __init__(self, settings: Optional[ClarabelSettings] = None) -> None:
        self.settings = ClarabelSettings()

    def _extract_qp_matrices(self, problem: OptimizationProblem) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrait les matrices P et q du problème QP depuis l'objectif.
 
        Parameters
        ----------
        problem : OptimizationProblem
            Problème d'optimisation contenant l'objectif.
 
        Returns
        -------
        P : np.ndarray
            Matrice quadratique (K, K), P = 2*Sigma (doit être symétrique définie positive).
        q : np.ndarray
            Vecteur linéaire (K,), q = -2*Sigma@b.
        """

        obj = problem.objective

         # Cas TrackingErrorObjective : Sigma et b directement accessibles
        if hasattr(obj, "cov") and hasattr(obj, "benchmark_weights"):
            cov = np.asarray(obj.cov, dtype=float)
            b   = np.asarray(obj.benchmark_weights, dtype=float)
            return 2.0 * cov, -2.0 * (cov @ b)

        # Cas TrackingErrorFullUniverseObjective : cov_full sur N, kept_idx pour réduire
        if hasattr(obj, "cov_full") and hasattr(obj, "benchmark_weights_full") and hasattr(obj, "kept_idx"):
            cov_full = np.asarray(obj.cov_full, dtype=float)
            b_full   = np.asarray(obj.benchmark_weights_full, dtype=float)
            idx      = np.asarray(obj.kept_idx, dtype=int)

            # sous-matrice K×K pour les actifs investissables
            cov_red  = cov_full[np.ix_(idx, idx)]

            # Sigma @ b sur N (vecteur complet)
            q_full   = cov_full @ b_full

            # extrait les K composantes kept
            return 2.0 * cov_red, -2.0 * q_full[idx]


        raise TypeError("[ClarabelOptimizer] Objective type not supported for direct QP extraction.")

    def solve(self, problem: OptimizationProblem, w0: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Résout le problème QP avec Clarabel.
 
        Parameters
        ----------
        problem : OptimizationProblem
            Problème d'optimisation.
        w0 : np.ndarray or None
            Point de départ (utilisé comme fallback si le solveur échoue).
 
        Returns
        -------
        OptimizationResult
            Poids optimaux, statut et diagnostics Clarabel.
        """
        
        n = problem.n_assets
        s = self.settings

        # Point de départ : uniforme si non fourni
        if w0 is None:
            w0 = np.full(n, 1.0 / n, dtype=float)
        else:
            w0 = np.asarray(w0, dtype=float)

        # Extrait les matrices QP depuis l'objectif
        P_dense, q = self._extract_qp_matrices(problem)

        # Régularisation optionnelle : P = P + 2*gamma*I pour améliorer le conditionnement
        gamma = float(s.regularization)
        if gamma > 0.0:

            # renforce la diagonale
            P_dense = P_dense + 2.0 * gamma * np.eye(n)

            # ajuste le terme linéaire
            q       = q - 2.0 * gamma * w0

        # True si bornes configurées (mode long-only)
        long_only = problem.bounds is not None

        # Convertit P en matrice creuse CSC (format requis par Clarabel)
        P_csc = _sp.csc_matrix(P_dense)

        if long_only:
            # Matrice de contraintes : [1' ; -I]
            # - ZeroConeT(1)         : 1'w = 1 (contrainte budget, égalité)
            # - NonnegativeConeT(n)  : -w <= 0  ↔  w >= 0 (long-only)
            A_eq   = np.ones((1, n))
            A_ineq = -np.eye(n)     
            A_dense = np.vstack([A_eq, A_ineq])
            b_cons = np.append([1.0], np.zeros(n))
            cones = [
                _clarabel.ZeroConeT(1), # contrainte d'égalité (budget = 1)
                _clarabel.NonnegativeConeT(n),  # contrainte inégalité (long-only)
            ]
        else:
            # Sans long-only : uniquement la contrainte budget
            A_dense = np.ones((1, n))
            b_cons  = np.array([1.0])
            cones   = [_clarabel.ZeroConeT(1)]

        # Convertit A en matrice creuse CSC
        A_csc = _sp.csc_matrix(A_dense)


        # Configure les paramètres Clarabel avec robustesse aux différentes versions
        settings_cl = _clarabel.DefaultSettings()
        for attr, val in [
            ("eps_abs", s.eps_abs),
            ("eps_prim_inf", s.eps_abs),   # nom alternatif selon version
            ("eps_rel", s.eps_rel),
            ("max_iter", s.max_iter),
            ("max_iters", s.max_iter),  # nom alternatif selon version
            ("verbose", s.verbose),
        ]:
            if hasattr(settings_cl, attr):
                # applique uniquement si l'attribut existe
                setattr(settings_cl, attr, val)

        # Lance le solveur Clarabel
        solver = _clarabel.DefaultSolver(P_csc, q, A_csc, b_cons, cones, settings_cl)
        res    = solver.solve()

        # Vérifie le statut
        status_str = str(res.status)
        solved = "Solved" in status_str 

        w_star = np.asarray(res.x, dtype=float)

        # Nettoyage numérique : clip long-only et renormalise pour sommer à 1
        if long_only:
            # élimine les poids légèrement négatifs
            w_star = np.clip(w_star, 0.0, 1.0)

        s_sum = float(w_star.sum())

        if s_sum > 1e-12:
            # renormalise pour sommer exactement à 1
            w_star = w_star / s_sum

        else:
            # fallback sur le point de départ
            w_star = w0.copy()
            solved = False

        # évalue l'objectif au point optimal
        obj = float(problem.objective.value(w_star))

        #Diagnostique, succès ou échec, nombre d'itérations
        diagnostics: Dict[str, float] = {"status": float(0 if solved else 1), "iter": float(getattr(res, "iterations", np.nan)), "obj_val": float(obj),}

        return OptimizationResult(w=w_star, success=bool(solved), message=status_str, objective_value=obj, diagnostics=diagnostics,)
    



# Fabrique
def make_optimizer(name: str = "osqp", **kwargs):
    """
    Instancie le bon solveur par nom de chaîne.
 
    Parameters
    ----------
    name : str
        Nom du solveur : 'slsqp' ou 'clarabel'.
    **kwargs :
        Paramètres additionnels passés aux settings du solveur.
 
    Returns
    -------
    SLSQPOptimizer or ClarabelOptimizer
        Instance du solveur configuré.
 
    Raises
    ------
    ValueError
        Si name n'est pas reconnu.
    """

    name = name.lower().strip()
    if name == "slsqp":
        return SLSQPOptimizer(SLSQPSettings(**kwargs) if kwargs else None)
    if name == "clarabel":
        return ClarabelOptimizer(ClarabelSettings(**kwargs) if kwargs else None)
    raise ValueError(f"Unknown optimizer name: {name!r}. " f"Choose from: slsqp, osqp, clarabel, active_set.")