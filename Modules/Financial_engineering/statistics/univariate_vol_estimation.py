
"""
Volatility models: GARCH / GJR / EGARCH / APARCH — package or scratch
=====================================================================

Architecture
------------
- VolModel (base) : interface commune .fit() / .conditional_sigma()
- GARCHPQ         : GARCH(p,q)   — scratch + package 
- GJRGARCHPQ      : GJR-GARCH(p,q) (leverage) — scratch + package 
- EGARCHPQ        : EGARCH(p,q)  — package, scratch = NotImplemented
- APARCHPQ        : APARCH(p,q)  — package, scratch = NotImplemented

Choix d'estimation
------------------
Dans .fit(r, use_package=bool):
- use_package=True  -> utilise `arch.univariate`
- use_package=False -> implémentation "scratch" (si dispo), sinon NotImplementedError.

Sélection d'ordres
------------------
- select_order(model_cls, r, p_grid, q_grid, criterion="aic"/"bic", use_package=True)

Viz (plots only)
----------------
- VolViz.overlay_conditional_vs_realized(...)
- VolViz.compare_models(...)

Hypothèses & limites
--------------------
- Innovations normales, μ constant (estimé).
- Stationnarité pénalisée (scratch) si somme alpha + beta >= 1.
- EGARCH/APARCH scratch non fournis (pour rester concis/robuste).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Type
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# utils génériques 
def _series_to_numpy(r: pd.Series) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    s = pd.Series(r).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        try: s.index = pd.to_datetime(s.index)
        except Exception: pass
    s = s.sort_index()
    return s.values.astype(float, copy=False), s.index

def annualize_vol(sigma: np.ndarray, ann_factor: int = 252) -> np.ndarray:
    return np.sqrt(ann_factor) * np.asarray(sigma, dtype=float)

def realized_vol_rolling(r: pd.Series, window: int = 21) -> pd.Series:
    x, idx = _series_to_numpy(r)
    return pd.Series(x, index=idx).rolling(window).std(ddof=1)





@dataclass
class FitResult:
    """Résultat d'ajustementd d'un modèle de volatilité."""
    params: Dict[str, float] # dictionnaire des paramètres estimés
    llf: float  # log-vraisemblance finale
    aic: float # critère AIC
    bic: float # critère BIC
    converged: bool # indicateur de convergence
    message: str # message du fit (info / erreur)


#  base class
class VolModel:
    """Interface commune des modèles de volatilité."""

    def __init__(self, p: int = 1, q: int = 1):
        self.p, self.q = int(p), int(q)
        self.params_: Dict[str, float]  = None
        self.mu_: float = None
        self.result_: FitResult = None
        self._scale_ = 1.0
        self._arch_res_ = None      # stocke le résultat arch (si use_package=True)
        self._fit_index_ = None 

    # API publique
    def fit(self, r: pd.Series, use_package: bool = True, mu: Optional[float] = None,rescale: str = "auto") -> "VolModel":
        """Estime le modèle sur la série de rendements r."""

        # autoscale si demandé
        x, _ = _series_to_numpy(r)
        x_scaled, sf = _autoscale(x, mode=rescale)
        self._scale_ = sf

        if use_package and self._has_arch():
            return self._fit_package_on_array(x_scaled, mu=mu)
        return self._fit_scratch_on_array(x_scaled, mu=mu)

    def conditional_sigma(self, r: pd.Series) -> pd.Series:
        """Trajectoire de sigma_t conditionnelle après fit."""
        raise NotImplementedError

    def _fit_package_on_array(self, x_scaled: np.ndarray, mu: Optional[float]) -> "VolModel":
        raise NotImplementedError

    def _fit_scratch_on_array(self, x_scaled: np.ndarray, mu: Optional[float]) -> "VolModel":
        raise NotImplementedError


    def _has_arch(self) -> bool:
        """Vérifie si le package `arch` est disponible."""
        try:
            import arch 
            return True
        except Exception:
            return False

    @staticmethod
    def _info_criteria(llf: float, nobs: int, k_params: int) -> Tuple[float, float]:
        """Calcule AIC et BIC."""
        aic = -2.0 * llf + 2.0 * k_params
        bic = -2.0 * llf + k_params * math.log(max(nobs, 1))
        return aic, bic





class GARCHPQ(VolModel):
    """
    GARCH(p,q):
    """

    # Estimation package 
    def _fit_package_on_array(self, x: np.ndarray, mu: Optional[float]) -> "GARCHPQ": #renvoit un objet GARCHPQ
        """Estimation via arch.univariate package."""

        # importe arch locallement
        from arch.univariate import ConstantMean, GARCH, Normal

        # setup arch model
        am = ConstantMean(x,rescale=False)
        am.volatility = GARCH(p=self.p, q=self.q)
        am.distribution = Normal()

        # fixe mu si demandé
        if mu is not None: 
            am.mean.fix(mu * self._scale_)

        # fit
        res = am.fit(disp="off")

        # récupère les paramètres
        mu_hat_scaled = float(res.params.get("mu", np.mean(x)))
        mu_hat = mu_hat_scaled / self._scale_
        pars = {"omega": float(res.params["omega"])}

        # ajoute alphas et betas
        pars.update({f"alpha{i+1}": float(res.params[f"alpha[{i+1}]"]) for i in range(self.p)})
        pars.update({f"beta{j+1}":  float(res.params[f"beta[{j+1}]"])  for j in range(self.q)})

        # stocke les résultats
        self.mu_, self.params_ = mu_hat, pars

        # log-likelihood et critères
        llf = float(res.loglikelihood)
        k = 1 + 1 + self.p + self.q  # mu + omega + alphas + betas

        # critères d'information
        aic, bic = self._info_criteria(llf, len(x), k)

        # stocke le résultat de fit
        self.result_ = FitResult(pars, llf, aic, bic, True, "arch fitted")
        self._arch_res_ = res

        return self

    # Estimation from scratch 
    def _fit_scratch_on_array(self, x: np.ndarray, mu: Optional[float]) -> "GARCHPQ": #renvoit un objet GARCHPQ
        """Estimation from scratch via MLE + optim SciPy."""

        # valeur initiale de mu
        mu0 = float(np.mean(x)) if mu is None else float(mu * self._scale_)

        # paramètres initiaux
        theta0 = self._start_params(x, mu0)

        #Fonction pour décompacter les paramètres
        def unpack(theta: np.ndarray) -> Tuple[float, Dict[str, float]]:

            # décompaction selon si mu est fixé ou non
            if mu is None: 
                mu_hat, rest = float(theta[0]), theta[1:]
            else:          
                mu_hat, rest = mu0, theta

            # omega positif
            omega = abs(rest[0]) + 1e-12

            # alphas et betas dans [0,1]
            alphas = np.clip(rest[1:1+self.p], 0.0, 1.0)
            betas  = np.clip(rest[1+self.p:], 0.0, 1.0)

            # construit le dictionnaire des paramètres
            pars = {"omega": float(omega)}

            # ajoute alphas et betas
            for i, a in enumerate(alphas): pars[f"alpha{i+1}"] = float(a)
            for j, b in enumerate(betas):  pars[f"beta{j+1}"]  = float(b)

            return mu_hat, pars

        # Fonction de log-vraisemblance négative
        def nll(theta: np.ndarray) -> float:

            # décompaction des paramètres
            mu_hat, pars = unpack(theta)

            # résidus
            eps = x - mu_hat

            # trajectoire de la variance conditionnelle
            var = self._var_path(eps, pars)
            var = np.maximum(var, 1e-12) # évite log(0) ou division par 0

            pen = 0.0

            # pénalisation si non-stationnarité (somme α + β >= 1)
            if self._sum_ab(pars) >= 0.999: 
                pen += 1e6 * (self._sum_ab(pars) - 0.999 + 1e-9)

            return 0.5 * np.sum(np.log(2*np.pi) + np.log(var) + (eps*eps)/var) + pen

        # optimisation via SciPy
        try:
            from scipy.optimize import minimize

            # minimise la log-vraisemblance négative
            res = minimize(nll, np.array(theta0), method="L-BFGS-B")

            # récupère les résultats
            theta_hat, success, nit, msg = res.x, bool(res.success), int(res.nit), str(res.message)

        except Exception as e:
            # en cas d'erreur, retourne les initiaux et un message d'erreur
            theta_hat, success, nit, msg = np.array(theta0), False, 0, f"SciPy failed: {e}"

        # décompaction finale des paramètres
        mu_hat_scaled, pars = unpack(theta_hat)
        self.mu_ = mu_hat_scaled / self._scale_  #  retour unités d’origine
        self.params_ = pars
        llf = -float(nll(theta_hat))
        k = 1 + 1 + self.p + self.q

        # critères d'information
        aic, bic = self._info_criteria(llf, len(x), k)

        # stocke le résultat de fit
        self.result_ = FitResult(pars, llf, aic, bic, success, msg)
        self._arch_res_ = res

        return self


    def conditional_sigma(self, r: pd.Series) -> pd.Series:
        """Trajectoire de sigma_t conditionnelle après fit."""

        if self.params_ is None or self.mu_ is None:
            raise RuntimeError("fit() d'abord.")
        
        # converti la série en numpy
        x, idx = _series_to_numpy(r)

        # résidus
        eps_scaled = (x * self._scale_) - (self.mu_ * self._scale_)

        # trajectoire de la variance conditionnelle
        var = self._var_path(eps_scaled, self.params_)  # params appris sur l’échelle rescalée

        sigma_scaled = np.sqrt(np.maximum(var, 0.0))
        sigma_orig = sigma_scaled / self._scale_       # <-- retour unités d’origine

        return pd.Series(sigma_orig, index=idx)

    # Méthodes internes au modèle
    def _start_params(self, x: np.ndarray, mu0: float) -> List[float]:
        """Paramètres initiaux pour l'optimisation."""

        v = float(np.var(x - mu0, ddof=1)) # variance empirique des résidus
        omega = 0.01 * v + 1e-8 # petite valeur pour omega
        alphas = [0.05 / max(self.p, 1)] * self.p # init alphas
        betas  = [0.90 / max(self.q, 1)] * self.q # init betas

        return ([mu0] + [omega] + alphas + betas) if True else [omega] + alphas + betas # si mu fixé

    def _sum_ab(self, pars: Dict[str, float]) -> float:
        """Somme des alpha_i et beta_i pour la stationnarité."""

        # somme des alphas et betas
        a = sum(pars.get(f"alpha{i+1}", 0.0) for i in range(self.p))
        b = sum(pars.get(f"beta{j+1}",  0.0) for j in range(self.q))

        return a + b


    def _var_path(self, eps: np.ndarray, pars: Dict[str, float]) -> np.ndarray:
        """Calcule la trajectoire de la variance conditionnelle sigma_t^2."""

        # longueur de la série
        T = len(eps)
        
        # paramètre omega
        omega = pars["omega"]

        # coefficients alpha et beta
        A = np.array([pars.get(f"alpha{i+1}", 0.0) for i in range(self.p)])
        B = np.array([pars.get(f"beta{j+1}",  0.0) for j in range(self.q)])

        # initialise la variance conditionnelle
        var = np.empty(T, float)

        # valeur initiale (variance empirique ou long-run)
        v0 = max(float(np.var(eps, ddof=1)), omega / max(1e-8, 1.0 - (A.sum()+B.sum())))
        var[:max(self.q,1)] = v0

        # itère pour le calcule de var[t]
        for t in range(max(1, max(self.p, self.q)), T):
            s = omega
            # ajoute les termes α_i ε_{t-i}^2
            for i in range(self.p): 
                s += A[i] * (eps[t-1-i]**2)
            # ajoute les termes β_j σ_{t-j}^2
            for j in range(self.q): 
                s += B[j] * var[t-1-j]
            # stocke la variance conditionnelle
            var[t] = s

        # gère les premiers instants t < max(p,q)
        for t in range(1, max(self.p, self.q)):
            var[t] = max(v0, omega
                         + (A[:min(self.p,t)] * (eps[t-1::-1][:min(self.p,t)]**2)).sum()
                         + (B[:min(self.q,t)] *  var[t-1::-1][:min(self.q,t)]).sum())
        return var



# GJR-GARCH(p,q) 
class GJRGARCHPQ(VolModel):
    """
    GJR-GARCH(p,q):
    (ici 1 terme de leverage sur t-1 pour rester compact)
    """

    def _fit_package_on_array(self, x: np.ndarray, mu: Optional[float]) -> "GJRGARCHPQ":
        """Estimation via arch.univariate package."""

        # importe arch locallement
        from arch.univariate import ConstantMean, GARCH, Normal

        # setup arch model
        am = ConstantMean(x,rescale=False)
        am.volatility = GARCH(p=self.p, o=1, q=self.q) # terme de leverage o=1 GJR
        am.distribution = Normal()

        # fixe mu si demandé
        if mu is not None: 
            am.mean.fix(mu * self._scale_)

        # fit
        res = am.fit(disp="off")

        # récupère les paramètres
        mu_hat_scaled = float(res.params.get("mu", np.mean(x)))
        mu_hat = mu_hat_scaled / self._scale_

        pars = {"omega": float(res.params["omega"]), "delta": float(res.params.get("gamma[1]", 0.0))}

        # ajoute alphas et betas
        pars.update({f"alpha{i+1}": float(res.params[f"alpha[{i+1}]"]) for i in range(self.p)})
        pars.update({f"beta{j+1}":  float(res.params[f"beta[{j+1}]"])  for j in range(self.q)})

        # stocke les résultats
        self.mu_, self.params_ = mu_hat, pars

        # log-likelihood et critères
        llf = float(res.loglikelihood)
        k = 1 + 1 + self.p + self.q + 1  # + delta
        aic, bic = self._info_criteria(llf, len(x), k)

        # stocke le résultat de fit
        self.result_ = FitResult(pars, llf, aic, bic, True, "arch fitted")
        self._arch_res_ = res

        return self

    def _fit_scratch_on_array(self,x: np.ndarray, mu: Optional[float]) -> "GJRGARCHPQ":
        """Estimation from scratch via MLE + optim SciPy."""

        # valeur initiale de mu
        mu0 = float(np.mean(x)) if mu is None else float(mu * self._scale_)

        # paramètres initiaux
        theta0 = self._start_params(x, mu0)

        #Fonction pour décompacter les paramètres
        def unpack(theta: np.ndarray) -> Tuple[float, Dict[str, float]]:
            if mu is None: 
                mu_hat, rest = float(theta[0]), theta[1:]
            else:          
                mu_hat, rest = mu0, theta
            
            # omega positif
            omega = abs(rest[0]) + 1e-12 

            # alphas et betas dans [0,1]
            alphas = np.clip(rest[1:1+self.p], 0.0, 1.0)
            betas  = np.clip(rest[1+self.p:1+self.p+self.q], 0.0, 1.0)

            # leverage delta >= 0
            delta  = max(0.0, rest[-1])

            pars = {"omega": float(omega), "delta": float(delta)}

            # ajoute alphas et betas
            for i, a in enumerate(alphas): pars[f"alpha{i+1}"] = float(a)
            for j, b in enumerate(betas):  pars[f"beta{j+1}"]  = float(b)


            return mu_hat, pars

        # Fonction de log-vraisemblance négative
        def nll(theta: np.ndarray) -> float:

            # décompaction des paramètres
            mu_hat, pars = unpack(theta)

            # résidus
            eps = x - mu_hat

            # trajectoire de la variance conditionnelle
            var = self._var_path(eps, pars)
            var = np.maximum(var, 1e-12)

            # pénalisation si non-stationnarité (somme paramètres >= 1)
            pen = 0.0
            if (self._sum_ab(pars) + 0.5*pars["delta"]) >= 0.999:
                pen += 1e6 * ((self._sum_ab(pars) + 0.5*pars["delta"]) - 0.999 + 1e-9)

            return 0.5 * np.sum(np.log(2*np.pi) + np.log(var) + (eps*eps)/var) + pen
        
        # optimisation via SciPy
        try:
            from scipy.optimize import minimize
            res = minimize(nll, np.array(theta0), method="L-BFGS-B")
            theta_hat, success, nit, msg = res.x, bool(res.success), int(res.nit), str(res.message)
        except Exception as e:
            theta_hat, success, nit, msg = np.array(theta0), False, 0, f"SciPy failed: {e}"

        # décompaction finale des paramètres
        mu_hat_scaled, pars = unpack(theta_hat)
        self.mu_ = mu_hat_scaled / self._scale_
        self.params_ = pars

        # log-likelihood et critères
        llf = -float(nll(theta_hat))
        k = 1 + 1 + self.p + self.q + 1
        aic, bic = self._info_criteria(llf, len(x), k)

        # stocke le résultat de fit
        self.result_ = FitResult(pars, llf, aic, bic, success, msg)
        self._arch_res_ = res
        return self

    def conditional_sigma(self, r: pd.Series) -> pd.Series:
        """Trajectoire de sigma_t conditionnelle après fit."""

        # vérifie que le modèle a été ajusté
        if self.params_ is None or self.mu_ is None:
            raise RuntimeError("fit() d'abord.")
        
        # converti la série en numpy
        x, idx = _series_to_numpy(r)

        # résidus
        eps_scaled = (x * self._scale_) - (self.mu_ * self._scale_)

        # trajectoire de la variance conditionnelle
        var = self._var_path(eps_scaled, self.params_)
        sigma_scaled = np.sqrt(np.maximum(var, 0.0))
        sigma_orig = sigma_scaled / self._scale_

        return pd.Series(sigma_orig, index=idx)

    def _start_params(self, x: np.ndarray, mu0: float) -> List[float]:
        """Paramètres initiaux pour l'optimisation."""

        # variance empirique des résidus
        v = float(np.var(x - mu0, ddof=1))

        # petite valeur pour omega
        omega = 0.01 * v + 1e-8

        # initialisation des alphas, betas et delta
        alphas = [0.05 / max(self.p, 1)] * self.p
        betas  = [0.90 / max(self.q, 1)] * self.q

        # initialisation de delta
        delta  = 0.05

        return [mu0, omega, *alphas, *betas, delta]

    def _sum_ab(self, pars: Dict[str, float]) -> float:
        """Somme des alpha_i et beta_i pour la stationnarité."""

        a = sum(pars.get(f"alpha{i+1}", 0.0) for i in range(self.p))
        b = sum(pars.get(f"beta{j+1}",  0.0) for j in range(self.q))

        return a + b


    def _var_path(self, eps: np.ndarray, pars: Dict[str, float]) -> np.ndarray:
        """Calcule la trajectoire de la variance conditionnelle sigme_t^2."""

        # longueur de la série
        T = len(eps)

        # paramètres
        omega = pars["omega"]; delta = pars["delta"]

        # coefficients alpha et beta
        A = np.array([pars.get(f"alpha{i+1}", 0.0) for i in range(self.p)])
        B = np.array([pars.get(f"beta{j+1}",  0.0) for j in range(self.q)])

        # initialise la variance conditionnelle
        var = np.empty(T, float)
        v0 = max(float(np.var(eps, ddof=1)), omega / max(1e-8, 1.0 - (A.sum()+B.sum()+0.5*delta)))

        # valeur initiale
        var[:max(self.q,1)] = v0

        # itère pour le calcule de var[t]
        for t in range(max(1, max(self.p, self.q)), T):
            s = omega
            for i in range(self.p):
                s += A[i] * (eps[t-1-i]**2)
            for j in range(self.q):
                s += B[j] * var[t-1-j]

            # leverage sur t-1
            s += delta * (eps[t-1]**2) * (1.0 if eps[t-1] < 0.0 else 0.0)
            var[t] = s

        # gère les premiers instants t < max(p,q)
        for t in range(1, max(self.p, self.q)):
            base = omega + (A[:min(self.p,t)]*(eps[t-1::-1][:min(self.p,t)]**2)).sum() \
                         + (B[:min(self.q,t)]* var[t-1::-1][:min(self.q,t)]).sum()
            base += delta * (eps[max(t-1,0)]**2) * (1.0 if eps[max(t-1,0)] < 0.0 else 0.0)
            var[t] = max(v0, base)

        return var


# EGARCH(p,q)
class EGARCHPQ(VolModel):
    """
    EGARCH(p,q) - package `arch` uniquement dans cette version concise.
    """

    def _fit_package_on_array(self, x: np.ndarray, mu: Optional[float]) -> "EGARCHPQ":
        """
        Estimation EGARCH via le package arch.
        - Si mu est fourni: on décale la série et on utilise ZeroMean (pas de .mean.fix ici).
        - Sinon: ConstantMean estime la moyenne.
        """
        from arch.univariate import ConstantMean, ZeroMean, EGARCH, Normal

        # rescale déjà géré en amont via self._scale_ ; ici on ne re-rescale pas
        if mu is not None and np.isfinite(mu):
            y = x - mu * float(self._scale_)  # cohérent avec ton autoscale
            am = ZeroMean(y, rescale=False)
            self.mu_ = float(mu)              # on mémorise la moyenne imposée (dé-scalée)
        else:
            am = ConstantMean(x, rescale=False)
            self.mu_ = None                   # mu estimée par le modèle

        am.volatility = EGARCH(p=self.p, o=0, q=self.q)
        am.distribution = Normal()

        res = am.fit(disp="off")

        # paramètres (on ignore 'mu' si ZeroMean)
        pars = {k: float(v) for k, v in res.params.items() if k != "mu"}
        self.params_ = pars

        # si mu n'était pas fixé, on le récupère et on le dé-scale
        if self.mu_ is None:
            self.mu_ = float(res.params.get("mu", np.mean(x))) / float(self._scale_)

        llf = float(res.loglikelihood)
        k   = len(res.params)
        aic, bic = self._info_criteria(llf, len(x), k)
        # arch n’expose pas toujours .iterations -> on n’enregistre pas
        self.result_ = FitResult(pars, llf, aic, bic, True, "arch fitted")

        # on garde le résultat arch pour réutiliser la trajectoire si même longueur
        self._arch_res_ = res
        return self


    def conditional_sigma(self, r: pd.Series) -> pd.Series:
        """
        Renvoie sigma_t (EGARCH) via arch.
        - Si on a un fit de même longueur -> on réutilise la trajectoire.
        - Sinon, on refit silencieusement la même spec (sans .mean.fix).
        """
        from arch.univariate import ConstantMean, ZeroMean, EGARCH, Normal

        x, idx = _series_to_numpy(r)
        if x.size == 0:
            return pd.Series(dtype=float, index=idx)

        # réutilisation si taille identique
        if (getattr(self, "_arch_res_", None) is not None
            and hasattr(self._arch_res_, "conditional_volatility")
            and len(self._arch_res_.conditional_volatility) == len(x)):
            sig_scaled = np.asarray(self._arch_res_.conditional_volatility, float)
            return pd.Series(sig_scaled / float(self._scale_), index=idx)

        # refit silencieux même spec (respecte mu_ si fixé)
        x_scaled = x * float(self._scale_)
        if self.mu_ is not None and np.isfinite(self.mu_):
            y = x_scaled - self.mu_ * float(self._scale_)
            am = ZeroMean(y, rescale=False)
        else:
            am = ConstantMean(x_scaled, rescale=False)

        am.volatility = EGARCH(p=self.p, o=0, q=self.q)
        am.distribution = Normal()
        res = am.fit(disp="off")
        self._arch_res_ = res

        sig_scaled = np.asarray(res.conditional_volatility, float)
        return pd.Series(sig_scaled / float(self._scale_), index=idx)





class APARCHPQ(VolModel):
    """
    APARCH(p,q) - package `arch` uniquement (scratch trop long).
    """

    def _fit_package_on_array(self, x: np.ndarray, *, mu: Optional[float]) -> "APARCHPQ":
        """
        Estimation APARCH via arch.
        - Si mu est fourni: décentrer et ZeroMean.
        - Sinon: ConstantMean.
        """
        from arch.univariate import ConstantMean, ZeroMean, APARCH, Normal

        if mu is not None and np.isfinite(mu):
            y = x - mu * float(self._scale_)
            am = ZeroMean(y, rescale=False)
            self.mu_ = float(mu)
        else:
            am = ConstantMean(x, rescale=False)
            self.mu_ = None

        am.volatility = APARCH(p=self.p, o=1, q=self.q)
        am.distribution = Normal()
        res = am.fit(disp="off")

        self._arch_res_ = res
        pars = {k: float(v) for k, v in res.params.items() if k != "mu"}
        self.params_ = pars

        if self.mu_ is None:
            self.mu_ = float(res.params.get("mu", np.mean(x))) / float(self._scale_)

        llf = float(res.loglikelihood)
        k   = len(res.params)
        aic, bic = self._info_criteria(llf, len(x), k)
        self.result_ = FitResult(pars, llf, aic, bic, True, "arch fitted")
        return self


    def conditional_sigma(self, r: pd.Series) -> pd.Series:
        """
        Renvoie sigma_t (APARCH) via arch.
        Recycle le fit si possible, sinon refit identique (ZeroMean si mu_ fixé).
        """
        from arch.univariate import ConstantMean, ZeroMean, APARCH, Normal

        x, idx = _series_to_numpy(r)
        if x.size == 0:
            return pd.Series(dtype=float, index=idx)

        if (getattr(self, "_arch_res_", None) is not None
            and hasattr(self._arch_res_, "conditional_volatility")
            and len(self._arch_res_.conditional_volatility) == len(x)):
            sig_scaled = np.asarray(self._arch_res_.conditional_volatility, float)
            return pd.Series(sig_scaled / float(self._scale_), index=idx)

        x_scaled = x * float(self._scale_)
        if self.mu_ is not None and np.isfinite(self.mu_):
            y = x_scaled - self.mu_ * float(self._scale_)
            am = ZeroMean(y, rescale=False)
        else:
            am = ConstantMean(x_scaled, rescale=False)

        am.volatility = APARCH(p=self.p, o=1, q=self.q)
        am.distribution = Normal()
        res = am.fit(disp="off")
        self._arch_res_ = res

        sig_scaled = np.asarray(res.conditional_volatility, float)
        return pd.Series(sig_scaled / float(self._scale_), index=idx)





def select_order(model_cls: Type[VolModel],r: pd.Series,p_grid: List[int] = [1,2,3],q_grid: List[int] = [1,2,3],
                criterion: str = "aic",use_package: bool = True, rescale: str = "auto") -> Tuple[Tuple[int,int], VolModel]:

    """Grid-search compacte pour la selection des ordres ((p,q)) -> ((p*,q*), modèle ajusté)."""

    #initialise la variable du meilleur modèle
    best = (math.inf, None, None)

    # boucle sur la grille
    for p in p_grid:
        for q in q_grid:
            #try:
                m = model_cls(p, q).fit(r, use_package=use_package, rescale=rescale)
                crit = m.result_.aic if criterion.lower()=="aic" else m.result_.bic # crit info
                # met à jour le meilleur modèle si besoin
                if crit < best[0]: 
                    best = (crit, (p,q), m)
    if best[1] is None:
        raise RuntimeError("Aucun modèle n'a convergé sur la grille.")
    return best[1], best[2]



def _autoscale(y: np.ndarray, mode: str = "auto") -> tuple[np.ndarray, float]:
    """
    Rescale commun pour stabiliser l'optimisation et rendre l'AIC comparable.
    """
    if mode == "off":
        return y, 1.0
    if mode == "x100":
        return (100.0 * y, 100.0)
    # auto
    s = float(np.std(y))
    if s == 0.0 or not np.isfinite(s):
        return y, 1.0
    if s < 1e-3:
        return (100.0 * y, 100.0)
    if s > 100.0:
        return (0.01 * y, 0.01)
    return y, 1.0



class VolViz:
    """Classe de visualisation des volatilités conditionnelles et réalisées et comparaison de modèles."""

    @staticmethod
    def _get_ax(ax):
        if ax is not None: 
            return ax, ax.figure
        fig, ax2 = plt.subplots(figsize=(8,4)); return ax2, fig


    def overlay_conditional_vs_realized(self, idx: pd.DatetimeIndex, sigma_t: pd.Series | np.ndarray,
                                        realized_vol: pd.Series, *, ann_factor: int = 252, ax=None,
                                        title: str = "Vol. annualisée : conditionnelle vs réalisée (rolling)"):
        """ Trace sigma_t (conditionnelle) et vol réalisée (rolling) sur le même graphique."""

        ax, fig = self._get_ax(ax)

        # prépare les séries
        s_cond = pd.Series(np.asarray(sigma_t), index=idx).dropna()
        s_real = realized_vol.reindex(s_cond.index).dropna()
        s_cond = s_cond.reindex(s_real.index)

        # trace
        ax.plot(s_cond.index, annualize_vol(s_cond.values, ann_factor), label="σ_t (cond.)")
        ax.plot(s_real.index, annualize_vol(s_real.values, ann_factor), label="Vol réalisée")
        ax.set_title(title)
        ax.set_ylabel("Vol ann."); ax.set_xlabel("Date"); ax.legend()
        fig.tight_layout()
        
        return ax

    def compare_models( self, idx: pd.DatetimeIndex, models_sigma: Dict[str, pd.Series | np.ndarray],
                        ann_factor: int = 252, ax=None, title: str = "Comparaison des volatilités ann. (σ_t)"):
        """Trace les volatilités conditionnelles de plusieurs modèles sur le même graphique."""
        
        # prépare le plot
        ax, fig = self._get_ax(ax)

        # trace chaque modèle
        for name, s in models_sigma.items():
            s = pd.Series(np.asarray(s), index=idx).dropna()
            ax.plot(s.index, annualize_vol(s.values, ann_factor), label=name)

        ax.set_title(title)
        ax.set_ylabel("Vol ann.")
        ax.set_xlabel("Date"); ax.legend(ncol=2)
        fig.tight_layout()
        
        return ax


