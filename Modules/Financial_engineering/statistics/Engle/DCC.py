
from __future__ import annotations
import sys
from typing import Dict, Tuple, Iterable,Optional,Literal
import numpy as np
import pandas as pd
import math
import warnings
from pathlib import Path
from dataclasses import dataclass



# -------- Roots & imports projet --------
ROOT = Path(__file__).resolve().parents[3]  # .../<ROOT>/Examples -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from Modules.Financial_engineering.statistics.multivariate_vol_estimation import MultiVolModel, CovariancePath, _safe_symmetrize, _chol_or_nearest_psd, _as_dt_df



@dataclass(frozen=True)
class DCCStatePath:
    """
    Path complet nécessaire pour une mise à jour DCC online.
    - cov  : CovariancePath Σ_t
    - Q    : ndarray (T,N,N)
    - sig2 : ndarray (T,N)   variances univariées
    - e    : ndarray (T,N)   résidus standardisés (e_t)
    """
    cov: "CovariancePath"
    Q: np.ndarray
    sig2: np.ndarray
    e: np.ndarray




    # Paramètre pour GARCH o
    mu: Optional[np.ndarray] = None          # (N,)
    omega: Optional[np.ndarray] = None       # (N,)
    alpha: Optional[np.ndarray] = None       # (N,)
    beta: Optional[np.ndarray] = None        # (N,)
    eps: Optional[np.ndarray] = None         # (T,N) ou au moins dernier eps_t


#  DCC 
class DCCGaussian(MultiVolModel):
    """
    DCC(1,1) gaussien.
      - DCC(1,1) "from scratch"
            * std univariée via EWMA(λ_std), puis standardisation
            * estimation (a,b) par MLE, contrainte a>=0, b>=0, a+b<1
    """

    def __init__(
        self,
        use_package: bool = True,
        lambda_std: float = 0.94,
        vol_model: str = "garch",
        # ---- nouveaux paramètres refit ----
        refit_enabled: bool = False,
        refit_lookback: int = 252,
        refit_mode: str = "rolling",   # "rolling" ou "expanding"
        refit_every: int | None = None, # ex: 21 (mensuel trading days), 63 (trimestriel)
        refit_dates: pd.DatetimeIndex | None = None, # pour backtest cohérent rebal
        refit_min_obs: int = 60,
        include_refit_date_in_forecast: bool = True,):
    

        self.lambda_std = float(lambda_std) # pour vols univariées EWMA si scratch
        self.vol_model = str(vol_model).lower() 

        # fit storage
        self._names: Tuple[str, ...] = tuple()
        self._fit_: Dict[str, float] | None = None
        self._Qbar_: np.ndarray | None = None
        self._sigma_t_: np.ndarray | None = None  # diag vols univariées (T,N)

        # refit options
        self.refit_enabled = bool(refit_enabled)
        self.refit_lookback = int(refit_lookback)
        self.refit_mode = str(refit_mode).lower()
        self.refit_every = None if refit_every is None else int(refit_every)
        self.refit_dates = refit_dates  # peut être None
        self.refit_min_obs = int(refit_min_obs)
        self.include_refit_date_in_forecast = bool(include_refit_date_in_forecast)
        


    def _default_refit_dates(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if self.refit_every is None or self.refit_every <= 0:
            raise ValueError("refit_every doit être un entier > 0 si refit_dates n'est pas fourni.")
        # exemple: tous les K points d'index (trading days)
        return idx[:: self.refit_every]




    # Scratch (DCC(1,1) MLE) 
    @staticmethod
    def _ewma_vols(X: np.ndarray, lmb: float) -> np.ndarray:
        """Vols univariées EWMA pour standardiser les rendements dans DCC."""

        T, N = X.shape # dimensions
        sig2 = np.var(X, axis=0, ddof=1) # initialisation
        out = np.empty((T, N), float) # stockage

        # boucle
        for t in range(T):
            sig2 = lmb * sig2 + (1.0 - lmb) * (X[t] ** 2) # mise à jour variance EWMA
            out[t] = np.sqrt(np.maximum(sig2, 1e-14)) # stocke écart-type avec plancher numérique
        return out
    
    @staticmethod
    def _garch_vols(X: np.ndarray) -> np.ndarray:
        """
        Vols univariées avec un GARCH(1,1) par série.
        Utilise le package `arch` (ConstantMean + GARCH(1,1) + Normal).

        Retourne un array (T, N) de *sigma_t* (écarts-types conditionnels).
        """
        try:
            from arch.univariate import ConstantMean, GARCH, Normal
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "vol_model='garch' requiert le package 'arch'. "
                "Installe-le via `pip install arch`."
            ) from e

        T, N = X.shape
        out = np.empty((T, N), float)

        for i in range(N):

            raw = X[:, i]

            # centre (optionnel mais propre)
            raw = raw - raw.mean()

            # rescale pour éviter le warning 
            scale = 100.0      # 0.001 -> 0.1 (0.1% devient 10% en échelle)
            y = pd.Series(scale * raw)

            # modèle GARCH sur la série rescalée
            am = ConstantMean(y, rescale=False)   # on gère nous-même l’échelle
            am.volatility = GARCH(1, 0, 1)
            am.distribution = Normal()

            res = am.fit(disp="off")

            # sigma_t en échelle "rescalée", qu'on ramène à l’échelle d’origine
            sigma_scaled = np.asarray(res.conditional_volatility, float)
            sigma_original = sigma_scaled / scale

            out[:, i] = sigma_original

        return out
    

    def _garch_fit_params(self, X: np.ndarray) -> dict:
        """
        Fit un GARCH(1,1) par série et retourne paramètres + états finaux
        dans l'échelle d'origine de X.
        """
        try:
            from arch.univariate import ConstantMean, GARCH, Normal
        except Exception as e:
            raise ImportError(
                "vol_model='garch' requiert le package 'arch' (`pip install arch`)."
            ) from e

        T, N = X.shape

        mu = np.zeros(N, float)
        omega = np.zeros(N, float)
        alpha = np.zeros(N, float)
        beta = np.zeros(N, float)

        # états fin de fenêtre
        h_last = np.zeros(N, float)
        eps_last = np.zeros(N, float)

        # option: garde le path complet de sig2 sur la fenêtre
        sig2_path = np.empty((T, N), float)
        eps_path = np.empty((T, N), float)

        for i in range(N):
            x = X[:, i].astype(float)

            # rescale pour stabilité numérique (comme tu fais)
            scale = 100.0
            y = pd.Series(scale * x)

            am = ConstantMean(y, rescale=False)
            am.volatility = GARCH(1, 0, 1)
            am.distribution = Normal()

            res = am.fit(disp="off")

            # paramètres côté "y"
            p = res.params
            mu_y = float(p.get("mu", 0.0))
            omega_y = float(p["omega"])
            alpha_y = float(p.get("alpha[1]", p.get("alpha1")))
            beta_y = float(p.get("beta[1]", p.get("beta1")))

            # conversion à l'échelle "x"
            # y = scale * x  =>  mu_x = mu_y/scale ; eps_x = eps_y/scale ; h_x = h_y/scale^2 ; omega_x = omega_y/scale^2
            mu_x = mu_y / scale
            omega_x = omega_y / (scale * scale)

            mu[i] = mu_x
            omega[i] = omega_x
            alpha[i] = alpha_y
            beta[i] = beta_y

            # états/pathed depuis res (échelle y)
            sigma_y = np.asarray(res.conditional_volatility, float)      # sqrt(h_y)
            h_y = sigma_y**2
            eps_y = np.asarray(res.resid, float)                         # y - mu_y (selon arch)

            sig2_path[:, i] = h_y / (scale * scale)
            eps_path[:, i] = eps_y / scale

            h_last[i] = sig2_path[-1, i]
            eps_last[i] = eps_path[-1, i]

        return {
            "mu": mu, "omega": omega, "alpha": alpha, "beta": beta,
            "sig2_path": sig2_path,
            "eps_path": eps_path,
            "h_last": h_last,
            "eps_last": eps_last,
        }



    def _compute_univariate_vols(self, X: np.ndarray) -> np.ndarray:
        """
        Helper pour choisir le modèle de vol univarié :
        - 'ewma'  -> EWMA RiskMetrics
        - 'garch' -> GARCH(1,1) univarié (via arch)
        """
        vm = getattr(self, "vol_model", "ewma")
        if vm == "ewma":
            return self._ewma_vols(X, self.lambda_std)
        if vm == "garch":
            return self._garch_vols(X)
        raise ValueError(f"vol_model inconnu: {vm!r} (attendu: 'ewma' ou 'garch')")


    @staticmethod
    def _dcc_recursion(e: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> Iterable[np.ndarray]:
        """
        Génère R_t via Q_t = (1-a-b)Qbar + a e_{t-1}e_{t-1}ᵀ + b Q_{t-1}, R_t = D^{-1} Q D^{-1}.
        e : (T,N) résidus standardisés.
        """
        T, N = e.shape # dimensions
        Q = Qbar.copy() # initialisation

        # boucle sur T
        for t in range(T):
            if t > 0: 
                Q = (1.0 - a - b) * Qbar + a * np.outer(e[t - 1], e[t - 1]) + b * Q # mise à jour Q_t
            d = np.sqrt(np.maximum(np.diag(Q), 1e-20))
            Dinv = np.diag(1.0 / d)
            R = _safe_symmetrize(Dinv @ Q @ Dinv)
            yield R


    def _negloglik_dcc(self, e: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> float:
        """
        Log-vraisemblance négative gaussienne de la *partie corrélation* du DCC :
            ℓ_c(a,b) = 1/2 ∑_t [ log|R_t| + e_t' R_t^{-1} e_t + const ]
        où e_t sont les résidus standardisés (var ≈ 1).
        """

        # Si invalides, grosse pénalité
        if (a < 0) or (b < 0) or (a + b >= 1.0):
            return 1e12

        T, N = e.shape
        ll = 0.0

        # boucle sur les dates avec la recursion DCC pour R_t
        for t, R in enumerate(self._dcc_recursion(e, a, b, Qbar)):
            # On travaille directement avec R_t (covariance des ε)
            try:
                L = _chol_or_nearest_psd(R)  # check SPD
            except np.linalg.LinAlgError:
                return 1e10

            # log|R_t| via Cholesky
            logdet = 2.0 * np.sum(np.log(np.diag(L)))

            # e_t' R_t^{-1} e_t
            quad = e[t] @ np.linalg.solve(R, e[t])

            # contribution à la log-vraisemblance
            ll += 0.5 * (logdet + quad + N * math.log(2 * math.pi))

        return float(ll)



    def _negloglik_dcc_last(self, X: np.ndarray, a: float, b: float, sigma_t: np.ndarray, Qbar: np.ndarray) -> float:
        """Log-vraisemblance négative gaussienne (DCC + vols fixées)."""

        # Si invalides, grosse pénalité
        if (a < 0) or (b < 0) or (a + b >= 1.0):
            return 1e8
        

        e = X / sigma_t  # standardisation élément par élément
        T, N = X.shape # dimensions

        #initialisation log-vraisemblance
        ll = 0.0

        # boucle sur chaque date, on estime la matrice de covariance conditionnelle a chaque date et on teste la vraisemblance
        for t, R in enumerate(self._dcc_recursion(e, a, b, Qbar)):
            D = np.diag(sigma_t[t]) # diag des écarts-types à t
            H = _safe_symmetrize(D @ R @ D) # symétrisation numérique de Σ_t

            # essai de la cholesky pour vérifier PSD 
            try:
                L = _chol_or_nearest_psd(H) #A approfondir
            except np.linalg.LinAlgError:
                return 1e10
            
            # log det via chol
            logdet = 2.0 * np.sum(np.log(np.diag(L)))

            # quadratic form
            quad = X[t] @ np.linalg.solve(H, X[t])

            # contribution à la log-vraisemblance
            ll += 0.5 * (logdet + quad + N * math.log(2 * math.pi))

        return float(ll) # log-vraisemblance négative totale

    def _fit_scratch(self, X: pd.DataFrame) -> "DCCGaussian":
        """Fit DCC(1,1) "from scratch" via MLE."""

        # données
        Xv = X.values

        # vols univariées via EWMA ou GARCH
        sigma_t = self._compute_univariate_vols(Xv)  # (T,N)
        e = Xv / sigma_t # Standardisation élément par élément des rendements

        # Corrélation empirique des residus standardisés
        Qbar = np.corrcoef(e.T) 
        Qbar = _safe_symmetrize(Qbar) # symétrisation numérique
        self._Qbar_ = Qbar
        self._sigma_t_ = sigma_t

        # optimisation des paramètres (a,b)
        try:
            from scipy.optimize import minimize

            # fonction objectif : log-vraisemblance négative
            def obj(theta):
                #return self._negloglik_dcc(Xv, theta[0], theta[1], sigma_t, Qbar)
                return self._negloglik_dcc(e, theta[0], theta[1], Qbar)
            
            # point de départ, bornes, contraintes
            #x0 = np.array([0.02, 0.97])  # proche RiskMetrics corr (a+b~0.99)
            x0 = np.array([0.05, 0.9])

            bnds = [(1e-6, 0.999), (1e-6, 0.999)]
            cons = ({'type': 'ineq', 'fun': lambda th: 0.999 - (th[0] + th[1])},) # a + b < 1

            # optimisation
            res = minimize(obj, x0, bounds=bnds, method="L-BFGS-B") 
            #res = minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP") 

            #Si échec, warning et fallback grille
            if not res.success:
                warnings.warn(f"DCC scratch: optimisation non-convergée ({res.message}). Essai grille.")
                raise RuntimeError("opt failed")
            
            # Récupère a,b optimaux
            a, b = float(res.x[0]), float(res.x[1])

        # Fallback grille si échec
        except Exception:
            # fallback petite grille
            grid = np.linspace(0.01, 0.2, 10)
            best = (1e12, 0.05, 0.9)

            # recherche grille
            for a in grid:
                for b in grid:
                    if a + b >= 0.999:
                        continue
                    #val = self._negloglik_dcc(Xv, a, b, sigma_t, Qbar) # évalue avec la log-vraisemblance negative
                    val = self._negloglik_dcc(e, a, b, Qbar)

                    # mise à jour du best
                    if val < best[0]:
                        best = (val, a, b)
            
            # Paramètres optimaux
            a, b = best[1], best[2]

        # stocke le fit
        self._fit_ = {"a": a, "b": b}
        self._names = tuple(X.columns)

        return self

    #  API 
    def fit(self, R: pd.DataFrame, **_) -> "DCCGaussian":
        """Fit DCC(1,1) sur les rendements R  via Scratch."""

        # données propres
        X = _as_dt_df(R).dropna()

        # vérifie au moins 2 séries
        if X.shape[1] < 2:
            raise ValueError("DCC requiert au moins 2 séries.")
        
        self._names = tuple(X.columns)

        # si mode batch
        if not getattr(self, "refit_enabled", False):
            return self._fit_scratch(X)

        # mode refit
        lookback = int(getattr(self, "refit_lookback", 252))
        min_obs = int(getattr(self, "refit_min_obs", 60))

        # seuil pour fit afin de pas faire le premier refit dans le model evaluator
        lazy_threshold = max(2 * lookback, min_obs)

        if len(X) >= lazy_threshold:
            self._fit_ = None
            self._Qbar_ = None
            self._sigma_t_ = None
            self._lazy_fit = True
            return self

        self._lazy_fit = False
        return self._fit_scratch(X)



    def _conditional_cov_batch(self, R: pd.DataFrame) -> CovariancePath:
        """Renvoie la trajectoire Σ_t alignée sur l'index de R."""

        X = _as_dt_df(R).dropna()

        # Scratch path
        if self._fit_ is None or self._Qbar_ is None or self._sigma_t_ is None:
            # pas de fit scratch ? on enchaîne
            self._fit_scratch(X)

        # paramètres
        a, b = self._fit_["a"], self._fit_["b"]

        # vols univariées (si dimensions ne matchent pas)
        sigma_t = (
            self._sigma_t_
            if self._sigma_t_.shape[0] == len(X)
            else self._compute_univariate_vols(X.values)
        )

        
        # Standardisation des rendements
        e = X.values / sigma_t

        mats = []
        Q = self._Qbar_.copy()

        # boucle DCC sur chaque date
        for t in range(len(X)):

            # mise à jour Q_t
            if t > 0:
                Q = (1.0 - a - b) * self._Qbar_ + a * np.outer(e[t - 1], e[t - 1]) + b * Q
            
            # diagonalisation pour R_t
            d = np.sqrt(np.maximum(np.diag(Q), 1e-20))

            #inversion diagonale
            Dinv = np.diag(1.0 / d) # D^{-1}

            # corrélation conditionnelle symétrisée
            Rcorr = _safe_symmetrize(Dinv @ Q @ Dinv)

            # diagonal des vols univariées à t
            D = np.diag(sigma_t[t])

            # covariance conditionnelle symetrisée
            H = _safe_symmetrize(D @ Rcorr @ D)

            # stocke
            mats.append(H)

        H = np.stack(mats, axis=0) # (T,N,N)

        return CovariancePath(H=H, index=X.index, names=tuple(X.columns)) # renvoie un objet CovariancePath
    

    def conditional_cov(self, R: pd.DataFrame) -> CovariancePath:
        """
        Routeur :
        - refit_enabled=False -> batch (comportement historique)
        - refit_enabled=True  -> refit périodique (Option B)
        """
        if not getattr(self, "refit_enabled", False):
            return self._conditional_cov_batch(R)

        Xdf = _as_dt_df(R).sort_index()

        # dates de refit : soit fournies, soit construites par refit_every
        if self.refit_dates is not None:
            refit_dates = pd.DatetimeIndex(self.refit_dates)
        else:
            refit_dates = self._default_refit_dates(Xdf.index)

        return self.conditional_cov_refit(
            R=Xdf,
            refit_dates=refit_dates,
            lookback=int(self.refit_lookback),
            window_mode="expanding" if str(self.refit_mode).lower() == "expanding" else "rolling",
            min_obs=int(self.refit_min_obs),
            include_refit_date_in_forecast=bool(self.include_refit_date_in_forecast),
        )



    def init_state_path_from_window(self, R_window: pd.DataFrame) -> DCCStatePath:
        Xdf = _as_dt_df(R_window)
        X = Xdf.values
        idx = Xdf.index
        names = tuple(Xdf.columns)

        # Fit DCC (a,b,Qbar) sur la fenêtre
        if self._fit_ is None or self._Qbar_ is None:
            self.fit(Xdf)

        a, b = float(self._fit_["a"]), float(self._fit_["b"])
        Qbar = self._Qbar_

        vm = str(self.vol_model).lower()

        if vm == "ewma":
            sigma = self._ewma_vols(X, self.lambda_std)
            sig2 = np.maximum(sigma**2, 1e-14)
            eps = X  # mu=0 implicite
        elif vm == "garch":
            g = self._garch_fit_params(X)
            sig2 = np.maximum(g["sig2_path"], 1e-14)
            eps = g["eps_path"]                     # eps_t = x_t - mu_t (mu constant ici)
        else:
            raise ValueError(f"vol_model inconnu: {vm!r}")

        e = eps / np.sqrt(sig2)

        # recursion Q_t sur la fenêtre
        T, N = X.shape
        Q = Qbar.copy()
        H = np.empty((T, N, N), float)
        Q_path = np.empty((T, N, N), float)  # garde si tu veux l'état complet

        for t in range(T):
            if t > 0:
                Q = (1.0 - a - b) * Qbar + a * np.outer(e[t-1], e[t-1]) + b * Q
                Q = _safe_symmetrize(Q)

            Q_path[t] = Q

            d = np.sqrt(np.maximum(np.diag(Q), 1e-20))
            invd = 1.0 / d

            Rcorr = _safe_symmetrize((Q * invd[None, :]) * invd[:, None])

            sigma_t = np.sqrt(np.maximum(sig2[t], 1e-14))
            H[t] = _safe_symmetrize((Rcorr * sigma_t[None, :]) * sigma_t[:, None])


        cov_path = CovariancePath(H=H, index=idx, names=names)

        if vm == "garch":
            return DCCStatePath(
                cov=cov_path, Q=Q_path, sig2=sig2, e=e,
                mu=g["mu"], omega=g["omega"], alpha=g["alpha"], beta=g["beta"],
                eps=eps,
            )

        return DCCStatePath(cov=cov_path, Q=Q_path, sig2=sig2, e=e)

    

    def update_next_from_state_path(self, prev: DCCStatePath, r_next: pd.Series, dt_next: pd.Timestamp):
        if self._fit_ is None or self._Qbar_ is None:
            raise RuntimeError("DCCGaussian doit être fit avant update_next_from_state_path.")

        a, b = float(self._fit_["a"]), float(self._fit_["b"])
        Qbar = self._Qbar_
        vm = str(self.vol_model).lower()

        names = list(prev.cov.names)
        x = pd.to_numeric(r_next.reindex(names), errors="coerce").fillna(0.0).values.astype(float)

        Q_prev = prev.Q[-1]
        e_prev = prev.e[-1]
        sig2_prev = prev.sig2[-1]

        if vm == "ewma":
            lmb = float(self.lambda_std)
            sig2_new = lmb * sig2_prev + (1.0 - lmb) * (x * x)
            sig2_new = np.maximum(sig2_new, 1e-14)
            eps_new = x  # mu=0
        elif vm == "garch":
            if prev.mu is None or prev.omega is None or prev.alpha is None or prev.beta is None or prev.eps is None:
                raise RuntimeError("State GARCH incomplet. Utilise init_state_path_from_window avec vol_model='garch'.")

            mu = prev.mu
            omega = prev.omega
            alpha = prev.alpha
            beta = prev.beta

            eps_prev = prev.eps[-1]              # dernier résidu (x_{t-1}-mu)
            # update variance GARCH
            sig2_new = omega + alpha * (eps_prev * eps_prev) + beta * sig2_prev
            sig2_new = np.maximum(sig2_new, 1e-14)

            eps_new = x - mu
        else:
            raise ValueError(f"vol_model inconnu: {vm!r}")

        sigma_new = np.sqrt(sig2_new)

        # update Q_t via e_{t-1}
        Q_new = (1.0 - a - b) * Qbar + a * np.outer(e_prev, e_prev) + b * Q_prev
        Q_new = _safe_symmetrize(Q_new)

        # corr -> Σ
        d = np.sqrt(np.maximum(np.diag(Q_new), 1e-20))
        Dinv = np.diag(1.0 / d)
        Rcorr = _safe_symmetrize(Dinv @ Q_new @ Dinv)
        D = np.diag(sigma_new)
        H_new = _safe_symmetrize(D @ Rcorr @ D)

        # e_t pour prochaine itération
        e_new = eps_new / sigma_new

        # append
        cov_prev = prev.cov
        H_all = np.concatenate([cov_prev.H, H_new[None, :, :]], axis=0)
        idx_all = cov_prev.index.append(pd.DatetimeIndex([pd.Timestamp(dt_next)]))
        cov_all = CovariancePath(H=H_all, index=idx_all, names=cov_prev.names)

        Q_all = np.concatenate([prev.Q, Q_new[None, :, :]], axis=0)
        sig2_all = np.concatenate([prev.sig2, sig2_new[None, :]], axis=0)
        e_all = np.concatenate([prev.e, e_new[None, :]], axis=0)

        if vm == "garch":
            eps_all = np.concatenate([prev.eps, eps_new[None, :]], axis=0)
            out = DCCStatePath(
                cov=cov_all, Q=Q_all, sig2=sig2_all, e=e_all,
                mu=prev.mu, omega=prev.omega, alpha=prev.alpha, beta=prev.beta,
                eps=eps_all,
            )
            diag = {"a": a, "b": b, "vol_model": "garch"}
            return out, diag

        out = DCCStatePath(cov=cov_all, Q=Q_all, sig2=sig2_all, e=e_all)
        diag = {"a": a, "b": b, "vol_model": "ewma", "lambda_std": float(self.lambda_std)}
        return out, diag

    

    def conditional_cov_refit(self,
        R: pd.DataFrame,
        refit_dates: pd.DatetimeIndex,
        lookback: int = 252,
        window_mode: Literal["rolling", "expanding"] = "rolling",
        min_obs: int = 60,
        include_refit_date_in_forecast: bool = True, ) -> CovariancePath:
        """
        Génère un chemin de covariances DCC avec re-fit périodique.
        """


        Xdf = _as_dt_df(R)
        Xdf = Xdf.sort_index()
        if Xdf.shape[1] < 2:
            raise ValueError("DCC requiert au moins 2 séries.")

        full_idx = Xdf.index
        names = tuple(Xdf.columns)

        # dates de refit valides = intersection avec l'index et triées
        refit_dates = pd.DatetimeIndex(sorted(set(refit_dates).intersection(set(full_idx))))
        if len(refit_dates) == 0:
            raise ValueError("Aucune refit_date n'appartient à l'index de R.")

        # helper pour découper la fenêtre de fit
        first_date = full_idx[0]

        def _get_fit_window_end_at(t: pd.Timestamp) -> pd.DataFrame:
            if window_mode == "expanding":
                w = Xdf.loc[first_date:t]
            else:
                # rolling
                pos = full_idx.get_loc(t)
                start_pos = max(0, pos - lookback + 1)
                w = Xdf.iloc[start_pos:pos + 1]
            # on exige un minimum d'observations non-NaN par colonne : ici on drop les lignes toutes-NaN puis dropna global
            w = w.dropna(how="all")
            return w

        # stockage résultats
        H_out = np.full((len(full_idx), len(names), len(names)), np.nan, dtype=float)

        # boucle sur segments entre refit_dates
        for k, t_refit in enumerate(refit_dates):
            t_next = refit_dates[k + 1] if (k + 1) < len(refit_dates) else None

            fit_win = _get_fit_window_end_at(t_refit)
            if len(fit_win) < min_obs:
                # pas assez de data -> on skip (laisse NaN)
                continue

            # (1) Fit + init state sur la fenêtre
            self.fit(fit_win)

            state = self.init_state_path_from_window(fit_win)

            # (2) définir la tranche de production
            if include_refit_date_in_forecast:
                seg_idx = full_idx[(full_idx >= t_refit) & (full_idx < (t_next if t_next is not None else full_idx[-1] + pd.Timedelta(days=1)))]
            else:
                seg_idx = full_idx[(full_idx > t_refit) & (full_idx < (t_next if t_next is not None else full_idx[-1] + pd.Timedelta(days=1)))]

            # si tranche vide, rien à faire
            if len(seg_idx) == 0:
                continue

            # (3) online update sur chaque date de seg_idx
            #    On update avec r_t et on stocke Σ_t (après update)
            for dt in seg_idx:
                r_t = Xdf.loc[dt]
                state, _ = self.update_next_from_state_path(state, r_next=r_t, dt_next=dt)

                # dernier état = Σ_dt
                H_dt = state.cov.H[-1]
                out_pos = full_idx.get_loc(dt)
                H_out[out_pos] = H_dt

        # post-traitement: pour les dates non remplies, tu choisis
        # - laisser NaN (utile pour diagnostiquer)
        # - forward-fill (souvent OK si tu veux toujours une Σ disponible)
        # Ici je laisse NaN; tu peux activer un ffill dans ton moteur si nécessaire.

        return CovariancePath(H=H_out, index=full_idx, names=names)


