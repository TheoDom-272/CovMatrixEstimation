
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Roots & imports projet
ROOT = Path(__file__).resolve().parents[3] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Modules.Financial_engineering.statistics.multivariate_vol_estimation import (_as_dt_df,_safe_symmetrize,)



#  Numba 
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False



# score QMV pour une grille de lambda (utilise Cholesky)
if _HAS_NUMBA:
    @njit(cache=True)
    def _qmv_scores_numba(lambdas: np.ndarray, S0: np.ndarray,S_emp_path: np.ndarray,R_path: np.ndarray,mask_path: np.ndarray, 
                            ridge: float,min_obs: int,) -> np.ndarray:
        """
        Pour chaque lambda :
          - reconstruit la trajectoire de covariance via EWMA sur S_emp_path
          - calcule une log-vraisemblance quasi-gaussienne moyenne
          - en ne scorant que les actifs observés (mask)
          - en ajoutant un ridge diagonal pour la stabilité (Cholesky)
        Retour : scores (L,) (plus grand = meilleur)
        """

        # dimensions
        L = lambdas.shape[0]
        K = S_emp_path.shape[0]
        N = S0.shape[0]

        #init scores
        scores = np.empty(L, dtype=np.float64)

        # buffer pour cov courante
        S = np.empty((N, N), dtype=np.float64)

        #iteration sur la grille de lambda
        for i in range(L):
            lmb = lambdas[i]

            # reset S = S0
            for a in range(N):
                for b in range(N):
                    S[a, b] = S0[a, b]

            ll_sum = 0.0
            n_used = 0

            # itération sur la trajectoire
            for k in range(K):
                # mise à jour EWMA 
                one_minus = 1.0 - lmb
                for a in range(N):
                    for b in range(N):
                        S[a, b] = lmb * S[a, b] + one_minus * S_emp_path[k, a, b]

                # construction de la liste des actifs observés ce jour k
                idx = np.empty(N, dtype=np.int64)
                m = 0
                for a in range(N):
                    if mask_path[k, a]:
                        idx[m] = a
                        m += 1

                if m < min_obs:
                    continue

                # sous-matrice cov_m + vecteur r_m 
                S_m = np.empty((m, m), dtype=np.float64)
                r_m = np.empty(m, dtype=np.float64)

                for a in range(m):
                    r_m[a] = R_path[k, idx[a]]
                    for b in range(m):
                        S_m[a, b] = S[idx[a], idx[b]]

                # ridge diagonal (stabilité SPD)
                for a in range(m):
                    S_m[a, a] += ridge

                # calcul logdet + quad form via Cholesky
                try:
                    Lm = np.linalg.cholesky(S_m)

                    logdet = 0.0
                    for a in range(m):
                        logdet += np.log(Lm[a, a])
                    logdet *= 2.0

                    y = np.linalg.solve(Lm, r_m)   # L y = r
                    quad = 0.0
                    for a in range(m):
                        quad += y[a] * y[a]

                except Exception:
                    continue

                ll_sum += -0.5 * (logdet + quad)
                n_used += 1

            # moyenne pour comparer équitablement
            if n_used > 0:
                scores[i] = ll_sum / n_used
            else:
                scores[i] = -1e300  # équivalent -inf

        return scores



# Pré-calcul des covariances empiriques rolling (pandas)
def _build_emp_cov_path_pandas( Xs: pd.DataFrame,window: int,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prépare les objets numpy nécessaires au kernel numba.
    """
    
    # init et dimensions
    Xs = Xs.sort_index()
    Xv = Xs.to_numpy(dtype=float)
    T, N = Xv.shape
    w = int(window)

    t_list = list(range(w + 1, T))
    K = len(t_list)

    S_emp_path = np.zeros((K, N, N), dtype=float)
    R_path = np.zeros((K, N), dtype=float)
    mask_path = np.zeros((K, N), dtype=bool)

    # boucle unique pour construire les trajectoires de cov empiriques + rendements + masques
    for k, t in enumerate(t_list):
        W = Xs.iloc[t - w : t]
        S_emp = W.cov(ddof=1).fillna(0.0).to_numpy()
        # symétrisation simple (équivalent safe_symmetrize)
        S_emp = 0.5 * (S_emp + S_emp.T)

        S_emp_path[k] = S_emp

        r = Xv[t]
        R_path[k] = r
        mask_path[k] = np.isfinite(r)

    return S_emp_path, R_path, mask_path



def tune_lambda_qmv_numba(self, X_slice: pd.DataFrame) -> float:
    """
   tuning QMV
    """


    Xs = _as_dt_df(X_slice)
    T, N = Xs.shape
    w = int(self.window)

    if T <= w + 5:
        return float(self.lmbda)

    # Grille lambda
    grid = self.lambda_grid if self.lambda_grid is not None else self._default_lambda_grid()
    grid = np.array([float(l) for l in grid if 0.80 <= float(l) < 1.0], dtype=np.float64)
    if grid.size == 0:
        return float(self.lmbda)

    ridge = float(getattr(self, "qmv_ridge", 1e-8))
    min_obs = int(getattr(self, "tune_min_obs", 30))

    # cov_0 sur première fenêtre 
    W0 = Xs.iloc[:w]
    S0 = W0.cov(ddof=1).fillna(0.0).to_numpy()
    if self.init == "diag":
        S0 = np.diag(np.diag(S0))
    S0 = _safe_symmetrize(S0)

    # Pré-calcul cov_emp une seule fois 
    S_emp_path, R_path, mask_path = _build_emp_cov_path_pandas(Xs, w)

    # Score numba pour tous les lambdas de la grille 
    scores = _qmv_scores_numba(grid, S0, S_emp_path, R_path, mask_path, ridge, min_obs)

    best_idx = int(np.argmax(scores))
    
    return float(grid[best_idx])
