# -*- coding: utf-8 -*-

"""
Moments de rendements : Empirique, MLE gaussien, Inférence & Viz
=================================================================

Contenu du module
-----------------
- Dataclasses résultats
  - `UnivariateMoments` : (mean, var, skew, kurt)
  - `MultivariateMoments` : (mean vector, covariance matrix)

- EmpiricalMoments
  - Estimateurs échantillonnaux (sample counterparts)
  - Univarié : moyenne, variance (ddof), skewness, kurtosis (Pearson/Fisher)
  - Multivarié : moyenne par colonne, covariance (`ddof` paramétrable)

- GaussianMLE
  - Hyp. i.i.d. gaussienne
  - Univarié 
  - Multivarié 

- Asymptotics (inférence)
  - SE / IC / z-tests pour la moyenne (i.i.d.)
  - SE pour variance de portefeuille et projection (gaussien)
  - Outil avancé : covariance projetée via cov_sample_cov_gaussian_proj
  - Helpers : ci_from_est_se, ztest_mean, ztest_portfolio_mean, required_T_for_target_se_mean, etc.

- MomentsViz (plots uniquement)
  - Histogramme avec superpositions et annotations (skew, kurt)
  - Barres d'IC pour moyennes par actif, whisker d'IC pour portefeuille
  - QQ-plot vs N(0,1), heatmap (cov/corr)
  - La classe ne calcule rien : elle affiche ce qui est déjà estimé

- Raccourcis
  - `estimate_univariate_empirical`, `estimate_multivariate_empirical`
  - `fit_gaussian_mle_univariate`, `fit_gaussian_mle_multivariate`

Conventions I/O
---------------
- Entrées : pd.Series ou pd.DataFrame indexés temps (le module tente pd.to_datetime).
- NaN : ignorés dans les calculs, les méthodes multivariées s'alignent par paires (pandas).

Hypothèses & limites
--------------------
- Partie moyenne : formules SE/IC/z basées sur i.i.d.
- Partie variance/projection : résultats sous gaussien (delta-method).
- Les IC sont asymptotiques (pertinents pour T raisonnablement grand).
"""



from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union, Tuple
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt




# dataclasses : Donne la possibilité de créer des instances immuables pour les résultats
@dataclass(frozen=True)
class UnivariateMoments:
    """Moments 1 à 4 pour une série : moyenne, variance, skewness, kurtosis."""
    mean: float
    var: float
    skew: float
    kurt: float 

@dataclass(frozen=True)
class MultivariateMoments:
    """Moments 1 et 2 pour un vecteur : moyenne vectorielle et covariance."""
    mean: pd.Series      # shape (N,)
    cov: pd.DataFrame    # shape (N,N)

# Type alias
NumericLike = Union[pd.Series, pd.DataFrame]


# Fonctions utilitaires
def _as_datetime_index(obj: NumericLike) -> NumericLike:
    """Tente d'imposer un DatetimeIndex, puis trie par date. N'altère pas les valeurs."""
    x = obj.copy()
    try:
        if not isinstance(x.index, pd.DatetimeIndex):
            x.index = pd.to_datetime(x.index)
    except Exception:
        pass
    return x.sort_index()

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numérique permissif colonne par colonne (non-numérique -> NaN)."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


#  Empirical moments 
class EmpiricalMoments:
    """
    Estimateurs empiriques des moments.
    """

    def __init__(self, *, ddof: int = 1, fisher: bool = False) -> None:
        self.ddof = ddof
        self.fisher = fisher

    # univarié (Series) 
    def univariate(self, r: pd.Series) -> UnivariateMoments:
        """
        Moments 1 à 4 pour une série (NaN ignorés). Variance avec ddof configuré.
        Skew/kurt basés sur moments centrés (asympt. non-biaisés).
        """

        # Nettoyage & conversion en datetime index
        s = _as_datetime_index(r.dropna())
        if s.empty:
            return UnivariateMoments(np.nan, np.nan, np.nan, np.nan)

        # mean
        mu = float(s.mean())

        # variance : 1/(T - ddof) * Σ (r - μ)^2
        var = float(s.var(ddof=self.ddof))

        # moments centrés
        c = s - mu
        m2 = float((c ** 2).mean())  # 1/T
        m3 = float((c ** 3).mean())
        m4 = float((c ** 4).mean())

        # skewness 
        skew = m3 / (m2 ** 1.5) if m2 > 0 else np.nan

        # kurtosis
        kurt_pearson = m4 / (m2 ** 2) if m2 > 0 else np.nan
        kurt = (kurt_pearson - 3.0) if self.fisher else kurt_pearson
        return UnivariateMoments(mu, var, skew, kurt) # renvoie l'objet UnivariateMoments avec les 4 moments

    #  multivarié (DataFrame)
    def multivariate(self, R: pd.DataFrame) -> MultivariateMoments:
        """
        Moments 1 et 2 pour un panel (colonnes=actifs).
        - mean : moyenne colonne par colonne
        - cov  : covariance (ddof paramétrable)
        NaN ignorés (alignement 'pairwise' via pandas.cov).
        """

        X = _to_numeric(_as_datetime_index(R)).dropna(how="all")
        if X.empty:
            return MultivariateMoments(pd.Series(dtype=float), pd.DataFrame(dtype=float))
        mu = X.mean(axis=0)
        Sigma = X.cov(ddof=self.ddof)
        return MultivariateMoments(mu, Sigma) # renvoie l'objet MultivariateMoments avec les 2 moments


#  Gaussian MLE
class GaussianMLE:
    """
    MLE sous hypothèse gaussienne i.i.d. N(μ, Ω).
    """

    #  univarié 
    def fit_univariate(self, r: pd.Series) -> UnivariateMoments:
        """Estimateur MLE gaussien + skew/kurt empiriques."""

        s = _as_datetime_index(r.dropna())
        T = len(s)

        # Cas vide
        if T == 0:
            return UnivariateMoments(np.nan, np.nan, np.nan, np.nan)
        
        mu = float(s.mean())                       
        var_mle = float(((s - mu) ** 2).sum() / T)  
        
        # compléter par moments empiriques (asymptotiques)
        em = EmpiricalMoments(ddof=1, fisher=False)
        uv = em.univariate(s)

        return UnivariateMoments(mu, var_mle, uv.skew, uv.kurt) # renvoie l'objet UnivariateMoments avec les 4 moments

    # multivarié  
    def fit_multivariate(self, R: pd.DataFrame) -> MultivariateMoments:
        """μmu (moyenne colonne) et sigma_MLE (1/T) pour un panel de rendements."""

        # Nettoyage & conversion en datetime index
        X = _to_numeric(_as_datetime_index(R)).dropna(how="all")
        T = len(X)

        # Cas vide
        if T == 0:
            return MultivariateMoments(pd.Series(dtype=float), pd.DataFrame(dtype=float))
        
        # MLE
        mu = X.mean(axis=0)
        C = X - mu
        Sigma_mle = (C.T @ C) / T  # 1/T :estimateur MLE de la covariance
        Sigma_mle.index = X.columns
        Sigma_mle.columns = X.columns
        return MultivariateMoments(mu, Sigma_mle)




# Asymptotic properties 
class Asymptotics:
    """
    SE (erreurs standards), IC (intervalles de confiance) et tests z
    pour la moyenne (i.i.d.) et des projections de covariance (gaussien).

    Implémentation :
      - Essaye SciPy (scipy.stats.norm) pour z-quantiles & p-values (précis).
      - Fallback sans SciPy via erf (approx) si SciPy indisponible.
    """

    #  moteur normal (SciPy si dispo)
    @staticmethod
    def _norm_ppf(p: float) -> float:
        """Quantile normal N(0,1) à proba p (ex: p=0.975 -> ~1.96)."""

        try:
            from scipy.stats import norm
            return float(norm.ppf(p))
        except Exception:

            # fallback erf
            import math

            # approximation par inversion simple (binaire) si besoin
            lo, hi = -10.0, 10.0
            for _ in range(60):
                mid = (lo + hi) / 2
                Phi = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
                if Phi < p:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2


    @staticmethod
    def _norm_cdf(z: float) -> float:
        """CDF normale standard."""
        try:
            from scipy.stats import norm
            return float(norm.cdf(z))
        except Exception:
            import math
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) # approx via erf

    @classmethod
    def _z_from_level(cls, level: float) -> float:
        """
        Quantile bilatéral : z_{1-(1-level)/2}.
        """
        p = 0.5 * (1.0 + level)
        return cls._norm_ppf(p)

    @classmethod
    def _two_sided_pvalue_from_z(cls, z: float) -> float:
        """p-value bilatérale """
        Phi = cls._norm_cdf(abs(z))
        return max(0.0, min(1.0, 2.0 * (1.0 - Phi)))

    #  Moyenne : i.i.d. 
    @staticmethod
    def var_mean_iid(Omega: pd.DataFrame) -> pd.DataFrame:
        """
        Variance de l'estimateur de la moyenne sous hypothèse i.i.d.
        """
        return Omega.copy()

    @staticmethod
    def var_mean_iid_univariate(sigma2: float, T: int) -> float:
        """Variance de la moyenne univariée sous hypothèse i.i.d. : sigma2 / T."""
        if T <= 0 or not np.isfinite(sigma2):
            return np.nan
        return sigma2 / float(T)

    @staticmethod
    def se_mean_iid(Omega: pd.DataFrame, T: int) -> pd.Series:
        """
        Calcul l'erreur standard (SE) de l'estimateur de la moyenne des rendements pour chaque actif sous hypothèse i.i.d.
        Sert notamment à la ocnstruction des intervalles de confiance (IC) et tests z.
        """
        if T <= 0 or Omega.empty:
            return pd.Series(dtype=float)
        diag = np.diag(Omega.values).astype(float)
        se = np.sqrt(diag / float(T))
        return pd.Series(se, index=Omega.index)

    @staticmethod
    def se_portfolio_mean(a: np.ndarray, Omega: pd.DataFrame, T: int) -> float:
        """
        Calcule l'erreur standard de la moyenne de portefeuille.
        """
        if T <= 0 or Omega.empty:
            return np.nan
        A = a.reshape(-1, 1)
        val = float(A.T @ Omega.values @ A)
        return np.sqrt(max(val, 0.0) / float(T))

    @classmethod
    def ci_from_est_se(cls, theta_hat: float, se: float, level: float = 0.95) -> tuple[float, float]:
        """
        Calcule un intervalle de confiance bilatéral pour un paramètre estimé avec
        z le quantile de la loi normale standard correspondant au niveau choisi.
        """
        if not np.isfinite(theta_hat) or not np.isfinite(se) or se < 0:
            return (np.nan, np.nan)
        z = cls._z_from_level(level)
        return (theta_hat - z * se, theta_hat + z * se)

    @classmethod
    def ztest_mean(cls, mu_hat: float, se: float, mu0: float = 0.0) -> tuple[float, float]:
        """
        Calcule un test bilatéral z pour la moyenne. 
        Permet de tester si la moyenne estimée est statistiquement différente d'une valeur donnée mu0.
        """
        if not np.isfinite(mu_hat) or not np.isfinite(se) or se <= 0:
            return (np.nan, np.nan)
        z = (mu_hat - mu0) / se
        p = cls._two_sided_pvalue_from_z(z)
        return (float(z), float(p))

    @classmethod
    def ztest_portfolio_mean(cls, a: np.ndarray, mu_hat: pd.Series, Omega: pd.DataFrame, T: int, mu0: float = 0.0) -> tuple[float, float, tuple[float, float]]:
        """
        Calcule le z-test pour la moyenne de portefeuille
        """
        est = float(a @ mu_hat.values)
        se = Asymptotics.se_portfolio_mean(a, Omega, T)
        z, p = cls.ztest_mean(est, se, mu0=mu0)
        ci = cls.ci_from_est_se(est, se, level=0.95)
        return (z, p, ci)


    # Covariance : cas gaussien 
    @staticmethod
    def cov_sample_cov_gaussian_proj(v1: np.ndarray, v2: np.ndarray, Omega: pd.DataFrame) -> np.ndarray:
        """
        Calcule sous hypothèse gaussienne la covariance de √T vec(Ŝ - Ω) projetée sur v1, v2 :
        
        """
        v1 = v1.reshape(-1, 1)
        v2 = v2.reshape(-1, 1)
        term1 = float(v1.T @ Omega.values @ v2) * Omega.values 
        term2 = (Omega.values) @ (v1.T @ Omega.values)
        return term1 + term2

    @staticmethod
    def se_portfolio_variance(v: np.ndarray, Omega: pd.DataFrame, T: int) -> float:
        """
        Calcule l'erreur standard de la variance de portefeuille.
        """
        if T <= 0 or Omega.empty:
            return np.nan
        v = v.reshape(-1, 1)
        port_var = float(v.T @ Omega.values @ v)
        port_var = max(port_var, 0.0)
        return np.sqrt(2.0 * (port_var ** 2) / float(T))

    @staticmethod
    def se_proj_covariance(v: np.ndarray, w: np.ndarray, Omega: pd.DataFrame, T: int) -> float:
        """
        Calcule l'erreur standard de la projection de la covariance de portefeuille sous hypothèse gaussienne.
        Sert a estimer l’incertitude d’une co-variance projetée
        """
        if T <= 0 or Omega.empty:
            return np.nan
        v = v.reshape(-1, 1); w = w.reshape(-1, 1)
        vv = float(v.T @ Omega.values @ v)
        ww = float(w.T @ Omega.values @ w)
        vw = float(v.T @ Omega.values @ w)
        var_scalar = (vv * ww + vw * vw) / float(T)
        return np.sqrt(max(var_scalar, 0.0))

    @classmethod
    def ci_portfolio_variance(cls, v: np.ndarray, Omega: pd.DataFrame, T: int, level: float = 0.95) -> tuple[float, float]:
        """
        Calcule un intervalle de confiance bilatéral pour la variance de portefeuille.
        Sert notamment à évaluer l’incertitude sur l'estimation de la variance de portefeuille.
        """
        est = float(v.reshape(1, -1) @ Omega.values @ v.reshape(-1, 1))  # plug-in
        se = cls.se_portfolio_variance(v, Omega, T)
        return cls.ci_from_est_se(est, se, level=level)

    # Planification
    @staticmethod
    def required_T_for_target_se_mean(se_target: float, Omega: pd.DataFrame, a: Optional[np.ndarray] = None) -> float:
        """
        Calcule le nombre d'observations T nécessaire pour atteindre une erreur standard cible sur la moyenne (i.i.d.).
        """
        if se_target <= 0 or Omega.empty:
            return np.nan
        if a is None:
            diag = np.diag(Omega.values).astype(float)
            Ti = diag / float(se_target ** 2)
            return float(np.nanmax(Ti))
        A = a.reshape(-1, 1)
        val = float(A.T @ Omega.values @ A)
        return float(val / (se_target ** 2))


#  Visualization (plot-only) 
class MomentsViz:
    """
    Visualisations.
    -> Tous les nombres (moyenne, sigma, skew, kurt, IC, SE, matrices) doivent être
       fournis par les autres modules (EmpiricalMoments, GaussianMLE, Asymptotics, etc.).

    Méthodes
    --------
    - hist_with_moments(r, moments)       : histogramme + overlays (µ, ±σ) + annotations (skew/kurt)
    - mean_confidence_bars(mu, lo, hi)    : barres d'erreur IC bilatéral par actif (déjà calculé)
    - portfolio_ci(mean, lo, hi)          : whisker horizontal pour IC de moyenne de portefeuille
    - qq_plot_normal(r)                   : QQ-plot vs N(0,1) (diagnostic visuel – pas de moments)
    - matrix_heatmap(M)                   : heatmap d'une matrice (covariance OU corrélation déjà donnée)
    """

    #  helpers affichage 
    @staticmethod
    def _get_ax(ax):
        """Retourne (ax, fig) à partir d'ax existants ou en crée de nouveaux."""
        if ax is not None:
            return ax, ax.figure
        fig, ax2 = plt.subplots(figsize=(7, 4))
        return ax2, fig

    @staticmethod
    def _fd_bins(x: pd.Series) -> int:
        """Freedman - Diaconis: uniquement pour choisir un nombre de bins (esthétique)."""
        x = pd.Series(x).dropna()
        n = x.size
        if n < 2:
            return 10
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr <= 0:
            return int(round(np.sqrt(n)))
        h = 2 * iqr / (n ** (1 / 3))
        if h <= 0:
            return int(round(np.sqrt(n)))
        span = x.max() - x.min()
        return max(5, int(math.ceil(span / h)))

    #  histogramme 
    def hist_with_moments(self, r: pd.Series, moments, bins = "fd", density: bool = True, ax=None, title: str | None = "Distribution des rendements (histogramme)", annotate: bool = True,):
        """
        Histogramme de la série 'r'.

        Paramètres
        ----------
        r        : Series de rendements (na déjà gérés en amont de préférence).
        moments  : Object UnivariateMoments (mean, var, skew, kurt).
        bins     : 'fd' (Freedman–Diaconis), int, ou None (par défaut matplotlib).
        density  : True -> densité ; False -> compte brut.
        ax       : axes existants (optionnel).
        title    : titre (optionnel).
        annotate : affiche le bloc texte des moments sur la figure.
        """

        # Nettoyage & conversion en datetime index
        x = pd.Series(r).dropna()
        if x.empty:
            return None

        # extraire les moments
        mu   = float(moments.mean)
        var  = float(moments.var)
        skew = float(moments.skew)
        kurt = float(moments.kurt)
        sigma = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan

        # bins
        if bins == "fd":
            bins = self._fd_bins(x)

        # plot
        ax, fig = self._get_ax(ax)
        ax.hist(x.values, bins=bins, density=density, alpha=0.75)
        ax.axvline(mu, linestyle="--", linewidth=1.5, label="moyenne")
        if np.isfinite(sigma) and sigma > 0:
            ax.axvspan(mu - sigma, mu + sigma, alpha=0.15, label="±1σ")

        ax.set_title(title or "")
        ax.set_xlabel("Rendement")
        ax.set_ylabel("Densité" if density else "Compte")
        if annotate:
            text = f"μ={mu:.5f}\nσ={sigma:.5f}\nskew={skew:.3f}\nkurt={kurt:.3f}"
            ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top")
        ax.legend(loc="best")
        fig.tight_layout()
        return ax
    

    #  barres de moyennes + IC 
    def mean_confidence_bars(self,mu: pd.Series,lo: pd.Series,hi: pd.Series,top_k: int | None = 25,ax=None,title: str | None = "Moyennes par actif avec IC"):
        """
        Barres d'erreur pour des IC bilatéraux (lo/hi).
        Aucune conversion z/SE ici.

        Paramètres
        ----------
        mu   : Series des estimations de moyenne (index=actifs)
        lo   : Series des bornes inférieures (même index)
        hi   : Series des bornes supérieures (même index)
        top_k: ne montre que les top_k |mu| pour lisibilité (None= tous)
        """

        # Nettoyage & alignement
        df = pd.concat({"mu": mu, "lo": lo, "hi": hi}, axis=1).dropna()

        # Cas vide
        if df.empty:
            return None
        
        # Trier par mu décroissant
        df = df.reindex(df["mu"].abs().sort_values(ascending=False).index)
        if top_k is not None:
            df = df.head(top_k)

        # plot
        ax, fig = self._get_ax(ax)
        x = np.arange(len(df))
        y = df["mu"].values
        yerr = np.vstack([y - df["lo"].values, df["hi"].values - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=90)
        ax.set_title(title or "")
        ax.set_ylabel("Moyenne (rendement)")
        fig.tight_layout()
        return ax

    # IC de moyenne de portefeuille 
    def portfolio_ci(self,mean: float,lo: float,hi: float,ax=None,title: str | None = "Moyenne de portefeuille : IC (fourni)",):
        """
        Whisker horizontal avec estimateur `mean` et IC [lo, hi].
        """
        # Cas vide
        if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
            return None
        
        # plot
        ax, fig = self._get_ax(ax)
        ax.plot([lo, hi], [0, 0], linewidth=3)   # segment IC
        ax.plot([mean], [0], marker="o")         # estimateur
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_xlabel("Rendement moyen")
        ax.set_title(title or "")
        fig.tight_layout()
        return ax

    # QQ-plot 
    def qq_plot_normal(self,r: pd.Series,ax=None,title: str | None = "QQ-plot vs normale"):
        """
        QQ-plot vs N(0,1) (diagnostic visuel de normalité).
        """
        # Nettoyage & conversion en datetime index
        x = pd.Series(r).dropna()
        if x.empty:
            return None
        
        x = np.sort(x.values)
        n = len(x)
        p = (np.arange(1, n + 1) - 0.5) / n

        # quantiles théoriques N(0,1) (plot-only, pas d'estimation de moments)
        try:
            from scipy.stats import norm
            q = norm.ppf(p)
        except Exception:
            # fallback bisection (visuel suffisant)
            def inv_norm(u):
                lo, hi = -10.0, 10.0
                for _ in range(40):
                    mid = (lo + hi) / 2
                    Phi = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
                    if Phi < u: lo = mid
                    else: hi = mid
                return (lo + hi) / 2
            q = np.array([inv_norm(ui) for ui in p])

        ax, fig = self._get_ax(ax)
        ax.plot(q, x, marker="o", linestyle="None", alpha=0.7)

        # droite de tendance (simple régression pour guider l'œil)
        a = np.polyfit(q, x, 1)
        xx = np.array([q.min(), q.max()])
        ax.plot(xx, a[0] * xx + a[1], linestyle="--")
        ax.set_xlabel("Quantiles théoriques N(0,1)")
        ax.set_ylabel("Quantiles empiriques")
        ax.set_title(title or "")
        fig.tight_layout()
        return ax

    # Heatmap 
    def matrix_heatmap(self,M: pd.DataFrame,ax=None,title: str | None = "Matrice"): 
        """
        Heatmap d'une matrice M (covariance OU corrélation).
        """
        if M.empty:
            return None
        
        ax, fig = self._get_ax(ax)
        im = ax.imshow(M.values, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(M.columns)))
        ax.set_yticks(np.arange(len(M.index)))
        ax.set_xticklabels(M.columns, rotation=90)
        ax.set_yticklabels(M.index)
        ax.set_title(title or "")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return ax
    

    #  helpers rapport (visuel) 
    def _plot_timeseries(self, r: pd.Series, ax, title: str = "Série temporelle"):
        """Trace la série telle quelle (plot-only)."""
        s = pd.Series(r).dropna()
        ax.plot(s.index, s.values, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Date"); ax.set_ylabel("Rendement")

    def _summary_box(self, ax, r: pd.Series, moments, mean_ci: tuple[float, float] | None):
        """Panneau texte : période, T, manquants, μ, σ, skew, kurt, IC si fourni (plot-only)."""
        s = pd.Series(r)
        T = int(s.dropna().shape[0])
        miss = int(s.isna().sum())
        start = str(s.first_valid_index()) if T > 0 else "—"
        end   = str(s.last_valid_index())  if T > 0 else "—"
        mu, var = float(moments.mean), float(moments.var)
        sigma = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
        skew, kurt = float(moments.skew), float(moments.kurt)
        ci_txt = "IC: n/a" if mean_ci is None else f"IC: [{mean_ci[0]:.5f}, {mean_ci[1]:.5f}]"

        text = (
            f"Période : {start} → {end}\n"
            f"T (valide) : {T}  |  Manquants : {miss}\n"
            f"μ = {mu:.6f}   σ = {sigma:.6f}\n"
            f"skew = {skew:.3f}   kurt = {kurt:.3f}\n"
            f"{ci_txt}"
        )
        ax.axis("off")
        ax.text(0.02, 0.98, text, va="top", ha="left", transform=ax.transAxes,
                fontsize=10, family="monospace")

    #  rapport 2×2 
    def report_univariate(self, r: pd.Series,moments,  mean_ci: tuple[float, float] | None = None, suptitle: str | None = "Rapport quantitatif (univarié)",
        show: bool = True,):
        """
        Compose un rapport 2x2 pour une série de rendements.:
          [0,0] timeseries   [0,1] histogramme (hist_with_moments)
          [1,0] QQ-plot      [1,1] résumé (texte + IC si fourni)
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        self._plot_timeseries(r, ax=axs[0, 0], title="Série temporelle")
        self.hist_with_moments(r, moments=moments, ax=axs[0, 1], title="Histogramme + moments")
        self.qq_plot_normal(r, ax=axs[1, 0], title="QQ-plot vs normale")
        self._summary_box(ax=axs[1, 1], r=r, moments=moments, mean_ci=mean_ci)
        
        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if show:
            plt.show()
        return fig





# Fonctions raccourcies d'estimation 
def estimate_univariate_empirical(r: pd.Series, *, ddof: int = 1, fisher: bool = False) -> UnivariateMoments:
    """
    Raccourci : moments empiriques 1 à 4 pour une série.
    ddof=1 => variance 'sample', fisher=False => kurtosis 'Pearson'.
    """
    return EmpiricalMoments(ddof=ddof, fisher=fisher).univariate(r)

def estimate_multivariate_empirical(R: pd.DataFrame, *, ddof: int = 1) -> MultivariateMoments:
    """Raccourci : moyenne vectorielle + covariance (ddof au choix)."""
    return EmpiricalMoments(ddof=ddof).multivariate(R)

def fit_gaussian_mle_univariate(r: pd.Series) -> UnivariateMoments:
    """Raccourci : MLE gaussien (μ̂, σ̂²_MLE=1/T) + skew/kurt empiriques."""
    return GaussianMLE().fit_univariate(r)

def fit_gaussian_mle_multivariate(R: pd.DataFrame) -> MultivariateMoments:
    """Raccourci : MLE gaussien (μ̂, Ω̂_MLE=1/T)."""
    return GaussianMLE().fit_multivariate(R)
