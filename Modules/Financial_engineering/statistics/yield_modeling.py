"""
Calcul de rendements arithmétiques ou logarithmiques depuis des séries ou tableaux de prix.

Ce fichier expose une seule classe (YieldModeler) qui prend en entrée des prix
(pd.Series ou pd.DataFrame) et retourne les rendements correspondants.
Il gère deux formats d'entrée : une série de prix bruts, et un tableau multi-actifs
(ou OHLCV avec colonnes 'Adj Close'/'Close'). Le nettoyage des lignes et colonnes
entièrement nulles est intégré pour éviter les biais sur les jours non ouvrés.

Classes
-------
YieldModeler :
    Classe principale de calcul de rendements. Expose compute_series() et compute_frame()
    comme interface publique, et _dispatch_returns() comme router interne.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union
import numpy as np
import pandas as pd

# Type alias pour les méthodes de calcul de rendements acceptées
ReturnMethod = Literal["arithmetic", "log"]

# Type alias pour les entrées acceptées : série ou tableau de prix
NumericLike = Union[pd.Series, pd.DataFrame]

# Noms de colonnes reconnus comme prix de clôture dans un fichier OHLCV
CANON_CLOSES = ("Adj Close", "Close")

@dataclass
class YieldModeler:
    """
    Classe contenant les méthodes de calcul de rendements depuis des prix.

    Gère deux cas d'usage : une série de prix unique (compute_series) et un tableau
    multi-actifs ou OHLCV (compute_frame). Le nettoyage des lignes et colonnes
    entièrement nulles est contrôlé par les attributs drop_all_zero_return_*.

    Attributes
    ----------
    prefer_adj : bool
        Si True, utilise 'Adj Close' en priorité sur 'Close' pour les fichiers OHLCV.
    drop_all_zero_return_rows : bool
        Si True, supprime les lignes où la quasi-totalité des rendements est nulle.
    drop_all_zero_return_cols : bool
        Si True, supprime les colonnes où la quasi-totalité des rendements est nulle.
    zero_tol : float
        Seuil de tolérance pour considérer un rendement comme nul (0.0 = strictement nul).

    Methods
    -------
    compute_series(prices, method, periods, dropna) -> pd.Series :
        Calcule les rendements d'une série de prix.
    compute_frame(prices, method, periods, dropna) -> pd.DataFrame :
        Calcule les rendements d'un tableau de prix (multi-actifs ou OHLCV).
    """

    prefer_adj: bool = True
    drop_all_zero_return_rows: bool = True
    drop_all_zero_return_cols: bool = True
    zero_tol: float = 0.0  


    def compute_series(self,prices: pd.Series,method: ReturnMethod = "arithmetic",periods: int = 1,dropna: bool = True,) -> pd.Series:
        """
        Calcule les rendements d'une série de prix.

        Parameters
        ----------
        prices : pd.Series
            Série de prix bruts indexée par dates.
        method : str
            Méthode de calcul : 'arithmetic' ou 'log'.
        periods : int
            Décalage temporel pour le calcul (1 = rendements journaliers).
        dropna : bool
            Si True, supprime les NaN en sortie (premier(s) rendement(s) manquants).

        Returns
        -------
        pd.Series
            Série de rendements.
        """

        # Prépare la série : conversion de l'index en datetime, tri et passage en numérique
        s = self._prep_series(prices)

        # Calcule les rendements selon la méthode choisie
        out = self._dispatch_returns(s, method=method, periods=periods)

        return out.dropna() if dropna else out

    def compute_frame(self,prices: pd.DataFrame,method: ReturnMethod = "arithmetic",periods: int = 1,dropna: bool = True,) -> pd.DataFrame:
        """
        Calcule les rendements d'un tableau de prix.

        Gère deux cas : tableau OHLCV (extrait Adj Close ou Close selon prefer_adj),
        et tableau multi-actifs (calcule colonne par colonne).

        Parameters
        ----------
        prices : pd.DataFrame
            Tableau de prix (multi-actifs ou OHLCV avec colonnes 'Adj Close'/'Close').
        method : str
            Méthode de calcul : 'arithmetic' ou 'log'.
        periods : int
            Décalage temporel pour le calcul (1 = rendements journaliers).
        dropna : bool
            Si True, supprime les lignes entièrement NaN en sortie.

        Returns
        -------
        pd.DataFrame
            Tableau de rendements.
        """

        # Prépare le DataFrame : conversion de l'index en datetime, tri et passage en numérique
        df = self._prep_frame(prices)

        # Construit un mapping en minuscules pour détecter les colonnes OHLCV sans tenir compte de la casse
        lowered = {c.lower(): c for c in df.columns}

        # Cas OHLCV : une seule colonne de prix à extraire avant le calcul
        if "adj close" in lowered or "close" in lowered:

            # Sélectionne Adj Close en priorité si prefer_adj est activé et la colonne existe
            col = lowered.get("adj close") if self.prefer_adj and "adj close" in lowered else lowered.get("close")
            if not col:
                col = lowered.get("adj close")

            series = df[col]

            # Calcule les rendements sur la série extraite et retourne un DataFrame à une colonne
            out = self._dispatch_returns(series, method=method, periods=periods).to_frame(col)

            # Sur OHLCV, on supprime uniquement les lignes all-zero (pas les colonnes, il n'y en a qu'une)
            if self.drop_all_zero_return_rows:
                out = self._drop_all_zero_returns(out)

            return out.dropna(how="all") if dropna else out

        # Cas multi-actifs : calcule les rendements sur toutes les colonnes simultanément
        out = self._dispatch_returns(df, method=method, periods=periods)

        # Supprime les lignes et colonnes dont la quasi-totalité des valeurs est nulle
        out = self._drop_all_zero_returns(out)

        # Supprime les lignes entièrement NaN si demandé
        return out.dropna(how="all") if dropna else out


    def _dispatch_returns(self, x: NumericLike, method: ReturnMethod, periods: int) -> NumericLike:
        """
        Route le calcul vers la méthode arithmétique ou logarithmique selon le paramètre method.

        Parameters
        ----------
        x : pd.Series or pd.DataFrame
            Données de prix à transformer en rendements.
        method : str
            'arithmetic' ou 'log'.
        periods : int
            Décalage temporel pour le calcul.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rendements calculés avec la méthode choisie.

        Raises
        ------
        ValueError
            Si method n'est ni 'arithmetic' ni 'log'.
        """
        if method == "arithmetic":
            return self._ret_arith(x, periods)
        if method == "log":
            return self._ret_log(x, periods)
        raise ValueError("method doit être 'arithmetic' ou 'log'.")

    def _ret_arith(self, x: NumericLike, periods: int) -> NumericLike:
        """
        Calcule les rendements arithmétiques via pct_change.

        Parameters
        ----------
        x : pd.Series or pd.DataFrame
            Données de prix.
        periods : int
            Décalage temporel.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rendements arithmétiques (x_t / x_{t-periods} - 1).
        """
        return x.pct_change(periods=periods)

    def _ret_log(self, x: NumericLike, periods: int) -> NumericLike:
        """
        Calcule les rendements logarithmiques en masquant les prix non positifs.

        Applique log(x_t / x_{t-periods}) uniquement aux positions où les deux prix
        sont strictement positifs. Les positions invalides restent NaN.

        Parameters
        ----------
        x : pd.Series or pd.DataFrame
            Données de prix.
        periods : int
            Décalage temporel.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rendements logarithmiques, NaN aux positions où un prix est nul ou négatif.
        """

        # Calcule les prix décalés de periods périodes
        prev = x.shift(periods)

        # Masque de validité : les deux prix doivent être strictement positifs pour calculer le log
        mask = (x > 0) & (prev > 0)

        # Cas Series : initialise avec NaN et ne calcule qu'aux positions valides
        if isinstance(x, pd.Series):

            out = pd.Series(np.nan, index=x.index, dtype="float64")

            # Calcule le log uniquement aux positions où les deux prix sont positifs
            out.loc[mask] = np.log(x.loc[mask] / prev.loc[mask])
            return out
        
        # Cas DataFrame : même logique, vectorisée sur toutes les colonnes simultanément
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype="float64")
        out[mask] = np.log(x[mask] / prev[mask])
        return out


    def _prep_series(self, s: pd.Series) -> pd.Series:
        """
        Prépare une série de prix pour le calcul de rendements.

        Convertit l'index en DatetimeIndex si possible, trie par date et convertit
        les valeurs en numérique en remplaçant les valeurs non convertibles par NaN.

        Parameters
        ----------
        s : pd.Series
            Série de prix bruts.

        Returns
        -------
        pd.Series
            Série préparée avec index datetime, triée et valeurs numériques.
        """
        s2 = s.copy()

        # Tente de convertir l'index en DatetimeIndex si ce n'est pas déjà le cas
        if not isinstance(s2.index, pd.DatetimeIndex):
            try:
                s2.index = pd.to_datetime(s2.index)
            except Exception:
                pass

        # Trie par ordre chronologique
        s2 = s2.sort_index()

        # Convertit les valeurs en numérique, les valeurs non convertibles deviennent NaN
        s2 = pd.to_numeric(s2, errors="coerce")

        return s2

    def _prep_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare un DataFrame de prix pour le calcul de rendements.

        Convertit l'index en DatetimeIndex si possible, trie par date et convertit
        chaque colonne en numérique en remplaçant les valeurs non convertibles par NaN.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de prix bruts.

        Returns
        -------
        pd.DataFrame
            DataFrame préparé avec index datetime, trié et valeurs numériques.
        """
        df2 = df.copy()

        # Tente de convertir l'index en DatetimeIndex si ce n'est pas déjà le cas
        if not isinstance(df2.index, pd.DatetimeIndex):
            try:
                df2.index = pd.to_datetime(df2.index)
            except Exception:
                pass

        # Trie par ordre chronologique
        df2 = df2.sort_index()

        # Convertit chaque colonne en numérique, les valeurs non convertibles deviennent NaN
        for c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

        return df2
        
    def _drop_all_zero_returns(self, r: pd.DataFrame, zero_ratio_threshold: float = 0.975) -> pd.DataFrame:
        """
        Supprime les lignes et colonnes dont la proportion de valeurs nulles dépasse le seuil.

        Une valeur est considérée nulle si elle est égale à 0 (ou inférieure à zero_tol
        en valeur absolue si zero_tol est positif). Les NaN sont traités comme des zéros.

        Parameters
        ----------
        r : pd.DataFrame
            DataFrame de rendements à nettoyer.
        zero_ratio_threshold : float
            Seuil de proportion de zéros au-delà duquel une ligne ou colonne est supprimée.
            Par exemple, 0.975 supprime les lignes/colonnes où 97.5% des valeurs sont nulles.

        Returns
        -------
        pd.DataFrame
            DataFrame nettoyé.
        """
        if r.empty:
            return r

        tol = float(self.zero_tol)

        # Remplace les NaN par 0 pour le calcul des ratios de zéros
        r0 = r.fillna(0.0)

        # Détermine le masque de zéros selon la tolérance configurée
        if tol > 0:

            # Avec tolérance : considère comme nul tout ce qui est inférieur à tol en valeur absolue
            is_zero = ~(r0.abs() > tol)

        else:
            # Sans tolérance : compare strictement à 0
            is_zero = (r0 == 0.0)

        # Supprime les lignes dont la proportion de zéros dépasse le seuil
        if zero_ratio_threshold is not None:
            zero_ratio_rows = is_zero.mean(axis=1)
            keep_rows = zero_ratio_rows < zero_ratio_threshold
            r = r.loc[keep_rows]

            # Recalcule le masque sur le sous-ensemble réduit avant le filtrage des colonnes
            r0 = r.fillna(0.0)
            if tol > 0:
                is_zero = ~(r0.abs() > tol)
            else:
                is_zero = (r0 == 0.0)

        # Supprime les colonnes dont la proportion de zéros dépasse le seuil
        if zero_ratio_threshold is not None:
            zero_ratio_cols = is_zero.mean(axis=0)
            keep_cols = zero_ratio_cols < zero_ratio_threshold
            r = r.loc[:, keep_cols]

        return r
        
    def _drop_all_zero_returns_last(self, r: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les lignes et colonnes où tous les rendements sont nuls ou NaN.

        Version stricte de _drop_all_zero_returns : supprime uniquement les lignes/colonnes
        dont la totalité des valeurs est nulle, sans seuil de proportion.
        Les NaN sont traités comme des zéros. Le paramètre zero_tol permet de considérer
        les valeurs proches de zéro comme nulles (utile pour le bruit numérique en virgule flottante).

        Parameters
        ----------
        r : pd.DataFrame
            DataFrame de rendements à nettoyer.

        Returns
        -------
        pd.DataFrame
            DataFrame sans lignes ni colonnes entièrement nulles.
        """
        if r.empty:
            return r

        # Remplace les NaN par 0 pour le test de nullité
        r0 = r.fillna(0.0)

        tol = float(self.zero_tol)

        # Construit le masque des valeurs considérées comme non nulles
        if tol > 0.0:

            # Avec tolérance : une valeur est non nulle si son absolu dépasse zero_tol
            nz = (r0.abs() > tol)
        else:
            
            # Sans tolérance : une valeur est non nulle si elle est strictement différente de 0
            nz = (r0 != 0.0)

        if self.drop_all_zero_return_rows:

            # Garde les lignes où au moins une valeur est non nulle
            keep_rows = nz.any(axis=1)
            r = r.loc[keep_rows]

            # Recalcule le masque sur le sous-ensemble réduit avant le filtrage des colonnes
            r0 = r.fillna(0.0)
            if tol > 0.0:
                nz = (r0.abs() > tol)
            else:
                nz = (r0 != 0.0)

        if self.drop_all_zero_return_cols:
            
            # Garde les colonnes où au moins une valeur est non nulle
            keep_cols = nz.any(axis=0)
            r = r.loc[:, keep_cols]

        return r