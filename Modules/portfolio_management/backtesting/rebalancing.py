"""
Définition du calendrier de rebalancement du portefeuille.
 
Ce fichier gère la génération des dates auxquelles le portefeuille doit être
rebalancé, selon une fréquence fixe (quotidienne, hebdomadaire, mensuelle, trimestrielle).
Il supporte aussi un mode ancré sur les dates de rebalancement du benchmark,
utile pour aligner le portefeuille sur les reconstitutions de l'indice.
 
Classes
-------
RebalanceSchedule :
    Définit la fréquence et le timing du rebalancement, et expose les méthodes
    pour générer les dates de rebalancement sur un index de trading donné.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


RebalanceTiming = Literal["close_to_next_open"]


@dataclass(frozen=True)
class RebalanceSchedule:
    """
    Calendrier de rebalancement du portefeuille.
 
    Définit à quelle fréquence et selon quel timing le portefeuille doit être rebalancé.
    Les dates sont toujours snappées sur l'index de trading réel (pas de dates calendaires
    hors marché).
 
    Attributes
    ----------
    freq : str
        Fréquence de rebalancement, exprimée en alias pandas :
        - 'D'     : chaque jour de trading.
        - 'W-FRI' : hebdomadaire, le vendredi.
        - 'M'     : fin de mois (dernier jour de trading du mois).
        - 'Q'     : fin de trimestre.
    timing : str
        Convention d'exécution des ordres. 'close_to_next_open' signifie que les poids
        sont décidés à la clôture de la date t et appliqués à partir de l'ouverture de t+1.
        En pratique et dans le engine, un lag est appliquable pour simuler les délais d'exécution.

    Methods
    -------
    rebalance_dates(index) -> pd.DatetimeIndex :
        Génère les dates de rebalancement selon self.freq, snappées sur l'index de trading.
    rebalance_dates_anchored(trading_index, bench_dates, k) -> pd.DatetimeIndex :
        Génère les dates de rebalancement ancrées sur les dates de reconstitution du benchmark, avec k jours de décalage après chaque date d'ancrage.
    """
    
    freq: str = "M"
    timing: RebalanceTiming = "close_to_next_open"
    
    
    def rebalance_dates_anchored(self,trading_index: pd.DatetimeIndex,bench_dates: pd.DatetimeIndex,k: int = 1,) -> pd.DatetimeIndex:
        """
        Génère les dates de rebalancement ancrées sur les dates de reconstitution du benchmark.
 
        Pour chaque intervalle (bench_t, bench_t+1), les dates de rebalancement sont générées
        selon self.freq à partir de bench_t + k jours de trading. Utile pour synchroniser le
        portefeuille sur les reconstitutions périodiques de l'indice.
 
        Comportement selon la fréquence :
        - 'Q' : une seule date par intervalle (premier jour de trading après bench_t + k).
        - 'M' : une date par début de mois dans l'intervalle.
        - 'W' : une date par lundi dans l'intervalle.
 
        Parameters
        ----------
        trading_index : pd.DatetimeIndex
            Index de trading complet (jours de bourse disponibles dans les données).
        bench_dates : pd.DatetimeIndex
            Dates de reconstitution du benchmark (ex : dates de rebalancement trimestrielles de l'indice).
        k : int
            Décalage en jours de trading après chaque date d'ancrage du benchmark.
            k=1 signifie que le premier rebalancement a lieu le jour de trading suivant la reconstitution du benchmark.
 
        Returns
        -------
        pd.DatetimeIndex
            Dates de rebalancement snappées sur l'index de trading, triées et sans doublons.
        """
        
        # Validation d'usage pour éviter les erreurs silencieuses
        if not isinstance(trading_index, pd.DatetimeIndex):
            raise TypeError("trading_index must be a pandas DatetimeIndex.")
        if not isinstance(bench_dates, pd.DatetimeIndex):
            raise TypeError("bench_dates must be a pandas DatetimeIndex.")

        # Si les index sont en dehors de la timezone locale, on les convertit pour éviter les problèmes de comparaison de dates
        idx = trading_index.sort_values().unique()
        anchors = bench_dates.sort_values().unique()

        # Sentinelle pour délimiter le dernier intervalle
        sentinel = idx[-1] + pd.Timedelta(days=1)
        anchors_ext = anchors.append(pd.DatetimeIndex([sentinel]))

        # On ne considère que les dates d'ancrage qui sont dans l'index de trading ou avant la dernière date de trading
        freq_map = {"Q": None, "M": "MS", "W": "W-MON"}
        pd_freq = freq_map.get(self.freq.upper())

        out: set = set()

        #Iteration sur les intervalles délimités par les dates d'ancrage du benchmark
        for i in range(len(anchors_ext) - 1):

            # Détermine les bornes de l'intervalle correspondant à l'ancrage du benchmark
            anchor_start = anchors_ext[i]
            anchor_end   = anchors_ext[i + 1]

            # Premier jour de trading dans cet intervalle + k pas
            pos = idx.searchsorted(anchor_start, side="left")

            # On ajoute k jours de trading pour trouver la première date de rebalancement post-benchmark
            pos_k = pos + k

            # Si on n'a pas assez de jours de trading après l'ancrage du benchmark, on passe à l'ancrage suivant
            if pos_k >= len(idx):
                continue
            first_rebal = idx[pos_k]

            #Si la fréquence est trimestrielle, on ne prend que la première date de rebalancement post-benchmark. Sinon, on génère les dates calendaires dans l'intervalle et on les snappe sur l'index de trading.
            if self.freq.upper() == "Q":
                out.add(first_rebal)
            else:
                # Génère les dates calendaires dans [first_rebal, anchor_end)
                candidates = pd.date_range(start=first_rebal,end=anchor_end - pd.Timedelta(days=1),freq=pd_freq,)
                for c in candidates:
                    pos_c = idx.searchsorted(c, side="left")
                    if pos_c < len(idx) and idx[pos_c] < anchor_end:
                        out.add(idx[pos_c])
                # Toujours inclure le premier rebal post-bench
                out.add(first_rebal)

        return pd.DatetimeIndex(sorted(out))
    

    def rebalance_dates(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Génère les dates de rebalancement selon la fréquence configurée.
 
        Pour chaque date calendaire générée par le resample, on snappe sur le dernier
        jour de trading disponible dans l'index (last trading date <= date calendaire).
 
        Parameters
        ----------
        index : pd.DatetimeIndex
            Index de trading complet (jours de bourse disponibles dans les données).
 
        Returns
        -------
        pd.DatetimeIndex
            Dates de rebalancement snappées sur l'index de trading, triées et sans doublons.
        """

        # Validation d'usage pour éviter les erreurs silencieuses
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("Returns index must be a pandas DatetimeIndex.")
        
        # Si l'index est en dehors de la timezone locale, on le convertit pour éviter les problèmes de comparaison de dates
        if index.tz is not None:
            index = index.tz_convert(index.tz)

        # Si la fréquence est quotidienne, on prend simplement l'index tel quel
        if self.freq.upper() == "D":
            return index

        # Utilise une serie fictive pour resampler selon la frequence choisie
        s = pd.Series(1.0, index=index)
        dates = s.resample(self.freq).last().index

        out = []
        for d in dates:
            # find last trading date <= d
            loc = index.searchsorted(d, side="right") - 1
            if loc >= 0:
                out.append(index[loc])
                
        out = pd.DatetimeIndex(out)
        return pd.DatetimeIndex(out.to_numpy()).unique().sort_values()

