# -*- coding: utf-8 -*-
"""
Structures de requêtes typées utilisées par la couche data pour l'extraction de données de marché.

Ce fichier définit les dataclasses immuables passées aux providers de données
(Yahoo Finance, Bloomberg, etc.) pour paramétrer les requêtes de prix et d'actions corporate.
Il ne contient aucune logique de chargement, uniquement les contrats de données.

Classes
-------
PriceRequest :
    Paramètres d'une requête de prix : ticker, période, granularité et ajustement des dividendes.
CorporateActionRequest :
    Paramètres d'une requête d'actions corporate : ticker et période.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

# Type alias limitant les valeurs acceptées pour la granularité temporelle des prix
PriceInterval = Literal["1d", "1wk", "1mo", "1h", "5m", "1m"]


@dataclass(frozen=True)
class PriceRequest:
    """
    Classe contenant les paramètres d'une requête de prix.

    Immuable (frozen=True) : une fois créée, la requête ne peut pas être modifiée.
    Passée telle quelle au provider de données pour extraire la série de prix demandée.

    Attributes
    ----------
    ticker : str
        Symbole de l'actif à charger (ex: 'AAPL', '^GSPC', 'EURUSD=X').
    start : str or None
        Date de début au format ISO 'YYYY-MM-DD'. Si None, laisse le provider décider.
    end : str or None
        Date de fin au format ISO 'YYYY-MM-DD'. Si None, laisse le provider décider.
    interval : PriceInterval
        Granularité temporelle des données (quotidien par défaut).
    adjusted : bool
        Si True, remplace le prix de clôture brut par le prix ajusté des dividendes et splits.
    """

    ticker: str
    start: Optional[str] = None # "YYYY-mm-dd"
    end: Optional[str] = None
    interval: PriceInterval = "1d"
    adjusted: bool = True


@dataclass(frozen=True)
class CorporateActionRequest:
    """
    Classe contenant les paramètres d'une requête d'actions corporate.

    Couvre les dividendes et les splits sur la période demandée.
    Immuable (frozen=True) : une fois créée, la requête ne peut pas être modifiée.

    Attributes
    ----------
    ticker : str
        Symbole de l'actif pour lequel extraire les actions corporate.
    start : str or None
        Date de début au format ISO 'YYYY-MM-DD'. Si None, laisse le provider décider.
    end : str or None
        Date de fin au format ISO 'YYYY-MM-DD'. Si None, laisse le provider décider.
    """

    ticker: str
    start: Optional[str] = None
    end: Optional[str] = None