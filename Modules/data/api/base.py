# -*- coding: utf-8 -*-
"""
Définit les contrats d'API de prix.


Contenu
-------
- `PriceAPI` : interface minimale (historique OHLCV, dernier prix, actions corporate).
- `BatchPriceAPI` : protocole pour chargements en lot.


Permet de changer de fournisseur (Yahoo, Bloomberg, CSV/Parquet) sans toucher
au code métier qui consomme l'interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Iterable
import pandas as pd
from ..models import PriceRequest, CorporateActionRequest


class PriceAPI(ABC):
    """Contrat minimal pour une API de prix."""


@abstractmethod
def get_price_history(self, req: PriceRequest) -> pd.DataFrame:
    """Retourne OHLCV indexé par date. Colonnes: Open, High, Low, Close, Adj Close, Volume."""
    raise NotImplementedError


@abstractmethod
def get_latest_price(self, ticker: str) -> float:
    """Dernier prix connu (float)."""
    raise NotImplementedError


@abstractmethod
def get_dividends(self, req: CorporateActionRequest) -> pd.DataFrame:
    """Dividendes historiques (DataFrame date→`dividend`)."""
    raise NotImplementedError


@abstractmethod
def get_splits(self, req: CorporateActionRequest) -> pd.DataFrame:
    """Splits historiques (DataFrame date→`split`)."""
    raise NotImplementedError


class BatchPriceAPI(Protocol):
    """Contrat optionnel pour récupérer plusieurs séries en lot."""
    def get_prices_batch(self, reqs: Iterable[PriceRequest]) -> dict[str, pd.DataFrame]:
        raise NotImplementedError
    