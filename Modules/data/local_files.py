"""
Source de données locale pour le chargement des prix, compositions et exclusions d'un indice.

Ce fichier expose une seule classe (LocalIndexFolderDataSource) qui lit les données
directement depuis un dossier structuré sur le disque, sans passer par une API.
Il gère trois types de données :
- Les prix journaliers : fichiers Excel/CSV annuels (un fichier par année).
- La composition trimestrielle du benchmark : fichier Compo avec blocs horizontaux par trimestre.
- Les exclusions et les secteurs : fichiers optionnels Exclusions.xlsx et sector_mapping.xlsx.

Le remappage des identifiants Bloomberg (ID BB Global, ID BB Company) vers les tickers
courants est géré via un fichier Mapping.xlsx, qui permet de suivre les changements de ticker
au fil du temps (spin-offs, renommages, etc.).

Classes
-------
LocalIndexFolderDataSource :
    Classe principale de chargement.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Literal, Iterable

import pandas as pd


# Type alias pour les extensions de fichiers supportées
FileExt = Literal["xlsx", "xlsm", "xls", "csv"]


@dataclass
class LocalIndexFolderDataSource:
    """
    Classe contenant les méthodes de chargement de données locales pour un indice.
    Basée sur un dossier par indice, sans API Bloomberg. Gère les prix annuels,
    la composition trimestrielle (Compo), les secteurs et les exclusions.

    Structure attendue du dossier:

        base_dir/
            <index_name>/
                2020.xlsx
                2021.xlsx
                2022.xlsx
                2023.xlsx
                2024.xlsx
                ...
                Compo.xlsx          (ou Compo.csv)
                sector.xlsx/csv     (optionnel)

    Formats attendus:
    - Prices (par année):
        * Fichier nommé par année (ex: "2023.xlsx" ou "2023.csv")
        * 1ère ligne: tickers
        * 1ère colonne: dates
        * cellules: prix
        * (Excel: 1ère feuille par défaut, ou prices_sheet si fourni)

    - Compo:
        * Un fichier "Compo" contenant des blocs par trimestre:
            - sur la 1ère ligne: une cellule sur deux est une DATE de début de trimestre
              et la colonne suivante a l'en-tête "Poids" (ou équivalent)
            - sous chaque paire de colonnes:
                col A: ticker
                col B: poids
            - les blocs se suivent horizontalement

        => Sortie attendue : DataFrame
            index = dates de début de trimestre (DatetimeIndex)
            columns = union de tous les tickers
            values = poids (0 si absent)
            (normalisé à 1 si normalize_weights=True)

    - Sector (optionnel):
        * 2 colonnes: [ticker, sector]

    Attributes
    ----------
    base_dir : Path
        Répertoire racine contenant les sous-dossiers par indice.
    index_name : str
        Nom du sous-dossier correspondant à l'indice.
    compo_filename : str or None
        Nom explicite du fichier Compo. Si None, auto-détection dans le dossier.
    sector_filename : str or None
        Nom explicite du fichier secteurs. Si None, auto-détection dans le dossier.
    prices_sheet : str or None
        Nom de la feuille Excel des prix. Si None, prend la première feuille.
    compo_sheet : str or None
        Nom de la feuille Excel de la composition. Si None, prend la première feuille.
    sector_sheet : str or None
        Nom de la feuille Excel des secteurs. Si None, prend la première feuille.
    require_weights : bool
        Si True, lève une erreur si un bloc Compo est vide ou illisible.
    normalize_weights : bool
        Si True, renormalise les poids pour que chaque ligne somme à 1.
    allow_negative_weights : bool
        Si False, lève une erreur si des poids négatifs sont détectés.
    coerce_prices_numeric : bool
        Si True, convertit toutes les colonnes de prix en numérique (NaN si échec).
    drop_missing_prices_rows : bool
        Si True, supprime les lignes où tous les prix sont manquants.
    drop_all_zero_rows : bool
        Si True, supprime les lignes où tous les prix valent 0.
    drop_all_zero_cols : bool
        Si True, supprime les colonnes où tous les prix valent 0.
    final_fillna_prices_with_zero : bool
        Si True, remplace les NaN restants par 0 en sortie finale.
    mapping_filename : str or None
        Nom du fichier de mapping ID BB Global vers ticker.
    compo_weight_header_candidates : tuple
        Liste des noms de colonnes reconnus comme header de poids dans le fichier Compo.

    Methods
    -------
    get_prices(start, end, columns) -> pd.DataFrame :
        Charge et retourne la matrice de prix sur la période demandée.
    get_composition_by_quarter() -> pd.DataFrame :
        Charge et retourne la composition trimestrielle du benchmark.
    get_weights_asof(rebal_dates, method) -> pd.DataFrame :
        Retourne les poids aux dates de rebalancement spécifiées.
    get_sectors() -> pd.Series :
        Retourne le mapping ticker vers secteur.
    get_exclusions() -> list[str] :
        Retourne la liste des tickers à exclure selon le fichier Exclusions.
    """

    base_dir: Path
    index_name: str

    # Fichiers 
    compo_filename: Optional[str] = None
    sector_filename: Optional[str] = None

    # Excel sheets 
    prices_sheet: Optional[str] = None
    compo_sheet: Optional[str] = None
    sector_sheet: Optional[str] = None

    require_weights: bool = True
    normalize_weights: bool = True
    allow_negative_weights: bool = False
    coerce_prices_numeric: bool = True
    drop_missing_prices_rows: bool = True

    # Prices cleaning
    drop_all_zero_rows: bool = True
    drop_all_zero_cols: bool = True
    final_fillna_prices_with_zero: bool = False

    mapping_filename: Optional[str] = "Mapping.xlsx" 

    # Heuristiques Compo
    compo_weight_header_candidates: tuple[str, ...] = ("poids", "Poids", "WEIGHT", "Weight", "w", "W")

    def __post_init__(self) -> None:
        """
        Valide les paramètres après initialisation et construit le chemin du dossier indice.
        Lève FileNotFoundError si le dossier de l'indice n'existe pas sur le disque.
        """

        # Convertit base_dir en Path si passé en string
        self.base_dir = Path(self.base_dir)

        # Construit le chemin complet du dossier de l'indice
        self.index_dir = self.base_dir / self.index_name
        
        # Vérifie que le dossier de l'indice existe avant toute lecture
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Dossier indice introuvable: {self.index_dir}")

    def get_prices(self,start: Optional[pd.Timestamp] = None,end: Optional[pd.Timestamp] = None,columns: Optional[Sequence[str]] = None,) -> pd.DataFrame:
        """
        Charge et retourne la matrice de prix sur la période demandée.

        Charge les fichiers annuels correspondant à la plage [start, end],
        les concatène et applique le filtre de dates et de colonnes si fournis.

        Parameters
        ----------
        start : pd.Timestamp or None
            Date de début (inclusive). Si None, charge depuis le premier fichier disponible.
        end : pd.Timestamp or None
            Date de fin (inclusive). Si None, charge jusqu'au dernier fichier disponible.
        columns : list[str] or None
            Liste de tickers à conserver. Si None, retourne toutes les colonnes.

        Returns
        -------
        pd.DataFrame
            Matrice de prix (index = dates, colonnes = tickers).
        """

        # Charge tous les fichiers annuels nécessaires et les concatène
        px = self._load_prices_by_year(start=start, end=end)

        # Applique le filtre de dates et de colonnes sur le DataFrame complet
        px = self._slice_df(px, start=start, end=end, columns=columns)

        return px

    def get_composition_by_quarter(self) -> pd.DataFrame:
        """
        Charge et retourne la composition trimestrielle du benchmark.

        Returns
        -------
        pd.DataFrame
            DataFrame wide (index = dates trimestrielles, colonnes = tickers, valeurs = poids).
        """
        return self._load_compo_blocks()

    def get_weights_asof(self,rebal_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,method: Literal["ffill", "exact"] = "ffill") -> pd.DataFrame:
        """
        Retourne les poids aux dates de rebalancement spécifiées.

        En mode 'ffill', propage vers l'avant la dernière composition connue.
        En mode 'exact', lève une erreur si une date demandée est absente.

        Parameters
        ----------
        rebal_dates : sequence of pd.Timestamp
            Dates auxquelles extraire les poids du benchmark.
        method : str
            'ffill' pour propagation avant, 'exact' pour correspondance exacte.

        Returns
        -------
        pd.DataFrame
            Poids du benchmark aux dates de rebalancement demandées.
        """
        # Charge la composition trimestrielle et la trie par date
        comp = self.get_composition_by_quarter().sort_index()

        # Convertit les dates de rebalancement en DatetimeIndex sans timezone
        rebal_idx = pd.DatetimeIndex(pd.to_datetime(list(rebal_dates))).tz_localize(None)

        if method == "exact":

            # Vérifie que toutes les dates demandées sont présentes dans la composition
            missing = rebal_idx.difference(comp.index)
            if len(missing) > 0:
                raise ValueError(f"Dates de rebal absentes dans la composition (method='exact'): "f"{missing[:10].tolist()}{'...' if len(missing) > 10 else ''}")
            out = comp.loc[rebal_idx]
        else:

            # Réindexe en ajoutant les dates manquantes, puis propage vers l'avant
            tmp = comp.reindex(comp.index.union(rebal_idx)).sort_index().ffill()
            out = tmp.loc[rebal_idx]

        return self._validate_and_clean_weights(out)

    def get_sectors(self) -> pd.Series:
        """
        Retourne le mapping ticker vers secteur depuis le fichier sector.

        Returns
        -------
        pd.Series
            Série indexée par ticker, valeurs = secteur. Série vide si fichier absent.
        """

        # Cherche le fichier secteur sans lever d'erreur s'il est absent
        path = self._resolve_sector_path(optional=True)

        # Retourne une série vide si aucun fichier secteur n'est trouvé
        if path is None:
            return pd.Series(dtype=object, name="sector")
        
        return self._load_sectors_from_path(path)

    def get_exclusions(self) -> list[str]:
        """
        Charge le fichier Exclusions et retourne la liste des tickers à exclure.

        Tente d'abord de résoudre les tickers via ID_BB_COMPANY et Mapping.xlsx.
        Se rabat sur la colonne TICKER si ID_BB_COMPANY est absent ou non résolu.

        Returns
        -------
        list[str]
            Liste triée des tickers à exclure. Liste vide si fichier absent.

        Raises
        ------
        ValueError
            Si le fichier existe mais ne contient pas la colonne 'EST EXCLUS'.
        """

        # Cherche le fichier Exclusions dans le dossier de l'indice
        path = self._resolve_exclusions_path()

        # Si aucun fichier n'est trouvé, retourne une liste vide sans erreur
        if path is None:
            return []

        # Charge le fichier selon l'extension détectée
        ext = path.suffix.lower()
        df  = pd.read_excel(path) if ext in {".xlsx", ".xlsm", ".xls"} else pd.read_csv(path)

        # Normalise les noms de colonnes en majuscules sans espaces superflus
        df.columns = [str(c).strip().upper() for c in df.columns]

        # Vérifie la présence de la colonne de marquage des exclusions
        if "EST EXCLUS" not in df.columns:
            raise ValueError("Fichier exclusions invalide : colonne 'EST EXCLUS' manquante")

        # Filtre les lignes dont la valeur EST EXCLUS est considérée comme vraie
        mask    = df["EST EXCLUS"].astype(str).str.lower().isin(["1", "true", "yes", "y", "oui"])
        df_excl = df.loc[mask].copy()

        # Si aucun actif n'est marqué comme exclu, retourne une liste vide
        if df_excl.empty:
            return []

        # Tente de résoudre les tickers via ID_BB_COMPANY et Mapping.xlsx
        if "ID_BB_COMPANY" in df_excl.columns:

            # Collecte les ID_BB_COMPANY uniques, en supprimant les valeurs nulles
            ids_excl = set(df_excl["ID_BB_COMPANY"].astype(str).str.strip().replace({"nan": None, "": None}).dropna().unique())
            if ids_excl:
                mapping = self._load_mapping()
                if mapping is not None and "id_bb_company" in mapping.columns:

                    # Prend le ticker le plus récent par ID_BB_COMPANY dans le mapping
                    latest = (mapping.sort_values("date").drop_duplicates(subset=["id_bb_company"], keep="last").set_index("id_bb_company")["ticker"])

                    # Résout chaque ID présent dans le mapping vers son ticker courant
                    resolved = [latest[i] for i in ids_excl if i in latest.index]
                    if resolved:
                        return sorted(set(resolved))

        # Fallback sur la colonne TICKER si ID_BB_COMPANY est absent ou non résolu
        if "TICKER" not in df_excl.columns:
            raise ValueError("Fichier exclusions invalide : ni 'ID_BB_COMPANY' ni 'TICKER' trouvé")

        return sorted(df_excl["TICKER"].astype(str).str.strip().unique())


    # Prices: fichiers annuels
    def _load_prices_by_year(self,start: Optional[pd.Timestamp],end: Optional[pd.Timestamp],) -> pd.DataFrame:
        """
        Charge et concatène les fichiers annuels de prix nécessaires.

        Déduit les années à charger depuis start/end, résout les chemins de fichiers,
        lit chaque fichier annuel et concatène les DataFrames.

        Parameters
        ----------
        start : pd.Timestamp or None
            Date de début pour déduire la première année à charger.
        end : pd.Timestamp or None
            Date de fin pour déduire la dernière année à charger.

        Returns
        -------
        pd.DataFrame
            Matrice de prix concaténée, nettoyée et triée par date.
        """

        # Déduit la liste des années à charger depuis start et end
        years = self._infer_years_needed(start=start, end=end)

        # Résout le chemin de fichier pour chaque année
        paths = self._resolve_year_price_paths(years)

        # Lit chaque fichier annuel et accumule les DataFrames
        frames: list[pd.DataFrame] = []
        for y in years:
            p = paths.get(int(y))
            if p is None:
                raise FileNotFoundError(f"Fichier prices annuel introuvable pour {y} dans {self.index_dir}")
            frames.append(self._read_prices_file(p))

        if not frames:
            raise ValueError("Aucun fichier annuel de prix trouvé.")

        # Concatène tous les fichiers annuels et trie par date
        px = pd.concat(frames, axis=0).sort_index()

        # Supprime les doublons de dates en gardant la dernière valeur par date
        if px.index.has_duplicates:
            px = px.groupby(level=0).last().sort_index()

        # Crée une version avec NaN remplis par 0 pour détecter les lignes/colonnes vides
        px0 = px.fillna(0.0)

        if self.drop_all_zero_rows:

            # Garde uniquement les lignes où au moins un actif a un prix non nul
            keep_rows = (px0 != 0.0).any(axis=1)
            px = px.loc[keep_rows]
            px0 = px0.loc[keep_rows]

        if self.drop_all_zero_cols:

            # Garde uniquement les colonnes où au moins une date a un prix non nul
            keep_cols = (px0 != 0.0).any(axis=0)
            px = px.loc[:, keep_cols]

        # Remplace les NaN restants par 0 si l'option finale est activée
        if self.final_fillna_prices_with_zero:
            px = px.fillna(0.0)

        # Supprime les colonnes entièrement NaN qui auraient pu survivre au filtrage
        px = px.dropna(axis=1, how="all")

        return px


    def _infer_years_needed(self,start: Optional[pd.Timestamp],end: Optional[pd.Timestamp],) -> list[int]:
        """
        Déduit les années à charger depuis start et end.

        Si start et end sont fournis, retourne l'intervalle [start.year, end.year].
        Si l'un ou les deux sont absents, complète avec les années disponibles sur le disque.

        Parameters
        ----------
        start : pd.Timestamp or None
            Date de début. Si None, utilise la plus ancienne année disponible.
        end : pd.Timestamp or None
            Date de fin. Si None, utilise la plus récente année disponible.

        Returns
        -------
        list[int]
            Liste ordonnée des années à charger.
        """

        # Cas où au moins une des bornes est fournie
        if start is not None or end is not None:
            s = pd.to_datetime(start) if start is not None else None
            e = pd.to_datetime(end) if end is not None else None

            if s is None:

                # start absent : prend la plus ancienne année disponible sur disque
                avail = self._list_available_years()
                if not avail:
                    raise FileNotFoundError(f"Aucun fichier annuel (YYYY.*) trouvé dans {self.index_dir}")
                s_year = min(avail)
            else:
                s_year = int(s.year)

            if e is None:

                # end absent : prend la plus récente année disponible sur disque
                avail = self._list_available_years()
                if not avail:
                    raise FileNotFoundError(f"Aucun fichier annuel (YYYY.*) trouvé dans {self.index_dir}")
                e_year = max(avail)
            else:
                e_year = int(e.year)

            if e_year < s_year:
                raise ValueError(f"end.year < start.year: {e_year} < {s_year}")

            return list(range(s_year, e_year + 1))

        # Fallback sans aucune borne : charge toutes les années disponibles sur disque
        years = self._list_available_years()
        if not years:
            raise FileNotFoundError(f"Aucun fichier annuel (YYYY.*) trouvé dans {self.index_dir}")
        return sorted(years)

    def _list_available_years(self) -> list[int]:
        """
        Liste les années pour lesquelles un fichier annuel de prix est présent dans le dossier.

        Détecte les fichiers dont le nom est un nombre à 4 chiffres (ex: 2023.xlsx).

        Returns
        -------
        list[int]
            Liste triée des années disponibles.
        """
        years: list[int] = []
        for p in self.index_dir.iterdir():

            # Ignore les sous-dossiers
            if not p.is_file():
                continue
            stem = p.stem.strip()

            # Retient uniquement les fichiers dont le nom est une année sur 4 chiffres
            if len(stem) == 4 and stem.isdigit() and p.suffix.lower() in {".xlsx", ".xlsm", ".xls", ".csv"}:
                years.append(int(stem))
        return sorted(set(years))

    def _resolve_year_price_paths(self, years: Iterable[int]) -> dict[int, Path]:
        """
        Pour chaque année, résout le chemin du fichier annuel de prix.

        Essaie les extensions xlsx, xlsm, xls, csv dans cet ordre et s'arrête au premier fichier trouvé.

        Parameters
        ----------
        years : iterable of int
            Années pour lesquelles chercher un fichier de prix.

        Returns
        -------
        dict[int, Path]
            Dictionnaire année vers chemin de fichier.
        """
        out: dict[int, Path] = {}
        for y in years:

            # Teste les extensions dans l'ordre de priorité
            candidates = [
                self.index_dir / f"{y}.xlsx",
                self.index_dir / f"{y}.xlsm",
                self.index_dir / f"{y}.xls",
                self.index_dir / f"{y}.csv",
            ]
            for p in candidates:
                if p.exists():
                    out[int(y)] = p
                    # S'arrête dès qu'un fichier est trouvé pour cette année
                    break
        return out

    def _read_prices_file(self, path: Path) -> pd.DataFrame:
        """
        Lit un fichier annuel de prix et retourne un DataFrame nettoyé.

        Gère les formats Excel et CSV, renomme la première colonne en 'Date',
        remape les ID Bloomberg en tickers via Mapping.xlsx, et nettoie les données.

        Parameters
        ----------
        path : Path
            Chemin du fichier annuel de prix à charger.

        Returns
        -------
        pd.DataFrame
            DataFrame de prix (index = dates, colonnes = tickers), nettoyé.
        """

        # Détermine le format de lecture selon l'extension du fichier
        ext = path.suffix.lower()

        if ext in {".xlsx", ".xlsm", ".xls"}:

            # Utilise la feuille configurée ou la première feuille par défaut
            sheet = self.prices_sheet if self.prices_sheet is not None else 0
            df = pd.read_excel(path, sheet_name=sheet, header=0)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Format prices non supporté: {path}")

        if df.shape[1] < 2:
            raise ValueError(f"Fichier prices invalide (pas assez de colonnes): {path}")

        # Renomme la première colonne en 'Date' quel que soit son nom d'origine
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})

        # Convertit la colonne de dates en Timestamp sans timezone et l'utilise comme index
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        # Nettoie les noms de colonnes en supprimant les espaces superflus
        df.columns = [str(c).strip() for c in df.columns]

        # Remplace les ID BB Global par leurs tickers courants via Mapping.xlsx
        year = int(path.stem)
        df = self._remap_bbg_to_ticker(df, year=year)

        # Convertit toutes les colonnes de prix en numérique (NaN si valeur non convertible)
        if self.coerce_prices_numeric:
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Crée une version avec NaN remplis par 0 pour détecter les lignes/colonnes vides
        df0 = df.fillna(0.0)

        if self.drop_all_zero_rows:

            # Supprime les lignes où tous les actifs valent 0 (jours non ouvrés)
            keep_rows = (df0 != 0.0).any(axis=1)
            df = df.loc[keep_rows]
            df0 = df0.loc[keep_rows]

        if self.drop_all_zero_cols:

            # Supprime les colonnes où tous les jours valent 0 (actifs jamais présents ou colonne vide)
            keep_cols = (df0 != 0.0).any(axis=0)
            df = df.loc[:, keep_cols]

        # Remplace les NaN restants par 0 si l'option finale est activée
        if self.final_fillna_prices_with_zero:
            df = df.fillna(0.0)

        # Supprime les doublons de dates en gardant la dernière valeur par date
        if df.index.has_duplicates:
            df = df.groupby(level=0).last().sort_index()

        return df


    def _load_compo_blocks(self) -> pd.DataFrame:
        """
        Charge les blocs de composition depuis le fichier Compo et retourne un DataFrame wide.

        Le fichier Compo est structuré horizontalement : chaque trimestre occupe 3 colonnes
        (ticker, id_bb_global, poids). La première ligne contient les dates de début de trimestre.
        L'id_bb_global est remappé vers le ticker via Mapping.xlsx si disponible.

        Returns
        -------
        pd.DataFrame
            DataFrame wide (index = dates trimestrielles, colonnes = tickers, valeurs = poids).

        Raises
        ------
        ValueError
            Si le fichier est vide ou si aucun bloc valide n'est détecté.
        """
        # Résout le chemin du fichier Compo et détermine son format
        path = self._resolve_compo_path()
        ext = path.suffix.lower()

        if ext in {".xlsx", ".xlsm", ".xls"}:

            # Utilise la feuille configurée ou la première feuille par défaut
            sheet = self.compo_sheet if self.compo_sheet is not None else 0

            # Charge sans header car la première ligne contient les dates
            raw = pd.read_excel(path, sheet_name=sheet, header=None)

        elif ext == ".csv":
            raw = pd.read_csv(path, header=None)
        else:
            raise ValueError(f"Format Compo non supporté: {path}")

        if raw.empty:
            raise ValueError(f"Fichier Compo vide: {path}")

        # La ligne 0 contient les dates de début de trimestre, une tous les 3 colonnes
        header = raw.iloc[0].tolist()
        ncols = raw.shape[1]

        # Charge le mapping une seule fois pour tous les blocs
        mapping = self._load_mapping()

        blocks: list[tuple[pd.Timestamp, pd.Series]] = []

        # Parcourt les blocs horizontaux, chaque bloc faisant 3 colonnes de large
        j = 0
        while j + 2 < ncols:

            # Tente de lire la date de début du bloc courant
            dt_cell = header[j] if j < len(header) else None
            dt = self._coerce_to_date(dt_cell)

            # Si la cellule n'est pas une date valide, avance d'une colonne
            if dt is None:
                j += 1
                continue

            # Extrait les 3 colonnes du bloc : tickers, id_bb_global, poids
            tickers     = raw.iloc[1:, j].astype(str).str.strip()
            id_bb_cols  = raw.iloc[1:, j + 1].astype(str).str.strip()
            weights     = raw.iloc[1:, j + 2]

            df_block = pd.DataFrame({"ticker": tickers.values, "id_bb_global": id_bb_cols.values, "weight": weights.values,})

            # Supprime les lignes avec des tickers vides ou invalides
            df_block = df_block[df_block["ticker"].notna() & (df_block["ticker"] != "") & (df_block["ticker"] != "nan")]

            # Convertit les poids en numérique et supprime les lignes avec poids invalide
            df_block["weight"] = pd.to_numeric(df_block["weight"], errors="coerce")
            df_block = df_block.dropna(subset=["weight"])

            if df_block.empty:
                if self.require_weights:
                    raise ValueError(f"Bloc Compo vide ou poids non lisibles pour date={dt.date()} dans {path}")
                j += 3
                continue

            # Remplace les id_bb_global par leurs tickers courants via le mapping
            if mapping is not None:

                # Filtre le mapping sur les dates antérieures ou égales à la date du bloc
                m = mapping[mapping["date"] <= dt]

                if not m.empty:
                    # Prend le ticker le plus récent par id_bb_global pour la date du bloc
                    latest = (m.sort_values("date").drop_duplicates(subset=["id_bb_global"], keep="last").set_index("id_bb_global")["ticker"])

                    # Applique le remapping et conserve le ticker original si absent du mapping
                    df_block["ticker"] = df_block["id_bb_global"].map(latest).fillna(df_block["ticker"])

            # Supprime les lignes dont le ticker est resté vide après remapping
            df_block = df_block[df_block["ticker"].notna() & (df_block["ticker"] != "")]

            # Agrège les poids par ticker en cas de doublons dans le bloc
            s = df_block.groupby("ticker")["weight"].sum()
            s.name = dt
            blocks.append((dt, s))

            # Avance au bloc suivant
            j += 3

        if not blocks:
            raise ValueError(f"Aucun bloc 'date + (ticker, id_bb_global, poids)' détecté dans Compo: {path}. "f"Vérifie que la ligne 1 a bien des dates en colonnes 1, 4, 7, ...")

        # Collecte l'union de tous les tickers présents dans tous les blocs
        all_tickers = sorted(set().union(*[set(s.index) for _, s in blocks]))

        # Construit le DatetimeIndex des dates de début de trimestre
        idx = pd.DatetimeIndex([dt for dt, _ in blocks]).tz_localize(None)

        # Initialise le DataFrame wide avec des zéros pour les actifs absents d'un trimestre
        W = pd.DataFrame(0.0, index=idx, columns=all_tickers, dtype=float)
        for dt, s in blocks:
            W.loc[pd.Timestamp(dt).tz_localize(None), s.index] = s.astype(float).values

        W = W.sort_index()

        # Supprime les doublons de dates en sommant les poids si nécessaire
        if W.index.has_duplicates:
            W = W.groupby(level=0).sum().sort_index()

        return self._validate_and_clean_weights(W)

    @staticmethod
    def _coerce_to_date(x) -> Optional[pd.Timestamp]:
        """
        Tente de convertir x en pd.Timestamp normalisé. Retourne None si échec.

        Parameters
        ----------
        x : any
            Valeur à convertir (chaîne, entier Excel, datetime, etc.).

        Returns
        -------
        pd.Timestamp or None
            Timestamp normalisé ou None si la conversion échoue.
        """
        if x is None:
            return None
        try:

            # Tente la conversion avec pandas et retourne None si le résultat est NaT
            dt = pd.to_datetime(x, errors="coerce")

            if pd.isna(dt):
                return None
            
            # Normalise à minuit pour éviter les problèmes d'heure dans les comparaisons
            return pd.Timestamp(dt).normalize()
        except Exception:
            return None

    def _looks_like_weight_header(self, s: str) -> bool:
        """
        Vérifie si une chaîne correspond à un header de poids connu.

        Compare sans tenir compte de la casse avec les candidats définis dans
        compo_weight_header_candidates.

        Parameters
        ----------
        s : str
            Chaîne à tester.

        Returns
        -------
        bool
            True si la chaîne correspond à un header de poids connu.
        """
        s0 = s.strip()

        # Compare sans tenir compte de la casse avec chaque candidat connu
        for cand in self.compo_weight_header_candidates:
            if s0.lower() == str(cand).lower():
                return True
        return False

    def _load_sectors_from_path(self, path: Path) -> pd.Series:
        """
        Charge les secteurs depuis le fichier sector et retourne une Series.

        La première colonne est traitée comme les tickers, la seconde comme les secteurs.

        Parameters
        ----------
        path : Path
            Chemin du fichier secteur à charger.

        Returns
        -------
        pd.Series
            Série indexée par ticker, valeurs = secteur.

        Raises
        ------
        ValueError
            Si le fichier est vide ou contient moins de 2 colonnes.
        """

        # Détermine le format de lecture selon l'extension du fichier
        ext = path.suffix.lower()

        if ext in {".xlsx", ".xlsm", ".xls"}:

            # Utilise la feuille configurée ou la première feuille par défaut
            sheet = self.sector_sheet if self.sector_sheet is not None else 0
            df = pd.read_excel(path, sheet_name=sheet, header=0)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Format sector non supporté: {path}")

        if df.empty or df.shape[1] < 2:
            raise ValueError(f"Fichier sector invalide: {path} (attendu >=2 colonnes).")

        # La première colonne est le ticker, la deuxième est le secteur
        t_col = df.columns[0]
        s_col = df.columns[1]

        # Ne conserve que les deux colonnes utiles
        out = df[[t_col, s_col]].copy()

        # Nettoie les valeurs en supprimant les espaces superflus
        out[t_col] = out[t_col].astype(str).str.strip()
        out[s_col] = out[s_col].astype(str).str.strip()

        # Supprime les lignes avec des valeurs manquantes ou vides
        out = out.dropna(subset=[t_col, s_col])
        out = out[(out[t_col] != "") & (out[s_col] != "")]

        # En cas de ticker dupliqué, garde la dernière occurrence
        out = out.drop_duplicates(subset=[t_col], keep="last")

        return pd.Series(out[s_col].values, index=out[t_col].values, name="sector")


    def _validate_and_clean_weights(self, w: pd.DataFrame) -> pd.DataFrame:
        """
        Valide et nettoie un DataFrame de poids chargé depuis Compo ou get_weights_asof.

        Vérifie l'absence de poids négatifs si configuré, normalise les poids à 1 par ligne,
        et fusionne les colonnes dupliquées.

        Parameters
        ----------
        w : pd.DataFrame
            DataFrame de poids bruts à valider et nettoyer.

        Returns
        -------
        pd.DataFrame
            DataFrame de poids nettoyé et normalisé.

        Raises
        ------
        ValueError
            Si le DataFrame est vide, contient des poids négatifs non autorisés,
            ou si certaines dates ont une somme de poids nulle.
        """

        # Supprime les colonnes entièrement NaN et fait une copie pour éviter les effets de bord
        w = w.copy().dropna(axis=1, how="all")
        if w.empty:
            raise ValueError("Composition vide après nettoyage.")

        # Remplace les NaN restants par 0 pour les calculs de somme et de normalisation
        w = w.fillna(0.0)

        # Détecte et signale les poids négatifs si l'option est désactivée
        if not self.allow_negative_weights and (w < -1e-12).any().any():
            bad = (w < -1e-12).stack()
            examples = bad[bad].index.tolist()[:10]
            raise ValueError(
                "Poids négatifs détectés alors que allow_negative_weights=False. " f"Exemples: {examples}")

        if self.normalize_weights:

            # Calcule la somme des poids par date pour la normalisation
            row_sum = w.sum(axis=1)

            # Détecte les dates avec une somme nulle où la normalisation est impossible
            zero_rows = row_sum.abs() < 1e-16
            if zero_rows.any():
                bad_dates = w.index[zero_rows].tolist()[:10]
                raise ValueError("Certaines dates ont une somme de poids nulle (impossible de normaliser). " f"Exemples: {bad_dates}")
            
            # Normalise chaque ligne pour que la somme des poids vaille 1
            w = w.div(row_sum, axis=0)

        if w.columns.has_duplicates:

            # Fusionne les colonnes dupliquées en sommant leurs poids
            w = w.groupby(level=0, axis=1).sum()

            if self.normalize_weights:
                # Renormalise après fusion des doublons
                w = w.div(w.sum(axis=1), axis=0)

        return w

    def _resolve_compo_path(self) -> Path:
        """
        Résout le chemin du fichier Compo dans le dossier de l'indice.

        Si compo_filename est explicitement fourni, l'utilise directement.
        Sinon, teste les noms standards (Compo.xlsx, compo.xlsx, etc.).

        Returns
        -------
        Path
            Chemin du fichier Compo trouvé.

        Raises
        ------
        FileNotFoundError
            Si aucun fichier Compo n'est trouvé.
        """
        # Si un nom de fichier explicite est fourni, l'utilise en priorité
        if self.compo_filename:
            p = self.index_dir / self.compo_filename
            if p.exists():
                return p
            raise FileNotFoundError(f"compo_filename introuvable: {p}")

        # Teste les noms de fichiers standards dans l'ordre de priorité
        candidates = [
            self.index_dir / "Compo.xlsx",
            self.index_dir / "Compo.xlsm",
            self.index_dir / "Compo.xls",
            self.index_dir / "Compo.csv",
            self.index_dir / "compo.xlsx",
            self.index_dir / "compo.xlsm",
            self.index_dir / "compo.xls",
            self.index_dir / "compo.csv",
        ]
        
        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(f"Impossible de trouver le fichier Compo dans {self.index_dir}.")

    def _resolve_sector_path(self, optional: bool = False) -> Optional[Path]:
        """
        Résout le chemin du fichier secteur dans le dossier de l'indice.

        Si sector_filename est explicitement fourni, l'utilise directement.
        Sinon, teste les noms standards (sector_mapping.xlsx, etc.).

        Parameters
        ----------
        optional : bool
            Si True, retourne None si le fichier est absent au lieu de lever une erreur.

        Returns
        -------
        Path or None
            Chemin du fichier secteur, ou None si absent et optional=True.
        """

        # Si un nom de fichier explicite est fourni, l'utilise en priorité
        if self.sector_filename:
            p = self.index_dir / self.sector_filename
            if p.exists():
                return p
            if optional:
                return None
            raise FileNotFoundError(f"sector_filename introuvable: {p}")

        # Teste les noms de fichiers standards dans l'ordre de priorité
        candidates = [
            self.index_dir / "sector_mapping.xlsx",
            self.index_dir / "sector_mapping.xlsm",
            self.index_dir / "sector_mapping.xls",
            self.index_dir / "sector_mapping.csv",
            self.index_dir / "sectors_mapping.xlsx",
            self.index_dir / "sectors_mapping.xlsm",
            self.index_dir / "sectors_mapping.xls",
            self.index_dir / "sectors_mapping.csv",
        ]
        for p in candidates:
            if p.exists():
                return p

        return None if optional else None  


    def _resolve_exclusions_path(self) -> Optional[Path]:
        """
        Cherche un fichier Exclusions (xlsx/csv) dans le dossier de l'indice.

        Format attendu : colonnes [TICKER, ISIN, ID_BB_COMPANY, EST EXCLUS, ...]

        Returns
        -------
        Path or None
            Chemin du fichier Exclusions trouvé, ou None si absent.
        """
        # Teste les noms de fichiers standards dans l'ordre de priorité
        candidates = [
            self.index_dir / "Exclusions.xlsx",
            self.index_dir / "Exclusions.xlsm",
            self.index_dir / "Exclusions.xls",
            self.index_dir / "Exclusions.csv",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None
    

    def _load_mapping(self) -> Optional[pd.DataFrame]:
        """
        Charge Mapping.xlsx et retourne le DataFrame de correspondance ID BB vers ticker.

        Cherche le fichier d'abord dans le dossier de l'indice, puis dans base_dir.
        Retourne None si mapping_filename est None ou si le fichier est introuvable.

        Format attendu : col 0 = Date rebal, col 1 = Ticker, col 2 = ID BB Global.

        Returns
        -------
        pd.DataFrame or None
            DataFrame avec colonnes [date, ticker, id_bb_global, id_bb_company],
            trié par date. None si le fichier est absent ou désactivé.

        Raises
        ------
        ValueError
            Si le fichier existe mais contient moins de 3 colonnes.
        """
        # Si mapping_filename est None, la fonctionnalité de remapping est désactivée
        if self.mapping_filename is None:
            return None

        # Cherche d'abord dans le dossier de l'indice, puis dans le répertoire racine
        path = self.index_dir / self.mapping_filename
        if not path.exists():
            path = self.base_dir / self.mapping_filename
        if not path.exists():
            return None

        # Charge le fichier selon son extension
        ext = path.suffix.lower()
        df = pd.read_excel(path, header=0) if ext in {".xlsx", ".xlsm", ".xls"} else pd.read_csv(path)

        if df.shape[1] < 3:
            raise ValueError(f"Mapping invalide : attendu ≥3 colonnes [Date, Ticker, ID_BB_Global], "f"trouvé {df.shape[1]} dans {path}.")

        # Renomme les 4 premières colonnes avec des noms standardisés
        base_cols = ["date", "ticker", "id_bb_global", "id_bb_company"]
        df.columns = base_cols[:df.shape[1]] + list(df.columns[df.shape[1]:])

        # Convertit et nettoie chaque colonne clé
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["id_bb_global"]  = df["id_bb_global"].astype(str).str.strip()

        # Nettoie id_bb_company uniquement si la colonne est présente
        if "id_bb_company" in df.columns:
            df["id_bb_company"] = df["id_bb_company"].astype(str).str.strip()

        # Supprime les lignes sans date ou sans ticker valide et trie par date
        df = df.dropna(subset=["date", "ticker"]).sort_values("date")
        
        return df
    

    def _remap_bbg_to_ticker(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Renomme les colonnes ID BB Global en tickers via Mapping.xlsx.

        Pour chaque ID, prend le ticker à la date de rebalancement la plus récente
        inférieure ou égale au 31 décembre de l'année. Les IDs sans correspondance
        sont conservés tels quels et seront supprimés si leur colonne est entièrement nulle.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de prix dont les colonnes sont des ID BB Global.
        year : int
            Année du fichier de prix (sert de cutoff pour le mapping).

        Returns
        -------
        pd.DataFrame
            DataFrame avec les colonnes renommées en tickers.
        """

        # Si pas de mapping disponible, retourne le DataFrame sans modification
        mapping = self._load_mapping()
        if mapping is None:
            return df

        # Définit le cutoff au 31 décembre de l'année du fichier
        cutoff = pd.Timestamp(f"{year}-12-31")

        # Filtre le mapping sur les entrées antérieures ou égales au cutoff
        m_year = mapping[mapping["date"] <= cutoff]

        # Prend le ticker le plus récent par ID pour cette année
        latest = (m_year.sort_values("date").drop_duplicates(subset=["id_bb_global"], keep="last").set_index("id_bb_global")["ticker"])

        # Fallback global pour les IDs hors plage (ex: fichier très ancien sans entrée dans l'année)
        fallback = (mapping.drop_duplicates(subset=["id_bb_global"], keep="last").set_index("id_bb_global")["ticker"])

        # Construit le dictionnaire de renommage : latest en priorité, fallback sinon, ID original en dernier recours
        rename_map = {col: latest.get(col) or fallback.get(col) or col for col in df.columns}
        return df.rename(columns=rename_map)
    
    
    # Utilities
    @staticmethod
    def _slice_df(df: pd.DataFrame,start: Optional[pd.Timestamp],end: Optional[pd.Timestamp],columns: Optional[Sequence[str]],) -> pd.DataFrame:
        """
        Filtre un DataFrame selon les bornes de dates et la liste de colonnes demandées.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame à filtrer.
        start : pd.Timestamp or None
            Date de début inclusive. Si None, pas de filtre sur le début.
        end : pd.Timestamp or None
            Date de fin inclusive. Si None, pas de filtre sur la fin.
        columns : list[str] or None
            Colonnes à conserver. Si None, conserve toutes les colonnes.

        Returns
        -------
        pd.DataFrame
            DataFrame filtré.

        Raises
        ------
        KeyError
            Si des colonnes demandées sont absentes du DataFrame.
        """
        out = df

        # Applique le filtre de date de début si fourni
        if start is not None:
            out = out.loc[out.index >= pd.to_datetime(start)]

        # Applique le filtre de date de fin si fourni
        if end is not None:
            out = out.loc[out.index <= pd.to_datetime(end)]

        if columns is not None:
            # Ne garde que les colonnes demandées qui existent dans le DataFrame
            cols = [c for c in columns if c in out.columns]
            missing = set(columns) - set(cols)
            
            # Signale les colonnes demandées mais absentes du DataFrame
            if missing:
                raise KeyError(f"Colonnes demandées absentes: {sorted(missing)}")
            out = out[cols]
        return out