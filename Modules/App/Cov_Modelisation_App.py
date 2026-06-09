# -*- coding: utf-8 -*-
"""
Application Tkinter principale pour l'évaluation des modèles de covariance.

Ce fichier constitue le point d'entrée de l'interface graphique QuantPortfolioEngine.
Il expose deux onglets : un backtest économique classique (EstimationTab) et un
Monte Carlo multi-modèles (MonteCarloTab) avec branches économique et statistique.

Les workers de simulation (_scenario_worker) sont définis au niveau module pour
être sérialisables par joblib.Parallel (requis par le backend loky).

Classes
-------
MCModelSelector :
    Sélecteur de modèle simplifié (un seul modèle, legacy).
EstimationTab :
    Onglet de backtest économique : chargement des données, configuration des modèles,
    lancement du backtest et génération du rapport PDF.
MonteCarloTab :
    Onglet Monte Carlo : grille de scénarios économiques multi-modèles et
    simulation statistique DGP avec export Excel checkpoint.
CovModelisationApp :
    Fenêtre principale Tkinter qui instancie les deux onglets et la console de log.
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

# Résolution de la racine du projet pour les imports absolus
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Modules.App.tkinter_app.widget import (LabeledEntry, LabeledCombobox, LabeledCheckbox, LogConsole, SectionFrame, ListEditor, RunButton,)
from Modules.App.tkinter_app.model_builder import ModelListPanel, COMPUTE_MODE_DEFAULT




def _filter_universe(universe_returns, exclude_frac: float, seed: int, forced_exclusions: set = None):
    """
    Filtre l'univers investissable en excluant des actifs forcés puis un tirage aléatoire.

    Les forced_exclusions (ex : indices benchmark pour ALM Classic) sont retirés en premier,
    avant d'appliquer le tirage aléatoire sur les actifs restants selon exclude_frac.

    Parameters
    ----------
    universe_returns : pd.DataFrame
        Rendements de l'univers complet (colonnes = tickers).
    exclude_frac : float
        Fraction d'actifs à exclure aléatoirement parmi les tickers non forcés.
    seed : int
        Graine aléatoire pour la reproductibilité du tirage.
    forced_exclusions : set or None
        Ensemble de tickers toujours exclus (indépendamment du tirage aléatoire).

    Returns
    -------
    pd.DataFrame
        Sous-ensemble des rendements après exclusions.

    Raises
    ------
    ValueError
        Si tous les actifs sont exclus après filtrage.
    """
    import numpy as np

    # Convertit en set vide si None pour simplifier les opérations suivantes
    forced_exclusions = set(forced_exclusions or [])

    # Retire les exclusions forcées avant le tirage aléatoire
    tickers = [t for t in universe_returns.columns if t not in forced_exclusions]

    # Calcule le nombre d'actifs à exclure aléatoirement
    k_excl = int(np.floor(exclude_frac * len(tickers)))

    if k_excl <= 0:

        # Aucune exclusion aléatoire demandée : garde tous les tickers non forcés
        kept = tickers
    else:
        rng  = __import__('numpy').random.default_rng(seed)

        # Tire k_excl tickers à exclure sans remise
        excl = set(rng.choice(tickers, size=k_excl, replace=False))

        # Garde les tickers non tirés
        kept = [t for t in tickers if t not in excl]

    if not kept:
        raise ValueError("Tous les actifs exclus — diminuer exclude_frac.")

    return universe_returns[kept].copy()


def _snap_to_index(dates, trading_index, snap="left"):
    """
    Projette une liste de dates sur l'index de trading le plus proche.

    Parameters
    ----------
    dates : iterable of date-like
        Dates à projeter sur l'index de trading.
    trading_index : array-like of date-like
        Index de dates de trading de référence.
    snap : str
        Direction de projection : 'left' pour la date précédente, 'right' pour la suivante.

    Returns
    -------
    pd.DatetimeIndex
        Dates projetées, dédupliquées et triées.
    """
    import pandas as pd

    # Construit l'index de trading trié et dédupliqué
    idx = __import__('pandas').DatetimeIndex(trading_index).sort_values().unique()
    out = []

    for d in pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None):
        if snap == "left":

            # Cherche la dernière date de trading inférieure ou égale à d
            pos = idx.searchsorted(d, side="right") - 1
        else:

            # Cherche la première date de trading supérieure ou égale à d
            pos = idx.searchsorted(d, side="left")

        # N'ajoute que si la position est valide dans l'index
        if 0 <= pos < len(idx):
            out.append(idx[pos])

    return __import__('pandas').DatetimeIndex(out).unique().sort_values()


def _scenario_worker(
    spec,
    rolling: int,
    exclude_frac: float,
    seed: int,
    rebal_freq: str,
    data_freq: str,
    key,
    all_returns,
    universe_returns,
    bench_weights,
    prices,
    rebal_dates_port,
    rebal_dates_bench,
    common_start,
    use_memmap: bool = False,
    memmap_path=None,
    memmap_shape=None,
    path_index=None,
    path_names=None,
    forced_exclusions=None,
):
    """
    Exécute un scénario Monte Carlo économique complet pour un ModelSpec et un seed donné.

    Filtre l'univers, aligne les dates de rebalancement, instancie le CovarianceProvider
    (mode memmap ou rebal), et lance le backtest via ModelEvaluator.full_evaluation().
    Définie au niveau module pour être picklable par joblib.Parallel (backend loky).

    Parameters
    ----------
    spec : ModelSpec
        Spécification du modèle (name, cov_cfg, optimizer_name).
    rolling : int
        Taille de la fenêtre rolling en nombre de périodes.
    exclude_frac : float
        Fraction d'actifs à exclure aléatoirement.
    seed : int
        Graine aléatoire pour le tirage de l'univers.
    rebal_freq : str
        Fréquence de rebalancement du portefeuille ('M', 'Q', 'W', etc.).
    data_freq : str
        Fréquence des données de covariance ('daily' ou 'weekly').
    key : ScenarioKey
        Clé du scénario pour l'export (checkpoint).
    all_returns : pd.DataFrame
        Rendements complets de tous les actifs.
    universe_returns : pd.DataFrame
        Rendements de l'univers avant filtrage.
    bench_weights : pd.DataFrame
        Poids du benchmark à chaque date.
    prices : pd.DataFrame
        Prix de l'univers (pour l'inventaire portefeuille).
    rebal_dates_port : pd.DatetimeIndex
        Dates de rebalancement du portefeuille.
    rebal_dates_bench : pd.DatetimeIndex
        Dates de rebalancement du benchmark.
    common_start : pd.Timestamp
        Date minimale commune (après burn-in de la fenêtre rolling).
    use_memmap : bool
        Si True, utilise un CovarianceProvider en mode memmap (path précomputé).
    memmap_path : Path or None
        Chemin du fichier memmap (requis si use_memmap=True).
    memmap_shape : tuple or None
        Dimensions du memmap (requis si use_memmap=True).
    path_index : array-like or None
        Index de dates du path memmap (requis si use_memmap=True).
    path_names : list or None
        Noms d'actifs du path memmap (requis si use_memmap=True).
    forced_exclusions : set or None
        Tickers toujours exclus de l'univers (ex : indices benchmark ALM Classic).

    Returns
    -------
    tuple
        (eval_res, key) où eval_res est le dict retourné par full_evaluation().
    """

    import gc
    import numpy as np
    from Modules.portfolio_management.backtesting.covariance_provider import CovarianceProvider
    from Modules.Financial_engineering.statistics.multivariate_vol_estimation import ModelEvaluator, DataFrequency
    from Modules.portfolio_management.backtesting.rebalancing import RebalanceSchedule

    # Facteur d'annualisation selon la fréquence des données de covariance
    ann_factor = DataFrequency(data_freq).ann_factor

    # Filtre l'univers : exclusions forcées puis tirage aléatoire selon exclude_frac
    univ_inv = _filter_universe(universe_returns, exclude_frac, seed, forced_exclusions)
    prices_inv = prices.reindex(index=univ_inv.index, columns=univ_inv.columns)

    # Le backtest tourne toujours en daily : on utilise les séries originales
    all_returns_eff  = all_returns
    bench_weights_eff  = bench_weights
    prices_eff  = prices_inv
    _rebal_dates_bench = rebal_dates_bench

    if rebal_freq == "Q":

        # Rebalancement trimestriel : utilise les dates du benchmark comme ancrage
        _rebal_dates_port = rebal_dates_port
    else:

        # Autre fréquence : génère les dates depuis l'index de l'univers filtré
        _rebal_dates_port = RebalanceSchedule(freq=rebal_freq).rebalance_dates(univ_inv.index)

    # Tronque les dates de rebalancement au-delà de common_start (après burn-in)
    _rebal_dates_port = _rebal_dates_port[_rebal_dates_port >= common_start]

    if use_memmap and path_index is not None:

        # En mode memmap, aligne aussi sur la première date disponible du path
        path_start        = pd.Timestamp(path_index[0])
        _rebal_dates_port  = _rebal_dates_port[_rebal_dates_port >= path_start]
        _rebal_dates_bench = _rebal_dates_bench[_rebal_dates_bench >= path_start]

    if len(_rebal_dates_port) == 0:
        raise ValueError(
            f"Aucune date de rebal après common_start={common_start.date()} "
            f"pour {spec.name} | rebal_freq={rebal_freq} | data_freq={data_freq}"
        )

    if use_memmap:

        # Mode path : lit la covariance depuis le memmap précomputé (lecture seule)
        provider = CovarianceProvider.from_memmap(cfg=spec.cov_cfg, memmap_path=memmap_path, memmap_shape=memmap_shape, path_index=path_index, path_names=path_names,)
    else:
        # Mode rebal : calcule la covariance à chaque date de rebalancement
        provider = CovarianceProvider(cfg=spec.cov_cfg)

    try:
        result = ModelEvaluator.full_evaluation(
            all_returns = all_returns_eff,
            universe_returns  = univ_inv,
            benchmark_weights  = bench_weights_eff,
            prices_inv      = prices_eff,
            rebal_dates_port    = _rebal_dates_port,
            rebal_dates_bench    = _rebal_dates_bench,
            model_specs        = [spec],
            engine_kwargs     = {"rebal_freq": rebal_freq, "verbose": False},
            ann_factor      = ann_factor,
            precomputed_cov_provider = provider,
        )
    finally:
        # Ferme le handle memmap en lecture seule et libère la mémoire
        if use_memmap:
            provider.close_readonly()
        gc.collect()

    return result, key


# Paramètres fixes par type de modèle pour le Monte Carlo (affichés dans MCModelSelector)
MC_MODEL_PARAMS = {
    "Rolling": [
        {"key": "rolling_ddof", "label": "ddof", "default": "1"},
    ],
    "EWMA": [
        {"key": "ewma_lambda", "label": "Lambda",                  "default": "0.94"},
        {"key": "ewma_init",   "label": "Init (scov / diag)",      "default": "scov"},
        {"key": "tune_lambda", "label": "Tune lambda (True/False)", "default": "False"},
    ],
    "LW_2004": [
        {"key": "use_package", "label": "Package sklearn (True/False)", "default": "False"},
        {"key": "lw_demean",   "label": "Demean (True/False)",          "default": "True"},
        {"key": "lw_ddof",     "label": "ddof",                         "default": "0"},
    ],
    "ANLS_2020": [
        {"key": "lw_demean",  "label": "Demean (True/False)", "default": "True"},
        {"key": "lw_ddof",    "label": "ddof",                "default": "0"},
        {"key": "chunk_size", "label": "Chunk size",          "default": "1024"},
    ],
    "QIS": [
        {"key": "lw_demean",  "label": "Demean (True/False)", "default": "True"},
        {"key": "lw_ddof",    "label": "ddof",                "default": "0"},
        {"key": "chunk_size", "label": "Chunk size",          "default": "1024"},
    ],
    "OAS": [
        {"key": "use_package", "label": "Package sklearn (True/False)", "default": "True"},
        {"key": "lw_demean",   "label": "Demean (True/False)",          "default": "True"},
        {"key": "lw_ddof",     "label": "ddof",                         "default": "0"},
    ],
}


class MCModelSelector(tk.Frame):
    """
    Classe contenant les widgets de sélection d'un modèle unique pour le Monte Carlo.

    Sélecteur simplifié (legacy) : un seul type de modèle avec ses paramètres fixes.
    La fenêtre rolling est balayée par la grille MC — elle n'est pas configurable ici.
    Remplacé dans l'onglet MC éco par ModelListPanel pour le multi-modèles.

    Methods
    -------
    get_config() -> dict :
        Retourne le dictionnaire de configuration du modèle sélectionné.
    """

    def __init__(self, parent, **kwargs):
        """Initialise le sélecteur et construit les widgets."""
        super().__init__(parent, **kwargs)
        self._param_widgets: dict = {}
        self._build()

    def _build(self):
        """Construit la ligne de sélection du type et du compute_mode, puis les paramètres."""
        top = tk.Frame(self)
        top.pack(fill="x", pady=2)

        tk.Label(top, text="Type de modèle :", width=20, anchor="w").pack(side="left")
        self.type_var = tk.StringVar(value="QIS")

        # Combobox de sélection du type de modèle parmi les types connus
        self.combo_type = ttk.Combobox(top, textvariable=self.type_var, values=list(MC_MODEL_PARAMS.keys()), state="readonly", width=14)
        self.combo_type.pack(side="left", padx=4)
        self.combo_type.bind("<<ComboboxSelected>>", self._refresh_params)

        tk.Label(top, text="compute_mode :", width=14, anchor="w").pack(side="left", padx=(10, 0))
        self.compute_mode_var = tk.StringVar(value="rebal")

        # Combobox de sélection du mode de calcul de covariance (rebal ou path)
        self.combo_mode = ttk.Combobox( top, textvariable=self.compute_mode_var, values=["rebal", "path"], state="readonly", width=6)
        self.combo_mode.pack(side="left", padx=2)

        # Cadre qui accueille les paramètres dynamiques du modèle sélectionné
        self.frame_params = tk.Frame(self)
        self.frame_params.pack(fill="x", pady=4)

        self._refresh_params()

    def _refresh_params(self, event=None):
        """Vide et reconstruit les widgets de paramètres selon le type de modèle sélectionné."""

        # Supprime les anciens widgets de paramètres
        for w in self.frame_params.winfo_children():
            w.destroy()
        self._param_widgets = {}

        model_type = self.type_var.get()

        # Met à jour le compute_mode par défaut selon le type de modèle
        self.compute_mode_var.set(COMPUTE_MODE_DEFAULT.get(model_type, "rebal"))

        # Crée une ligne Entry par paramètre fixe du modèle sélectionné
        params = MC_MODEL_PARAMS.get(model_type, [])
        for p in params:
            row = tk.Frame(self.frame_params)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=p["label"] + " :", width=30, anchor="w").pack(side="left")
            var = tk.StringVar(value=p["default"])
            tk.Entry(row, textvariable=var, width=12).pack(side="left")
            self._param_widgets[p["key"]] = var

    def get_config(self) -> dict:
        """
        Retourne le dictionnaire de configuration du modèle sélectionné.

        Returns
        -------
        dict
            Configuration avec les clés : model_type, compute_mode, et tous les
            paramètres fixes (rolling_ddof, ewma_lambda, lw_demean, etc.).
        """
        model_type = self.type_var.get()
        cfg = { "model_type" : model_type, "compute_mode" : self.compute_mode_var.get(), }

        # Ajoute chaque paramètre fixe du modèle à la configuration
        for key, var in self._param_widgets.items():
            cfg[key] = var.get()

        return cfg


class EstimationTab(tk.Frame):
    """
    Classe contenant les widgets et la logique de l'onglet de backtest économique.

    Permet de configurer l'indice, les modèles de covariance, la fréquence de
    rebalancement et l'optimiseur, puis de lancer un backtest TE-min et de générer
    un rapport PDF.

    Methods
    -------
    _build_sections() -> None :
        Construit tous les widgets de l'onglet.
    _run() -> None :
        Valide les paramètres et lance le backtest dans un thread daemon.
    _run_in_thread(params) -> None :
        Exécute le backtest complet (chargement, estimation, rapport PDF).
    _build_model_specs(model_configs, cov_data_freq, default_optimizer) -> list :
        Construit la liste de ModelSpec depuis les configs de ModelListPanel.
    """

    def __init__(self, parent, log: LogConsole, **kwargs):
        """Initialise l'onglet avec un canvas scrollable et construit les sections."""
        super().__init__(parent, **kwargs)
        self.log = log

        # Canvas + scrollbar vertical pour permettre le défilement de l'onglet
        canvas  = tk.Canvas(self, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas)

        # Met à jour la scrollregion quand le contenu change de taille
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Active le scroll souris uniquement quand le curseur est dans le canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _bind(event):   canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind(event): canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _bind)
        canvas.bind("<Leave>", _unbind)

        self._build_sections()

    def _build_sections(self):
        """Construit les 5 sections de l'onglet : données, univers, modèles, backtest, rapport."""
        p = self.inner

        # Section A : configuration des données et de l'indice
        sec_data = SectionFrame(p, "A. Données & univers")
        sec_data.pack(fill="x", padx=10, pady=6)

        # Ligne de sélection du dossier racine des indices
        frame_dir = tk.Frame(sec_data)
        frame_dir.pack(fill="x", pady=2)
        tk.Label(frame_dir, text="Dossier indices :", width=25, anchor="w").pack(side="left")
        self.var_indices_root = tk.StringVar(value=str(ROOT / "Données" / "IndicesLocaux"))
        tk.Entry(frame_dir, textvariable=self.var_indices_root, width=50).pack(side="left", padx=4)
        tk.Button(frame_dir, text="📂", command=self._browse_indices_dir, relief="flat", cursor="hand2").pack(side="left")

        # Ligne de sélection de l'indice (liste des sous-dossiers détectés)
        frame_idx = tk.Frame(sec_data)
        frame_idx.pack(fill="x", pady=2)
        tk.Label(frame_idx, text="Indice :", width=25, anchor="w").pack(side="left")
        self.var_index_name = tk.StringVar()
        self.combo_index = ttk.Combobox(frame_idx, textvariable=self.var_index_name, width=35, state="readonly")
        self.combo_index.pack(side="left", padx=4)
        tk.Button(frame_idx, text="🔄 Actualiser", command=self._refresh_index_list, relief="flat", cursor="hand2").pack(side="left", padx=4)

        # Dates de début et de fin de la période de backtest
        self.entry_start = LabeledEntry(sec_data, "Date début (YYYY-MM-DD) :", "2017-01-03")
        self.entry_start.pack(fill="x", pady=2)
        self.entry_end   = LabeledEntry(sec_data, "Date fin   (YYYY-MM-DD) :", "2025-12-31")
        self.entry_end.pack(fill="x", pady=2)

        # Fréquence des données et méthode de calcul des rendements
        self.combo_data_freq = LabeledCombobox(sec_data, "Fréquence données :", options=["daily", "weekly"], default="daily")
        self.combo_data_freq.pack(fill="x", pady=2)
        self.combo_method = LabeledCombobox(sec_data, "Méthode rendements :", options=["arithmetic", "log"], default="arithmetic")
        self.combo_method.pack(fill="x", pady=2)

        # Section B : configuration des exclusions d'actifs
        sec_univ = SectionFrame(p, "B. Univers investissable (exclusions)")
        sec_univ.pack(fill="x", padx=10, pady=6)

        # Choix entre exclusions aléatoires ou depuis le fichier Exclusions
        self.check_random_univ = LabeledCheckbox(sec_univ, "Exclusions aléatoires (sinon : fichier Exclusions)", default=False)
        self.check_random_univ.pack(fill="x", pady=2)
        self.entry_exclude_frac = LabeledEntry(sec_univ, "Fraction exclue (si random) :", "0.50")
        self.entry_exclude_frac.pack(fill="x", pady=2)
        self.entry_seed = LabeledEntry(sec_univ, "Seed (si random) :", "1234")
        self.entry_seed.pack(fill="x", pady=2)

        # Section C : liste des modèles de covariance à évaluer
        sec_models = SectionFrame(p, "C. Modèles de covariance à évaluer")
        sec_models.pack(fill="x", padx=10, pady=6)
        self.model_panel = ModelListPanel(sec_models)
        self.model_panel.pack(fill="both", expand=True, pady=4)

        # Section D : paramètres du backtest économique
        sec_eco = SectionFrame(p, "D. Backtest économique")
        sec_eco.pack(fill="x", padx=10, pady=6)
        self.combo_rebal_freq = LabeledCombobox(sec_eco, "Fréquence de rebalancement :", options=["D", "W", "M", "Q", "A"], default="Q")
        self.combo_rebal_freq.pack(fill="x", pady=2)
        self.combo_optimizer = LabeledCombobox(sec_eco, "Optimiseur :", options=["clarabel", "slsqp", "osqp"], default="clarabel")
        self.combo_optimizer.pack(fill="x", pady=2)

        # Section E : dossier de sortie du rapport PDF
        sec_report = SectionFrame(p, "E. Rapport PDF")
        sec_report.pack(fill="x", padx=10, pady=6)
        frame_out = tk.Frame(sec_report)
        frame_out.pack(fill="x", pady=2)
        tk.Label(frame_out, text="Dossier de sortie :", width=25, anchor="w").pack(side="left")
        self.var_output_dir = tk.StringVar(value=str(ROOT / "Reports" / "outputs"))
        tk.Entry(frame_out, textvariable=self.var_output_dir, width=50).pack(side="left", padx=4)
        tk.Button(frame_out, text="📂", command=self._browse_output_dir, relief="flat", cursor="hand2").pack(side="left")

        # Bouton de lancement du backtest
        frame_run = tk.Frame(p)
        frame_run.pack(fill="x", padx=10, pady=10)
        self.run_btn = RunButton(frame_run, "▶  Lancer", command=self._run)
        self.run_btn.pack(side="left")

        self._refresh_index_list()

    def _browse_indices_dir(self):
        """Ouvre un dialog de sélection de dossier et met à jour la liste des indices."""
        path = filedialog.askdirectory(title="Choisir le dossier des indices")
        if path:
            self.var_indices_root.set(path)
            self._refresh_index_list()

    def _browse_output_dir(self):
        """Ouvre un dialog de sélection et met à jour le dossier de sortie du rapport."""
        path = filedialog.askdirectory(title="Choisir le dossier de sortie")
        if path:
            self.var_output_dir.set(path)

    def _refresh_index_list(self):
        """
        Scanne le dossier racine des indices et met à jour la combobox.

        Liste les sous-dossiers du répertoire configuré et les propose en sélection.
        """
        root_path = Path(self.var_indices_root.get())
        if not root_path.exists():
            self.log.log(f"[WARN] Dossier introuvable : {root_path}", "WARNING")
            return

        # Trie les sous-dossiers par ordre alphabétique
        indices = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
        self.combo_index["values"] = indices
        if indices:
            self.combo_index.set(indices[0])
            self.log.log(f"[INFO] {len(indices)} indice(s) trouvé(s)", "INFO")

    def _collect_params(self) -> dict:
        """
        Collecte et convertit tous les paramètres de l'onglet en dictionnaire.

        Returns
        -------
        dict
            Paramètres de configuration du backtest (chemins, dates, modèles, etc.).
        """
        return {
            "indices_root"  : Path(self.var_indices_root.get()),
            "index_name"    : self.var_index_name.get().strip(),
            "start_date"    : self.entry_start.get(),
            "end_date"      : self.entry_end.get(),
            "data_freq"     : self.combo_data_freq.get(),
            "method"        : self.combo_method.get(),
            "use_random"    : self.check_random_univ.get(),
            "exclude_frac"  : float(self.entry_exclude_frac.get()),
            "seed"          : int(self.entry_seed.get()),
            "rebal_freq"    : self.combo_rebal_freq.get(),
            "optimizer"     : self.combo_optimizer.get(),
            "output_dir"    : Path(self.var_output_dir.get()),
            "model_configs" : self.model_panel.get_all_configs(),
        }

    def _run(self):
        """
        Valide les paramètres et lance le backtest dans un thread daemon.

        Affiche un message d'avertissement si l'indice ou les modèles sont manquants.
        """
        try:
            params = self._collect_params()
        except Exception as e:
            messagebox.showerror("Erreur de paramètres", str(e))
            return

        if not params["index_name"]:
            messagebox.showwarning("Paramètre manquant", "Veuillez sélectionner un indice.")
            return
        if not params["model_configs"]:
            messagebox.showwarning("Paramètre manquant", "Veuillez ajouter au moins un modèle.")
            return

        # Lance le backtest dans un thread daemon pour ne pas bloquer l'UI
        self.run_btn.set_running(True)
        threading.Thread(target=self._run_in_thread, args=(params,), daemon=True).start()

    def _run_in_thread(self, params: dict):
        """
        Exécute le backtest complet dans le thread daemon.

        Enchaîne : chargement des données, calcul des rendements, alignement des dates,
        construction des ModelSpec, lancement du backtest via ModelEvaluator,
        génération du rapport PDF.

        Parameters
        ----------
        params : dict
            Paramètres collectés par _collect_params().
        """
        try:
            self.log.log("=" * 60, "INFO")
            self.log.log(f"[Estimation] Démarrage pour : {params['index_name']}", "INFO")
            self.log.log(f"[Estimation] Modèles : {[c['name'] for c in params['model_configs']]}", "INFO")

            # Imports locaux pour éviter les imports circulaires au chargement du module
            from Modules.data.local_files import LocalIndexFolderDataSource
            from Modules.portfolio_management.backtesting.covariance_provider import make_cov_config
            from Modules.Financial_engineering.statistics.multivariate_vol_estimation import (
                ModelEvaluator, DataFrequency, DAILY, WEEKLY
            )
            from Modules.Financial_engineering.statistics.yield_modeling import YieldModeler
            from Modules.portfolio_management.backtesting.rebalancing import RebalanceSchedule
            from Modules.Reports.Estimation_cov import CovarianceEstimationReport
            import numpy as np
            import pandas as pd

            self.log.log("[Données] Chargement en cours...", "INFO")
            data_freq_obj = DataFrequency(params["data_freq"])

            # Instancie la source de données locale pour l'indice sélectionné
            src = LocalIndexFolderDataSource( base_dir=params["indices_root"], index_name=params["index_name"],  prices_sheet=None, compo_sheet=None, sector_sheet=None, )

            # Charge la matrice de prix sur la période configurée
            prices = src.get_prices(start=params["start_date"], end=params["end_date"])
            prices.columns = [str(c).strip().removesuffix(" Equity").strip() for c in prices.columns]
            if prices.empty:
                raise RuntimeError("Matrice de prix vide après chargement.")

            # Calcule les rendements avec la méthode configurée
            ym = YieldModeler(prefer_adj=True)
            R  = ym.compute_frame(prices, method=params["method"], periods=1, dropna=True)

            # Convertit en rendements simples si la méthode est log
            universe_returns = np.expm1(R) if params["method"] == "log" else R.copy()
            returns_all  = R.copy()

            # Charge la composition trimestrielle du benchmark
            Wq = src.get_composition_by_quarter()
            if Wq is None or Wq.empty:
                raise RuntimeError("Composition benchmark vide.")

            def align_rebal(wq_index, univ_index):
                """Projette les dates de rebalancement benchmark sur l'index de trading."""
                wq_index   = pd.to_datetime(wq_index)
                univ_index = pd.to_datetime(univ_index).sort_values()
                out = []
                for d in wq_index:
                    pos = univ_index.searchsorted(d, side="left")
                    out.append(univ_index[min(pos, len(univ_index) - 1)])
                return pd.DatetimeIndex(out)

            # Aligne les dates de rebalancement benchmark et portefeuille sur l'index de trading
            rebal_dates_bench = align_rebal(Wq.index, universe_returns.index)
            rebal_dates_port  = RebalanceSchedule(freq=params["rebal_freq"]).rebalance_dates_anchored(trading_index=universe_returns.index, bench_dates=rebal_dates_bench, k=1,)

            # Charge les poids du benchmark et les aligne sur l'univers
            bench_weights_raw = src.get_weights_asof(rebal_dates=universe_returns.index, method="ffill")
            bench_weights_raw.columns = [str(c).strip().removesuffix(" Equity").strip() for c in bench_weights_raw.columns]
            bench_weights_raw = bench_weights_raw.reindex(columns=universe_returns.columns).fillna(0.0)
            rs = bench_weights_raw.fillna(0).sum(axis=1)

            # Normalise les poids pour que chaque ligne somme à 1
            bench_weights = bench_weights_raw.div(rs, axis=0).fillna(0.0)

            self.log.log(f"[Données] {universe_returns.shape[1]} actifs chargés.", "INFO")

            if params["use_random"]:
                self.log.log(f"[Univers] Exclusions aléatoires : {params['exclude_frac']:.0%}", "INFO")
                tickers = list(universe_returns.columns)
                k_excl  = int(__import__('numpy').floor(params["exclude_frac"] * len(tickers)))
                rng  = __import__('numpy').random.default_rng(params["seed"])
                excl = set(rng.choice(tickers, size=k_excl, replace=False))
                kept   = [t for t in tickers if t not in excl]
                universe_returns_inv = universe_returns[kept].copy()
            else:
                self.log.log("[Univers] Exclusions depuis fichier.", "INFO")
                excluded = src.get_exclusions()

                # Normalise les tickers exclus pour la comparaison (majuscules, sans " EQUITY")
                excl_set = {t.strip().upper().removesuffix(" EQUITY").strip() for t in excluded}
                kept = [t for t in universe_returns.columns if str(t).strip().upper() not in excl_set]
                universe_returns_inv = universe_returns[kept].copy()

            # Aligne les prix sur l'univers filtré
            prices_inv = prices.reindex(index=universe_returns_inv.index, columns=universe_returns_inv.columns)
            sectors    = src.get_sectors()
            sector_map = {t: str(sectors.get(t, "Unknown")) for t in universe_returns_inv.columns}

            self.log.log("[Modèles] Construction des ModelSpec...", "INFO")
            data_freq_obj = DataFrequency(params["data_freq"])
            model_specs = self._build_model_specs(params["model_configs"], params["data_freq"], params["optimizer"])

            # Détermine le common_start après le burn-in de la plus grande fenêtre
            max_window = max(getattr(spec.cov_cfg, "lw_window", None) or  getattr(spec.cov_cfg, "rolling_window", None) or 0  for spec in model_specs)
            if max_window > 0 and max_window < len(universe_returns):
                common_start  = universe_returns.index[max_window]
                rebal_dates_port = rebal_dates_port[rebal_dates_port >= common_start]
                self.log.log(f"[Alignment] common_start={common_start.date()}", "INFO")

            # Crée le répertoire de sortie des inventaires portefeuille
            out_dir_port = ROOT / "PORT" / "Integration"
            out_dir_port.mkdir(parents=True, exist_ok=True)

            self.log.log("[Évaluation] Lancement du backtest économique...", "INFO")

            result = ModelEvaluator.full_evaluation(
                model_specs       = model_specs,
                all_returns       = returns_all,
                universe_returns  = universe_returns_inv,
                benchmark_weights = bench_weights,
                prices_inv        = prices_inv,
                rebal_dates_port  = rebal_dates_port,
                rebal_dates_bench = rebal_dates_bench,
                engine_kwargs     = {"rebal_freq": params["rebal_freq"], "verbose": True},
                port_root         = out_dir_port,
                enabled_export    = True,
                ann_factor        = DataFrequency("daily").ann_factor,
                indice_name       = params["index_name"].replace("/", "_"),
                make_stats        = False,
                make_eco          = True,
            )

            self.log.log("[Rapport] Génération du PDF...", "INFO")

            # Extrait les résultats de backtest pour le rapport
            eco_res  = result.get("economic")
            bt_results = eco_res.get("bt_results", {}) if eco_res else {}

            params["output_dir"].mkdir(parents=True, exist_ok=True)
            tag  = params["index_name"].replace("/", "_")
            pdf_path = params["output_dir"] / f"Covariance_Estimation_Report_{tag}.pdf"

            # Génère le rapport PDF avec les résultats du backtest
            report = CovarianceEstimationReport(te_window=DataFrequency("daily").ann_factor, include_multistart=True, multistart_n_perturbations=20,)
            report.to_pdf(
                pdf_path,
                index_name        = params["index_name"],
                date_range        = (params["start_date"], params["end_date"]),
                metrics_table     = None,
                loss_std_table    = None,
                n_scenarios_stats = 0,
                backtest_results  = bt_results,
                model_specs       = model_specs,
                sector_map        = sector_map,
                universe_returns  = universe_returns,
                ann_factor        = DataFrequency("daily").ann_factor,
                stat_sim_cfg      = None,
            )

            self.log.log(f"[Rapport] PDF généré : {pdf_path}", "INFO")
            self.log.log("=" * 60, "INFO")
            self.log.log("Évaluation terminée avec succès !", "INFO")

        except Exception:
            self.log.log("[ERREUR] " + traceback.format_exc(), "ERROR")
        finally:
            # Réactive le bouton même en cas d'erreur
            self.run_btn.set_running(False)

    def _build_model_specs(self, model_configs: list, cov_data_freq, default_optimizer: str):
        """
        Construit la liste de ModelSpec depuis les configurations de ModelListPanel.

        Le backtest tourne toujours en daily (ann_factor=252). La fréquence des données
        de covariance (cov_data_freq) est transmise dans la CovConfig pour le resampling.

        Parameters
        ----------
        model_configs : list of dict
            Configurations brutes retournées par ModelListPanel.get_all_configs().
        cov_data_freq : str
            Fréquence des données pour l'estimation de covariance ('daily' ou 'weekly').
        default_optimizer : str
            Optimiseur par défaut si non spécifié dans la config.

        Returns
        -------
        list of ModelSpec
            Liste de specs prêtes à être passées à ModelEvaluator.full_evaluation().
        """
        from Modules.portfolio_management.backtesting.covariance_provider import make_cov_config, CovConfig
        from Modules.Financial_engineering.statistics.multivariate_vol_estimation import DataFrequency
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ModelSpec:
            """Spécification d'un modèle pour le backtest : nom, config covariance, optimiseur."""
            name: str
            cov_cfg: CovConfig
            optimizer_name: str = "clarabel"

        specs = []
        # Le backtest tourne toujours en daily : la fenêtre rolling est en jours de bourse
        ann = DataFrequency("daily").ann_factor

        for cfg in model_configs:
            method        = cfg["method"]
            window_factor = float(cfg.get("window_factor", 1))

            # Convertit la fenêtre exprimée en années en nombre de jours de bourse
            window        = int(window_factor * ann)
            optimizer     = cfg.get("optimizer", default_optimizer)
            compute_mode  = cfg.get("compute_mode", "rebal")

            if method == "rolling":
                cov_cfg = make_cov_config(
                    method="rolling", compute_mode=compute_mode,
                    rolling_window=window, rolling_ddof=int(cfg.get("rolling_ddof", 1)),
                    cov_data_freq=cov_data_freq,
                )
            elif method == "ewma":
                tune = cfg.get("tune_lambda", True)
                if isinstance(tune, str): tune = tune.lower() == "true"
                cov_cfg = make_cov_config(
                    method="ewma", compute_mode=compute_mode,
                    rolling_window=window,
                    ewma_lambda=float(cfg.get("ewma_lambda", 0.94)),
                    ewma_init=str(cfg.get("ewma_init", "scov")),
                    tune_lambda=tune, path_scope="decision_dates",
                    cov_data_freq=cov_data_freq,
                )
            elif method == "ledoit_wolf":
                use_pkg = cfg.get("use_package", False)
                if isinstance(use_pkg, str): use_pkg = use_pkg.lower() == "true"
                cov_cfg = make_cov_config(
                    method="ledoit_wolf", compute_mode=compute_mode,
                    lw_variant=cfg.get("lw_variant", "lw_2004"),
                    lw_window=window, use_package=use_pkg,
                    lw_demean=bool(cfg.get("lw_demean", True)),
                    lw_ddof=int(cfg.get("lw_ddof", 0)),
                    chunk_size=int(cfg.get("chunk_size", 1024)),
                    cov_data_freq=cov_data_freq,
                )
            else:
                self.log.log(f"[WARN] Méthode inconnue : {method}, ignorée.", "WARNING")
                continue

            specs.append(ModelSpec(name=cfg["name"], cov_cfg=cov_cfg, optimizer_name=optimizer))

        return specs


class MonteCarloTab(tk.Frame):
    """
    Classe contenant les widgets et la logique de l'onglet Monte Carlo.

    Deux branches configurables indépendamment :
    - Monte Carlo économique : grille de scénarios (rolling, data_freq, exclude_frac,
      seed, rebal_freq) sur plusieurs modèles, avec export Excel checkpoint.
    - Monte Carlo statistique : simulation DGP (static_oracle ou factor_shock)
      avec calcul des pertes matricielles (Frobenius, spectral, Stein).

    Methods
    -------
    _build_sections() -> None :
        Construit tous les widgets des sous-sections.
    _run() -> None :
        Valide les paramètres et lance le Monte Carlo dans un thread daemon.
    _run_in_thread(params) -> None :
        Exécute les branches économique et/ou statistique.
    _build_spec_from_model_cfg(model_cfg, rolling, default_optimizer, cov_data_freq) -> ModelSpec :
        Construit un ModelSpec depuis une config de ModelListPanel pour une fenêtre donnée.
    """

    def __init__(self, parent, log: LogConsole, **kwargs):
        """Initialise l'onglet avec un canvas scrollable et construit les sections."""
        super().__init__(parent, **kwargs)
        self.log = log

        # Canvas + scrollbar vertical pour le défilement de l'onglet
        canvas    = tk.Canvas(self, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas)
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Scroll souris actif uniquement dans le canvas
        def _mw_mc(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")
        def _bind_mc(event):   canvas.bind_all("<MouseWheel>", _mw_mc)
        def _unbind_mc(event): canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _bind_mc)
        canvas.bind("<Leave>", _unbind_mc)

        self._build_sections()

    def _build_sections(self):
        """Construit toutes les sous-sections de l'onglet Monte Carlo."""
        p = self.inner

        # Sélection du mode : éco, stats, ou les deux
        sec_mode = SectionFrame(p, "Que souhaitez-vous lancer ?")
        sec_mode.pack(fill="x", padx=10, pady=6)
        check_row = tk.Frame(sec_mode)
        check_row.pack(anchor="w")
        self.check_mc_eco = LabeledCheckbox(check_row, "Monte Carlo économique (modèles)", default=True)
        self.check_mc_eco.pack(side="left", padx=(0, 30))
        self.check_mc_eco.check.config(command=self._on_mode_change)
        self.check_mc_stats = LabeledCheckbox(check_row, "Monte Carlo statistique (DGP)", default=False)
        self.check_mc_stats.pack(side="left")
        self.check_mc_stats.check.config(command=self._on_mode_change)

        # Section A : données (visible uniquement si MC éco coché)
        self.sec_data_mc = SectionFrame(p, "A. Données")
        frame_dir = tk.Frame(self.sec_data_mc)
        frame_dir.pack(fill="x", pady=2)
        tk.Label(frame_dir, text="Dossier indices :", width=25, anchor="w").pack(side="left")
        self.var_indices_root = tk.StringVar(value=str(ROOT / "Données" / "IndicesLocaux"))
        tk.Entry(frame_dir, textvariable=self.var_indices_root, width=50).pack(side="left", padx=4)
        tk.Button(frame_dir, text="📂", relief="flat", cursor="hand2", command=lambda: self.var_indices_root.set(
             filedialog.askdirectory(title="Choisir dossier indices") or self.var_indices_root.get()),).pack(side="left")
        frame_idx = tk.Frame(self.sec_data_mc)
        frame_idx.pack(fill="x", pady=2)
        tk.Label(frame_idx, text="Indice :", width=25, anchor="w").pack(side="left")
        self.var_index_name = tk.StringVar()
        self.combo_index = ttk.Combobox(frame_idx, textvariable=self.var_index_name, width=35, state="readonly")
        self.combo_index.pack(side="left", padx=4)
        tk.Button(frame_idx, text="🔄 Actualiser", command=self._refresh_index_list, relief="flat", cursor="hand2").pack(side="left", padx=4)
        self.entry_start = LabeledEntry(self.sec_data_mc, "Date début :", "2017-01-03")
        self.entry_start.pack(fill="x", pady=2)
        self.entry_end   = LabeledEntry(self.sec_data_mc, "Date fin :", "2025-12-31")
        self.entry_end.pack(fill="x", pady=2)

        # Section B : grille MC économique (visible uniquement si MC éco coché)
        self.sec_eco_mc = SectionFrame(p, "B. Monte Carlo économique — grille de scénarios")
        frame_out_eco = tk.Frame(self.sec_eco_mc)
        frame_out_eco.pack(fill="x", pady=2)
        tk.Label(frame_out_eco, text="Dossier sortie Excel :", width=25, anchor="w").pack(side="left")
        self.var_output_dir = tk.StringVar(value=str(ROOT / "Reports" / "outputs" / "montecarlo"))
        tk.Entry(frame_out_eco, textvariable=self.var_output_dir, width=50).pack(side="left", padx=4)
        tk.Button(frame_out_eco, text="📂", relief="flat", cursor="hand2", command=lambda: self.var_output_dir.set(
                filedialog.askdirectory(title="Choisir dossier sortie") or self.var_output_dir.get()),).pack(side="left")

        # Grille de paramètres MC éco : deux colonnes (gauche/droite)
        cols  = tk.Frame(self.sec_eco_mc); cols.pack(fill="x", pady=4)
        left  = tk.Frame(cols); left.pack(side="left",  fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(cols); right.pack(side="left", fill="both", expand=True)

        # Colonne gauche : fréquences données, fenêtres rolling, fréquences de rebalancement
        self.list_data_freqs    = ListEditor(left,  "Fréquences données",           default_values=["daily", "weekly"],                   height=3)
        self.list_data_freqs.pack(fill="x", pady=4)
        self.list_rolling_years = ListEditor(left,  "Fenêtres rolling (en années)", default_values=[0.5, 1, 1.5, 2, 3],                  height=5)
        self.list_rolling_years.pack(fill="x", pady=4)
        self.list_rebal_freqs   = ListEditor(left,  "Fréquences de rebalancement",  default_values=["M", "Q", "W"],                       height=3)
        self.list_rebal_freqs.pack(fill="x", pady=4)

        # Colonne droite : fractions d'exclusion et seeds
        self.list_exclude_fracs = ListEditor(right, "Fractions d'exclusion",        default_values=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50], height=6)
        self.list_exclude_fracs.pack(fill="x", pady=4)
        self.list_seeds         = ListEditor(right, "Seeds (exclusion univers)",     default_values=list(range(1000, 1100)),                height=6)
        self.list_seeds.pack(fill="x", pady=4)

        # Optimiseur et liste de modèles à tester
        self.combo_mc_optimizer = LabeledCombobox(self.sec_eco_mc, "Optimiseur :", options=["clarabel", "slsqp"], default="clarabel")
        self.combo_mc_optimizer.pack(fill="x", pady=2)
        tk.Label(self.sec_eco_mc, anchor="w", fg="#333", text="Modèles à tester (la fenêtre rolling est balayée par la grille) :", font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(8, 2), padx=4)
        
        # ModelListPanel permet de configurer plusieurs modèles simultanément
        self.model_panel_eco = ModelListPanel(self.sec_eco_mc)
        self.model_panel_eco.pack(fill="both", expand=True, padx=4, pady=2)

        # Section C : Monte Carlo statistique (DGP)
        self.sec_stat_mc = SectionFrame(p, "C. Monte Carlo statistique — simulation DGP")
        frame_out_stat = tk.Frame(self.sec_stat_mc)
        frame_out_stat.pack(fill="x", pady=2)
        tk.Label(frame_out_stat, text="Fichier Excel stats :", width=25, anchor="w").pack(side="left")
        self.var_mc_stat_output = tk.StringVar(
            value=str(ROOT / "Reports" / "outputs" / "montecarlo" / "stat_results.xlsx")
        )
        tk.Entry(frame_out_stat, textvariable=self.var_mc_stat_output, width=45).pack(side="left", padx=4)
        tk.Button(
            frame_out_stat, text="📂", relief="flat", cursor="hand2",
            command=lambda: self.var_mc_stat_output.set(
                filedialog.asksaveasfilename(
                    title="Fichier Excel stats", defaultextension=".xlsx",
                    filetypes=[("Excel", "*.xlsx")]
                ) or self.var_mc_stat_output.get()
            ),
        ).pack(side="left")

        # Sélection du type de DGP avec description dynamique
        self.combo_mc_dgp = LabeledCombobox(
            self.sec_stat_mc, "DGP :", options=["factor_shock", "static_oracle"], default="factor_shock"
        )
        self.combo_mc_dgp.pack(fill="x", pady=2)
        self.combo_mc_dgp.combo.bind("<<ComboboxSelected>>", self._on_mc_dgp_change)
        self.label_mc_dgp_desc = tk.Label(
            self.sec_stat_mc, text="", fg="#555",
            font=("Segoe UI", 8, "italic"), anchor="w", wraplength=700
        )
        self.label_mc_dgp_desc.pack(fill="x", padx=4, pady=(0, 4))

        # Paramètres communs au MC statistique (deux colonnes)
        frame_mc_stat_common = tk.Frame(self.sec_stat_mc)
        frame_mc_stat_common.pack(fill="x", pady=2)
        col1 = tk.Frame(frame_mc_stat_common); col1.pack(side="left", fill="y", padx=(0, 20))
        col2 = tk.Frame(frame_mc_stat_common); col2.pack(side="left", fill="y")

        # Colonne gauche : N_sim, facteurs, scénarios, random state
        self.entry_mc_n_sim        = LabeledEntry(col1, "N_sim (actifs) :", "300");        self.entry_mc_n_sim.pack(fill="x", pady=1)
        self.entry_mc_n_factors    = LabeledEntry(col1, "Facteurs K :", "10");             self.entry_mc_n_factors.pack(fill="x", pady=1)
        self.entry_mc_n_scenarios  = LabeledEntry(col1, "Scénarios par modèle :", "20");   self.entry_mc_n_scenarios.pack(fill="x", pady=1)
        self.entry_mc_random_state = LabeledEntry(col1, "Random state :", "42");           self.entry_mc_random_state.pack(fill="x", pady=1)

        # Colonne droite : type d'innovation
        self.combo_mc_innovation = LabeledCombobox(col2, "Innovation :", options=["gaussian", "student"], default="gaussian")
        self.combo_mc_innovation.pack(fill="x", pady=1)

        # Paramètres spécifiques au DGP static_oracle
        self.frame_mc_oracle = SectionFrame(self.sec_stat_mc, "Paramètres static_oracle")
        self.frame_mc_oracle.pack(fill="x", pady=4)
        self.entry_mc_p_ratio = LabeledEntry(self.frame_mc_oracle, "p_ratio (N/T) :", "1.0")
        self.entry_mc_p_ratio.pack(fill="x", pady=1)

        # Paramètres spécifiques au DGP factor_shock
        self.frame_mc_shock = SectionFrame(self.sec_stat_mc, "Paramètres factor_shock")
        self.frame_mc_shock.pack(fill="x", pady=4)
        frame_shock = tk.Frame(self.frame_mc_shock); frame_shock.pack(fill="x")
        cs1 = tk.Frame(frame_shock); cs1.pack(side="left", fill="y", padx=(0, 20))
        cs2 = tk.Frame(frame_shock); cs2.pack(side="left", fill="y")
        self.entry_mc_rho_B   = LabeledEntry(cs1, "rho_B :", "0.95");   self.entry_mc_rho_B.pack(fill="x", pady=1)
        self.entry_mc_sigma_B = LabeledEntry(cs1, "sigma_B :", "0.05"); self.entry_mc_sigma_B.pack(fill="x", pady=1)
        self.entry_mc_rho_d   = LabeledEntry(cs2, "rho_d :", "0.97");   self.entry_mc_rho_d.pack(fill="x", pady=1)
        self.entry_mc_sigma_d = LabeledEntry(cs2, "sigma_d :", "0.10"); self.entry_mc_sigma_d.pack(fill="x", pady=1)

        # Liste de modèles pour le MC statistique (fenêtre configurable par modèle)
        tk.Label(
            self.sec_stat_mc, anchor="w", fg="#333",
            text="Modèles à comparer (fenêtre configurable par modèle — même seeds pour tous) :",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(8, 2), padx=4)
        self.model_panel_stat = ModelListPanel(self.sec_stat_mc)
        self.model_panel_stat.pack(fill="both", expand=True, pady=2)

        # Section D : paramètres d'exécution (toujours visible)
        self.sec_exec = SectionFrame(p, "D. Exécution")
        self.entry_n_jobs      = LabeledEntry(self.sec_exec, "Nombre de jobs parallèles (-1 = tous) :", "4")
        self.entry_n_jobs.pack(fill="x", pady=2)
        self.entry_flush_every = LabeledEntry(self.sec_exec, "Sauvegarder tous les N scénarios :", "100")
        self.entry_flush_every.pack(fill="x", pady=2)

        # Bouton de lancement
        self.frame_run = tk.Frame(p)
        self.run_btn = RunButton(self.frame_run, "▶  Lancer le Monte Carlo", command=self._run)
        self.run_btn.pack(side="left")

        self._refresh_index_list()
        self._on_mode_change()
        self._on_mc_dgp_change()

    def _on_mode_change(self, event=None):
        """
        Affiche ou masque les sections selon les cases cochées (éco et/ou stats).

        Toujours affichées : section Exécution et bouton Lancer.
        """
        show_eco   = self.check_mc_eco.get()
        show_stats = self.check_mc_stats.get()

        # Masque toutes les sections conditionnelles avant de les réafficher selon le mode
        for w in (self.sec_data_mc, self.sec_eco_mc, self.sec_stat_mc, self.sec_exec, self.frame_run):
            w.pack_forget()

        if show_eco:
            # Affiche les sections données et grille éco
            self.sec_data_mc.pack(fill="x", padx=10, pady=6)
            self.sec_eco_mc.pack(fill="x", padx=10, pady=6)
        if show_stats:
            # Affiche la section DGP statistique
            self.sec_stat_mc.pack(fill="x", padx=10, pady=6)

        # La section exécution et le bouton sont toujours visibles
        self.sec_exec.pack(fill="x", padx=10, pady=6)
        self.frame_run.pack(fill="x", padx=10, pady=10)

    def _on_mc_dgp_change(self, event=None):
        """
        Affiche les paramètres spécifiques au DGP sélectionné et met à jour la description.

        static_oracle : affiche p_ratio, masque les paramètres AR(1).
        factor_shock  : affiche les paramètres AR(1), masque p_ratio.
        """
        dgp = self.combo_mc_dgp.get()
        descs = {
            "static_oracle": "Σ_true fixe par seed, T obs i.i.d. — cadre Ledoit-Wolf. Paramètre clé : p_ratio.",
            "factor_shock":  "Σ_t varie via loadings AR(1) — teste l'adaptation temporelle des estimateurs.",
        }
        self.label_mc_dgp_desc.config(text=descs.get(dgp, ""))

        if dgp == "static_oracle":
            self.frame_mc_oracle.pack(fill="x", pady=4)
            self.frame_mc_shock.pack_forget()
        else:
            self.frame_mc_oracle.pack_forget()
            self.frame_mc_shock.pack(fill="x", pady=4)

    def _refresh_index_list(self):
        """Scanne le dossier racine et met à jour la combobox des indices disponibles."""
        root_path = Path(self.var_indices_root.get())
        if not root_path.exists():
            return
        indices = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
        self.combo_index["values"] = indices
        if indices:
            self.combo_index.set(indices[0])

    def _collect_params(self) -> dict:
        """
        Collecte et convertit tous les paramètres de l'onglet Monte Carlo.

        Returns
        -------
        dict
            Paramètres complets pour les branches éco et/ou statistique.
        """
        p_ratio_str = self.entry_mc_p_ratio.get()
        # Convertit en float ou None si la valeur est vide ou "none"
        p_ratio = float(p_ratio_str) if p_ratio_str and p_ratio_str.lower() != "none" else None

        return {
            "indices_root"           : Path(self.var_indices_root.get()),
            "index_name"             : self.var_index_name.get().strip(),
            "start_date"             : self.entry_start.get(),
            "end_date"               : self.entry_end.get(),
            "make_mc_eco"            : self.check_mc_eco.get(),
            "make_mc_stats"          : self.check_mc_stats.get(),
            # Paramètres MC économique
            "output_dir"             : Path(self.var_output_dir.get()),
            "data_freqs"             : self.list_data_freqs.get_values(),
            "rolling_years"          : self.list_rolling_years.get_floats(),
            "rebal_freqs"            : self.list_rebal_freqs.get_values(),
            "exclude_fracs"          : self.list_exclude_fracs.get_floats(),
            "seeds"                  : self.list_seeds.get_ints(),
            "optimizer"              : self.combo_mc_optimizer.get(),
            "model_configs_eco"      : self.model_panel_eco.get_all_configs(),
            # Paramètres MC statistique
            "mc_stat_output"         : Path(self.var_mc_stat_output.get()),
            "mc_dgp_type"            : self.combo_mc_dgp.get(),
            "mc_n_sim"               : int(self.entry_mc_n_sim.get()),
            "mc_n_factors"           : int(self.entry_mc_n_factors.get()),
            "mc_n_scenarios"         : int(self.entry_mc_n_scenarios.get()),
            "mc_random_state"        : int(self.entry_mc_random_state.get()),
            "mc_innovation"          : self.combo_mc_innovation.get(),
            "mc_p_ratio"             : p_ratio,
            "mc_rho_B"               : float(self.entry_mc_rho_B.get()),
            "mc_sigma_B"             : float(self.entry_mc_sigma_B.get()),
            "mc_rho_d"               : float(self.entry_mc_rho_d.get()),
            "mc_sigma_d"             : float(self.entry_mc_sigma_d.get()),
            "model_configs_stat"     : self.model_panel_stat.get_all_configs(),
            # Paramètres d'exécution
            "n_jobs"                 : int(self.entry_n_jobs.get()),
            "flush_every"            : int(self.entry_flush_every.get()),
        }

    def _run(self):
        """
        Valide les paramètres et lance le Monte Carlo dans un thread daemon.

        Vérifie que le mode, l'indice et les modèles sont correctement configurés
        avant de démarrer le thread.
        """
        try:
            params = self._collect_params()
        except Exception as e:
            messagebox.showerror("Erreur de paramètres", str(e))
            return

        if params["make_mc_eco"] and not params["index_name"]:
            messagebox.showwarning("Paramètre manquant", "Veuillez sélectionner un indice pour le Monte Carlo économique.")
            return
        if not params["make_mc_eco"] and not params["make_mc_stats"]:
            messagebox.showwarning("Paramètre manquant", "Cochez au moins un mode.")
            return
        if params["make_mc_eco"] and not params["model_configs_eco"]:
            messagebox.showwarning("Paramètre manquant", "Ajoutez au moins un modèle pour le MC économique.")
            return
        if params["make_mc_stats"] and not params["model_configs_stat"]:
            messagebox.showwarning("Paramètre manquant", "Ajoutez au moins un modèle pour le MC statistique.")
            return

        # Lance le Monte Carlo dans un thread daemon pour ne pas bloquer l'UI
        self.run_btn.set_running(True)
        threading.Thread(target=self._run_in_thread, args=(params,), daemon=True).start()

    def _build_spec_from_model_cfg(self, model_cfg: dict, rolling: int, default_optimizer: str, cov_data_freq: str = "daily"):
        """
        Construit un ModelSpec depuis une configuration de ModelListPanel pour une fenêtre donnée.

        Supporte deux sources de config : ModelListPanel (clé 'method') et MCModelSelector (clé 'model_type').
        La fenêtre rolling passée en paramètre écrase celle éventuellement présente dans model_cfg.

        Parameters
        ----------
        model_cfg : dict
            Configuration d'un modèle retournée par ModelListPanel.get_all_configs()[i].
        rolling : int
            Taille de la fenêtre rolling en nombre de périodes (écrase la config).
        default_optimizer : str
            Optimiseur à utiliser si non spécifié dans la config.
        cov_data_freq : str
            Fréquence des données pour l'estimation de covariance ('daily' ou 'weekly').

        Returns
        -------
        ModelSpec
            Spécification du modèle prête à être passée à _scenario_worker.
        """
        from Modules.portfolio_management.backtesting.covariance_provider import make_cov_config, CovConfig
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ModelSpec:
            """Spécification d'un modèle pour le backtest : nom, config covariance, optimiseur."""
            name: str
            cov_cfg: CovConfig
            optimizer_name: str = "clarabel"

        model_type   = model_cfg.get("model_type", "")
        compute_mode = model_cfg.get("compute_mode", COMPUTE_MODE_DEFAULT.get(model_type, "rebal"))

        # Mapping model_type (MCModelSelector) vers method + lw_variant
        METHOD_MAP = {
            "Rolling"  : ("rolling",     None),
            "EWMA"     : ("ewma",        None),
            "LW_2004"  : ("ledoit_wolf", "lw_2004"),
            "ANLS_2020": ("ledoit_wolf", "anls_2020"),
            "QIS"      : ("ledoit_wolf", "QIS"),
            "OAS"      : ("ledoit_wolf", "OAS"),
        }

        # Priorité à la clé 'method' (ModelListPanel) sur 'model_type' (MCModelSelector)
        if "method" in model_cfg and model_cfg["method"] != "ledoit_wolf":
            method     = model_cfg["method"]
            lw_variant = model_cfg.get("lw_variant", "lw_2004")
        elif "method" in model_cfg and model_cfg["method"] == "ledoit_wolf":
            method     = "ledoit_wolf"
            lw_variant = model_cfg.get("lw_variant", "lw_2004")
        else:
            # Fallback sur MODEL_TYPE pour MCModelSelector
            method, lw_variant = METHOD_MAP.get(model_type, ("ledoit_wolf", "lw_2004"))

        if method == "rolling":
            cov_cfg = make_cov_config(
                method="rolling", compute_mode=compute_mode,
                rolling_window=rolling,
                rolling_ddof=int(model_cfg.get("rolling_ddof", 1)),
                cov_data_freq=cov_data_freq,
            )
        elif method == "ewma":
            tune = model_cfg.get("tune_lambda", False)
            if isinstance(tune, str): tune = tune.lower() == "true"
            cov_cfg = make_cov_config(
                method="ewma", compute_mode=compute_mode,
                rolling_window=rolling,
                ewma_lambda=float(model_cfg.get("ewma_lambda", 0.94)),
                ewma_init=str(model_cfg.get("ewma_init", "scov")),
                tune_lambda=tune, path_scope="decision_dates",
                cov_data_freq=cov_data_freq,
            )
        else:
            use_pkg = model_cfg.get("use_package", False)
            if isinstance(use_pkg, str): use_pkg = use_pkg.lower() == "true"
            cov_cfg = make_cov_config(
                method="ledoit_wolf", compute_mode=compute_mode,
                lw_variant=lw_variant,
                lw_window=rolling,
                use_package=use_pkg,
                lw_demean=bool(model_cfg.get("lw_demean", True)),
                lw_ddof=int(model_cfg.get("lw_ddof", 0)),
                chunk_size=int(model_cfg.get("chunk_size", 1024)),
                cov_data_freq=cov_data_freq,
            )

        # Utilise le nom de la config ou génère un nom par défaut
        name = model_cfg.get("name") or f"{model_type}_{rolling}"
        return ModelSpec(name=name, cov_cfg=cov_cfg, optimizer_name=default_optimizer)

    def _run_in_thread(self, params: dict):
        """
        Exécute les branches économique et/ou statistique du Monte Carlo dans le thread daemon.

        Branche économique : charge les données réelles, construit la grille de scénarios,
        lance les backtests en parallèle (joblib) et exporte les résultats au fil de l'eau
        dans un fichier Excel avec checkpoint (reprise automatique si interruption).

        Branche statistique : construit un R_ref synthétique, configure StatSimConfig,
        et lance run_stat_evaluation() avec export checkpoint.

        Parameters
        ----------
        params : dict
            Paramètres collectés par _collect_params().
        """
        try:
            self.log.log("=" * 60, "INFO")
            self.log.log(f"[MC] Démarrage Monte Carlo pour : {params.get('index_name', 'stats seul')}", "INFO")
            mode_label = (
                'éco + stats' if params['make_mc_eco'] and params['make_mc_stats']
                else 'éco seul' if params['make_mc_eco']
                else 'stats seul'
            )
            self.log.log(f"[MC] Mode : {mode_label}", "INFO")

            # Imports locaux pour éviter les imports circulaires au chargement du module
            from Modules.Financial_engineering.statistics.multivariate_vol_estimation import (
                ModelEvaluator, DataFrequency, StatSimConfig
            )
            from Modules.Financial_engineering.Export.Montecarlo_cov_export import (
                MonteCarloExporter, ScenarioKey, StatMonteCarloExporter,
            )
            from itertools import product
            import numpy as np
            import pandas as pd
            import time
            from joblib import Parallel, delayed

            # ================================================================
            # Branche Monte Carlo économique
            # ================================================================
            if params["make_mc_eco"]:
                from Modules.data.local_files import LocalIndexFolderDataSource
                from Modules.Financial_engineering.statistics.yield_modeling import YieldModeler
                from Modules.portfolio_management.backtesting.rebalancing import RebalanceSchedule

                self.log.log("[MC Éco] Chargement des données...", "INFO")

                # Instancie la source de données pour l'indice sélectionné
                src = LocalIndexFolderDataSource(
                    base_dir=params["indices_root"], index_name=params["index_name"],
                    prices_sheet=None, compo_sheet=None, sector_sheet=None
                )
                prices = src.get_prices(start=params["start_date"], end=params["end_date"])
                prices.columns = [str(c).strip().removesuffix(" Equity").strip() for c in prices.columns]

                # Calcule les rendements arithmétiques
                ym = YieldModeler(prefer_adj=True)
                R  = ym.compute_frame(prices, method="arithmetic", periods=1, dropna=True)
                all_returns = universe_returns = R.copy()

                Wq = src.get_composition_by_quarter()

                def align_rebal(wq_idx, univ_idx):
                    """Projette les dates de rebalancement benchmark sur l'index de trading."""
                    wq_idx   = pd.to_datetime(wq_idx)
                    univ_idx = pd.to_datetime(univ_idx).sort_values()
                    out = []
                    for d in wq_idx:
                        pos = univ_idx.searchsorted(d, side="left")
                        out.append(univ_idx[min(pos, len(univ_idx) - 1)])
                    return pd.DatetimeIndex(out)

                # Aligne les dates de rebalancement benchmark et portefeuille sur l'index de trading
                rebal_dates_bench = align_rebal(Wq.index, universe_returns.index)
                rebal_dates_port  = RebalanceSchedule(freq="Q").rebalance_dates_anchored(
                    trading_index=universe_returns.index, bench_dates=rebal_dates_bench, k=1
                )

                # Charge et normalise les poids du benchmark
                bench_weights_raw = src.get_weights_asof(rebal_dates=universe_returns.index, method="ffill")
                bench_weights_raw.columns = [str(c).strip().removesuffix(" Equity").strip() for c in bench_weights_raw.columns]
                bench_weights_raw = bench_weights_raw.reindex(columns=universe_returns.columns).fillna(0.0)
                rs = bench_weights_raw.fillna(0).sum(axis=1)
                bench_weights = bench_weights_raw.div(rs, axis=0).fillna(0.0)

                # Récupère la grille de paramètres MC éco
                model_configs_eco = params["model_configs_eco"]
                data_freqs        = params["data_freqs"]
                rolling_years     = params["rolling_years"]

                # Calcule le common_start après le plus long burn-in (fenêtre rolling en daily)
                ann_daily = DataFrequency("daily").ann_factor
                max_roll  = max(int(ry * ann_daily) for ry in rolling_years)
                common_start = (
                    universe_returns.index[max_roll]
                    if max_roll < len(universe_returns)
                    else universe_returns.index[-1]
                )

                total = (
                    len(model_configs_eco) * len(rolling_years) * len(data_freqs) *
                    len(params["exclude_fracs"]) * len(params["seeds"]) * len(params["rebal_freqs"])
                )
                self.log.log(f"[MC Éco] {len(model_configs_eco)} modèle(s) | common_start={common_start.date()} | {total} scénarios.", "INFO")

                # Crée l'exporter Excel avec checkpoint pour la reprise automatique
                params["output_dir"].mkdir(parents=True, exist_ok=True)
                tag        = params["index_name"].replace("/", "_")
                excel_path = params["output_dir"] / f"montecarlo_{tag}.xlsx"
                exporter   = MonteCarloExporter(path=excel_path, flush_every=params["flush_every"])
                self.log.log(f"[MC Éco] {exporter.n_done()} scénarios déjà calculés (checkpoint).", "INFO")

                # Pour ALM Classic : exclut toujours les indices du benchmark de l'univers
                bench_tickers     = set(bench_weights.columns[bench_weights.any()].astype(str))
                universe_tickers  = set(universe_returns.columns.astype(str))
                forced_exclusions = (
                    bench_tickers & universe_tickers
                    if "ALM CLASSIC" in params["index_name"].upper()
                    else set()
                )

                # Boucle sur les modèles, puis sur la grille (rolling_yr, data_freq)
                for model_cfg_eco in model_configs_eco:
                    for rolling_yr, data_freq in product(rolling_years, data_freqs):
                        # La fenêtre rolling est toujours calculée en daily (ann_factor=252)
                        ann_factor_daily = DataFrequency("daily").ann_factor
                        rolling = int(rolling_yr * ann_factor_daily)
                        spec    = self._build_spec_from_model_cfg(
                            model_cfg_eco, rolling, params["optimizer"], cov_data_freq=data_freq
                        )

                        # Filtre les scénarios déjà calculés via le checkpoint
                        scenarios_todo = []
                        for excl_frac, seed, rebal_freq in product(
                            params["exclude_fracs"], params["seeds"], params["rebal_freqs"]
                        ):
                            key = ScenarioKey(
                                model_name=spec.name, rolling=rolling, exclude_frac=excl_frac,
                                seed=seed, rebal_freq=rebal_freq, data_freq=data_freq
                            )
                            if not exporter.already_done(key):
                                scenarios_todo.append((key, excl_frac, seed, rebal_freq))

                        if not scenarios_todo:
                            # Tous les scénarios de cette combinaison sont déjà calculés
                            continue

                        n_batch      = len(scenarios_todo)
                        n_done_local = 0
                        compute_mode = getattr(spec.cov_cfg, "compute_mode", "rebal")
                        self.log.log(f"[MC Éco] {spec.name} | data={data_freq} | mode={compute_mode} | {n_batch} scénarios...", "INFO")

                        if compute_mode == "path":
                            # Mode path : précompute le path de covariance sur all_returns
                            # puis le partage en lecture seule entre tous les workers
                            from Modules.portfolio_management.backtesting.covariance_provider import CovarianceProvider
                            import gc as _gc

                            provider = CovarianceProvider(cfg=spec.cov_cfg)
                            self.log.log(f"[MC Éco] Précompute path {spec.name}...", "INFO")
                            provider.precompute_path(all_returns)

                            # Récupère les métadonnées du memmap pour les passer aux workers
                            memmap_path  = provider._covariance_path._memmap_path
                            memmap_shape = provider._path_H.shape
                            path_index   = provider._path_index
                            path_names   = provider._full_cols

                            # Tronque les dates de rebalancement à la première date du path
                            path_start            = pd.Timestamp(path_index[0])
                            rebal_dates_port_eff  = rebal_dates_port[rebal_dates_port >= path_start]
                            rebal_dates_bench_eff = rebal_dates_bench[rebal_dates_bench >= path_start]

                            raw_results = Parallel(n_jobs=params["n_jobs"], backend="loky", return_as="generator")(
                                delayed(_scenario_worker)(
                                    spec=spec, rolling=rolling, exclude_frac=excl, seed=seed,
                                    rebal_freq=freq, data_freq=data_freq, key=key,
                                    all_returns=all_returns, universe_returns=universe_returns,
                                    bench_weights=bench_weights, prices=prices,
                                    rebal_dates_port=rebal_dates_port_eff,
                                    rebal_dates_bench=rebal_dates_bench_eff,
                                    common_start=common_start, use_memmap=True,
                                    memmap_path=memmap_path, memmap_shape=memmap_shape,
                                    path_index=path_index, path_names=path_names,
                                    forced_exclusions=forced_exclusions,
                                )
                                for key, excl, seed, freq in scenarios_todo
                            )

                            # Collecte les résultats au fil de l'eau et les écrit dans l'exporter
                            for eval_res, key in raw_results:
                                n_done_local += 1
                                t0 = time.perf_counter()
                                try:
                                    result = exporter.build_result(key, eval_res, ann_factor_daily, duration=time.perf_counter() - t0)
                                except Exception as exc:
                                    result = exporter.build_error(key, exc, duration=time.perf_counter() - t0)
                                exporter.write(result)
                                self.log.log(f"[MC Éco] {n_done_local}/{n_batch} | {key}", "NORMAL")

                            # Libère le memmap et force le garbage collector
                            provider._covariance_path.close()
                            provider._path_H = None
                            _gc.collect()

                        else:
                            # Mode rebal : chaque worker calcule la covariance à la volée
                            raw_results = Parallel(n_jobs=params["n_jobs"], backend="loky", return_as="generator")(
                                delayed(_scenario_worker)(
                                    spec=spec, rolling=rolling, exclude_frac=excl, seed=seed,
                                    rebal_freq=freq, data_freq=data_freq, key=key,
                                    all_returns=all_returns, universe_returns=universe_returns,
                                    bench_weights=bench_weights, prices=prices,
                                    rebal_dates_port=rebal_dates_port,
                                    rebal_dates_bench=rebal_dates_bench,
                                    common_start=common_start, use_memmap=False,
                                    forced_exclusions=forced_exclusions,
                                )
                                for key, excl, seed, freq in scenarios_todo
                            )

                            # Collecte les résultats au fil de l'eau et les écrit dans l'exporter
                            for eval_res, key in raw_results:
                                n_done_local += 1
                                t0 = time.perf_counter()
                                try:
                                    result = exporter.build_result(key, eval_res, ann_factor=ann_factor_daily, duration=time.perf_counter() - t0)
                                except Exception as exc:
                                    result = exporter.build_error(key, exc, duration=time.perf_counter() - t0)
                                exporter.write(result)
                                self.log.log(f"[MC Éco] {n_done_local}/{n_batch} | {key}", "NORMAL")

                # Flush final pour s'assurer que tous les résultats bufferisés sont écrits
                exporter.flush()
                self.log.log(f"[MC Éco] ✅ {exporter.n_done()} scénarios exportés → {excel_path}", "INFO")

            # ================================================================
            # Branche Monte Carlo statistique
            # ================================================================
            if params["make_mc_stats"]:
                self.log.log("[MC Stats] Démarrage de l'évaluation statistique...", "INFO")

                params["mc_stat_output"].parent.mkdir(parents=True, exist_ok=True)
                stat_exporter = StatMonteCarloExporter(
                    path=params["mc_stat_output"], flush_every=params["flush_every"],
                )
                self.log.log(f"[MC Stats] {stat_exporter.n_done()} scénarios déjà calculés (checkpoint).", "INFO")

                # Construit la configuration de simulation statistique depuis les paramètres UI
                stat_cfg = StatSimConfig(
                    dgp_type     = params["mc_dgp_type"],
                    N_sim        = params["mc_n_sim"],
                    n_factors    = params["mc_n_factors"],
                    n_scenarios  = params["mc_n_scenarios"],
                    innovation   = params["mc_innovation"],
                    p_ratio      = params["mc_p_ratio"],
                    rho_B        = params["mc_rho_B"],
                    sigma_B      = params["mc_sigma_B"],
                    rho_d        = params["mc_rho_d"],
                    sigma_d      = params["mc_sigma_d"],
                    random_state = params["mc_random_state"],
                    add_drift    = False,
                    metrics      = ("frobenius", "spectral", "stein"),
                )

                ann_factor_stat    = DataFrequency("daily").ann_factor
                model_configs_stat = params["model_configs_stat"]

                # Construit les ModelSpec depuis la liste (fenêtre en années * ann_factor)
                model_specs_stat = []
                for cfg_s in model_configs_stat:
                    wf      = float(cfg_s.get("window_factor", 1))
                    rolling = int(wf * ann_factor_stat)
                    spec    = self._build_spec_from_model_cfg(cfg_s, rolling, "clarabel")
                    model_specs_stat.append(spec)

                self.log.log(
                    f"[MC Stats] {len(model_specs_stat)} modèle(s) | DGP={params['mc_dgp_type']} | "
                    f"N={params['mc_n_sim']} | scénarios={params['mc_n_scenarios']}...",
                    "INFO"
                )

                # R_ref synthétique : sert uniquement à fournir N, les noms et l'index de dates
                import numpy as np
                _N   = params["mc_n_sim"]
                _T   = max(int(float(c.get("window_factor", 1)) * ann_factor_stat) for c in model_configs_stat) + 300
                _idx = pd.bdate_range("2000-01-01", periods=_T)
                _R_ref = pd.DataFrame(
                    np.zeros((_T, _N)),
                    index=_idx,
                    columns=[f"A{i:04d}" for i in range(_N)]
                )

                try:
                    ModelEvaluator.run_stat_evaluation(
                        R_ref       = _R_ref,
                        model_specs = model_specs_stat,
                        cfg         = stat_cfg,
                        exporter    = stat_exporter,
                    )
                except Exception as e:
                    self.log.log(f"[MC Stats] Erreur : {e}", "WARNING")

                # Flush final pour s'assurer que tous les résultats bufferisés sont écrits
                stat_exporter.flush()
                self.log.log(
                    f"[MC Stats] ✅ {stat_exporter.n_done()} scénarios exportés → {params['mc_stat_output']}",
                    "INFO"
                )

            self.log.log("=" * 60, "INFO")
            self.log.log("Monte Carlo terminé avec succès !", "INFO")

        except Exception:
            self.log.log("[ERREUR] " + traceback.format_exc(), "ERROR")
        finally:
            # Réactive le bouton même en cas d'erreur
            self.run_btn.set_running(False)




class CovModelisationApp(tk.Tk):
    """
    Classe principale de l'application Tkinter QuantPortfolioEngine.

    Instancie la fenêtre principale avec un PanedWindow vertical contenant :
    - Un Notebook avec les onglets Estimation et Monte Carlo.
    - Une console de log en bas pour le suivi des opérations.

    Methods
    -------
    _build_ui() -> None :
        Construit la structure principale de la fenêtre.
    """

    def __init__(self):
        """Initialise la fenêtre principale, configure les dimensions et construit l'UI."""
        super().__init__()
        self.title("QuantPortfolioEngine — Évaluation des modèles de covariance")
        self.geometry("1050x820")
        self.minsize(900, 600)

        # Charge l'icône si disponible (ignore l'erreur si absente)
        try:
            self.iconbitmap(str(Path(__file__).parent / "icon.ico"))
        except Exception:
            pass

        self._build_ui()

    def _build_ui(self):
        """
        Construit la structure principale de la fenêtre.

        Crée un PanedWindow vertical avec le Notebook (onglets) en haut
        et la console de log en bas. Le sash est positionné après rendu
        pour que la console occupe une petite zone initiale.
        """
        # PanedWindow vertical : onglets en haut, console en bas
        main_pane = tk.PanedWindow(self, orient="vertical", sashrelief="raised")
        main_pane.pack(fill="both", expand=True, padx=6, pady=6)

        # Frame supérieure pour le Notebook
        notebook_frame = tk.Frame(main_pane)
        main_pane.add(notebook_frame, minsize=400)
        notebook = ttk.Notebook(notebook_frame)
        notebook.pack(fill="both", expand=True)

        # Console de log en bas du PanedWindow
        log_frame = SectionFrame(main_pane, "Console de log")
        main_pane.add(log_frame, minsize=60)
        self.log = LogConsole(log_frame, height=4)
        self.log.pack(fill="both", expand=True)
        tk.Button(
            log_frame, text="🗑  Vider la console", relief="flat",
            command=self.log.clear, cursor="hand2", fg="gray"
        ).pack(anchor="e", padx=4, pady=2)

        # Création et ajout des deux onglets
        tab_estimation = EstimationTab(notebook, log=self.log)
        notebook.add(tab_estimation, text="  📊  Estimation  ")

        tab_montecarlo = MonteCarloTab(notebook, log=self.log)
        notebook.add(tab_montecarlo, text="  🎲  Monte Carlo  ")

        # Messages de démarrage dans la console
        self.log.log(" Application démarrée. Configurez les paramètres et cliquez sur Lancer.", "INFO")
        self.log.log(f"    Racine projet : {ROOT}", "NORMAL")

        # Positionne le sash après rendu pour que la console occupe environ 110px
        def _set_sash():
            total = main_pane.winfo_height()
            if total > 200:
                main_pane.sash_place(0, 0, total - 110)
        self.after(100, _set_sash)


if __name__ == "__main__":
    app = CovModelisationApp()
    app.mainloop()