# -*- coding: utf-8 -*-
"""
Configuration des modèles de covariance depuis l'interface graphique.

Ce fichier expose les composants Tkinter permettant à l'utilisateur de sélectionner,
configurer et gérer une liste de modèles de covariance à évaluer.
Il est utilisé dans les deux onglets de l'application (Estimation et Monte Carlo).

Constantes
----------
COMPUTE_MODE_DEFAULT :
    Mapping model_type -> compute_mode par défaut. 
MODEL_CATALOG :
    Catalogue des modèles disponibles avec leurs paramètres configurables.
    Chaque entrée définit le label affiché, la méthode covariance_provider,
    la variante LW (si applicable), et la liste des paramètres avec leurs valeurs par défaut.

Classes
-------
ModelRow :
    Ligne de configuration d'un modèle unique dans l'interface.
    Affiche dynamiquement les paramètres du modèle sélectionné.
ModelListPanel :
    Panneau de gestion de la liste complète des modèles.
    Permet d'ajouter, configurer et supprimer des modèles.
    Prépopulé avec les modèles par défaut du projet.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any


# Compute mode par défaut par type de modèle.
# EWMA doit impérativement être en mode 'path' car sa récurrence exige
# de calculer la covariance sur le path complet des rendements.
COMPUTE_MODE_DEFAULT = {
    "Rolling"  : "rebal",
    "EWMA"     : "path",
    "LW_2004"  : "rebal",
    "ANLS_2020": "rebal",
    "QIS"  : "rebal",
    "OAS" : "rebal",
}


# Catalogue complet des modèles disponibles dans l'interface.
# Structure de chaque entrée :
#   label      : nom affiché à l'utilisateur dans la combobox
#   method     : méthode passée à make_cov_config() ('rolling', 'ledoit_wolf', 'ewma')
#   params     : liste des paramètres configurables, chacun avec key, label et default
#   lw_variant : variante Ledoit-Wolf (uniquement pour method='ledoit_wolf')

MODEL_CATALOG = {
    "Rolling": {
        "label"  : "Rolling (fenêtre glissante)",
        "method" : "rolling",
        "params" : [
            {"key": "window_factor", "label": "Fenêtre (× ann_factor)", "default": "1"},
            {"key": "rolling_ddof",  "label": "ddof",                   "default": "1"},
        ],
    },
    "EWMA": {
        "label"  : "EWMA (lissage exponentiel)",
        "method" : "ewma",
        "params" : [
            {"key": "window_factor", "label": "Fenêtre warmup (× ann_factor)", "default": "1"},
            {"key": "ewma_lambda",   "label": "Lambda",                        "default": "0.94"},
            {"key": "ewma_init",     "label": "Init (scov / diag)",            "default": "scov"},
            {"key": "tune_lambda",   "label": "Tune lambda (True/False)",      "default": "False"},
        ],
    },
    "LW_2004": {
        "label"  : "Ledoit-Wolf 2004 (linéaire)",
        "method" : "ledoit_wolf",
        "params" : [
            {"key": "window_factor", "label": "Fenêtre (× ann_factor)",       "default": "1"},
            {"key": "use_package",   "label": "Package sklearn (True/False)", "default": "False"},
            {"key": "lw_demean",     "label": "Demean (True/False)",          "default": "True"},
            {"key": "lw_ddof",       "label": "ddof",                         "default": "0"},
        ],
        "lw_variant": "lw_2004",
    },
    "ANLS_2020": {
        "label"  : "ANLS 2020 (non-linéaire)",
        "method" : "ledoit_wolf",
        "params" : [
            {"key": "window_factor", "label": "Fenêtre (× ann_factor)", "default": "1"},
            {"key": "lw_demean",     "label": "Demean (True/False)",    "default": "True"},
            {"key": "lw_ddof",       "label": "ddof",                   "default": "0"},
            {"key": "chunk_size",    "label": "Chunk size",             "default": "1024"},
        ],
        "lw_variant": "anls_2020",
    },
    "QIS": {
        "label"  : "QIS 2022 (Bernoulli)",
        "method" : "ledoit_wolf",
        "params" : [
            {"key": "window_factor", "label": "Fenêtre (× ann_factor)", "default": "1"},
            {"key": "lw_demean",     "label": "Demean (True/False)",    "default": "True"},
            {"key": "lw_ddof",       "label": "ddof",                   "default": "0"},
            {"key": "chunk_size",    "label": "Chunk size",             "default": "1024"},
        ],
        "lw_variant": "QIS",
    },
    "OAS": {
        "label"  : "OAS (Oracle Approximating Shrinkage)",
        "method" : "ledoit_wolf",
        "params" : [
            {"key": "window_factor", "label": "Fenêtre (× ann_factor)",       "default": "1"},
            {"key": "use_package",   "label": "Package sklearn (True/False)", "default": "True"},
            {"key": "lw_demean",     "label": "Demean (True/False)",          "default": "True"},
            {"key": "lw_ddof",       "label": "ddof",                         "default": "0"},
        ],
        "lw_variant": "OAS",
    },
}


class ModelRow(tk.Frame):
    """
    Classe contenant les widgets de configuration d'un modèle unique dans l'interface.

    Affiche une ligne avec un nom personnalisable, un sélecteur de type de modèle,
    un sélecteur de compute_mode, et des champs de paramètres générés dynamiquement
    depuis MODEL_CATALOG selon le type sélectionné.

    Methods
    -------
    get_config() -> dict :
        Collecte et retourne la configuration du modèle sous forme de dictionnaire.
    get_name() -> str :
        Retourne le nom saisi par l'utilisateur.
    """

    def __init__(self, parent, on_delete_callback, row_index: int, **kwargs):
        """
        Initialise la ligne avec les widgets de base et affiche les paramètres par défaut.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        on_delete_callback : callable
            Fonction appelée avec self comme argument quand l'utilisateur clique sur Supprimer.
        row_index : int
            Indice de la ligne utilisé pour générer le nom par défaut ('Modele_{row_index+1}').
        """
        super().__init__(parent, relief="groove", bd=1, padx=6, pady=4, **kwargs)

        self.on_delete_callback = on_delete_callback
        self.row_index          = row_index

        # Dictionnaire des widgets de paramètres : key -> StringVar
        self._param_widgets: Dict[str, tk.StringVar] = {}

        # Ligne du haut : nom personnalisé, type de modèle, compute_mode, bouton supprimer
        top = tk.Frame(self)
        top.pack(fill="x")

        tk.Label(top, text="Nom :", width=5).pack(side="left")

        # Champ de saisie du nom personnalisé du modèle
        self.name_var = tk.StringVar(value=f"Modele_{row_index + 1}")
        tk.Entry(top, textvariable=self.name_var, width=16).pack(side="left", padx=4)

        tk.Label(top, text="Type :").pack(side="left", padx=(10, 2))

        # Combobox de sélection du type de modèle parmi ceux du catalogue
        self.type_var = tk.StringVar(value=list(MODEL_CATALOG.keys())[0])
        self.combo_type = ttk.Combobox(
            top, textvariable=self.type_var,
            values=list(MODEL_CATALOG.keys()), state="readonly", width=14
        )
        self.combo_type.pack(side="left", padx=4)
        # Rafraîchit les paramètres dynamiques quand le type change
        self.combo_type.bind("<<ComboboxSelected>>", self._refresh_params)

        tk.Label(top, text="Mode :").pack(side="left", padx=(10, 2))

        # Combobox du compute_mode : mis à jour automatiquement selon le type de modèle
        self.compute_mode_var = tk.StringVar(value="rebal")
        self.combo_mode = ttk.Combobox(
            top, textvariable=self.compute_mode_var,
            values=["rebal", "path"], state="readonly", width=6
        )
        self.combo_mode.pack(side="left", padx=2)

        # Bouton rouge de suppression de la ligne
        tk.Button(
            top, text="Supprimer", fg="white", bg="#d13438", relief="flat",
            command=lambda: self.on_delete_callback(self), cursor="hand2"
        ).pack(side="right", padx=4)

        # Cadre qui accueille les paramètres dynamiques du modèle sélectionné
        self.params_frame = tk.Frame(self)
        self.params_frame.pack(fill="x", pady=(4, 0))

        # Affiche les paramètres du modèle sélectionné par défaut
        self._refresh_params()

    def _refresh_params(self, event=None):
        """
        Détruit les anciens widgets de paramètres et crée les nouveaux selon le type sélectionné.

        Appelée automatiquement à l'initialisation et à chaque changement de type de modèle.
        Les paramètres sont affichés sur 3 colonnes pour économiser l'espace vertical.
        """
        # Supprime les anciens widgets de paramètres avant de les recréer
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self._param_widgets.clear()

        model_type = self.type_var.get()
        if model_type not in MODEL_CATALOG:
            return

        # Met à jour le compute_mode par défaut selon le type de modèle sélectionné
        self.compute_mode_var.set(COMPUTE_MODE_DEFAULT.get(model_type, "rebal"))

        params = MODEL_CATALOG[model_type]["params"]

        # Dispose les paramètres sur 3 colonnes pour économiser l'espace vertical
        for i, param in enumerate(params):
            col = i % 3
            row = i // 3

            # Chaque paramètre est dans sa propre cellule de grille
            cell = tk.Frame(self.params_frame)
            cell.grid(row=row, column=col, padx=6, pady=1, sticky="w")

            tk.Label(cell, text=param["label"] + " :", width=26, anchor="w", font=("Segoe UI", 8)).pack(side="left")

            # Variable et champ de saisie pour la valeur du paramètre
            var = tk.StringVar(value=str(param["default"]))
            tk.Entry(cell, textvariable=var, width=10, font=("Segoe UI", 8)).pack(side="left")

            # Stocke la variable pour la récupérer dans get_config()
            self._param_widgets[param["key"]] = var

    def get_config(self) -> Dict[str, Any]:
        """
        Collecte la configuration de ce modèle sous forme de dictionnaire.

        Tente de convertir automatiquement les valeurs saisies en bool, int ou float.
        Les valeurs non convertibles sont conservées en string.

        Returns
        -------
        dict
            Configuration complète du modèle avec les clés : name, model_type, method,
            compute_mode, lw_variant (si applicable), et tous les paramètres saisis.
        """
        model_type = self.type_var.get()
        catalog    = MODEL_CATALOG[model_type]

        # Construit le dictionnaire de base avec les champs communs à tous les modèles
        config = {
            "name" : self.name_var.get().strip(),
            "model_type" : model_type,
            "method" : catalog["method"],
            "compute_mode" : self.compute_mode_var.get(),
        }

        # Ajoute la variante LW uniquement pour les modèles Ledoit-Wolf
        if "lw_variant" in catalog:
            config["lw_variant"] = catalog["lw_variant"]

        # Récupère et convertit chaque paramètre saisi
        for key, var in self._param_widgets.items():
            val_str = var.get().strip()

            if val_str.lower() in ("true", "false"):
                # Convertit les chaînes booléennes en bool Python
                config[key] = val_str.lower() == "true"
            else:
                try:
                    # Convertit en float si la valeur contient un point, sinon en int
                    config[key] = float(val_str) if "." in val_str else int(val_str)
                except ValueError:
                    # Conserve en string si la conversion échoue
                    config[key] = val_str

        return config

    def get_name(self) -> str:
        """Retourne le nom saisi par l'utilisateur pour ce modèle."""
        return self.name_var.get().strip()


class ModelListPanel(tk.Frame):
    """
    Classe contenant les widgets de gestion de la liste complète des modèles à évaluer.

    Affiche un panneau scrollable avec une ligne ModelRow par modèle configuré.
    Prépopulé au démarrage avec les modèles par défaut du projet (QIS, ANLS, LW, Rolling, EWMA).

    Methods
    -------
    get_all_configs() -> list :
        Retourne la liste des configurations de tous les modèles actifs.
    """

    def __init__(self, parent, **kwargs):
        """Initialise le panneau avec la barre d'ajout et la zone scrollable."""
        super().__init__(parent, **kwargs)

        # Liste des lignes de modèles actives
        self._rows: List[ModelRow] = []

        # Compteur pour générer des noms par défaut uniques
        self._counter = 0

        # Barre du haut avec le titre et le bouton d'ajout
        bar = tk.Frame(self)
        bar.pack(fill="x", pady=(0, 6))
        tk.Label(bar, text="Modèles à évaluer :", font=("Segoe UI", 9, "bold")).pack(side="left")
        tk.Button(bar, text="＋  Ajouter un modèle", command=self._add_row, bg="#107c10", fg="white", relief="flat", padx=8, cursor="hand2").pack(side="left", padx=10)

        # Conteneur avec relief pour délimiter visuellement la zone scrollable
        container = tk.Frame(self, relief="sunken", bd=1)
        container.pack(fill="both", expand=True)

        # Canvas + scrollbar pour permettre le défilement quand beaucoup de modèles
        canvas    = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas)

        # Met à jour la scrollregion quand le contenu change de taille
        self.scroll_frame.bind( "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")) )

        # Crée la fenêtre interne du canvas pointant sur scroll_frame
        self.canvas_window = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        def _on_canvas_configure(event):
            """Adapte la largeur de scroll_frame à celle du canvas."""
            canvas.itemconfig(self.canvas_window, width=event.width)

        # S'assure que les ModelRow occupent toute la largeur disponible
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Prépopule avec les modèles par défaut du projet
        self._add_default_models()

    def _add_default_models(self):
        """
        Prépopule la liste avec les modèles utilisés par défaut dans le projet.

        Crée une ligne pour chaque combinaison (modèle, fenêtre) standard :
        QIS, ANLS, LW, Rolling et EWMA en fenêtres 1 an (252j) et 2 ans (504j).
        """
        defaults = [
            ("QIS_252",     "QIS",      "1"),
            ("QIS_504",     "QIS",      "2"),
            ("ANLS_252",    "ANLS_2020","1"),
            ("ANLS_504",    "ANLS_2020","2"),
            ("LW_252",      "LW_2004",  "1"),
            ("LW_504",      "LW_2004",  "2"),
            ("Rolling_252", "Rolling",  "1"),
            ("Rolling_504", "Rolling",  "2"),
            ("EWMA_252",    "EWMA",     "1"),
            ("EWMA_504",    "EWMA",     "2"),
        ]

        for name, model_type, window_factor in defaults:
            row = self._add_row(name=name, model_type=model_type)

            # Force la valeur de window_factor après la création de la ligne
            if "window_factor" in row._param_widgets:
                row._param_widgets["window_factor"].set(window_factor)

    def _add_row(self, name: str = None, model_type: str = None) -> ModelRow:
        """
        Ajoute une nouvelle ligne de modèle dans le panneau scrollable.

        Parameters
        ----------
        name : str or None
            Nom personnalisé à pré-remplir. Si None, utilise le nom par défaut.
        model_type : str or None
            Type de modèle à pré-sélectionner dans la combobox. Si None, utilise le premier.

        Returns
        -------
        ModelRow
            L'instance de la ligne créée (permet de modifier ses widgets après création).
        """

        # Crée la ligne avec le compteur courant comme indice
        row = ModelRow(self.scroll_frame, on_delete_callback=self._delete_row,  row_index=self._counter,)
        row.pack(fill="x", padx=4, pady=3)

        # Applique les valeurs personnalisées si fournies
        if name:
            row.name_var.set(name)

        if model_type and model_type in MODEL_CATALOG:
            row.type_var.set(model_type)

            # Rafraîchit les paramètres pour correspondre au type imposé
            row._refresh_params()

        self._rows.append(row)
        self._counter += 1
        return row

    def _delete_row(self, row: ModelRow):
        """
        Supprime une ligne de modèle après confirmation de l'utilisateur.

        Parameters
        ----------
        row : ModelRow
            Ligne à supprimer de la liste et de l'interface.
        """
        if messagebox.askyesno("Confirmer", f"Supprimer le modèle « {row.get_name()} » ?"):
            # Détruit les widgets Tkinter et retire la ligne de la liste
            row.destroy()
            self._rows.remove(row)

    def get_all_configs(self) -> List[Dict[str, Any]]:
        """
        Collecte et retourne la configuration de tous les modèles actifs.

        Ignore les lignes dont les widgets ont été détruits (sécurité Tkinter).

        Returns
        -------
        list of dict
            Liste des configurations prêtes à être transformées en ModelSpec
            par _build_model_specs() ou _build_spec_from_model_cfg().
        """
        configs = []
        for row in self._rows:

            # Vérifie que la ligne n'a pas été détruite entre-temps
            if row.winfo_exists():
                configs.append(row.get_config())
        return configs