# -*- coding: utf-8 -*-
"""
Composants Tkinter génériques et réutilisables pour l'interface graphique.

Ce fichier regroupe tous les widgets personnalisés partagés entre les onglets
de l'application. Chaque composant encapsule un pattern d'interface récurrent
(label + champ, label + menu, console de log, etc.) pour éviter la duplication
de code dans les onglets.

Composants disponibles
----------------------
LabeledEntry :
    Champ texte (Entry) avec label sur la gauche. Pour les paramètres simples
    (dates, fenêtres de rolling, fractions d'exclusion, etc.).
LabeledCombobox :
    Menu déroulant (Combobox) avec label sur la gauche. Pour les choix fermés
    (fréquence, méthode, optimiseur, etc.).
LabeledCheckbox :
    Case à cocher (Checkbutton) avec label. Pour les booléens (activer/désactiver
    une option).
LogConsole :
    Console de log défilante à fond sombre (style terminal). Supporte des messages
    colorés par niveau (INFO, WARNING, ERROR, NORMAL).
SectionFrame :
    Cadre LabelFrame avec titre en gras. Pour regrouper visuellement des paramètres
    de même nature.
ListEditor :
    Liste éditable avec boutons Ajouter/Supprimer et champ de saisie. Pour les listes
    de valeurs numériques (seeds, fenêtres, fréquences, fractions).
RunButton :
    Bouton de lancement avec gestion de l'état d'exécution (désactivation + label
    de statut pendant le calcul).
"""

import tkinter as tk
from tkinter import ttk, scrolledtext


class LabeledEntry(tk.Frame):
    """
    Classe contenant un champ texte (Entry) avec un label fixe sur la gauche.

    Composant de base pour la saisie de paramètres simples : dates, nombres,
    chemins, etc. Le label est aligné à gauche sur une largeur fixe pour que
    plusieurs LabeledEntry empilés restent alignés verticalement.

    Attributes
    ----------
    var : tk.StringVar
        Variable Tkinter liée au champ de saisie.
    entry : tk.Entry
        Widget Entry sous-jacent (accessible pour binding éventuel).

    Methods
    -------
    get() -> str :
        Retourne la valeur saisie sans espaces superflus.
    set(value) -> None :
        Définit la valeur affichée dans le champ.
    """

    def __init__(self, parent, label: str, default: str = "", width: int = 20, **kwargs):
        """
        Initialise le composant avec son label et sa valeur par défaut.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        label : str
            Texte du label affiché à gauche du champ.
        default : str
            Valeur initiale du champ de saisie.
        width : int
            Largeur en caractères du champ Entry.
        """
        super().__init__(parent, **kwargs)

        # Label aligné à gauche sur une largeur fixe pour l'alignement vertical
        tk.Label(self, text=label, anchor="w", width=25).pack(side="left")

        # Variable et champ de saisie liés
        self.var   = tk.StringVar(value=str(default))
        self.entry = tk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side="left", padx=5)

    def get(self) -> str:
        """Retourne la valeur saisie sans espaces superflus."""
        return self.var.get().strip()

    def set(self, value):
        """Définit la valeur affichée dans le champ."""
        self.var.set(str(value))


class LabeledCombobox(tk.Frame):
    """
    Classe contenant un menu déroulant (Combobox) en lecture seule avec un label sur la gauche.

    Utilisé pour les choix fermés parmi une liste fixe : fréquence de données,
    méthode de calcul des rendements, optimiseur, etc.

    Attributes
    ----------
    var : tk.StringVar
        Variable Tkinter liée à la valeur sélectionnée.
    combo : ttk.Combobox
        Widget Combobox sous-jacent (accessible pour binding éventuel).

    Methods
    -------
    get() -> str :
        Retourne la valeur sélectionnée sans espaces superflus.
    set(value) -> None :
        Définit la valeur sélectionnée.
    """

    def __init__(self, parent, label: str, options: list, default: str = "", width: int = 18, **kwargs):
        """
        Initialise le composant avec son label, ses options et sa valeur par défaut.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        label : str
            Texte du label affiché à gauche du menu.
        options : list of str
            Liste des valeurs proposées dans le menu déroulant.
        default : str
            Valeur sélectionnée par défaut. Si vide, sélectionne la première option.
        width : int
            Largeur en caractères du menu déroulant.
        """
        super().__init__(parent, **kwargs)

        # Label aligné à gauche sur une largeur fixe pour l'alignement vertical
        tk.Label(self, text=label, anchor="w", width=25).pack(side="left")

        # Sélectionne default si fourni, sinon la première option disponible
        self.var   = tk.StringVar(value=default if default else (options[0] if options else ""))
        self.combo = ttk.Combobox(self, textvariable=self.var, values=options, state="readonly", width=width)
        self.combo.pack(side="left", padx=5)

    def get(self) -> str:
        """Retourne la valeur sélectionnée sans espaces superflus."""
        return self.var.get().strip()

    def set(self, value: str):
        """Définit la valeur sélectionnée."""
        self.var.set(str(value))


class LabeledCheckbox(tk.Frame):
    """
    Classe contenant une case à cocher (Checkbutton) avec un label intégré.

    Utilisé pour les paramètres booléens : activer/désactiver les exclusions
    aléatoires, les exports, les modes de calcul, etc.

    Attributes
    ----------
    var : tk.BooleanVar
        Variable Tkinter liée à l'état de la case.
    check : tk.Checkbutton
        Widget Checkbutton sous-jacent (accessible pour config() éventuel).

    Methods
    -------
    get() -> bool :
        Retourne l'état courant de la case (True si cochée).
    set(value) -> None :
        Définit l'état de la case.
    """

    def __init__(self, parent, label: str, default: bool = False, **kwargs):
        """
        Initialise le composant avec son label et son état par défaut.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        label : str
            Texte affiché à droite de la case à cocher.
        default : bool
            État initial de la case (True = cochée, False = décochée).
        """
        super().__init__(parent, **kwargs)

        # Variable booléenne liée à l'état de la case
        self.var   = tk.BooleanVar(value=default)
        self.check = tk.Checkbutton(self, text=label, variable=self.var, anchor="w")
        self.check.pack(side="left")

    def get(self) -> bool:
        """Retourne l'état courant de la case (True si cochée)."""
        return self.var.get()

    def set(self, value: bool):
        """Définit l'état de la case."""
        self.var.set(bool(value))


class LogConsole(tk.Frame):
    """
    Classe contenant une console de log défilante à fond sombre, style terminal.

    Affiche les messages de l'application en lecture seule avec un code couleur
    par niveau : vert pour INFO, jaune pour WARNING, rouge pour ERROR, gris pour NORMAL.
    Le scroll est automatique vers le bas à chaque nouveau message.

    Attributes
    ----------
    text : scrolledtext.ScrolledText
        Zone de texte scrollable sous-jacente.

    Methods
    -------
    log(message, level) -> None :
        Ajoute une ligne dans la console avec la couleur du niveau.
    clear() -> None :
        Vide complètement la console.
    """

    def __init__(self, parent, height: int = 12, **kwargs):
        """
        Initialise la console avec le fond sombre et les tags de couleur.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        height : int
            Hauteur en lignes de la zone de texte.
        """
        super().__init__(parent, **kwargs)

        # Zone de texte scrollable en lecture seule, style terminal (fond sombre)
        self.text = scrolledtext.ScrolledText(self, state="disabled", height=height, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9), wrap="word")
        self.text.pack(fill="both", expand=True)

        # Tags de couleur pour chaque niveau de log
        self.text.tag_config("INFO",    foreground="#4ec9b0")  # vert cyan
        self.text.tag_config("WARNING", foreground="#dcdcaa")  # jaune
        self.text.tag_config("ERROR",   foreground="#f48771")  # rouge-orangé
        self.text.tag_config("NORMAL",  foreground="#d4d4d4")  # gris clair

    def log(self, message: str, level: str = "NORMAL"):
        """
        Ajoute une ligne dans la console avec la couleur correspondant au niveau.

        Active temporairement l'édition de la zone de texte (state="normal"),
        insère le message, fait défiler vers le bas, puis verrouille à nouveau.

        Parameters
        ----------
        message : str
            Texte du message à afficher.
        level : str
            Niveau du message : 'INFO', 'WARNING', 'ERROR' ou 'NORMAL'.
        """
        # Déverrouille temporairement la zone de texte pour l'écriture
        self.text.config(state="normal")
        self.text.insert("end", message + "\n", level)

        # Fait défiler automatiquement vers le bas pour afficher le dernier message
        self.text.see("end")

        # Reverrouille en lecture seule
        self.text.config(state="disabled")

    def clear(self):
        """Vide complètement la console en supprimant tout le contenu."""
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")


class SectionFrame(tk.LabelFrame):
    """
    Classe contenant un cadre avec titre en gras pour regrouper des paramètres.

    Hérite de tk.LabelFrame et applique un style uniforme (titre en gras,
    marges internes standardisées) pour toutes les sections de l'interface.

    Le titre est entouré d'espaces pour l'aérer visuellement dans le cadre.
    """

    def __init__(self, parent, title: str, **kwargs):
        """
        Initialise le cadre avec son titre et le style standard.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        title : str
            Titre affiché dans le bord supérieur du cadre.
        """
        # Passe le titre entouré d'espaces directement au LabelFrame pour l'aération
        super().__init__(parent, text=f"  {title}  ",  padx=8, pady=6, font=("Segoe UI", 9, "bold"), **kwargs)


class ListEditor(tk.Frame):
    """
    Classe contenant une liste éditable avec champ de saisie et boutons Ajouter/Supprimer.

    Utilisé pour les paramètres multi-valeurs : seeds du Monte Carlo, fenêtres de rolling,
    fréquences de rebalancement, fractions d'exclusion, etc. L'utilisateur peut modifier
    la liste à la volée sans reconfigurer l'application.

    Attributes
    ----------
    listbox : tk.Listbox
        Widget Listbox sous-jacent contenant les valeurs.
    entry_var : tk.StringVar
        Variable liée au champ de saisie pour l'ajout de nouvelles valeurs.

    Methods
    -------
    get_values() -> list :
        Retourne les valeurs sous forme de liste de strings.
    get_floats() -> list :
        Retourne les valeurs converties en float.
    get_ints() -> list :
        Retourne les valeurs converties en int.
    """

    def __init__(self, parent, label: str, default_values: list = None, height: int = 5, width: int = 15, **kwargs):
        """
        Initialise la liste avec son label, ses valeurs par défaut et sa taille.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        label : str
            Titre affiché au-dessus de la liste.
        default_values : list or None
            Valeurs initiales à pré-remplir dans la liste.
        height : int
            Hauteur en lignes de la Listbox.
        width : int
            Largeur en caractères de la Listbox.
        """
        super().__init__(parent, **kwargs)

        # Titre de la liste en gras
        tk.Label(self, text=label, anchor="w", font=("Segoe UI", 9, "bold")).pack(anchor="w")

        # Zone principale : listbox + scrollbar vertical
        frame_list = tk.Frame(self)
        frame_list.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame_list, orient="vertical")
        self.listbox = tk.Listbox(frame_list, height=height, width=width,  yscrollcommand=scrollbar.set, selectmode="extended")

        # Lie la scrollbar à la listbox
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Zone de saisie et boutons Ajouter / Supprimer
        frame_buttons = tk.Frame(self)
        frame_buttons.pack(fill="x", pady=3)

        self.entry_var = tk.StringVar()
        # Champ de saisie pour la valeur à ajouter
        entry = tk.Entry(frame_buttons, textvariable=self.entry_var, width=12)
        entry.pack(side="left", padx=(0, 4))

        tk.Button(frame_buttons, text="Ajouter",   command=self._add,    width=8).pack(side="left", padx=2)
        tk.Button(frame_buttons, text="Supprimer", command=self._remove, width=8).pack(side="left", padx=2)

        # Prérempli la liste avec les valeurs par défaut si fournies
        if default_values:
            for v in default_values:
                self.listbox.insert("end", str(v))

    def _add(self):
        """
        Ajoute la valeur saisie dans le champ Entry à la fin de la liste.

        Ignore les valeurs vides. Vide le champ après l'ajout.
        """
        val = self.entry_var.get().strip()
        if val:
            self.listbox.insert("end", val)

            # Vide le champ de saisie après ajout pour la prochaine valeur
            self.entry_var.set("")

    def _remove(self):
        """
        Supprime tous les éléments actuellement sélectionnés dans la liste.

        Itère en ordre inverse pour que la suppression ne décale pas les indices.
        """
        for idx in reversed(self.listbox.curselection()):
            self.listbox.delete(idx)

    def get_values(self) -> list:
        """
        Retourne toutes les valeurs de la liste sous forme de strings.

        Returns
        -------
        list of str
            Valeurs dans l'ordre d'insertion.
        """
        return list(self.listbox.get(0, "end"))

    def get_floats(self) -> list:
        """
        Retourne les valeurs converties en float.

        Utilisé pour les fractions d'exclusion, fenêtres en années, etc.

        Returns
        -------
        list of float
            Valeurs converties.
        """
        return [float(v) for v in self.get_values()]

    def get_ints(self) -> list:
        """
        Retourne les valeurs converties en int.

        Utilisé pour les seeds, les nombres de scénarios, etc.

        Returns
        -------
        list of int
            Valeurs converties.
        """
        return [int(v) for v in self.get_values()]


class RunButton(tk.Frame):
    """
    Classe contenant un bouton de lancement avec gestion de l'état pendant l'exécution.

    Affiche un bouton bleu "Lancer" et un label de statut à sa droite.
    Pendant l'exécution, le bouton est désactivé et grisé pour éviter les doubles
    clics, et le label indique que le calcul est en cours. À la fin, le bouton est
    réactivé et le label affiche "Terminé ✓".

    Attributes
    ----------
    btn : tk.Button
        Widget Button sous-jacent.
    status_label : tk.Label
        Label de statut affiché à droite du bouton.

    Methods
    -------
    set_running(is_running) -> None :
        Active ou désactive l'état d'exécution du bouton.
    """

    def __init__(self, parent, label: str = "▶  Lancer", command=None, **kwargs):
        """
        Initialise le bouton avec son label, sa commande et le label de statut.

        Parameters
        ----------
        parent : tk.Widget
            Widget parent dans la hiérarchie Tkinter.
        label : str
            Texte affiché sur le bouton au repos.
        command : callable or None
            Fonction à appeler quand l'utilisateur clique sur le bouton.
        """
        super().__init__(parent, **kwargs)

        # Stocke la commande pour l'appeler dans _run()
        self.command = command

        # Bouton bleu avec style uniforme
        self.btn = tk.Button(self, text=label, command=self._run, bg="#0078d4", fg="white",font=("Segoe UI", 10, "bold"),padx=16, pady=6, relief="flat", cursor="hand2")
        self.btn.pack(side="left", padx=(0, 12))

        # Label de statut vide au démarrage, mis à jour par set_running()
        self.status_label = tk.Label(self, text="", fg="gray", font=("Segoe UI", 9))
        self.status_label.pack(side="left")

    def _run(self):
        """Appelle la commande configurée si elle est définie."""
        if self.command:
            self.command()

    def set_running(self, is_running: bool):
        """
        Active ou désactive l'état d'exécution du bouton.

        Quand is_running=True : désactive le bouton, affiche le message d'attente
        en orange. Quand is_running=False : réactive le bouton, affiche "Terminé ✓"
        en vert.

        Parameters
        ----------
        is_running : bool
            True pour indiquer qu'un calcul est en cours, False pour indiquer la fin.
        """
        if is_running:
            # Désactive le bouton et affiche le message d'attente
            self.btn.config(state="disabled", text="⏳  En cours...")
            self.status_label.config(text="Calcul en cours, merci de patienter...", fg="#f0a500")
        else:
            # Réactive le bouton et affiche la confirmation de fin
            self.btn.config(state="normal", text="▶  Lancer")
            self.status_label.config(text="Terminé ✓", fg="#107c10")