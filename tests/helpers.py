"""
Outils partagés pour les tests de l'Oracle DP.

Importé à la fois par :
  - la suite pytest (tests/test_*.py)
  - les notebooks d'exploration 20, 21, 22 (via `from tests.helpers import ...`)

Le cœur est `exact_theoretical_wr`, un solveur récursif EXACT qui énumère tout
l'arbre combinatoire (action agent × booster × issue réelle × pari de Bob). Il
sert d'oracle de référence pour valider les moteurs DP (qui, eux, passent par une
grille d'états). Le peloton (gap_2) est supposé neutralisé côté DP (saturé ou
fantôme), ce qui réduit le problème à un suivi 1D de l'écart avec Bob.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Chemins projet (réutilisés par les tests et les notebooks)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_TESTS_DIR = DATA_DIR / "tests"


# ---------------------------------------------------------------------------
# Conditions terminales : gap final (en points réels) -> win rate
# ---------------------------------------------------------------------------
def terminal_ge_zero(gap):
    """Terminal des phases finales : V_term construit avec gap >= 0 -> victoire.

    Correspond aux notebooks 20/21 où l'on injecte une matrice terminale
    `(g - offset) >= 0 -> 1.0` dans solve_endgame_dp.
    """
    return 1.0 if gap >= 0 else 0.0


def terminal_strict(gap):
    """Terminal interne de solve_dp_coarse / solve_dp_with_full_empirical_distribution.

    Départage les égalités de points : gap == 0 -> 1/2 (on partage la 1re place).
    """
    if gap > 0:
        return 1.0
    if gap == 0:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Solveur théorique EXACT (arbre combinatoire complet, peloton ignoré)
# ---------------------------------------------------------------------------
def exact_theoretical_wr(gap, n_matchs, true_proba, crowd, gains,
                         has_booster=False, terminal=terminal_ge_zero):
    """
    Win rate exact d'un agent affrontant un unique adversaire (Bob) sur
    `n_matchs` matchs identiques, par énumération exhaustive de l'arbre.

    Paramètres
    ----------
    gap : int
        Écart initial de points avec Bob (positif = l'agent est devant).
    n_matchs : int
        Nombre de matchs restants jusqu'au terminal.
    true_proba, crowd, gains : array-like de taille 3
        Probabilités réelles, répartition de la foule (= proba de pari de Bob),
        et gains MPP pour les issues [1, N, 2].
    has_booster : bool
        L'agent dispose-t-il encore du booster x2 (utilisable une seule fois) ?
    terminal : callable
        Fonction gap_final -> win_rate. Utiliser `terminal_ge_zero` pour les
        phases finales, `terminal_strict` pour le moteur de poules coarse.

    Retour
    ------
    float : win rate optimal (l'agent joue parfaitement action + timing du booster).
    """
    def rec(t, g, hb):
        if t == n_matchs:
            return terminal(g)

        best = 0.0
        booster_options = (False, True) if hb else (False,)
        for my_action in range(3):
            for use_now in booster_options:
                wr = 0.0
                for out in range(3):
                    p_out = true_proba[out]
                    if p_out == 0.0:
                        continue
                    base_g = gains[out] if my_action == out else 0
                    my_g = base_g * 2 if use_now else base_g
                    for bob_a in range(3):
                        p_bob = crowd[bob_a]
                        if p_bob == 0.0:
                            continue
                        bob_g = gains[out] if bob_a == out else 0
                        wr += p_out * p_bob * rec(t + 1, g + my_g - bob_g,
                                                  hb and not use_now)
                if wr > best:
                    best = wr
        return best

    return rec(0, int(gap), has_booster)


# ---------------------------------------------------------------------------
# Constructeurs de fixtures "pipeau"
# ---------------------------------------------------------------------------
def build_pipeau_dataframe(n_matchs, true_proba, crowd, gains, phase="poules"):
    """
    Construit un DataFrame de `n_matchs` matchs IDENTIQUES, compatible avec
    run_daily_pipeline (colonnes phase, cote_*, crowd_*, gain_mpp_*).

    Les cotes sont "justes" (sans marge) : 1/cote normalisé == true_proba, donc
    calculate_true_outcome_probas_from_odds restitue exactement true_proba.
    """
    true_proba = np.asarray(true_proba, dtype=float)
    cotes = 1.0 / true_proba
    rows = [{
        "phase": phase,
        "cote_1": cotes[0], "cote_N": cotes[1], "cote_2": cotes[2],
        "crowd_1": crowd[0], "crowd_N": crowd[1], "crowd_2": crowd[2],
        "gain_mpp_1": int(gains[0]), "gain_mpp_N": int(gains[1]), "gain_mpp_2": int(gains[2]),
    } for _ in range(n_matchs)]
    return pd.DataFrame(rows)


def build_ghost_peloton(n_matchs, max_gain=250):
    """
    Distribution de peloton FANTÔME : toute la masse sur delta_gain = 0.
    Le peloton ne bouge jamais -> couplé à un gap_2 saturé, il n'influence pas
    la décision, ce qui permet la comparaison directe avec le solveur 1D.
    Shape (n_matchs, 3, max_gain).
    """
    p = np.zeros((n_matchs, 3, max_gain), dtype=np.float32)
    p[:, :, 0] = 1.0
    return p
