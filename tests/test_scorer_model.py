"""
Tests du moteur Poisson du meilleur buteur (distribution conjointe Favori/Buteur).

Couvre :
  - top_scorer_joint_probs : loi sommant à 1, gel par élimination (M=0), cas
    déterministe alpha=0 (= current_goals), gestion des égalités (ex-aequo).
  - build_terminal_state : demi-points (bonus /2), conditions de victoire, et
    cohérence avec la loi jointe injectée.
  - calibrate_scorer_alphas : retrouve une cible P(meilleur buteur) imposée.
  - load_scorer_players : lecture du State-Tracker (ghosts / 'autre').
"""
import numpy as np
import pandas as pd
import pytest

from mpp_project.end_game_solver import build_terminal_state, GAP_OFFSET
from mpp_project.scorer_model import (
    top_scorer_joint_probs, calibrate_scorer_alphas, load_scorer_players, _calib_top_counts,
)

RNG_SEED = 12345


def _members(n, idx):
    m = np.zeros(n, dtype=np.bool_)
    if idx is not None:
        m[idx] = True
    return m


# ---------------------------------------------------------------------------
# 1. LOI CONJOINTE
# ---------------------------------------------------------------------------
def test_joint_sums_to_one():
    np.random.seed(RNG_SEED)
    n = 5
    alpha = np.array([0.6, 0.5, 0.4, 0.3, 0.2])
    cg = np.zeros(n)
    M = np.array([5.0, 5.0, 4.0, 3.0, 2.0])
    j = top_scorer_joint_probs(alpha, cg, M, _members(n, 0), _members(n, 1), _members(n, 2), 5000)
    assert j.shape == (8,)
    assert j.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(j >= 0.0)


def test_eliminated_player_never_top():
    """Joueur gelé (M=0, current_goals=0) face à des actifs : ~jamais meilleur buteur."""
    np.random.seed(RNG_SEED)
    n = 3
    alpha = np.array([0.5, 0.5, 0.5])
    cg = np.zeros(n)
    M = np.array([0.0, 5.0, 5.0])          # joueur 0 éliminé
    j = top_scorer_joint_probs(alpha, cg, M, _members(n, 0), _members(n, 1), _members(n, 2), 20000)
    p_my_top = j[4] + j[5] + j[6] + j[7]   # my_top bit
    assert p_my_top < 0.02


def test_alpha_zero_is_deterministic_current_goals():
    """alpha=0 -> Y_final = current_goals (déterministe) : le plus de buts est top."""
    np.random.seed(RNG_SEED)
    n = 3
    alpha = np.zeros(n)
    cg = np.array([3.0, 1.0, 0.0])         # joueur 0 a le plus de buts
    M = np.array([5.0, 5.0, 5.0])
    j = top_scorer_joint_probs(alpha, cg, M, _members(n, 0), _members(n, 1), _members(n, 2), 1000)
    assert j[4] == pytest.approx(1.0, abs=1e-9)   # my_top seul


def test_ties_make_multiple_top():
    """Égalité au sommet (deux joueurs à current_goals max, alpha=0) -> les deux top."""
    np.random.seed(RNG_SEED)
    n = 3
    alpha = np.zeros(n)
    cg = np.array([2.0, 2.0, 1.0])         # joueurs 0 et 1 ex-aequo
    M = np.array([5.0, 5.0, 5.0])
    j = top_scorer_joint_probs(alpha, cg, M, _members(n, 0), _members(n, 1), _members(n, 2), 1000)
    assert j[6] == pytest.approx(1.0, abs=1e-9)   # my_top=1, bob_top=1, pack_top=0 (combo 110)


def test_shared_pick_correlated():
    """Bob et le peloton parient le MÊME buteur -> leurs indicateurs sont identiques."""
    np.random.seed(RNG_SEED)
    n = 4
    alpha = np.array([0.4, 0.4, 0.4, 0.4])
    cg = np.zeros(n)
    M = np.array([5.0, 5.0, 5.0, 5.0])
    same = _members(n, 1)                  # Bob et peloton -> joueur 1
    j = top_scorer_joint_probs(alpha, cg, M, _members(n, 0), same, same, 8000)
    # bob_top != pack_top est impossible : combos où bits bob (val 2) et pack (val 1)
    # diffèrent -> idx ∈ {1 (001), 2 (010), 5 (101), 6 (110)}.
    p_bob_ne_pack = j[1] + j[2] + j[5] + j[6]
    assert p_bob_ne_pack == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. ÉTAT TERMINAL (demi-points + loi jointe)
# ---------------------------------------------------------------------------
def test_terminal_half_points_favorite():
    """Bonus favori 90 (plein) -> +45 demi-points : à gap -44 demi-pts, l'agent passe
    devant si son favori est champion (axe my_fav=1)."""
    jz = np.zeros(8)
    jz[0] = 1.0                            # aucun buteur top
    V = build_terminal_state(90.0, 0.0, 0.0, 0.0, 0.0, 0.0, jz)
    g1 = -44 + GAP_OFFSET
    g2 = 200 + GAP_OFFSET                  # peloton neutralisé (très en retard)
    # favori mort -> reste à -44 < 0 -> défaite ; favori champion -> -44+45=+1 > 0 -> victoire
    assert float(V[g1, g2, 0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(V[g1, g2, 0, 1, 0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_terminal_scorer_joint_weighting():
    """V_term = moyenne pondérée par la loi jointe des issues buteur."""
    # 60% mon buteur top (+100 plein = +50 demi), 40% personne -> à gap 0 :
    #   - mon buteur top : +50 vs Bob et peloton -> victoire (1.0)
    #   - personne : gap 0/0 -> 1/3
    j = np.zeros(8)
    j[4] = 0.6     # my_top
    j[0] = 0.4     # rien
    V = build_terminal_state(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, j)
    val = float(V[GAP_OFFSET, GAP_OFFSET, 0, 0, 0, 0])
    assert val == pytest.approx(0.6 * 1.0 + 0.4 * (1.0 / 3.0), abs=1e-4)


def test_terminal_scorer_punishes_agent():
    """Le buteur de Bob est top (et pas celui de l'agent) -> l'agent recule."""
    j = np.zeros(8)
    j[2] = 1.0     # bob_top
    V = build_terminal_state(0.0, 0.0, 0.0, 0.0, 100.0, 0.0, j)
    # gap 0 vs Bob, Bob +50 demi -> gap -50 < 0 -> défaite
    assert float(V[GAP_OFFSET, GAP_OFFSET, 0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. CALIBRATION
# ---------------------------------------------------------------------------
def test_calibration_recovers_target():
    """Avec un nombre de matchs IDENTIQUE pour tous, la calibration (lien paramétrique
    + champ de fond) doit retrouver l'ORDRE d'une cible P(meilleur buteur) imposée et
    rester proche en valeur."""
    nation_id = np.array([0, 1, 2], dtype=np.int64)
    field_nation_id = np.array([0, 1, 2], dtype=np.int64)
    team_matches_runs = np.full((1, 3), 5.0)           # M=5 identique
    target = np.array([0.5, 0.3, 0.2])

    alpha, info = calibrate_scorer_alphas(
        target, nation_id, team_matches_runs, field_nation_id, g_fav=4.0,
        p_grid=np.arange(0.3, 1.01, 0.1), field_grid=np.arange(0.0, 0.6, 0.1),
        n_poisson_per_run=3000, refine=True, verbose=False, seed=99,
    )
    sim = info["sim_probs"]
    assert np.argsort(-sim).tolist() == np.argsort(-target).tolist()
    assert info["rankcorr"] > 0.95
    assert info["mae"] < 0.05
    assert alpha.shape == (3,)


def test_calibration_monotone_alpha():
    """À M égal, P(meilleur buteur) simulé croît avec alpha."""
    nation_id = np.array([0, 1, 2], dtype=np.int64)
    M = np.full((1, 3), 5.0)
    cg = np.zeros(3)
    low = _calib_top_counts(np.array([0.1, 0.5, 0.9]), cg, nation_id, M, 4000)
    assert low[0] < low[1] < low[2]


# ---------------------------------------------------------------------------
# 4. LECTURE DU STATE-TRACKER
# ---------------------------------------------------------------------------
def test_load_scorer_players():
    df = pd.DataFrame({
        "category": ["favorite", "scorer", "scorer", "scorer", "scorer", "field"],
        "selection": ["france", "harry_kane", "autre", "michael_olise", "kylian_mbappe", "field"],
        "cote": [5.75, 7.5, np.nan, 26.0, 6.49, np.nan],
        "gain_mpp": [90, 100, 150, np.nan, 80, np.nan],
        "crowd": [0.35, 0.1, np.nan, 0.09, 0.51, np.nan],
        "nation": ["france", "angleterre", "", "france", "france", ""],
        "current_goals": [np.nan, 1, np.nan, 0, 2, np.nan],
        "is_eliminated": ["", "", "", "true", "", ""],
        "alpha": [np.nan, 0.5, np.nan, 0.3, 0.6, 0.12],
    })
    players, autre_gain, field_alpha = load_scorer_players(df)
    assert autre_gain == 150
    assert field_alpha == pytest.approx(0.12)
    assert len(players) == 3                         # named + ghost, hors favori et 'autre'
    assert set(players["selection"]) == {"harry_kane", "michael_olise", "kylian_mbappe"}
    olise = players[players["selection"] == "michael_olise"].iloc[0]
    assert bool(olise["is_ghost"]) is True           # pas de gain_mpp
    assert bool(olise["is_eliminated"]) is True
    kane = players[players["selection"] == "harry_kane"].iloc[0]
    assert bool(kane["is_ghost"]) is False
    assert kane["current_goals"] == 1.0
