"""
Tests du moteur des phases finales solve_endgame_dp (notebooks 20 & 21).

On confronte la grille DP (501x501, demi-points) au solveur récursif EXACT
(tests/helpers.exact_theoretical_wr) avec terminal `gap >= 0 -> victoire`,
identique à la matrice terminale V_term injectée.

Convention d'indexation (comme les notebooks 20/21) :
    idx = gap_reel // 2 + 250   (la grille stocke les points divisés par 2)
    gap_2 saturé a l'index max (500) -> seul l'ecart avec Bob compte.
"""
import numpy as np
import pytest

from mpp_project.end_game_solver import solve_endgame_dp
from tests.helpers import exact_theoretical_wr, terminal_ge_zero

TOL = 0.005
OFFSET = 250          # demi-grille : gap 0 -> index 250 (cf. notebooks 20/21)
IDX_G2_SAFE = 500     # peloton/2e adversaire neutralisé (avance maximale)


def _build_endgame_inputs(n_matchs, true_proba, crowd, gains):
    """Prépare les 32 matchs (seuls les n_matchs derniers sont actifs) + V_term."""
    stop_t = 32 - n_matchs

    match_probs = np.zeros((32, 3), dtype=np.float32)
    crowds = np.zeros((32, 3), dtype=np.float32)
    gains_1N2 = np.zeros((32, 3), dtype=np.int32)
    for t in range(stop_t, 32):
        match_probs[t] = true_proba
        crowds[t] = crowd
        gains_1N2[t] = gains

    p_empirique_1D = np.zeros((32, 3, 250), dtype=np.float32)
    p_empirique_1D[:, :, 0] = 1.0          # peloton fantôme
    alphas = np.ones(32, dtype=np.float32)
    roles_mock = np.full(32, -1, dtype=np.int32)

    V_term = np.zeros((501, 501, 2, 2, 2, 2), dtype=np.float32)
    for g1 in range(501):
        for g2 in range(501):
            if (g1 - OFFSET) >= 0 and (g2 - OFFSET) >= 0:
                V_term[g1, g2, :, :, :, :] = 1.0

    return match_probs, crowds, gains_1N2, p_empirique_1D, alphas, roles_mock, V_term, stop_t


@pytest.fixture(scope="module")
def params():
    return {
        "true_proba": np.array([0.60, 0.25, 0.15], dtype=np.float32),
        "crowd": np.array([0.70, 0.20, 0.10], dtype=np.float32),
        "gains": np.array([20, 50, 90], dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# 1. MATCH UNIQUE (notebook 20) : base et booster
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def q_single(params):
    inp = _build_endgame_inputs(1, params["true_proba"], params["crowd"], params["gains"])
    match_probs, crowds, gains_1N2, p_emp, alphas, roles, V_term, stop_t = inp
    V_history = solve_endgame_dp(match_probs, crowds, gains_1N2, p_emp, alphas,
                                 roles, roles, roles, V_term, stop_t=stop_t)
    return V_history[stop_t]


@pytest.mark.parametrize("gap", [-100, -70, -40, 10, 50])
def test_endgame_single_match(q_single, params, gap):
    tp, cr, ga = params["true_proba"], params["crowd"], params["gains"]
    idx_g1 = gap // 2 + OFFSET

    wr_base = float(q_single[idx_g1, IDX_G2_SAFE, 0, 0, 0, 0])
    wr_boost = float(max(q_single[idx_g1, IDX_G2_SAFE, 1, 0, 0, 0],
                         q_single[idx_g1, IDX_G2_SAFE, 0, 0, 0, 0]))

    th_base = exact_theoretical_wr(gap, 1, tp, cr, ga,
                                   has_booster=False, terminal=terminal_ge_zero)
    th_boost = exact_theoretical_wr(gap, 1, tp, cr, ga,
                                    has_booster=True, terminal=terminal_ge_zero)

    assert wr_base == pytest.approx(th_base, abs=TOL)
    assert wr_boost == pytest.approx(th_boost, abs=TOL)


# ---------------------------------------------------------------------------
# 2. PROFONDEUR TEMPORELLE 4 MATCHS (notebook 21) : base et booster
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def q_depth4(params):
    inp = _build_endgame_inputs(4, params["true_proba"], params["crowd"], params["gains"])
    match_probs, crowds, gains_1N2, p_emp, alphas, roles, V_term, stop_t = inp
    V_history = solve_endgame_dp(match_probs, crowds, gains_1N2, p_emp, alphas,
                                 roles, roles, roles, V_term, stop_t=stop_t)
    return V_history[stop_t]


@pytest.mark.parametrize("gap", [-460, -400, -300, -150, 50])
def test_endgame_depth4(q_depth4, params, gap):
    tp, cr, ga = params["true_proba"], params["crowd"], params["gains"]
    idx_g1 = gap // 2 + OFFSET

    wr_base = float(q_depth4[idx_g1, IDX_G2_SAFE, 0, 0, 0, 0])
    wr_boost = float(max(q_depth4[idx_g1, IDX_G2_SAFE, 1, 0, 0, 0],
                         q_depth4[idx_g1, IDX_G2_SAFE, 0, 0, 0, 0]))

    th_base = exact_theoretical_wr(gap, 4, tp, cr, ga,
                                   has_booster=False, terminal=terminal_ge_zero)
    th_boost = exact_theoretical_wr(gap, 4, tp, cr, ga,
                                    has_booster=True, terminal=terminal_ge_zero)

    assert wr_base == pytest.approx(th_base, abs=TOL)
    assert wr_boost == pytest.approx(th_boost, abs=TOL)
