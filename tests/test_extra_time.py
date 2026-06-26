"""
Tests de la correction « 120 minutes » des phases finales (mpp_project/extra_time.py).

Couvre :
  - get_120m_outcome_probas : conservation de masse, invariant Q_A = P_A_120 + 0.5·P_N_120,
    fallback prior z=0.45 (Q_A None) et singularité P_A≈P_B, cas P_N=0 ;
  - devig_to_qualify : Shin 2 issues, cas indisponible ;
  - build_p120_grid : conservation de masse (Σ=1), m=0 == grille 90', diagonale exclue,
    masse nulle décroissante en m ;
  - fit_goal_rates : moment-matching ;
  - calibrate_extra_time_multiplier : la masse nulle de P_120 colle à la cible, unicité,
    cas cible >= masse nulle 90' -> m=0 ;
  - correct_cond_crowd_120 : conditionnel normalisé par outcome, reversement de l'excédent
    de nul vers les scores +1 but ;
  - build_exact_score_market_120 : bout-en-bout, régions de p_score == 1N2@120, crowd
    conditionnel, bonus barème, retombée sur le 1N2 quand m=0.
"""
import numpy as np
import pytest

from mpp_project.extra_time import (
    get_120m_outcome_probas,
    devig_to_qualify,
    fit_goal_rates,
    build_p120_grid,
    calibrate_extra_time_multiplier,
    correct_cond_crowd_120,
    build_exact_score_market_120,
    _draw_mass,
)
from mpp_project.core import exact_score_bonus


# ==========================================================================
# get_120m_outcome_probas
# ==========================================================================
def test_get_120m_conservation_de_masse():
    P_A, P_N, P_B = 0.50, 0.28, 0.22
    pA, pN, pB = get_120m_outcome_probas(P_A, P_B, P_N, Q_A=0.62)
    assert pA + pN + pB == pytest.approx(P_A + P_N + P_B, abs=1e-12)


def test_get_120m_invariant_qualif():
    # Par construction : Q_A = P(A gagne en 120') + 0.5·P(nul 120') = P_A_120 + 0.5·P_N_120.
    # Inputs choisis dans le domaine non-clippé (x, y, z tous dans [0, 1]).
    P_A, P_N, P_B, Q_A = 0.45, 0.25, 0.30, 0.59
    pA, pN, pB = get_120m_outcome_probas(P_A, P_B, P_N, Q_A=Q_A)
    assert pA + 0.5 * pN == pytest.approx(Q_A, abs=1e-9)


def test_get_120m_deflate_les_nuls():
    # Le nul 120' est forcément <= nul 90' (une partie se décide en prolongation).
    P_A, P_N, P_B = 0.45, 0.25, 0.30
    pA, pN, pB = get_120m_outcome_probas(P_A, P_B, P_N, Q_A=0.59)
    assert pN < P_N
    assert pA >= P_A and pB >= P_B


def test_get_120m_fallback_z_prior():
    # Q_A None -> z = 0.45 : P_N_120 = 0.45 · P_N, split (1-z) au prorata de P_A, P_B.
    P_A, P_N, P_B = 0.50, 0.30, 0.20
    pA, pN, pB = get_120m_outcome_probas(P_A, P_B, P_N, Q_A=None)
    assert pN == pytest.approx(0.45 * P_N, abs=1e-12)
    x = (pA - P_A) / P_N
    y = (pB - P_B) / P_N
    assert x == pytest.approx((1 - 0.45) * P_A / (P_A + P_B), abs=1e-9)
    assert y == pytest.approx((1 - 0.45) * P_B / (P_A + P_B), abs=1e-9)


def test_get_120m_singularite_egalite():
    # P_A ≈ P_B + Q_A fourni -> bascule sur le fallback (split équiprobable).
    P_A, P_N, P_B = 0.39, 0.22, 0.39
    pA, pN, pB = get_120m_outcome_probas(P_A, P_B, P_N, Q_A=0.55)
    assert pN == pytest.approx(0.45 * P_N, abs=1e-12)
    assert pA == pytest.approx(pB, abs=1e-9)


def test_get_120m_pn_nul():
    pA, pN, pB = get_120m_outcome_probas(0.6, 0.4, 0.0, Q_A=0.7)
    assert (pA, pN, pB) == (0.6, 0.0, 0.4)


# ==========================================================================
# devig_to_qualify
# ==========================================================================
def test_devig_to_qualify_shin_2_issues():
    Q_A = devig_to_qualify(1.5, 2.5)
    Q_B = devig_to_qualify(2.5, 1.5)
    assert 0.0 < Q_A < 1.0
    assert Q_A + Q_B == pytest.approx(1.0, abs=1e-9)
    assert Q_A > 0.5  # cote plus courte -> plus probable


def test_devig_to_qualify_indisponible():
    assert devig_to_qualify(None, 2.0) is None
    assert devig_to_qualify(1.5, None) is None
    assert devig_to_qualify(-1.0, 2.0) is None
    assert devig_to_qualify(0.0, 2.0) is None


# ==========================================================================
# build_p120_grid
# ==========================================================================
def _grid_pipeau():
    """Petite grille 90' (4x4) normalisée, avec masse sur nuls et décisifs."""
    g = np.zeros((4, 4))
    g[1, 0] = 0.25  # 1-0
    g[2, 1] = 0.10  # 2-1
    g[0, 0] = 0.15  # 0-0
    g[1, 1] = 0.12  # 1-1
    g[0, 1] = 0.20  # 0-1
    g[1, 2] = 0.08  # 1-2
    g[2, 0] = 0.10  # 2-0
    return g / g.sum()


def test_build_p120_conservation():
    g = _grid_pipeau()
    p120 = build_p120_grid(g, lam_A=1.3, lam_B=1.0, m=0.15)
    assert p120.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(p120 >= 0.0)


def test_build_p120_m_zero_identite():
    # m=0 -> P_30 = δ_{0,0} -> P_120 == grille 90' au bit près.
    g = _grid_pipeau()
    p120 = build_p120_grid(g, lam_A=1.3, lam_B=1.0, m=0.0)
    assert np.allclose(p120, g, atol=1e-9)


def test_build_p120_masse_nulle_decroissante():
    g = _grid_pipeau()
    masses = [_draw_mass(build_p120_grid(g, 1.3, 1.0, m)) for m in (0.0, 0.1, 0.3, 0.6, 1.2)]
    assert masses[0] == pytest.approx(_draw_mass(g), abs=1e-12)
    assert all(masses[i] > masses[i + 1] for i in range(len(masses) - 1))


# ==========================================================================
# fit_goal_rates
# ==========================================================================
def test_fit_goal_rates_moment_matching():
    scores = ["1-0", "0-1", "2-1"]
    p = np.array([0.5, 0.3, 0.2])
    lam_A, lam_B = fit_goal_rates(scores, p)
    assert lam_A == pytest.approx(0.5 * 1 + 0.3 * 0 + 0.2 * 2, abs=1e-9)
    assert lam_B == pytest.approx(0.5 * 0 + 0.3 * 1 + 0.2 * 1, abs=1e-9)


# ==========================================================================
# calibrate_extra_time_multiplier
# ==========================================================================
def test_calibrate_atteint_la_cible():
    # Grille paddée à 11x11 (comme en production G>=10) : le repli de la queue Poisson
    # est négligeable -> la masse nulle de la grille colle à la cible analytique.
    g = np.zeros((11, 11))
    g[1, 0] = 0.25; g[2, 1] = 0.10; g[0, 0] = 0.15; g[1, 1] = 0.12
    g[0, 1] = 0.20; g[1, 2] = 0.08; g[2, 0] = 0.10
    g /= g.sum()
    draw90 = _draw_mass(g)
    target = 0.45 * draw90               # déflation réaliste
    m = calibrate_extra_time_multiplier(g, 1.3, 1.0, target)
    got = _draw_mass(build_p120_grid(g, 1.3, 1.0, m))
    assert got == pytest.approx(target, abs=1e-3)
    assert m > 0.0


def test_calibrate_cible_superieure_masse_nulle():
    # Cible >= masse nulle 90' -> aucune déflation -> m = 0.
    g = _grid_pipeau()
    m = calibrate_extra_time_multiplier(g, 1.3, 1.0, _draw_mass(g) + 0.1)
    assert m == 0.0


# ==========================================================================
# correct_cond_crowd_120
# ==========================================================================
def test_correct_cond_crowd_120_normalise_par_outcome():
    scores = ["1-0", "2-0", "2-1", "0-0", "1-1", "0-1", "1-2"]
    outc = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int8)
    crowds = np.array([10.0, 4.0, 6.0, 20.0, 12.0, 8.0, 3.0])  # nuls sur-joués (Winamax)
    q120 = np.array([0.50, 0.18, 0.32])  # MPP 120' : moins de nuls
    cc = correct_cond_crowd_120(crowds, scores, outc, q120, mpp_outcome_crowd_90=None)
    for o in (0, 1, 2):
        mask = outc == o
        assert cc[mask].sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(cc >= 0.0)


def test_correct_cond_crowd_120_reverse_excedent_nul():
    # Beaucoup de crowd sur 0-0/1-1 (90') ; après correction, les scores +1 but
    # (1-0, 0-1) doivent capter une part conditionnelle plus forte que sans reversement.
    scores = ["1-0", "0-1", "0-0", "1-1", "2-0", "0-2"]
    outc = np.array([0, 2, 1, 1, 0, 2], dtype=np.int8)
    crowds = np.array([5.0, 5.0, 30.0, 20.0, 2.0, 2.0])
    q120_defl = np.array([0.45, 0.10, 0.45])   # peu de nuls -> gros excédent reversé
    q120_neutre = np.array([0.34, 0.5, 0.16])  # nul "déjà" élevé -> peu d'excédent
    cc_defl = correct_cond_crowd_120(crowds, scores, outc, q120_defl, mpp_outcome_crowd_90=None)
    cc_neutre = correct_cond_crowd_120(crowds, scores, outc, q120_neutre, mpp_outcome_crowd_90=None)
    i_10 = scores.index("1-0")
    i_01 = scores.index("0-1")
    # Le reversement gonfle la part conditionnelle des scores +1 but de chaque outcome.
    assert cc_defl[i_10] > cc_neutre[i_10]
    assert cc_defl[i_01] > cc_neutre[i_01]


# ==========================================================================
# build_exact_score_market_120 (bout-en-bout)
# ==========================================================================
def _match_pipeau():
    # Match A favori. Cotes scores ~ plausibles (le détail importe peu : ancrage 1N2).
    data = {
        "1-0": (6.5, 17.0), "2-0": (8.5, 13.0), "2-1": (9.5, 24.0), "3-1": (18.0, 3.0),
        "1-1": (7.0, 18.0), "0-0": (9.5, 6.0), "2-2": (20.0, 3.0),
        "0-1": (12.0, 2.0), "1-2": (17.0, 6.0), "0-2": (26.0, 1.0),
    }
    # 1N2@90 + Q_A dans le domaine non-clippé (déflation nette du nul).
    op90 = np.array([0.45, 0.25, 0.30])
    crowd120 = np.array([0.55, 0.18, 0.27])
    return data, op90, crowd120


_Q_A_PIPEAU = 0.59


def test_market_120_pscore_regions_collent_1n2_120():
    data, op90, crowd120 = _match_pipeau()
    Q_A = _Q_A_PIPEAU
    market, info = build_exact_score_market_120(
        data, outcome_probas_90=op90, mpp_outcome_crowd_120=crowd120, Q_A=Q_A
    )
    p = np.asarray(market.p_score)
    outc = np.asarray(market.outcomes)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    pA120, pN120, pB120 = info["p1n2_120"]
    assert p[outc == 0].sum() == pytest.approx(pA120, abs=1e-6)
    assert p[outc == 1].sum() == pytest.approx(pN120, abs=1e-6)
    assert p[outc == 2].sum() == pytest.approx(pB120, abs=1e-6)


def test_market_120_regions_90_collent_1n2_90():
    # L'ancrage 90' fixe les régions de la grille 90' au 1N2@90 (indépendant des cotes).
    data, op90, crowd120 = _match_pipeau()
    _, info = build_exact_score_market_120(
        data, outcome_probas_90=op90, mpp_outcome_crowd_120=crowd120, Q_A=_Q_A_PIPEAU
    )
    r1, rN, r2 = info["region_sums_90"]
    assert (r1, rN, r2) == pytest.approx(tuple(op90), abs=1e-6)


def test_market_120_deflate_le_nul_vs_90():
    data, op90, crowd120 = _match_pipeau()
    market, info = build_exact_score_market_120(
        data, outcome_probas_90=op90, mpp_outcome_crowd_120=crowd120, Q_A=_Q_A_PIPEAU
    )
    # Q_A favorable -> nul à 120' nettement < nul à 90'.
    assert info["p1n2_120"][1] < info["p1n2_90"][1]
    assert info["m"] > 0.0


def test_market_120_crowd_et_bonus_coherents():
    data, op90, crowd120 = _match_pipeau()
    market, _ = build_exact_score_market_120(
        data, outcome_probas_90=op90, mpp_outcome_crowd_120=crowd120, Q_A=_Q_A_PIPEAU
    )
    cc = np.asarray(market.cond_crowd)
    outc = np.asarray(market.outcomes)
    for o in (0, 1, 2):
        mask = outc == o
        if mask.any():
            assert cc[mask].sum() == pytest.approx(1.0, abs=1e-6)
    # Bonus = barème appliqué au crowd conditionnel corrigé.
    expected = [exact_score_bonus(c) for c in cc]
    assert list(market.bonus) == expected


def test_market_120_sans_qualif_utilise_prior():
    # Q_A None -> prior z=0.45 : le marché reste construit, nul 120' = 0.45·nul 90'.
    data, op90, crowd120 = _match_pipeau()
    _, info = build_exact_score_market_120(
        data, outcome_probas_90=op90, mpp_outcome_crowd_120=crowd120, Q_A=None
    )
    assert info["p1n2_120"][1] == pytest.approx(0.45 * op90[1], abs=1e-3)
