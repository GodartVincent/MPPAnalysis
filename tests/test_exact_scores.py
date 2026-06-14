"""
Tests de la décision SCORES EXACTS du match du jour.

Trois niveaux :
  1. BARÈME / MARCHÉ : exact_score_bonus + build_exact_score_market (pur Python).
  2. ORACLE EXACT    : evaluate_exact_score_day (numba) comparé à une réf. Python
                       brute-force qui énumère explicitement tout l'arbre
                       (score réel × pari de Bob × bonus de Bob × bin peloton ×
                       bonus peloton), bonus inclus.
  3. INTÉGRATION     : run_daily_pipeline(..., exact_score_data=...) sur un pipeau.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mpp_project.core import (
    ExactScoreMarket,
    build_exact_score_market,
    exact_score_bonus,
    expected_mpp_points,
    expected_simple_points,
    load_exact_scores,
)
from mpp_project.core import load_exact_scores_by_match
from mpp_project.oracle_dp import (
    evaluate_exact_score_day, solve_dp_coarse, solve_dp_coarse_exact,
    GAP_OFFSET, GAP_MIN, GAP_MAX,
)
from mpp_project.daily_pipeline import run_daily_pipeline
from tests.helpers import build_pipeau_dataframe, build_ghost_peloton, build_terminal_horizon

TOL = 1e-4


# ===========================================================================
# 1. BARÈME & CONSTRUCTION DU MARCHÉ
# ===========================================================================
@pytest.mark.parametrize("cc, expected", [
    (0.35, 20), (0.30, 30), (0.25, 30), (0.20, 50), (0.10, 50),
    (0.05, 70), (0.02, 70), (0.005, 100), (0.001, 100), (0.0, 100),
])
def test_bonus_tiers(cc, expected):
    assert exact_score_bonus(cc) == expected


def test_build_market():
    data = {
        "1-0": (3.0, 30.0),   # outcome 0
        "2-0": (6.0, 10.0),   # outcome 0
        "0-0": (8.0, 15.0),   # outcome 1 (nul)
        "0-1": (7.0, 12.0),   # outcome 2
        "9-9": (None, None),  # ignoré (cote absente)
    }
    m = build_exact_score_market(data)

    assert m.scores == ["1-0", "2-0", "0-0", "0-1"]
    assert list(m.outcomes) == [0, 0, 1, 2]

    # Probas Shin somment à 1
    assert m.p_score.sum() == pytest.approx(1.0, abs=1e-9)

    # Crowd conditionnel normalisé par outcome
    assert m.cond_crowd[0] == pytest.approx(30.0 / 40.0)  # 0.75
    assert m.cond_crowd[1] == pytest.approx(10.0 / 40.0)  # 0.25
    assert m.cond_crowd[2] == pytest.approx(1.0)          # seul score 'nul'
    assert m.cond_crowd[3] == pytest.approx(1.0)          # seul score outcome 2

    # Bonus cohérents avec le barème
    assert list(m.bonus) == [20, 30, 20, 20]


def test_build_market_creux_renormalise():
    # Marché très partiel (somme 1/cote = 0.4) : les probas listées sont quand même
    # renormalisées à 1 (masse résiduelle repliée sur les scores listés).
    m = build_exact_score_market({"1-0": (5.0, 50.0), "0-1": (5.0, 50.0)})
    assert m.p_score.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.isfinite(m.p_score)) and np.all(m.p_score > 0.0)


def test_build_market_vide_rejete():
    with pytest.raises(ValueError):
        build_exact_score_market({"1-0": (None, None), "0-1": (0, 5.0)})


def test_build_market_ancrage_1n2():
    """Avec outcome_probas, la somme des probas de scores par outcome == P(outcome)."""
    data = {
        "1-0": (3.0, 30.0), "2-0": (6.0, 10.0),   # outcome 0
        "0-0": (8.0, 15.0),                         # outcome 1
        "0-1": (7.0, 12.0), "0-2": (9.0, 8.0),     # outcome 2
    }
    op = np.array([0.50, 0.30, 0.20])
    m = build_exact_score_market(data, outcome_probas=op)
    for o in (0, 1, 2):
        s = m.p_score[m.outcomes == o].sum()
        assert s == pytest.approx(op[o], abs=1e-9), f"outcome {o}: somme={s} != {op[o]}"
    assert m.p_score.sum() == pytest.approx(1.0, abs=1e-9)


def test_expected_simple_points():
    """Barème 1/2/3 sur un marché construit à la main (probas contrôlées)."""
    m = ExactScoreMarket(
        scores=["1-0", "2-1", "0-1"],
        outcomes=np.array([0, 0, 2], dtype=np.int8),
        p_score=np.array([0.4, 0.3, 0.3]),
        cond_crowd=np.zeros(3),
        bonus=np.zeros(3, dtype=np.int64),
    )
    E = expected_simple_points(m)
    # "1-0"(o0,d1): 3*.4 + 2*.3 (2-1 même diff) + 0 = 1.8
    # "2-1"(o0,d1): 2*.4 + 3*.3 + 0 = 1.7
    # "0-1"(o2,d-1): 0 + 0 + 3*.3 = 0.9
    assert E == pytest.approx([1.8, 1.7, 0.9], abs=1e-9)


def test_expected_mpp_points():
    """E[pts MPP] = P(outcome)*gain_mpp[outcome] + P(score exact)*bonus."""
    m = ExactScoreMarket(
        scores=["1-0", "2-1", "0-1"],
        outcomes=np.array([0, 0, 2], dtype=np.int8),
        p_score=np.array([0.4, 0.3, 0.3]),
        cond_crowd=np.zeros(3),
        bonus=np.array([10, 20, 30], dtype=np.int64),
    )
    E = expected_mpp_points(m, gains_1N2=np.array([50, 40, 60]))
    # P(o0)=0.7, P(o2)=0.3
    # "1-0": 0.7*50 + 0.4*10 = 39 ; "2-1": 0.7*50 + 0.3*20 = 41 ; "0-1": 0.3*60 + 0.3*30 = 27
    assert E == pytest.approx([39.0, 41.0, 27.0], abs=1e-9)


# ===========================================================================
# 1ter. CHARGEMENT CSV + VALIDATION 1N2
# ===========================================================================
def test_load_exact_scores(tmp_path):
    df = pd.DataFrame({
        "match_id": [3, 3, 3, 5],
        "score": ["1-0", "0-1", "6-0", "2-2"],
        "cote": [6.5, 12.0, np.nan, 9.0],     # 6-0 sans cote
        "crowd": [15.0, 8.0, np.nan, 3.0],
    })
    path = tmp_path / "exact_scores.csv"
    df.to_csv(path, index=False)

    data = load_exact_scores(path, 3)
    assert set(data.keys()) == {"1-0", "0-1", "6-0"}
    assert data["1-0"] == (6.5, 15.0)
    assert data["6-0"] == (None, None)        # cote/crowd vides -> None
    # match_id 5 isolé
    assert set(load_exact_scores(path, 5).keys()) == {"2-2"}


def test_load_exact_scores_match_absent(tmp_path):
    path = tmp_path / "e.csv"
    pd.DataFrame({"match_id": [1], "score": ["1-0"], "cote": [2.0], "crowd": [50.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_exact_scores(path, 99)


def test_load_exact_scores_colonnes_manquantes(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"match_id": [1], "score": ["1-0"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_exact_scores(path, 1)


def test_validation_1n2_warns():
    """1N2 brut des scores très éloigné du CDM -> warning de saisie."""
    data = {  # 5 scores (>3 -> garde-fou marge de Shin inactif), très orienté outcome 0
        "1-0": (2.0, 40.0), "2-0": (3.0, 30.0),
        "0-1": (15.0, 20.0), "0-2": (20.0, 10.0), "1-1": (12.0, 5.0),
    }
    with pytest.warns(UserWarning, match="1N2 agrégé"):
        build_exact_score_market(data, outcome_probas=np.array([0.40, 0.30, 0.30]))


def test_validation_1n2_silencieuse_si_coherent():
    """Si outcome_probas == agrégation brute, aucun warning de validation."""
    data = {
        "1-0": (3.0, 30.0), "2-0": (6.0, 10.0),
        "0-0": (8.0, 15.0),
        "0-1": (7.0, 12.0), "0-2": (9.0, 8.0),
    }
    m0 = build_exact_score_market(data)  # renorm global, pas d'ancrage
    op = np.array([m0.p_score[m0.outcomes == o].sum() for o in (0, 1, 2)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # tout warning devient une erreur
        build_exact_score_market(data, outcome_probas=op)


# ===========================================================================
# 2. ORACLE EXACT — evaluate_exact_score_day vs brute-force Python
# ===========================================================================
def _lookup_py(val_g1, val_g2, agent_pts, bob_g, pel_delta, alpha, V_next, layer):
    ng_bob = val_g1 + agent_pts - bob_g
    ng_pel = val_g2 + agent_pts - pel_delta
    vg1 = min(ng_bob, ng_pel)
    vg2 = alpha * max(ng_bob, ng_pel) + (1.0 - alpha) * ng_pel
    i1 = max(GAP_MIN, min(GAP_MAX, int(round(vg1)))) + GAP_OFFSET
    i2 = max(GAP_MIN, min(GAP_MAX, int(round(vg2)))) + GAP_OFFSET
    return float(V_next[i1, i2, layer])


def _bruteforce_exact_day(g1_idx, g2_idx, m, gains, crowd, p_emp_day, alpha, V_next):
    """Référence indépendante : énumère explicitement tout l'arbre du match du jour."""
    K = len(m.scores)
    max_gain = p_emp_day.shape[1]
    val_g1 = g1_idx - GAP_OFFSET
    val_g2 = g2_idx - GAP_OFFSET
    Q = np.zeros((K, 3))

    for ka in range(K):
        a_o = int(m.outcomes[ka])
        a_bonus = float(m.bonus[ka])
        for ks in range(K):
            p_s = float(m.p_score[ks])
            if p_s == 0.0:
                continue
            o = int(m.outcomes[ks])
            cc = float(m.cond_crowd[ks])
            b_s = float(m.bonus[ks])
            true_gain = float(gains[o])

            agent_pts = (true_gain if a_o == o else 0.0) + (a_bonus if ka == ks else 0.0)
            agent_pts_boost = 2.0 * agent_pts

            for bob_a in range(3):
                p_bob = float(crowd[bob_a])
                if p_bob == 0.0:
                    continue
                if bob_a == o:
                    bob_cases = [(cc, true_gain + b_s), (1.0 - cc, true_gain)]
                else:
                    bob_cases = [(1.0, 0.0)]

                for p_be, bob_g in bob_cases:
                    if p_be == 0.0:
                        continue
                    for dg in range(max_gain):
                        p_pel = float(p_emp_day[o, dg])
                        if p_pel == 0.0:
                            continue
                        if dg != 0:
                            pel_cases = [(cc, dg + b_s), (1.0 - cc, float(dg))]
                        else:
                            pel_cases = [(1.0, 0.0)]
                        for p_pe, pel_delta in pel_cases:
                            if p_pe == 0.0:
                                continue
                            w = p_s * p_bob * p_be * p_pel * p_pe
                            Q[ka, 0] += w * _lookup_py(val_g1, val_g2, agent_pts,
                                                       bob_g, pel_delta, alpha, V_next, 0)
                            Q[ka, 1] += w * _lookup_py(val_g1, val_g2, agent_pts,
                                                       bob_g, pel_delta, alpha, V_next, 1)
                            Q[ka, 2] += w * _lookup_py(val_g1, val_g2, agent_pts_boost,
                                                       bob_g, pel_delta, alpha, V_next, 0)
    return Q


@pytest.fixture(scope="module")
def synthetic_market():
    data = {
        "1-0": (3.0, 30.0),
        "2-0": (6.0, 10.0),
        "0-0": (8.0, 15.0),
        "0-1": (7.0, 12.0),
        "0-2": (9.0, 8.0),
    }
    return build_exact_score_market(data)


@pytest.fixture(scope="module")
def synthetic_hist():
    """Histogramme peloton (3, max_gain) : moitié à dg=0, moitié à dg=30 (exerce le bonus peloton)."""
    max_gain = 60
    p = np.zeros((3, max_gain), dtype=np.float64)
    p[:, 0] = 0.5
    p[:, 30] = 0.5
    return p


@pytest.mark.parametrize("g1, g2", [(0, 0), (-40, 0), (-120, 80), (50, -50)])
def test_evaluate_vs_bruteforce(synthetic_market, synthetic_hist, g1, g2):
    m = synthetic_market
    gains = np.array([50.0, 36.0, 45.0])
    crowd = np.array([0.45, 0.30, 0.25])
    alpha = 0.4
    V_next = build_terminal_horizon()  # (1001, 1001, 2) float32

    g1_idx = g1 + GAP_OFFSET
    g2_idx = g2 + GAP_OFFSET

    Q = evaluate_exact_score_day(
        g1_idx, g2_idx, m.outcomes, m.p_score, m.cond_crowd, m.bonus,
        gains, crowd, synthetic_hist, alpha, V_next
    )
    Q_ref = _bruteforce_exact_day(g1_idx, g2_idx, m, gains, crowd, synthetic_hist, alpha, V_next)

    assert np.allclose(Q, Q_ref, atol=TOL), f"écart max {np.abs(Q - Q_ref).max():.2e}"


def test_no_bonus_wr_is_floor(synthetic_market, synthetic_hist):
    """agent_bonus_factor=0 (l'agent loupe son score) : WR <= WR avec bonus, et
    identique pour tous les scores d'un même outcome (ne dépend que de l'outcome)."""
    m = synthetic_market
    gains = np.array([50.0, 36.0, 45.0])
    crowd = np.array([0.45, 0.30, 0.25])
    V_next = build_terminal_horizon()
    g1_idx, g2_idx = -40 + GAP_OFFSET, GAP_OFFSET

    full = evaluate_exact_score_day(g1_idx, g2_idx, m.outcomes, m.p_score, m.cond_crowd,
                                    m.bonus, gains, crowd, synthetic_hist, 0.4, V_next)
    nb = evaluate_exact_score_day(g1_idx, g2_idx, m.outcomes, m.p_score, m.cond_crowd,
                                  m.bonus, gains, crowd, synthetic_hist, 0.4, V_next,
                                  agent_bonus_factor=0.0)

    assert np.all(nb <= full + 1e-6), "retirer le bonus agent ne peut pas augmenter le WR"
    for o in (0, 1, 2):
        idx = np.where(m.outcomes == o)[0]
        if len(idx) > 1:
            assert np.allclose(nb[idx], nb[idx[0]], atol=1e-6), f"WR sans bonus non constant sur outcome {o}"


def test_evaluate_booster_value(synthetic_market, synthetic_hist):
    """Avec booster, max(keep, use) sur les actions >= meilleur WR base."""
    m = synthetic_market
    gains = np.array([50.0, 36.0, 45.0])
    crowd = np.array([0.45, 0.30, 0.25])
    V_next = build_terminal_horizon()
    for g1 in (-200, -80, 0, 60):
        Q = evaluate_exact_score_day(
            g1 + GAP_OFFSET, GAP_OFFSET, m.outcomes, m.p_score, m.cond_crowd, m.bonus,
            gains, crowd, synthetic_hist, 0.4, V_next
        )
        wr_base = Q[:, 0].max()
        wr_boost = max(Q[:, 1].max(), Q[:, 2].max())
        assert wr_boost >= wr_base - 1e-5, f"g1={g1}: booster destructeur"


# ===========================================================================
# 3. INTÉGRATION via run_daily_pipeline
# ===========================================================================
EXACT_DATA = {
    "1-0": (6.5, 15.0), "0-0": (9.0, 5.0), "0-1": (12.0, 8.0),
    "2-0": (8.0, 18.0), "1-1": (6.0, 20.0), "0-2": (15.0, 4.0),
    "2-1": (8.5, 22.0), "2-2": (15.0, 2.0), "1-2": (16.0, 3.0),
}


def test_pipeline_integration_exact(tmp_path):
    df = build_pipeau_dataframe(
        n_matchs=4,
        true_proba=np.array([0.50, 0.30, 0.20]),
        crowd=np.array([0.45, 0.30, 0.25]),
        gains=np.array([50, 60, 70], dtype=np.int64),
    )
    path = tmp_path / "exact.csv"
    df.to_csv(path, index=False)

    reco, wr, market_df, q_jour = run_daily_pipeline(
        csv_path=path,
        match_id_cible=1,
        mon_gap_1=-30,
        mon_gap_2=0,
        has_booster=1,
        use_drift=False,
        horizon_nuit=1,
        nb_scenarios=1,
        v_horizon_override=build_terminal_horizon(),
        save_abaques=False,
        validate_input=False,
        exact_score_data=EXACT_DATA,
    )

    listed = set(EXACT_DATA.keys())
    assert reco.split(" ")[0] in listed, f"reco={reco!r} hors marché"
    assert 0.0 <= wr <= 1.0
    # market_df bien formé : tous les scores listés (cotes valides) présents
    assert set(market_df["Score"]) == listed
    assert len(market_df) == len(EXACT_DATA)
    for col in ("WR base (%)", "WR keep (%)", "WR x2 (%)"):
        assert market_df[col].between(0.0, 100.0).all()
    # Q 1N2 du jour toujours renvoyée en slot 4
    assert q_jour.shape == (1001, 1001, 3, 3)


# ===========================================================================
# 4. MULTI-MATCHS : DP coarse exact-aware + reco par match
# ===========================================================================
def test_load_exact_scores_by_match(tmp_path):
    df = pd.DataFrame({
        "match_id": [3, 3, 4, 4],
        "score": ["1-0", "0-1", "2-0", "1-1"],
        "cote": [6.5, 12.0, 8.0, np.nan],
        "crowd": [15.0, 8.0, 18.0, np.nan],
    })
    path = tmp_path / "exact_scores.csv"
    df.to_csv(path, index=False)

    by_match = load_exact_scores_by_match(path)
    assert set(by_match.keys()) == {3, 4}
    assert by_match[3]["1-0"] == (6.5, 15.0)
    assert by_match[4]["1-1"] == (None, None)


def _coarse_1N2_sc(true_probas):
    """Tableaux de scores paddés représentant des matchs 1N2 purs (bonus 0)."""
    n = true_probas.shape[0]
    sc_o = np.zeros((n, 3), dtype=np.int8)
    sc_o[:, 1] = 1
    sc_o[:, 2] = 2
    sc_p = np.ascontiguousarray(true_probas, dtype=np.float64)
    sc_cc = np.zeros((n, 3), dtype=np.float64)
    sc_b = np.zeros((n, 3), dtype=np.float64)
    sc_n = np.full(n, 3, dtype=np.int64)
    return sc_o, sc_p, sc_cc, sc_b, sc_n


def test_coarse_exact_equals_coarse_1N2():
    """RÉGRESSION CRITIQUE : solve_dp_coarse_exact avec entrées tout-1N2 (bonus 0)
    reproduit solve_dp_coarse sur le même scénario."""
    n, max_gain = 3, 20
    tp = np.array([[0.50, 0.30, 0.20]] * n, dtype=np.float64)
    cr = np.array([[0.45, 0.35, 0.20]] * n, dtype=np.float64)
    ga = np.array([[40, 30, 50]] * n, dtype=np.float64)
    hist = np.zeros((n, 3, max_gain), dtype=np.float64)
    hist[:, :, 0] = 0.5
    hist[:, :, 6] = 0.5
    al = np.full(n, 0.3, dtype=np.float64)
    V0 = np.zeros((501, 501, 2), dtype=np.float32)  # écrasé par le terminal (start_t=-1)

    sc_o, sc_p, sc_cc, sc_b, sc_n = _coarse_1N2_sc(tp)
    V_exact = solve_dp_coarse_exact(sc_o, sc_p, sc_cc, sc_b, sc_n, ga, cr, hist, al,
                                    V0, stop_t=0, start_t=-1)
    V_ref, _ = solve_dp_coarse(tp, cr, ga, hist, al, V0, stop_t=0, horizon_nuit=0, start_t=-1)

    assert np.allclose(V_exact, V_ref, atol=1e-4), f"écart max {np.abs(V_exact - V_ref).max():.2e}"


MULTI_DATA = {
    1: {"1-0": (3.0, 30.0), "2-0": (6.0, 10.0), "0-0": (8.0, 15.0), "0-1": (7.0, 12.0)},
    2: {"1-0": (3.0, 30.0), "1-1": (5.0, 25.0), "0-1": (7.0, 20.0)},
}


def test_pipeline_multi_exact(tmp_path):
    df = build_pipeau_dataframe(
        n_matchs=4,
        true_proba=np.array([0.50, 0.30, 0.20]),
        crowd=np.array([0.45, 0.30, 0.25]),
        gains=np.array([50, 40, 60], dtype=np.int64),
    )
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)

    out = run_daily_pipeline(
        csv_path=path,
        match_id_cible=1,
        mon_gap_1=-30,
        mon_gap_2=0,
        has_booster=1,
        use_drift=False,
        horizon_nuit=2,
        nb_scenarios=1,
        p_empirique_override=build_ghost_peloton(4, max_gain=250),
        v_horizon_override=build_terminal_horizon(),
        save_abaques=False,
        validate_input=False,
        exact_scores_by_match=MULTI_DATA,
    )
    assert len(out) == 5, "le mode multi-matchs renvoie 5 éléments"
    reco, wr, market_df, q_jour, night_markets = out

    assert set(night_markets.keys()) == {1, 2}
    for mid, (reco_m, wr_m, df_m) in night_markets.items():
        assert reco_m.split(" ")[0] in set(MULTI_DATA[mid].keys())
        assert 0.0 <= wr_m <= 1.0
        assert {"Score", "WR base (%)", "WR outcome (%)", "E[pts MPP]"} <= set(df_m.columns)
    # le 1er élément = reco du match courant (id=1)
    assert (reco, wr) == (night_markets[1][0], night_markets[1][1])
