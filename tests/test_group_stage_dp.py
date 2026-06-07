"""
Tests du moteur de poules via run_daily_pipeline (notebook 22).

Deux niveaux :
  1. ORACLE EXACT  : pipeline déterministe (use_drift=False) sur un pipeau de 4 matchs
                     identiques, peloton fantôme + gap_2 saturé -> comparable au
                     solveur récursif exact (tests/helpers.exact_theoretical_wr).
  2. INTÉGRATION    : le pipeline tourne sur les CSV réalistes de data/tests/ et
                     respecte les invariants structurels (WR dans [0,1], monotonie,
                     valeur du booster >= 0).
"""
import numpy as np
import pandas as pd
import pytest

from mpp_project.daily_pipeline import run_daily_pipeline
from mpp_project.oracle_dp import GAP_OFFSET, GAP_MIN, GAP_MAX
from tests.helpers import exact_theoretical_wr, terminal_strict, DATA_TESTS_DIR

TOL = 0.005
GAP_PELOTON_SAFE = 400  # +400 réel : peloton distancé pour toujours (alpha sans effet)


# ===========================================================================
# 1. ORACLE EXACT
# ===========================================================================
@pytest.fixture(scope="module")
def q_table_jour(pipeau_csv, ghost_peloton_4):
    """Exécute le pipeline déterministe une fois et renvoie (q_table, n_matchs)."""
    path, n_matchs = pipeau_csv
    _, _, _, q = run_daily_pipeline(
        csv_path=path,
        match_id_cible=1,
        mon_gap_1=0,
        mon_gap_2=GAP_PELOTON_SAFE,
        has_booster=1,
        use_drift=False,        # déterministe : comparable à la théorie exacte
        horizon_nuit=1,         # seule la Q-table du jour ; profondeur = n_matchs
        nb_scenarios=1,
        p_empirique_override=ghost_peloton_4,
        save_abaques=False,
        validate_input=False,   # fixture synthétique (pas de colonnes équipes, phase libre)
    )
    return q, n_matchs


@pytest.mark.parametrize("nom, gap_initial", [
    ("Mission impossible (meme x2)", -460),
    ("Le booster, seule issue",      -400),
    ("Droit a l'erreur elargi",      -300),
    ("Retard modere",                -150),
    ("Gestion de l'avance",           50),
])
def test_pipeline_vs_oracle_exact(q_table_jour, pipeau_params, nom, gap_initial):
    q, n_matchs = q_table_jour
    tp, cr, ga = pipeau_params["true_proba"], pipeau_params["crowd"], pipeau_params["gains"]

    idx_g1 = gap_initial + GAP_OFFSET
    idx_g2 = GAP_PELOTON_SAFE + GAP_OFFSET

    # DP : meilleure action sur chaque mode booster
    wr_base = float(np.max(q[idx_g1, idx_g2, :, 0]))
    wr_keep = float(np.max(q[idx_g1, idx_g2, :, 1]))
    wr_use = float(np.max(q[idx_g1, idx_g2, :, 2]))
    wr_boost = max(wr_keep, wr_use)

    # Théorie exacte (terminal strict, comme solve_dp_coarse)
    th_base = exact_theoretical_wr(gap_initial, n_matchs, tp, cr, ga,
                                   has_booster=False, terminal=terminal_strict)
    th_boost = exact_theoretical_wr(gap_initial, n_matchs, tp, cr, ga,
                                    has_booster=True, terminal=terminal_strict)

    assert wr_base == pytest.approx(th_base, abs=TOL), f"{nom} [base] DP={wr_base} vs th={th_base}"
    assert wr_boost == pytest.approx(th_boost, abs=TOL), f"{nom} [x2] DP={wr_boost} vs th={th_boost}"


def test_booster_a_de_la_valeur(q_table_jour):
    """Avec le booster dispo, le WR ne peut jamais être inférieur au WR sans booster."""
    q, _ = q_table_jour
    idx_g2 = GAP_PELOTON_SAFE + GAP_OFFSET
    for gap in range(-400, 200, 20):
        i1 = gap + GAP_OFFSET
        wr_base = np.max(q[i1, idx_g2, :, 0])
        wr_boost = max(np.max(q[i1, idx_g2, :, 1]), np.max(q[i1, idx_g2, :, 2]))
        assert wr_boost >= wr_base - 1e-5, f"gap={gap}: booster destructeur"


# ===========================================================================
# 1bis. NON-RÉGRESSION : match du jour = DERNIER match du tournoi
# ===========================================================================
# Avant le correctif, V_near[match_idx + k + 1] débordait (IndexError) quand
# l'horizon atteignait le dernier match. Le V_next du dernier match doit être
# la condition terminale → la Q-table doit coïncider avec la théorie à 1 match.
def test_dernier_match_pas_d_overflow(pipeau_csv, ghost_peloton_4, pipeau_params):
    path, n_matchs = pipeau_csv
    tp, cr, ga = pipeau_params["true_proba"], pipeau_params["crowd"], pipeau_params["gains"]

    # match_id_cible = n_matchs → match du jour = dernier match ; horizon volontairement
    # plus grand que ce qui reste, pour forcer l'ancien dépassement d'index.
    _, _, _, q = run_daily_pipeline(
        csv_path=path,
        match_id_cible=n_matchs,
        mon_gap_1=0,
        mon_gap_2=GAP_PELOTON_SAFE,
        has_booster=1,
        use_drift=False,
        horizon_nuit=5,        # > matchs restants (1) → exerce le cas terminal
        nb_scenarios=1,
        p_empirique_override=ghost_peloton_4,
        save_abaques=False,
        validate_input=False,
    )

    idx_g2 = GAP_PELOTON_SAFE + GAP_OFFSET
    for gap in (-180, -90, -40, 0, 50):
        i1 = gap + GAP_OFFSET
        wr_base = float(np.max(q[i1, idx_g2, :, 0]))
        wr_boost = max(float(np.max(q[i1, idx_g2, :, 1])), float(np.max(q[i1, idx_g2, :, 2])))
        # Un seul match restant face au terminal strict
        th_base = exact_theoretical_wr(gap, 1, tp, cr, ga,
                                       has_booster=False, terminal=terminal_strict)
        th_boost = exact_theoretical_wr(gap, 1, tp, cr, ga,
                                        has_booster=True, terminal=terminal_strict)
        assert wr_base == pytest.approx(th_base, abs=TOL), f"gap={gap} base"
        assert wr_boost == pytest.approx(th_boost, abs=TOL), f"gap={gap} x2"


# ===========================================================================
# 1ter. RECOMMANDATION : le pipeline ne peut recommander que l'outcome viable
# ===========================================================================
from tests.helpers import build_pipeau_dataframe, build_ghost_peloton  # noqa: E402


def _run_reco(tmp_path, true_proba, crowd, gains, n_matchs=3):
    """Construit un CSV pipeau et renvoie la chaîne `reco` du pipeline déterministe."""
    df = build_pipeau_dataframe(n_matchs, true_proba, crowd, gains)
    path = tmp_path / "reco.csv"
    df.to_csv(path, index=False)
    reco, _, _, _ = run_daily_pipeline(
        csv_path=path,
        match_id_cible=1,
        mon_gap_1=0,
        mon_gap_2=GAP_PELOTON_SAFE,
        has_booster=1,
        use_drift=False,
        horizon_nuit=1,
        nb_scenarios=1,
        p_empirique_override=build_ghost_peloton(n_matchs, max_gain=250),
        save_abaques=False,
        validate_input=False,
    )
    return reco


def test_reco_unique_outcome_gainful(tmp_path):
    """Si seul l'outcome 1 (Dom) rapporte des points, c'est le seul levier => reco '1'."""
    reco = _run_reco(
        tmp_path,
        true_proba=np.array([0.50, 0.30, 0.20]),
        crowd=np.array([0.40, 0.35, 0.25]),
        gains=np.array([50, 0, 0], dtype=np.int64),   # N et 2 ne rapportent rien
    )
    assert reco.startswith("1 (Dom)"), f"reco={reco!r} (attendu outcome 1)"


def test_reco_outcome_quasi_certain(tmp_path):
    """Si l'outcome 2 (Ext) est quasi certain (proba ~1) et gains égaux, reco '2'."""
    reco = _run_reco(
        tmp_path,
        true_proba=np.array([0.01, 0.01, 0.98]),      # issue 2 quasi sûre
        crowd=np.array([0.40, 0.35, 0.25]),
        gains=np.array([50, 50, 50], dtype=np.int64),  # gains égaux : seul la proba décide
    )
    assert reco.startswith("2 (Ext)"), f"reco={reco!r} (attendu outcome 2)"


# ===========================================================================
# 2. INTÉGRATION sur les CSV réalistes de data/tests/
# ===========================================================================
FIXTURE_CSVS = sorted(DATA_TESTS_DIR.glob("*.csv")) if DATA_TESTS_DIR.exists() else []


@pytest.mark.skipif(not FIXTURE_CSVS, reason="Aucun CSV dans data/tests/.")
@pytest.mark.parametrize("csv_path", FIXTURE_CSVS, ids=lambda p: p.stem)
def test_pipeline_integration_invariants(csv_path):
    """
    Le pipeline tourne sur un CSV réaliste (données + p_empirique + horizon réels)
    et produit des sorties saines. Déterministe (use_drift=False) pour la rapidité.
    """
    reco, wr, ev, q = run_daily_pipeline(
        csv_path=csv_path,
        match_id_cible=1,
        mon_gap_1=0,
        mon_gap_2=0,
        has_booster=1,
        use_drift=False,
        horizon_nuit=1,
        nb_scenarios=1,
        save_abaques=False,
    )

    # Sorties dans les bornes
    assert 0.0 <= wr <= 1.0
    assert q.shape == (1001, 1001, 3, 3)
    assert np.all(q >= -1e-6) and np.all(q <= 1.0 + 1e-6)
    assert isinstance(reco, str) and len(reco) > 0

    # Invariant "golden rule" : à peloton fixé, avoir plus d'avance sur Bob
    # ne peut pas diminuer le win rate (monotonie en gap_1).
    tranche = q[:, GAP_OFFSET, :, 0].max(axis=1)  # WR(meilleure action) le long de gap_1
    diffs = np.diff(tranche)
    assert np.sum(diffs < -1e-4) == 0, "Violation de monotonie en gap_1"
