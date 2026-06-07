"""
Tests du chemin use_drift=True (le multivers) de run_daily_pipeline.

C'est le chemin de PRODUCTION (architecture 3-phases : Phase 1 lointaine avec
drift × nb_scenarios, Phase 2 proche, split), absent des autres tests qui
utilisent tous use_drift=False.

Stochastique -> on fixe la graine numpy (apply_temporal_drift /
apply_heteroscedastic_noise utilisent le RNG global numpy) pour la reproductibilité.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpp_project.daily_pipeline import run_daily_pipeline
from mpp_project.oracle_dp import GAP_OFFSET
from tests.helpers import build_ghost_peloton

GAP_PELOTON_SAFE = 400
N_MATCHS = 18
NEAR_HORIZON = 5          # < N_MATCHS -> use_split=True (Phase 1 lointaine activée)


@pytest.fixture(scope="module")
def drift_csv():
    """18 matchs avec phases J1/J2/J3 (le drift dépend du niveau de phase)."""
    phases = ["Poule_J1"] * 6 + ["Poule_J2"] * 6 + ["Poule_J3"] * 6
    tp = np.array([0.5, 0.3, 0.2])
    cotes = 1.0 / tp
    crowd = np.array([0.45, 0.35, 0.20])
    rows = [{
        "phase": phases[i],
        "cote_1": cotes[0], "cote_N": cotes[1], "cote_2": cotes[2],
        "crowd_1": crowd[0], "crowd_N": crowd[1], "crowd_2": crowd[2],
        "gain_mpp_1": 20, "gain_mpp_N": 50, "gain_mpp_2": 90,
    } for i in range(N_MATCHS)]
    path = Path(tempfile.gettempdir()) / "_mpp_multiverse.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(path, use_drift, seed=0):
    np.random.seed(seed)
    _, _, _, q = run_daily_pipeline(
        csv_path=path,
        match_id_cible=1,
        mon_gap_1=0,
        mon_gap_2=GAP_PELOTON_SAFE,
        has_booster=1,
        use_drift=use_drift,
        horizon_nuit=1,
        nb_scenarios=4,
        near_horizon=NEAR_HORIZON,
        p_empirique_override=build_ghost_peloton(N_MATCHS, 250),
        save_abaques=False,
        validate_input=False,
    )
    return q


@pytest.fixture(scope="module")
def q_drift(drift_csv):
    return _run(drift_csv, use_drift=True)


@pytest.fixture(scope="module")
def q_nodrift(drift_csv):
    return _run(drift_csv, use_drift=False)


def test_multivers_sortie_valide(q_drift):
    """Le chemin use_drift=True s'exécute et produit une Q-table saine."""
    assert q_drift.shape == (1001, 1001, 3, 3)
    assert q_drift.min() >= -1e-6 and q_drift.max() <= 1.0 + 1e-6


def test_multivers_ne_panique_pas(q_drift, q_nodrift):
    """
    Cœur du multivers : à déficit modéré en début de tournoi, l'agent qui
    anticipe la forte incertitude future (drift) garde un meilleur win rate
    que l'agent déterministe -> il ne panique pas, il sait que des opportunités
    arriveront. WR(drift) doit dominer WR(no_drift) aux gaps négatifs.
    """
    idx_g2 = GAP_PELOTON_SAFE + GAP_OFFSET
    for gap in (-150, -100, -50):
        i1 = gap + GAP_OFFSET
        wr_drift = q_drift[i1, idx_g2, :, 0].max()
        wr_nodrift = q_nodrift[i1, idx_g2, :, 0].max()
        assert wr_drift >= wr_nodrift - 5e-3, (
            f"gap={gap}: drift={wr_drift:.4f} < no_drift={wr_nodrift:.4f}"
        )


def test_multivers_reproductible(drift_csv, q_drift):
    """Même graine -> même résultat (déterminisme sous seed)."""
    q_again = _run(drift_csv, use_drift=True, seed=0)
    assert np.allclose(q_drift, q_again)
