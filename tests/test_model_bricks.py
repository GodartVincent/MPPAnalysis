"""
Tests unitaires des briques du modèle (core.py) jusqu'ici non couvertes :
  - apply_temporal_drift  : drift temporel des probabilités
  - estimate_crowd_3D     : modèle softmax de répartition de la foule
  - calculate_mpp_gains   : conversion cubique proba -> gain MPP
"""
import numpy as np
import pytest

from mpp_project.core import (
    apply_temporal_drift,
    estimate_crowd_3D,
    calculate_mpp_gains,
    MIN_MPP_GAIN, MAX_MPP_GAIN, MIN_TRUE_PROBA, MAX_TRUE_PROBA, CROWD_EPSILON,
)


# ---------------------------------------------------------------------------
# apply_temporal_drift
# ---------------------------------------------------------------------------
def test_drift_passe_inchange_et_normalise():
    phases = ["Poule_J1", "Poule_J2", "Poule_J3"]
    tp = np.array([[0.5, 0.3, 0.2]] * 3, dtype=np.float64)
    np.random.seed(0)
    out = np.array([apply_temporal_drift(tp, phases, current_match_idx=0) for _ in range(500)])
    # Le match courant/passé (i <= current_match_idx) n'est jamais modifié
    assert np.allclose(out[:, 0, :], tp[0])
    # Sorties toujours normalisées et bornées
    assert np.allclose(out.sum(axis=2), 1.0)
    assert out.min() >= MIN_TRUE_PROBA - 1e-6
    assert out.max() <= MAX_TRUE_PROBA + 1e-6


def test_drift_plus_fort_pour_matchs_lointains():
    """La variance du drift croît avec l'éloignement (J3 > J2 depuis J1)."""
    phases = ["Poule_J1", "Poule_J2", "Poule_J3"]
    tp = np.array([[0.5, 0.3, 0.2]] * 3, dtype=np.float64)
    np.random.seed(0)
    out = np.array([apply_temporal_drift(tp, phases, current_match_idx=0) for _ in range(4000)])
    var_proche = out[:, 1, :].var(axis=0).mean()   # J2
    var_lointain = out[:, 2, :].var(axis=0).mean()  # J3
    assert var_lointain > var_proche


# ---------------------------------------------------------------------------
# apply_temporal_drift — régime DATE (diffusion + phase, variances additives)
# ---------------------------------------------------------------------------
def test_drift_date_match_courant_fige():
    """Le match courant/passé (i <= current_match_idx) reste figé même en régime date."""
    phases = ["Poule_J1"] * 3
    tp = np.array([[0.5, 0.3, 0.2]] * 3, dtype=np.float64)
    dates = np.array(["2026-06-17", "2026-06-20", "2026-06-24"], dtype="datetime64[ns]")
    np.random.seed(0)
    out = np.array([
        apply_temporal_drift(tp, phases, current_match_idx=0,
                             match_dates=dates, reference_date="2026-06-17")
        for _ in range(300)
    ])
    assert np.allclose(out[:, 0, :], tp[0])
    # Les matchs futurs (1 et 2) sont bien driftés
    assert out[:, 1, :].var(axis=0).mean() > 0.0
    assert out[:, 2, :].var(axis=0).mean() > 0.0


def test_drift_date_diffusion_ajoute_de_la_variance():
    """À phase ÉGALE, un match plus lointain en jours est plus drifté (composante diffusion)."""
    phases = ["Poule_J1"] * 3  # même phase -> composante phase constante
    tp = np.array([[0.5, 0.3, 0.2]] * 3, dtype=np.float64)
    dates = np.array(["2026-06-17", "2026-06-18", "2026-07-05"], dtype="datetime64[ns]")  # +1 j / +18 j
    np.random.seed(0)
    out = np.array([
        apply_temporal_drift(tp, phases, current_match_idx=0,
                             match_dates=dates, reference_date="2026-06-17")
        for _ in range(6000)
    ])
    var_proche = out[:, 1, :].var(axis=0).mean()    # +1 j
    var_lointain = out[:, 2, :].var(axis=0).mean()  # +18 j
    assert var_lointain > var_proche
    # Sorties bornées et normalisées
    assert np.allclose(out.sum(axis=2), 1.0)
    assert out.min() >= MIN_TRUE_PROBA - 1e-6
    assert out.max() <= MAX_TRUE_PROBA + 1e-6


def test_drift_date_combine_phase_et_diffusion():
    """Régime date = sqrt(diffusion^2 + phase^2) > la composante phase seule (d > 0)."""
    phases = ["Poule_J1", "Poule_J2"]  # match 1 : 1 phase d'écart
    tp = np.array([[0.5, 0.3, 0.2]] * 2, dtype=np.float64)
    dates = np.array(["2026-06-17", "2026-06-24"], dtype="datetime64[ns]")  # +7 j
    n = 8000
    np.random.seed(0)
    out_date = np.array([
        apply_temporal_drift(tp, phases, current_match_idx=0,
                             match_dates=dates, reference_date="2026-06-17")
        for _ in range(n)
    ])
    np.random.seed(0)
    out_phase = np.array([
        apply_temporal_drift(tp, phases, current_match_idx=0)  # phase seule (pas de dates)
        for _ in range(n)
    ])
    var_date = out_date[:, 1, :].var(axis=0).mean()
    var_phase = out_phase[:, 1, :].var(axis=0).mean()
    assert var_date > var_phase  # la diffusion ajoute de la variance par-dessus la phase


def test_drift_date_reference_today_par_defaut_ok():
    """reference_date=None -> date.today() : exécution sans erreur, sortie normalisée/bornée."""
    phases = ["Poule_J1", "Poule_J2", "Poule_J3"]
    tp = np.array([[0.5, 0.3, 0.2]] * 3, dtype=np.float64)
    dates = np.array(["2026-06-11", "2026-06-18", "2026-06-24"], dtype="datetime64[ns]")
    np.random.seed(0)
    out = apply_temporal_drift(tp, phases, current_match_idx=0, match_dates=dates)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# estimate_crowd_3D
# ---------------------------------------------------------------------------
def test_crowd_somme_a_un_et_plancher():
    p1 = np.array([0.60, 0.40, 0.33])
    pN = np.array([0.25, 0.35, 0.33])
    p2 = np.array([0.15, 0.25, 0.34])
    c1, cN, c2 = estimate_crowd_3D(p1, pN, p2, add_noise=False)
    crowds = np.column_stack((c1, cN, c2))
    assert np.allclose(crowds.sum(axis=1), 1.0)
    # Plancher incompressible (eps) respecté sur chaque issue
    assert crowds.min() >= CROWD_EPSILON - 1e-9


def test_crowd_amplifie_le_favori():
    """Avec beta > 1, le favori est sur-représenté : p plus haut -> crowd plus haut."""
    c1_fav, _, _ = estimate_crowd_3D(np.array([0.60]), np.array([0.25]), np.array([0.15]),
                                     add_noise=False)
    c1_serre, _, _ = estimate_crowd_3D(np.array([0.40]), np.array([0.35]), np.array([0.25]),
                                       add_noise=False)
    assert c1_fav[0] > c1_serre[0]


# ---------------------------------------------------------------------------
# calculate_mpp_gains
# ---------------------------------------------------------------------------
def test_gains_dans_les_bornes():
    probas = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.95])
    gains = calculate_mpp_gains(probas, add_noise=False)
    assert gains.min() >= MIN_MPP_GAIN
    assert gains.max() <= MAX_MPP_GAIN
    assert gains.dtype.kind in "iu"   # entiers


def test_gains_decroissants_avec_la_proba():
    """Plus une issue est probable (favori), moins elle rapporte de points."""
    probas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    gains = calculate_mpp_gains(probas, add_noise=False)
    assert np.all(np.diff(gains) <= 0), f"gains non décroissants : {gains}"
