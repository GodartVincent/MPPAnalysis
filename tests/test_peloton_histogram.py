"""
Tests de extract_peloton_full_distribution (histogrammes de transition de gap_2).

Cette fonction Monte Carlo (1M runs) est le SEUL maillon non couvert par les
autres tests : run_daily_pipeline charge p_empirique_1D depuis le disque et ne
l'appelle jamais. On la teste donc directement, avec des tableaux construits.

delta_pack = (meilleur score du peloton HORS leader) après le match
           - (idem) avant le match.

Rappels de la mécanique (cf. oracle_dp.extract_peloton_full_distribution) :
  - n_players joueurs ; le leader (bob) est exclu du "peloton".
  - chaque joueur parie selon crowd (seuils CUMULÉS -> crowd doit sommer à 1).
  - p_empirique[t, out, :] = distribution de delta_pack | (issue du match t == out).
"""
import numpy as np
import pytest

from mpp_project.oracle_dp import extract_peloton_full_distribution

N_PLAYERS = 11
N_OTHERS = N_PLAYERS - 1           # le peloton exclut le leader
MAX_GAIN = 250


def _tile(vec, n_matches):
    """(3,) -> (1, n_matches, 3) : un seul univers, n_matches identiques."""
    return np.tile(np.asarray(vec, np.float32), (n_matches, 1)).reshape(1, n_matches, 3)


# ===========================================================================
# 1. PREMIER MATCH : histogramme BINAIRE et valeurs prédictibles
# ===========================================================================
# À t=0 tous les scores valent 0, donc le peloton (hors leader) ne progresse
# que si AU MOINS un des N_OTHERS joueurs trouve l'issue :
#   P(delta == 0)          = (1 - crowd[out]) ** N_OTHERS
#   P(delta == gains[out]) = 1 - (1 - crowd[out]) ** N_OTHERS
# et AUCUN autre bin n'est atteignable.
def test_premier_match_binaire():
    true_proba = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    crowd = np.array([0.5, 0.3, 0.2], dtype=np.float32)   # normalisé (somme = 1)
    gains = np.array([10, 20, 30], dtype=np.int32)
    n_matches = 3

    p = extract_peloton_full_distribution(
        _tile(true_proba, n_matches), _tile(crowd, n_matches),
        np.tile(gains, (n_matches, 1)).astype(np.int32),
        max_gain=MAX_GAIN, n_runs=500_000, n_players=N_PLAYERS,
    )

    for out in range(3):
        hist = p[0, out]
        bins_non_nuls = set(np.nonzero(hist)[0].tolist())
        # (a) Strictement binaire : seuls 0 et gains[out] sont possibles
        assert bins_non_nuls <= {0, int(gains[out])}, (
            f"out={out}: bins inattendus {bins_non_nuls - {0, int(gains[out])}}"
        )
        # (b) Valeurs conformes à la combinatoire
        pred_zero = (1.0 - crowd[out]) ** N_OTHERS
        assert hist[0] == pytest.approx(pred_zero, abs=0.01)
        assert hist[int(gains[out])] == pytest.approx(1.0 - pred_zero, abs=0.01)
        # (c) Distribution normalisée
        assert hist.sum() == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# 2. CONFIG DÉTERMINISTE : delta_pack connu exactement à chaque match
# ===========================================================================
# crowd = proba = [1, 0, 0] : tout le monde parie 0, l'issue est toujours 0,
# tous les joueurs gagnent gains[t, 0] à chaque match -> le peloton progresse
# d'exactement gains[t, 0]. L'histogramme est un Dirac sur ce bin (valeur 1.0),
# quelle que soit la part aléatoire (déterminisme total).
def test_deterministe_delta_connu():
    n_matches = 3
    gains = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int32)

    tp = np.zeros((1, n_matches, 3), np.float32); tp[:, :, 0] = 1.0
    cr = np.zeros((1, n_matches, 3), np.float32); cr[:, :, 0] = 1.0

    p = extract_peloton_full_distribution(
        tp, cr, gains, max_gain=MAX_GAIN, n_runs=2000, n_players=N_PLAYERS,
    )

    for t in range(n_matches):
        g0 = int(gains[t, 0])
        hist = p[t, 0]
        assert set(np.nonzero(hist)[0].tolist()) == {g0}, f"t={t}: doit être un Dirac sur {g0}"
        assert hist[g0] == pytest.approx(1.0, abs=1e-6)
        # Les issues 1 et 2 ne se produisent jamais -> histogramme vide
        assert p[t, 1].sum() == pytest.approx(0.0, abs=1e-6)
        assert p[t, 2].sum() == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 3. PROPRIÉTÉS GÉNÉRALES (sur une config réaliste)
# ===========================================================================
def test_proprietes_generales():
    true_proba = np.array([0.45, 0.30, 0.25], dtype=np.float32)
    crowd = np.array([0.40, 0.35, 0.25], dtype=np.float32)
    gains = np.array([15, 25, 40], dtype=np.int32)
    n_matches = 4

    p = extract_peloton_full_distribution(
        _tile(true_proba, n_matches), _tile(crowd, n_matches),
        np.tile(gains, (n_matches, 1)).astype(np.int32),
        max_gain=MAX_GAIN, n_runs=300_000, n_players=N_PLAYERS,
    )

    # Probabilités valides
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    # Chaque distribution (t, out) somme à 1 (issue forcément observée avec ces probas)
    sums = p.sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-4)
    # delta_pack >= 0 (le peloton ne peut pas reculer) : bin 0 inclus, jamais négatif
    # (garanti par construction ; on vérifie qu'aucune masse n'est "perdue").
    assert p.shape == (n_matches, 3, MAX_GAIN)


# ===========================================================================
# 4. ISSUES RÉELLES CONNUES (known_outcomes — mode horizon glissant)
# ===========================================================================
def test_known_outcomes_fixe_les_issues_passees():
    """
    Pour un match passé (known_outcomes >= 0), seule l'issue réelle est peuplée ;
    un match futur (-1) garde ses 3 issues possibles. Le défaut (None) est inchangé.
    """
    tp = _tile([0.5, 0.3, 0.2], 3)
    cr = _tile([0.5, 0.3, 0.2], 3)
    g = np.tile([10, 20, 30], (3, 1)).astype(np.int32)
    ko = np.array([0, 2, -1])   # match 0 -> '1', match 1 -> '2', match 2 -> futur

    p = extract_peloton_full_distribution(
        tp, cr, g, max_gain=MAX_GAIN, n_runs=100_000, n_players=N_PLAYERS,
        known_outcomes=ko,
    )

    # Match 0 joué (issue 0) : seule out=0 peuplée
    assert p[0, 0].sum() == pytest.approx(1.0, abs=1e-9)
    assert p[0, 1].sum() == 0.0 and p[0, 2].sum() == 0.0
    # Match 1 joué (issue 2) : seule out=2 peuplée
    assert p[1, 2].sum() == pytest.approx(1.0, abs=1e-9)
    assert p[1, 0].sum() == 0.0 and p[1, 1].sum() == 0.0
    # Match 2 futur : les 3 issues restent possibles
    assert all(p[2, o].sum() == pytest.approx(1.0, abs=1e-9) for o in range(3))
