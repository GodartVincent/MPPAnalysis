"""
Tests de validation/normalisation des données d'entrée (crowds & cotes).

- normalize_crowds : renormalise par ligne, alerte si la somme brute est aberrante.
- calculate_true_outcome_probas_from_odds : alerte si marge bookmaker négative.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mpp_project.core import (
    normalize_crowds,
    calculate_true_outcome_probas_from_odds,
    validate_match_dataframe,
    validate_team_consistency,
)
from tests.helpers import DATA_DIR


# ---------------------------------------------------------------------------
# normalize_crowds
# ---------------------------------------------------------------------------
def test_normalize_crowds_renormalise():
    raw = np.array([[0.82, 0.10, 0.08],   # somme 1.00
                    [0.40, 0.30, 0.29]])  # somme 0.99 (arrondi)
    out = normalize_crowds(raw)
    assert np.allclose(out.sum(axis=1), 1.0)
    # Les proportions relatives sont préservées
    assert out[1, 0] == pytest.approx(0.40 / 0.99, abs=1e-9)


def test_normalize_crowds_pas_de_warning_dans_la_tolerance():
    raw = np.array([[0.50, 0.30, 0.19],   # 0.99 -> OK
                    [0.50, 0.30, 0.21]])  # 1.01 -> OK
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # tout warning fait échouer
        normalize_crowds(raw)              # ne doit rien lever


def test_normalize_crowds_warning_hors_tolerance():
    raw = np.array([[0.50, 0.30, 0.05],    # 0.85 -> faute de saisie
                    [0.50, 0.30, 0.20]])   # 1.00 -> OK
    with pytest.warns(UserWarning, match="index 0"):
        out = normalize_crowds(raw)
    # Malgré l'alerte, la sortie est bien normalisée
    assert np.allclose(out.sum(axis=1), 1.0)


def test_normalize_crowds_1d():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # entrée non normalisée volontaire
        out = normalize_crowds(np.array([2.0, 1.0, 1.0]))
    assert out.shape == (1, 3)
    assert out[0] == pytest.approx([0.5, 0.25, 0.25])


# ---------------------------------------------------------------------------
# Marge bookmaker (somme des 1/cote)
# ---------------------------------------------------------------------------
def test_odds_marge_positive_pas_de_warning():
    # Marge réaliste positive : somme(1/cote) > 1 (cas NORMAL, aucune alerte)
    odds = np.array([[1.5, 4.1, 6.75]])    # ~ 1.05
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = calculate_true_outcome_probas_from_odds(odds)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_odds_marge_negative_warning():
    # Marge négative impossible : somme(1/cote) < 1 -> alerte
    odds = np.array([[3.0, 4.0, 5.0]])     # 0.333+0.25+0.2 = 0.783 < 1
    with pytest.warns(UserWarning, match="marge bookmaker négative"):
        calculate_true_outcome_probas_from_odds(odds)


def test_odds_marge_excessive_warning():
    # Marge > 1.15 : cote probablement mal saisie -> alerte
    odds = np.array([[1.5, 3.0, 3.5]])     # 0.667+0.333+0.286 = 1.286 > 1.15
    with pytest.warns(UserWarning, match="marge bookmaker excessive"):
        calculate_true_outcome_probas_from_odds(odds)


# ---------------------------------------------------------------------------
# validate_match_dataframe (phases, gains, équipes)
# ---------------------------------------------------------------------------
def _df_ok():
    return pd.DataFrame({
        "phase": ["Poule_J1", "Poule_J1"],
        "team_A": ["france", "bresil"],
        "team_B": ["bresil", "france"],
        "gain_mpp_1": [40, 50], "gain_mpp_N": [80, 90], "gain_mpp_2": [120, 110],
    })


def test_validate_df_propre_silencieux():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_match_dataframe(_df_ok())


def test_validate_df_phase_inattendue():
    df = _df_ok(); df.loc[0, "phase"] = "Poule_J9"
    with pytest.warns(UserWarning, match="Phase"):
        validate_match_dataframe(df)


def test_validate_df_gain_aberrant():
    df = _df_ok()
    df.loc[0, "gain_mpp_1"] = 5      # < 14
    df.loc[1, "gain_mpp_2"] = 300    # > 250
    with pytest.warns(UserWarning, match="Gain MPP aberrant"):
        validate_match_dataframe(df)


def test_validate_df_equipe_unique():
    df = _df_ok(); df.loc[0, "team_A"] = "francee"   # faute -> france n'apparaît plus que 1x, francee 1x
    with pytest.warns(UserWarning, match="une seule fois"):
        validate_match_dataframe(df)


def test_validate_df_colonnes_absentes_ok():
    """DataFrame minimal (sans équipes) : pas d'erreur, colonnes ignorées."""
    df = pd.DataFrame({"phase": ["Poule_J1"], "gain_mpp_1": [40],
                       "gain_mpp_N": [80], "gain_mpp_2": [120]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_match_dataframe(df)


# ---------------------------------------------------------------------------
# validate_team_consistency (cohérence inter-fichiers)
# ---------------------------------------------------------------------------
def _matches_3x():
    # france et bresil jouent 3 matchs chacun (poules)
    return pd.DataFrame({
        "team_A": ["france", "bresil", "france", "bresil", "france", "bresil"],
        "team_B": ["bresil", "france", "bresil", "france", "bresil", "france"],
    })


def test_team_consistency_ok():
    matches = _matches_3x()
    odds = pd.DataFrame({"team": ["france", "bresil"]})
    market = pd.DataFrame({"category": ["favorite", "scorer"],
                           "selection": ["france", "kylian_mbappe"]})  # joueur ignoré
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_team_consistency(matches, odds, market)


def test_team_consistency_typo_dans_odds():
    matches = _matches_3x()
    odds = pd.DataFrame({"team": ["france", "bresill"]})   # faute -> 0 apparition
    with pytest.warns(UserWarning, match="bresill"):
        validate_team_consistency(matches, odds)


def test_team_consistency_favorite_absent():
    matches = _matches_3x()
    market = pd.DataFrame({"category": ["favorite"], "selection": ["espagne"]})  # absente
    with pytest.warns(UserWarning, match="espagne"):
        validate_team_consistency(matches, df_market=market)


def test_team_consistency_scorer_jamais_verifie():
    """Un joueur (scorer) absent du CSV des matchs ne doit JAMAIS déclencher d'alerte."""
    matches = _matches_3x()
    market = pd.DataFrame({"category": ["scorer"], "selection": ["erling_haaland"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_team_consistency(matches, df_market=market)


# ---------------------------------------------------------------------------
# Data-lint : cohérence des VRAIS fichiers du projet (régression données)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (DATA_DIR / "CDM_2026.csv").exists()
    or not (DATA_DIR / "CDM_2026_group_stage_odds.csv").exists()
    or not (DATA_DIR / "CDM_2026_goal_scorer_and_favorite.csv").exists(),
    reason="Fichiers CDM_2026 absents.",
)
def test_donnees_reelles_coherentes():
    """Les équipes des fichiers cotes/marché doivent exister dans CDM_2026.

    (Aurait échoué sur le typo 'parnama' -> 'panama' dans group_stage_odds.csv.)
    """
    matches = pd.read_csv(DATA_DIR / "CDM_2026.csv")
    odds = pd.read_csv(DATA_DIR / "CDM_2026_group_stage_odds.csv")
    market = pd.read_csv(DATA_DIR / "CDM_2026_goal_scorer_and_favorite.csv")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        validate_team_consistency(matches, odds, market)
    messages = [str(w.message) for w in captured]
    assert not messages, "Incohérences inter-fichiers détectées :\n" + "\n".join(messages)


def test_odds_probas_normalisees():
    """Quelle que soit la marge, les vraies probas somment à 1."""
    odds = np.array([[2.0, 3.5, 4.0], [1.2, 7.0, 12.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = calculate_true_outcome_probas_from_odds(odds)
    assert np.allclose(p.sum(axis=1), 1.0)
