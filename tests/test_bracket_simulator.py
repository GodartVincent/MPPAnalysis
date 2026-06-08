"""
Smoke tests de bracket_simulator (simulation poules + bracket knockout).

Consommateur de CDM_2026_group_stage_odds.csv (où vivait le typo 'parnama').
Stochastique -> graine fixée. Vérifie surtout l'intégrité structurelle
(qualifiés, slots, mapping des équipes) et l'absence de KeyError d'incohérence.
"""
import numpy as np
import pandas as pd
import pytest

from mpp_project.bracket_simulator import (
    simulate_group_stages,
    generate_bracket_scenario,
    poules_horizon_from_full,
)
from tests.helpers import DATA_DIR

ODDS_PATH = DATA_DIR / "CDM_2026_group_stage_odds.csv"

pytestmark = pytest.mark.skipif(
    not ODDS_PATH.exists(), reason="CDM_2026_group_stage_odds.csv absent."
)


@pytest.fixture(scope="module")
def df_odds():
    return pd.read_csv(ODDS_PATH)


def test_structure_des_groupes(df_odds):
    """48 équipes réparties en 12 groupes de 4 (pré-requis des tirages)."""
    assert len(df_odds) == 48
    tailles = df_odds.groupby("group").size()
    assert len(tailles) == 12
    assert (tailles == 4).all(), f"groupes non standards : {tailles.to_dict()}"


def test_simulate_group_stages(df_odds):
    np.random.seed(0)
    q1, q2, q3 = simulate_group_stages(df_odds)
    # 12 premiers, 12 deuxièmes, 8 meilleurs troisièmes
    assert len(q1) == 12 and len(q2) == 12 and len(q3) == 8
    # Équipes valides et distinctes
    teams_valides = set(df_odds["team"])
    selectionnes = [d["team"] for d in q1] + [d["team"] for d in q2] + list(q3)
    assert all(t in teams_valides for t in selectionnes)
    assert len(set(selectionnes)) == len(selectionnes), "équipe qualifiée en double"


def test_generate_bracket_scenario(df_odds):
    """Le bracket Monte-Carlo se construit sans erreur et produit des matrices saines."""
    np.random.seed(0)
    match_probs, mpp_gains, crowds, match_teams, team_to_id = generate_bracket_scenario(df_odds)
    assert match_probs.shape == (32, 3)
    assert mpp_gains.shape == (32, 3)
    assert crowds.shape == (32, 3)
    # Probabilités et foules valides (lignes renseignées normalisées)
    assert np.all(match_probs >= -1e-6) and np.all(match_probs <= 1.0 + 1e-6)
    assert np.all(crowds >= -1e-6) and np.all(crowds <= 1.0 + 1e-6)
    assert len(team_to_id) == 48


def test_override_affiches_et_propagation(df_odds):
    """
    Reality Override : les affiches connues de CDM_2026.csv et leurs résultats
    sont pris en compte, et les vainqueurs se propagent au tour suivant.

    On force les 16 matchs des 16es (team_A gagne toujours) et on vérifie :
      (1) les affiches des 16es écrasent bien le bracket simulé ;
      (2) le vainqueur de chaque 16e remonte dans le bon slot du 8e
          (next_match_map : matchs 0 et 1 -> les deux équipes du match 16).
    """
    teams = list(df_odds["team"])
    rows = [{
        "phase": "16e",
        "team_A": teams[2 * m], "team_B": teams[2 * m + 1],
        "cote_1": np.nan, "cote_N": np.nan, "cote_2": np.nan,
        "result": teams[2 * m],          # team_A gagne
    } for m in range(16)]
    df_tournoi = pd.DataFrame(rows)

    np.random.seed(0)
    _, _, _, match_teams, t2id = generate_bracket_scenario(df_odds, df_tournoi=df_tournoi)

    # (1) affiches des 16es respectées
    for m in range(16):
        assert match_teams[m, 0] == t2id[teams[2 * m]]
        assert match_teams[m, 1] == t2id[teams[2 * m + 1]]

    # (2) propagation : vainqueurs des matchs 0 et 1 -> les 2 équipes du match 16
    assert match_teams[16, 0] == t2id[teams[0]]
    assert match_teams[16, 1] == t2id[teams[2]]


def test_override_insensible_a_la_casse(df_odds):
    """
    Robustesse à la casse : une phase 'quart' (minuscule, convention CSV) doit
    être prise en compte par le Reality Override (régression du bug
    case-sensitive du masque 'Quart|Demi|Finale').
    """
    teams = list(df_odds["team"])
    # 28 lignes en ordre de bracket (16x 16e, 8x 8e, 4x quart minuscule).
    rows = []
    for m in range(16):
        rows.append({"phase": "16e", "team_A": teams[2 * m], "team_B": teams[2 * m + 1],
                     "cote_1": np.nan, "result": teams[2 * m]})
    for m in range(8):
        rows.append({"phase": "8e", "team_A": teams[m], "team_B": teams[m + 8],
                     "cote_1": np.nan, "result": teams[m]})
    quart_teams = []
    for m in range(4):
        a, b = teams[m], teams[m + 4]
        quart_teams.append((a, b))
        rows.append({"phase": "quart", "team_A": a, "team_B": b,   # minuscule !
                     "cote_1": np.nan, "result": a})
    df_tournoi = pd.DataFrame(rows)

    np.random.seed(0)
    _, _, _, match_teams, t2id = generate_bracket_scenario(df_odds, df_tournoi=df_tournoi)

    # Les matchs 24..27 sont les quarts : leurs affiches minuscules doivent être prises.
    for i, (a, b) in enumerate(quart_teams):
        assert match_teams[24 + i, 0] == t2id[a], f"quart {i}: team_A non overridée (casse ?)"
        assert match_teams[24 + i, 1] == t2id[b], f"quart {i}: team_B non overridée (casse ?)"


# ---------------------------------------------------------------------------
# poules_horizon_from_full : dérivation de l'horizon poules depuis _full
# ---------------------------------------------------------------------------
def test_poules_horizon_7d():
    """7D (32, g1, g2, booster, fav×3) -> (32, g1, g2, booster), tranche favoris vivants."""
    full = np.zeros((32, 6, 6, 2, 2, 2, 2), dtype=np.float32)
    full[..., 1, 1, 1] = 0.42        # favoris vivants
    full[..., 0, 0, 0] = 0.99        # favoris morts (ne doit PAS être pris)
    out = poules_horizon_from_full(full)
    assert out.shape == (32, 6, 6, 2)
    assert np.allclose(out, 0.42)


def test_poules_horizon_6d_ajoute_axe_match():
    """6D (matrice unique) -> (1, g1, g2, booster) : axe match ajouté pour daily_pipeline[0]."""
    full = np.zeros((6, 6, 2, 2, 2, 2), dtype=np.float32)
    full[..., 1, 1, 1] = 0.7
    out = poules_horizon_from_full(full)
    assert out.shape == (1, 6, 6, 2)
    assert np.allclose(out[0], 0.7)
