"""
Configuration pytest partagée + fixtures pour la suite de tests de l'Oracle DP.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Racine du projet (un niveau au-dessus de tests/) sur le sys.path pour importer mpp_project
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.helpers import build_pipeau_dataframe, build_ghost_peloton, DATA_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Paramètres du match "pipeau" canonique (partagés par les tests d'oracle exact)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeau_params():
    """Paramètres d'un match type. Gains PAIRS -> pas d'arrondi en grille coarse."""
    return {
        "true_proba": np.array([0.60, 0.25, 0.15], dtype=float),
        "crowd": np.array([0.70, 0.20, 0.10], dtype=float),
        "gains": np.array([20, 50, 90], dtype=np.int64),
    }


@pytest.fixture(scope="session")
def pipeau_csv(tmp_path_factory, pipeau_params):
    """
    Écrit un CSV temporaire de 4 matchs identiques et renvoie (chemin, n_matchs).
    scope=session : généré une seule fois pour toute la session de test.
    """
    n_matchs = 4
    df = build_pipeau_dataframe(
        n_matchs,
        pipeau_params["true_proba"],
        pipeau_params["crowd"],
        pipeau_params["gains"],
    )
    path = tmp_path_factory.mktemp("pipeau") / "pipeau_4matchs.csv"
    df.to_csv(path, index=False)
    return path, n_matchs


@pytest.fixture(scope="session")
def ghost_peloton_4():
    """Peloton fantôme pour 4 matchs (max_gain=250)."""
    return build_ghost_peloton(4, max_gain=250)


# ---------------------------------------------------------------------------
# Skips conditionnels selon la présence des gros artefacts data/
# ---------------------------------------------------------------------------
def require_file(relpath):
    """Renvoie un marqueur skipif si le fichier data/ est absent (gros artefacts)."""
    full = DATA_DIR / relpath
    return pytest.mark.skipif(
        not full.exists(),
        reason=f"Artefact absent : data/{relpath} (générer avant de lancer ce test).",
    )
