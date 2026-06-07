"""
Tests d'intégrité (invariants) de la matrice Oracle sauvegardée.

Convertit l'ancien script de diagnostic en tests pytest assert-based. On vérifie
trois propriétés structurelles sur chaque matrice match de
data/expected_V_phases_finales_full.npy :

  1. MONOTONIE   : à peloton fixé, plus d'avance sur Bob => WR non décroissant.
  2. BOOSTER     : disposer du booster a une valeur strictement positive.
  3. ÉVIDENCE    : être largement devant vaut mieux qu'être à égalité.

Le fichier faisant ~2 Go, il est chargé en mmap (lecture paresseuse) et le test
est ignoré automatiquement s'il est absent.
"""
import numpy as np
import pytest

from mpp_project.oracle_dp import GAP_OFFSET
from tests.helpers import DATA_DIR

MATRIX_PATH = DATA_DIR / "expected_V_phases_finales_full.npy"

pytestmark = pytest.mark.skipif(
    not MATRIX_PATH.exists(),
    reason="data/expected_V_phases_finales_full.npy absent (générer la matrice Oracle).",
)


@pytest.fixture(scope="module")
def V_full():
    """Charge la matrice en lecture seule (mmap) et garantit le format 7D."""
    V = np.load(MATRIX_PATH, mmap_mode="r")
    if V.ndim == 6:          # une seule matrice -> on ajoute l'axe match
        V = np.expand_dims(V, axis=0)
    assert V.ndim == 7, f"Dimensions inattendues : {V.ndim}D (attendu 6 ou 7)."
    return V


def test_monotonie_gap1(V_full):
    """Le WR doit croître (au sens large) avec l'avance sur Bob, pour chaque match."""
    for m in range(V_full.shape[0]):
        tranche = np.asarray(V_full[m, :, GAP_OFFSET, 0, 1, 1, 1])
        diffs = np.diff(tranche)
        violations = int(np.sum(diffs < -1e-5))   # tolérance float32
        assert violations == 0, f"Matrice {m}: {violations} violations de monotonie."


def test_valeur_du_booster(V_full):
    """Au point neutre (gap 0/0), avoir le booster doit valoir > sans booster."""
    for m in range(V_full.shape[0]):
        with_booster = float(V_full[m, GAP_OFFSET, GAP_OFFSET, 1, 1, 1, 1])
        without_booster = float(V_full[m, GAP_OFFSET, GAP_OFFSET, 0, 1, 1, 1])
        assert with_booster - without_booster > 0, (
            f"Matrice {m}: booster sans valeur ({(with_booster - without_booster) * 100:.3f}%)."
        )


def test_evidence_du_gain(V_full):
    """Être largement devant (gap +50) vaut strictement mieux qu'à égalité (gap 0)."""
    for m in range(V_full.shape[0]):
        wr_egalite = float(V_full[m, GAP_OFFSET, GAP_OFFSET, 0, 1, 1, 1])
        safe = min(GAP_OFFSET + 50, V_full.shape[1] - 1)
        wr_devant = float(V_full[m, safe, safe, 0, 1, 1, 1])
        assert wr_devant > wr_egalite, (
            f"Matrice {m}: WR devant ({wr_devant:.4f}) <= WR égalité ({wr_egalite:.4f})."
        )
