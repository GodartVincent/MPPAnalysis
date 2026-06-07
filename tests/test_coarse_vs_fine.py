"""
Accord entre solve_dp_coarse (501², demi-points) et
solve_dp_with_full_empirical_distribution (1001², pleine résolution).

La grille coarse fonde le gain de vitesse ×4 du pipeline ; ce test garantit
qu'elle reste une approximation fidèle de la grille fine. Avec des gains PAIRS,
l'accord est exact (les transitions tombent sur des points de grille).
"""
import numpy as np
import pytest

from mpp_project.oracle_dp import (
    solve_dp_coarse,
    solve_dp_with_full_empirical_distribution,
)


def _inputs(n_matches=3, gains=(20, 50, 90)):
    tp = np.array([[0.5, 0.3, 0.2]] * n_matches, dtype=np.float32)
    cr = np.array([[0.5, 0.3, 0.2]] * n_matches, dtype=np.float32)
    g = np.array([list(gains)] * n_matches, dtype=np.int32)
    pe = np.zeros((n_matches, 3, 250), dtype=np.float32)
    pe[:, :, 0] = 1.0                      # peloton fantôme (isole gap_1)
    al = np.ones(n_matches, dtype=np.float32)
    return tp, cr, g, pe, al


def test_coarse_concorde_avec_fine():
    tp, cr, g, pe, al = _inputs()

    V_fine, _ = solve_dp_with_full_empirical_distribution(
        tp, cr, g, pe, al, np.zeros((1001, 1001, 2), np.float32), stop_t=0)
    V_coarse, _ = solve_dp_coarse(
        tp, cr, g, pe, al, np.zeros((501, 501, 2), np.float32), stop_t=0)

    # On compare la valeur du match 0, sur le triangle valide (g1 <= g2),
    # fine downsamplée [::2, ::2] vs coarse.
    fine0 = V_fine[0][::2, ::2, :]
    coarse0 = V_coarse[0]
    iu = np.triu_indices(501)
    for canal in (0, 1):
        diff = np.abs(fine0[:, :, canal][iu] - coarse0[:, :, canal][iu])
        assert diff.max() < 1e-3, f"canal {canal}: écart max coarse/fine = {diff.max():.4f}"
