"""
Garantie d'inoffensivité de la copie-symétrie du triangle inférieur.

Les moteurs DP remplissent le triangle supérieur (g1 <= g2) puis recopient en
miroir vers le triangle inférieur. Cette copie est mathématiquement FAUSSE
(gap_1 = écart à Bob et gap_2 = écart au peloton jouent des rôles asymétriques
via l'alpha-gating), mais INOFFENSIVE car les états atteignables vérifient
toujours g1 <= g2 : le triangle inférieur n'est jamais lu.

Ce test prouve l'invariant g1_final <= g2_final pour la formule de transition
(min / alpha-gating) sur tout le domaine, donc qu'aucun état (g1 > g2) n'est
jamais produit ni interrogé.
"""
import numpy as np


def _transition(new_gap_bob, new_gap_peloton, alpha):
    """Réplique la formule de transition des moteurs DP (oracle_dp / end_game_solver)."""
    g1_final = min(new_gap_bob, new_gap_peloton)
    g2_final = alpha * max(new_gap_bob, new_gap_peloton) + (1.0 - alpha) * new_gap_peloton
    return g1_final, g2_final


def test_g1_toujours_inferieur_ou_egal_a_g2():
    rng = np.random.default_rng(0)
    a = rng.uniform(-600, 400, 20000)      # new_gap_bob
    b = rng.uniform(-600, 400, 20000)      # new_gap_peloton
    alpha = rng.uniform(0.0, 1.0, 20000)
    for ai, bi, al in zip(a, b, alpha):
        g1, g2 = _transition(ai, bi, al)
        assert g1 <= g2 + 1e-9, f"violation: a={ai}, b={bi}, alpha={al} -> g1={g1} > g2={g2}"


def test_invariant_preserve_apres_clamp_et_arrondi():
    """Le clamp + arrondi entier (monotones) préservent g1_idx <= g2_idx."""
    rng = np.random.default_rng(1)
    for _ in range(20000):
        a = rng.uniform(-700, 500)
        b = rng.uniform(-700, 500)
        al = rng.uniform(0.0, 1.0)
        g1, g2 = _transition(a, b, al)
        g1_idx = max(-600, min(400, int(round(g1)))) + 600
        g2_idx = max(-600, min(400, int(round(g2)))) + 600
        assert g1_idx <= g2_idx
