# Suite de tests — Oracle DP

Tests automatisés de non-régression pour les moteurs de programmation dynamique.
Chaque moteur DP (qui passe par une grille d'états) est confronté à un **solveur
récursif exact** (`helpers.exact_theoretical_wr`) qui énumère tout l'arbre
combinatoire — l'oracle de référence.

## Lancer les tests

```bash
# Tout (≈ 8-9 min : dominé par la compilation Numba de l'endgame DP)
.venv/Scripts/python.exe -m pytest

# Rapide : poules uniquement (oracle exact + intégration sur fixtures)
.venv/Scripts/python.exe -m pytest tests/test_group_stage_dp.py

# Invariants sur la matrice Oracle 2 Go (chargée en mmap)
.venv/Scripts/python.exe -m pytest tests/test_oracle_invariants.py
```

## Structure

| Fichier | Rôle | Source |
|---|---|---|
| `helpers.py` | Solveur récursif exact, conditions terminales, constructeurs de fixtures pipeau | factorisé depuis NB 21/22 |
| `conftest.py` | Fixtures pytest partagées (CSV pipeau, peloton fantôme) | — |
| `test_endgame_dp.py` | `solve_endgame_dp` (match unique + profondeur 4), base & booster | NB 20/21 |
| `test_group_stage_dp.py` | `run_daily_pipeline` : oracle exact (pipeau) + reco forcée + non-régression fin de tournoi + intégration sur `data/tests/*.csv` | NB 22 |
| `test_peloton_histogram.py` | `extract_peloton_full_distribution` : histogrammes de transition de gap_2 (1er match binaire, config déterministe, propriétés) | — |
| `test_data_validation.py` | `normalize_crowds`, marge bookmaker, `validate_match_dataframe` (phases, gains, équipes), cohérence inter-fichiers (`validate_team_consistency` + data-lint réel) | — |
| `test_multiverse_drift.py` | Chemin **use_drift=True** (production) : smoke, propriété « ne panique pas », reproductibilité | — |
| `test_model_bricks.py` | Briques core : `apply_temporal_drift`, `estimate_crowd_3D`, `calculate_mpp_gains` | — |
| `test_coarse_vs_fine.py` | Accord `solve_dp_coarse` ≈ grille fine (fonde le ×4) | — |
| `test_symmetry_invariant.py` | Invariant g1 ≤ g2 : la copie-symétrie du triangle inférieur est inoffensive | — |
| `test_bracket_simulator.py` | Smoke `bracket_simulator` (poules + bracket knockout) | — |
| `test_oracle_invariants.py` | Invariants structurels (monotonie, valeur booster, évidence) sur la matrice sauvegardée | ancien script |

> **Histogrammes de peloton** : `run_daily_pipeline` *charge* `p_empirique_1D.npy` et n'appelle jamais
> `extract_peloton_full_distribution`. Cette fonction est donc testée directement (tableaux construits),
> pas via les CSV. Les crowds doivent **sommer à 1** (seuils cumulés : le dernier outcome absorbe le reste).

## Conventions des tests d'oracle exact

- **Gains pairs** (`[20, 50, 90]`) → pas d'arrondi dans la grille coarse (demi-points).
- **Peloton fantôme** (`build_ghost_peloton`, masse sur δ=0) + **gap_2 saturé (+400)**
  → le 2ᵉ adversaire est neutralisé, le problème se réduit à un suivi 1D de Bob,
  directement comparable au solveur récursif.
- **`use_drift=False`** → exécution déterministe.
- Deux terminaux : `terminal_ge_zero` (phases finales) et `terminal_strict`
  (poules, départage les égalités à 0,5).

## Fixtures

- Les notebooks 21 et 22 importent `tests.helpers` (plus de duplication du solveur).
- `data/tests/*.csv` : scénarios réalistes à 72 matchs pour les tests d'intégration.
