"""
Moteur Poisson dynamique du meilleur buteur (distribution conjointe Favori/Buteur,
"in-play").

Idée : le pronostic "meilleur buteur" n'est plus une probabilité STATIQUE figée
avant le tournoi, mais une variable aléatoire dépendant de l'arbre réalisé. Pour
chaque joueur i :

    Y_final,i = current_goals_i + Poisson(alpha_i * M_restants,i)

où `alpha_i` est le taux de but intrinsèque (buts/match, calibré en "Phase A" avant
le tournoi, cf. `calibrate_scorer_alphas`), `current_goals_i` les buts déjà marqués,
et `M_restants,i` le nombre de matchs FUTURS que l'équipe du joueur va jouer dans
l'arbre simulé (matchs de poule restants + matchs knockout jusqu'à élimination ;
0 si l'équipe est déjà éliminée). Le joueur au `Y_final` maximum est meilleur buteur
(égalités : tous les ex-aequo sont meilleurs buteurs).

Corrélation native : si l'arbre élimine la France en 8es, `M_restants` de Mbappé
tombe à 0 et son `Y_final` est gelé à `current_goals` — la punition conjointe
(perte du bonus favori ET du bonus buteur) émerge sans la coder explicitement.

Ce module fournit :
  - `simulate_team_matches_played` : distribution Monte-Carlo du nombre TOTAL de
    matchs joués par équipe sur un tournoi complet (poule + knockout), pour la
    calibration des alpha (Phase A, avant-tournoi).
  - `top_scorer_joint_probs` : loi conjointe (mon buteur top ?, buteur de Bob top ?,
    buteur du peloton top ?) pour UN arbre réalisé (M par joueur fixé), via tirage
    Poisson Monte-Carlo. Consommée par `build_terminal_state`.
  - `calibrate_scorer_alphas` : calibre les alpha pour que la fréquence simulée de
    "meilleur buteur" colle aux cotes buteur dé-viggées (Shin).
  - `load_scorer_players` : lit les colonnes State-Tracker du CSV buteurs/favoris.
"""

import numpy as np
import pandas as pd
from numba import njit

from mpp_project.core import MIN_TRUE_PROBA, MAX_TRUE_PROBA, calculate_true_outcome_probas_from_odds

# Bornes du modèle d'avancement knockout (identiques à simulate_champion_distribution).
_LO = 10.0 * MIN_TRUE_PROBA
_HI = MAX_TRUE_PROBA ** 4
_SIG = 0.03

# Nombre de matchs de poule par équipe (CDM : 3) — utilisé comme base avant-tournoi.
N_GROUP_MATCHES = 3


@njit
def _seed_numba(s):
    """Graine le RNG INTERNE de Numba. Indispensable : np.random.seed() appelé en
    Python ne touche PAS l'état RNG des fonctions @njit (états séparés). Sans cela,
    _calib_top_counts / les simulations njit sont non reproductibles d'un run à l'autre."""
    np.random.seed(s)


# ==========================================================================
# 1. ÉCHANTILLONNAGE PONDÉRÉ SANS REMISE (Gumbel-top-k = Plackett-Luce)
# ==========================================================================
@njit(inline="always")
def _gumbel_key(logw):
    """Clé de Gumbel pour un poids en log : logw + (-log(-log(U))).
    Le tri décroissant de ces clés reproduit un tirage Plackett-Luce (séquentiel
    sans remise proportionnel au poids), équivalent njit de np.random.choice(...,
    replace=False, p=w)."""
    u = np.random.random()
    if u < 1e-12:
        u = 1e-12
    return logw - np.log(-np.log(u))


# ==========================================================================
# 2. SIMULATION DU NOMBRE DE MATCHS JOUÉS PAR ÉQUIPE (cœur njit)
# ==========================================================================
@njit
def _simulate_matches_played_core(
    group_team_ids, group_logw, group_slot1, group_slot2, group_gamma,
    cv_true, qualif_true, next_m, next_slot, n_teams, n_runs, beta,
):
    """Pour n_runs tournois complets (poules Plackett-Luce + knockout à force
    conditionnelle), renvoie counts (n_runs, n_teams) = nombre de matchs KNOCKOUT
    joués par chaque équipe (les matchs de poule, constants à 3, sont ajoutés par
    l'appelant). Mirroir njit de bracket_simulator.simulate_champion_distribution."""
    n_groups = group_team_ids.shape[0]
    out = np.zeros((n_runs, n_teams), dtype=np.int64)

    for run in range(n_runs):
        bracket = np.full((32, 2), -1, dtype=np.int64)
        surv = qualif_true.copy()

        thirds = np.empty(n_groups, dtype=np.int64)
        thirds_gamma = np.empty(n_groups, dtype=np.float64)

        # --- Poules : 1er / 2e / 3e de chaque groupe via Gumbel-top-3 ---
        for grp in range(n_groups):
            keys = np.empty(4, dtype=np.float64)
            for j in range(4):
                keys[j] = _gumbel_key(group_logw[grp, j])
            # top-3 ordonné (sélection par maxima successifs)
            order = np.full(3, -1, dtype=np.int64)
            used = np.zeros(4, dtype=np.bool_)
            for r in range(3):
                best = -1
                best_k = -1e18
                for j in range(4):
                    if not used[j] and keys[j] > best_k:
                        best_k = keys[j]
                        best = j
                order[r] = best
                used[best] = True

            first = group_team_ids[grp, order[0]]
            second = group_team_ids[grp, order[1]]
            s1 = group_slot1[grp, order[0]]
            s2 = group_slot2[grp, order[1]]
            for team, slot in ((first, s1), (second, s2)):
                if bracket[slot, 0] == -1:
                    bracket[slot, 0] = team
                else:
                    bracket[slot, 1] = team
            thirds[grp] = group_team_ids[grp, order[2]]
            thirds_gamma[grp] = group_gamma[grp]

        # --- 8 meilleurs 3es via Gumbel-top-8 sur les 12 (poids = gamma) ---
        tkeys = np.empty(n_groups, dtype=np.float64)
        for grp in range(n_groups):
            tkeys[grp] = _gumbel_key(np.log(thirds_gamma[grp]))
        used_t = np.zeros(n_groups, dtype=np.bool_)
        q3 = np.empty(8, dtype=np.int64)
        for r in range(8):
            best = -1
            best_k = -1e18
            for grp in range(n_groups):
                if not used_t[grp] and tkeys[grp] > best_k:
                    best_k = tkeys[grp]
                    best = grp
            q3[r] = thirds[best]
            used_t[best] = True

        qi = 0
        for m in range(16):
            for s in range(2):
                if bracket[m, s] == -1:
                    bracket[m, s] = q3[qi]
                    qi += 1

        # --- Knockout : force conditionnelle cv_true / surv**beta + bruit ---
        for m in range(32):
            t1 = bracket[m, 0]
            t2 = bracket[m, 1]
            s_a = cv_true[t1] / surv[t1] ** beta
            s_b = cv_true[t2] / surv[t2] ** beta
            p1 = s_a / (s_a + s_b)
            p1 = p1 + np.random.normal(0.0, _SIG)
            if p1 < _LO:
                p1 = _LO
            elif p1 > _HI:
                p1 = _HI
            if np.random.random() < p1:
                winner = t1
                pw = p1
            else:
                winner = t2
                pw = 1.0 - p1
            surv[winner] *= pw

            nm = next_m[m]
            if nm >= 0:
                bracket[nm, next_slot[m]] = winner
            elif m == 28:
                bracket[31, 0] = winner
                bracket[30, 0] = t2 if winner == t1 else t1
            elif m == 29:
                bracket[31, 1] = winner
                bracket[30, 1] = t2 if winner == t1 else t1

        # Comptage des matchs knockout joués (apparitions dans le bracket réalisé)
        for m in range(32):
            out[run, bracket[m, 0]] += 1
            out[run, bracket[m, 1]] += 1

    return out


def compute_group_matches_remaining(df_tournoi, team_names):
    """
    Nombre de matchs de POULE RESTANTS (résultat non renseigné) par équipe, aligné
    sur l'ordre de `team_names` (= ordre des lignes de df_odds des groupes).

    Lit `data/CDM_2026.csv` (colonnes team_A/team_B/phase/result) : compte, pour
    chaque équipe, ses matchs de phase `Poule_*` sans résultat. Sert à la calibration
    IN-PLAY (composante poule de M_restants ; les matchs déjà joués sont dans
    `current_goals`). Pré-tournoi -> 3 par équipe. Noms comparés via normalize_team_name.
    """
    from mpp_project.core import normalize_team_name
    t2i = {str(t).strip().lower(): i for i, t in enumerate(team_names)}
    rem = np.zeros(len(team_names), dtype=np.int64)
    cols = {"team_A", "team_B", "phase"}
    if not cols <= set(df_tournoi.columns):
        return rem
    for _, r in df_tournoi.iterrows():
        if not str(r["phase"]).lower().startswith("poule"):
            continue
        res = str(r.get("result", "")).strip().lower()
        if res not in ("", "nan", "none"):
            continue  # match déjà joué -> buts dans current_goals
        for col in ("team_A", "team_B"):
            nm = normalize_team_name(r[col])
            if nm in t2i:
                rem[t2i[nm]] += 1
    return rem


def simulate_team_matches_played(df_odds, n_runs=20000, beta=0.95, seed=None,
                                 include_group_matches=True, group_matches_per_team=None):
    """
    Distribution Monte-Carlo du nombre de matchs joués par équipe (poules + knockout),
    pour la calibration des alpha.

    Reproduit le modèle d'avancement de `bracket_simulator.simulate_champion_distribution`
    (Plackett-Luce sur cote_1er, équation des 3es sur la qualif VRAIE, force
    conditionnelle dé-viggée + bruit), en njit.

    `group_matches_per_team` (n_teams,) optionnel : composante POULE par équipe
    (alignée sur df_odds). Fournir les matchs de poule RESTANTS (cf.
    compute_group_matches_remaining) pour une calibration IN-PLAY -> matchs FUTURS.
    Si None : `N_GROUP_MATCHES` (=3) constant si include_group_matches, sinon 0
    (régime pré-tournoi). NB : la composante knockout vient de la simulation (le
    bracket est re-simulé depuis les cotes des groupes ; valable tant que le tableau
    est ouvert — poules / début knockout sans éliminations connues).

    Retour : (matches (n_runs, n_teams) int, team_names (n_teams,)).
    """
    from mpp_project.bracket_simulator import devig_group_odds, N_QUALIFIED

    if seed is not None:
        np.random.seed(seed)
        _seed_numba(seed)

    team_names = df_odds["team"].astype(str).str.strip().str.lower().values
    n_teams = len(team_names)
    name_to_id = {n: i for i, n in enumerate(team_names)}

    cv_true, qualif_true = devig_group_odds(df_odds)
    cv_true = cv_true.astype(np.float64)
    qualif_true = qualif_true.astype(np.float64)

    # Structures de groupes (4 équipes / groupe)
    grp_keys = list(dict.fromkeys(df_odds["group"].tolist()))
    n_groups = len(grp_keys)
    group_team_ids = np.full((n_groups, 4), -1, dtype=np.int64)
    group_logw = np.zeros((n_groups, 4), dtype=np.float64)
    group_slot1 = np.zeros((n_groups, 4), dtype=np.int64)
    group_slot2 = np.zeros((n_groups, 4), dtype=np.int64)
    group_gamma = np.zeros(n_groups, dtype=np.float64)

    for gi, gkey in enumerate(grp_keys):
        grp = df_odds[df_odds["group"] == gkey]
        ids = np.array([name_to_id[str(t).strip().lower()] for t in grp["team"]], dtype=np.int64)
        w = calculate_true_outcome_probas_from_odds(grp["cote_1er"].values)
        w = w / w.sum()
        group_team_ids[gi, :len(ids)] = ids
        group_logw[gi, :len(ids)] = np.log(np.clip(w, 1e-12, None))
        group_slot1[gi, :len(ids)] = grp["slot_1er"].values.astype(np.int64)
        group_slot2[gi, :len(ids)] = grp["slot_2e"].values.astype(np.int64)
        group_gamma[gi] = max(0.01, float(qualif_true[ids].sum()) - 2.0)

    # next_match_map -> arrays
    next_match_map = {0: (16, 0), 1: (16, 1), 2: (17, 0), 3: (17, 1), 4: (18, 0), 5: (18, 1),
                      6: (19, 0), 7: (19, 1), 8: (20, 0), 9: (20, 1), 10: (21, 0), 11: (21, 1),
                      12: (22, 0), 13: (22, 1), 14: (23, 0), 15: (23, 1), 16: (24, 0), 17: (24, 1),
                      18: (25, 0), 19: (25, 1), 20: (26, 0), 21: (26, 1), 22: (27, 0), 23: (27, 1),
                      24: (28, 0), 25: (28, 1), 26: (29, 0), 27: (29, 1)}
    next_m = np.full(32, -1, dtype=np.int64)
    next_slot = np.full(32, -1, dtype=np.int64)
    for m, (nm, slot) in next_match_map.items():
        next_m[m] = nm
        next_slot[m] = slot

    ko = _simulate_matches_played_core(
        group_team_ids, group_logw, group_slot1, group_slot2, group_gamma,
        cv_true, qualif_true, next_m, next_slot, n_teams, int(n_runs), float(beta),
    )
    if group_matches_per_team is not None:
        base = np.asarray(group_matches_per_team, dtype=np.int64)
        return ko + base[None, :], team_names
    base = N_GROUP_MATCHES if include_group_matches else 0
    return ko + base, team_names


# ==========================================================================
# 3. LOI CONJOINTE DU MEILLEUR BUTEUR POUR UN ARBRE (Poisson Monte-Carlo)
# ==========================================================================
@njit
def top_scorer_joint_probs(alpha, current_goals, M_per_player,
                           my_member, bob_member, pack_member, n_poisson):
    """
    Pour UN arbre réalisé (M_per_player = matchs futurs de l'équipe de chaque joueur,
    déjà fixé), estime par tirage Poisson la loi CONJOINTE des indicateurs
    (mon buteur est meilleur buteur ?, celui de Bob ?, celui du peloton ?).

    Paramètres (tous alignés sur les n_players joueurs/buteurs simulés) :
      alpha, current_goals, M_per_player : (n_players,) float/float/float.
      my_member / bob_member / pack_member : (n_players,) bool — appartenance du
        joueur au "pick" de l'agent / Bob / peloton. Un pick named = un seul membre ;
        le pick "autre" = tous les ghost scorers ; deux joueurs peuvent partager un
        pick (Bob et le peloton parient le même buteur).

    Égalités : tous les buteurs au maximum de buts sont meilleurs buteurs (un pick
    est "top" si au moins un de ses membres atteint le maximum).

    Retour : joint (8,) float64 sommant à 1, indexée par
      idx = my_top*4 + bob_top*2 + pack_top.
    """
    n = alpha.shape[0]
    joint = np.zeros(8, dtype=np.float64)
    goals = np.empty(n, dtype=np.float64)

    for _ in range(n_poisson):
        max_g = -1.0
        for i in range(n):
            lam = alpha[i] * M_per_player[i]
            g = current_goals[i]
            if lam > 0.0:
                g += np.random.poisson(lam)
            goals[i] = g
            if g > max_g:
                max_g = g

        my_top = 0
        bob_top = 0
        pack_top = 0
        for i in range(n):
            if goals[i] == max_g:
                if my_member[i]:
                    my_top = 1
                if bob_member[i]:
                    bob_top = 1
                if pack_member[i]:
                    pack_top = 1
        joint[my_top * 4 + bob_top * 2 + pack_top] += 1.0

    return joint / n_poisson


# ==========================================================================
# 4. CALIBRATION DES ALPHA (Phase A, avant-tournoi)
# ==========================================================================
@njit
def _calib_top_counts(alpha, current_goals, nation_id, team_matches_runs, n_poisson_per_run):
    """Probabilité simulée que le joueur i remporte le titre de meilleur buteur, en
    convention DEAD-HEAT (part 1/k partagée entre les k ex-aequo au sommet). Cette
    convention somme à 1 par tirage — cohérente avec la cible Shin (P(top), somme 1)
    et MONOTONE en alpha (à l'inverse du comptage 'tous les ex-aequo = 1', pathologique
    quand les buts sont rares : tout le monde à 0 -> tout le monde ex-aequo).

    NB : le MOTEUR de jeu (top_scorer_joint_probs / build_terminal_state) applique lui
    la règle MPP 'tous les ex-aequo touchent le bonus PLEIN' ; seule la calibration des
    taux intrinsèques alpha utilise la part dead-heat (= façon dont le marché règle)."""
    n_runs = team_matches_runs.shape[0]
    n_players = alpha.shape[0]
    counts = np.zeros(n_players, dtype=np.float64)
    goals = np.empty(n_players, dtype=np.float64)

    for r in range(n_runs):
        for _ in range(n_poisson_per_run):
            max_g = -1.0
            for i in range(n_players):
                M = team_matches_runs[r, nation_id[i]]
                lam = alpha[i] * M
                g = current_goals[i]
                if lam > 0.0:
                    g += np.random.poisson(lam)
                goals[i] = g
                if g > max_g:
                    max_g = g
            k = 0
            for i in range(n_players):
                if goals[i] == max_g:
                    k += 1
            share = 1.0 / k
            for i in range(n_players):
                if goals[i] == max_g:
                    counts[i] += share

    return counts / (n_runs * n_poisson_per_run)


def calibrate_scorer_alphas(target_probs, nation_id, team_matches_runs, field_nation_id,
                            current_goals=None, g_fav=6.0, p_grid=None, field_grid=None,
                            n_poisson_per_run=2, refine=True, verbose=True, seed=None):
    """
    Calibre les taux de but `alpha` (buts/match) pour que la fréquence simulée de
    "meilleur buteur" (convention dead-heat, cf. _calib_top_counts) colle à
    `target_probs` (cotes buteur dé-viggées par Shin).

    Schéma BIEN POSÉ à 2 degrés de liberté + ancrage réaliste (l'inverse libre
    par joueur est instable : sur ~6 matchs un joueur "contend" ou est hors course,
    sans milieu lisse -> dynamique à falaise) :
      - LIEN PARAMÉTRIQUE : E[buts_i] = c · target_i**p (alpha_i = E[buts_i] / E[M_i]).
        `p` règle la concentration (shape), `c` l'échelle. On ÉPINGLE l'échelle au
        réalisme : c tel que le favori du marché marque `g_fav` buts en espérance
        (Soulier d'or WC ~ 5-7). Sinon l'optimiseur s'échappe vers des buts ~1
        (régime dégénéré qui matche les parts mais tue la dynamique de catch-up).
      - CHAMP DE FOND : `field_nation_id` liste les nations des buteurs GÉNÉRIQUES de
        fond (typiquement un par équipe). Leur taux commun `alpha_field` est calibré.
        Le field peut remporter le titre (-> "aucun de mes 3 picks") ; sa part CROÎT
        en in-play à mesure que les équipes des stars sont éliminées (M -> 0).

    On ajuste (p, alpha_field) par grille (puis raffinement local) en minimisant le
    MAE entre parts simulées des joueurs LISTÉS et `target_probs`.

    Paramètres :
      target_probs      : (n_listed,) P(meilleur buteur) cible (dé-viggée Shin) ;
                          en IN-PLAY = devig des cotes buteur COURANTES.
      nation_id         : (n_listed,) index d'équipe de chaque joueur listé.
      team_matches_runs : (n_runs, n_teams) matchs FUTURS joués par équipe et par
                          tournoi simulé. Pré-tournoi = tournoi complet (3 poule + KO) ;
                          IN-PLAY = matchs RESTANTS (cf. simulate_team_matches_played
                          avec group_matches_per_team=compute_group_matches_remaining).
      field_nation_id   : (n_field,) nations des buteurs de fond (ex. np.arange(n_teams)).
      current_goals     : (n_listed,) buts DÉJÀ marqués (réels en in-play ; 0 avant
                          tournoi ; le field est supposé à 0). Combinés aux matchs
                          futurs : Y_final = current_goals + Poisson(alpha × M_restants).
      g_fav             : buts TOTAUX (en fin de tournoi) attendus du favori du marché
                          (ancre de réalisme). Sa part FUTURE calibrée = g_fav moins ses
                          buts déjà marqués (évite le double comptage en in-play).

    Retour : (alpha (n_listed,), info dict {alpha_field, p, c, g_fav, mae, rankcorr,
              sim_probs, field_share}).
    """
    if seed is not None:
        np.random.seed(seed)
        _seed_numba(seed)

    target = np.asarray(target_probs, dtype=np.float64)
    nation_id = np.asarray(nation_id, dtype=np.int64)
    field_nation_id = np.asarray(field_nation_id, dtype=np.int64)
    team_matches_runs = np.ascontiguousarray(team_matches_runs, dtype=np.float64)
    n_listed = target.shape[0]
    n_field = field_nation_id.shape[0]
    if current_goals is None:
        current_goals = np.zeros(n_listed, dtype=np.float64)
    else:
        current_goals = np.asarray(current_goals, dtype=np.float64)

    EM = np.clip(team_matches_runs.mean(axis=0)[nation_id], 1e-6, None)
    s = np.clip(target, 1e-12, None)
    tmax = s.max()
    nat_ext = np.concatenate([nation_id, field_nation_id])
    cg_ext = np.concatenate([current_goals, np.zeros(n_field)])

    # Ancre de réalisme par TOTAL : le favori du marché finit à `g_fav` buts au total.
    # Sa composante FUTURE (à modéliser par Poisson) = g_fav - ses buts DÉJÀ marqués.
    # Pré-tournoi (current_goals=0) total = future. En in-play, retrancher les buts
    # déjà acquis évite le DOUBLE COMPTAGE (sinon alpha capte la cote courte d'un
    # joueur en tête alors qu'elle vient surtout de ses buts déjà marqués).
    g_future_fav = max(0.5, g_fav - float(current_goals[int(np.argmax(s))]))

    if p_grid is None:
        # Plancher 0.15 : l'optimum typique (p≈0.27) doit être INTÉRIEUR à la grille
        # (sinon il est rattrapé de justesse par le refine et le balayage affiche un
        # p* figé au plancher). p<0.20 dégrade nettement le MAE -> jamais retenu.
        p_grid = np.arange(0.15, 1.01, 0.05)
    if field_grid is None:
        field_grid = np.arange(0.00, 0.81, 0.05)

    def _eval(p, af):
        c = g_future_fav / (tmax ** p)
        alpha = (c * s ** p) / EM
        al_ext = np.concatenate([alpha, np.full(n_field, af)])
        sim = _calib_top_counts(al_ext, cg_ext, nat_ext, team_matches_runs, int(n_poisson_per_run))
        listed = sim[:n_listed]
        mae = float(np.mean(np.abs(listed - target)))
        return mae, c, alpha, listed, float(sim[n_listed:].sum())

    best = None  # (mae, p, af, c, alpha, listed, field_share)
    for p in p_grid:
        for af in field_grid:
            mae, c, alpha, listed, fld = _eval(p, af)
            if best is None or mae < best[0]:
                best = (mae, float(p), float(af), c, alpha, listed, fld)
        if verbose:
            print(f"  [calib] p={p:.2f} -> meilleur MAE courant {best[0]:.4f} "
                  f"(p*={best[1]:.2f}, alpha_field*={best[2]:.2f})")

    # Raffinement local autour de l'optimum de grille
    if refine:
        p0, af0 = best[1], best[2]
        for p in np.linspace(max(0.05, p0 - 0.05), p0 + 0.05, 5):
            for af in np.linspace(max(0.0, af0 - 0.05), af0 + 0.05, 5):
                mae, c, alpha, listed, fld = _eval(p, af)
                if mae < best[0]:
                    best = (mae, float(p), float(af), c, alpha, listed, fld)

    mae, p_best, af_best, c_best, alpha_best, listed_best, field_share = best
    rank = lambda x: np.argsort(np.argsort(x))
    rankcorr = float(np.corrcoef(rank(target), rank(listed_best))[0, 1])
    info = {
        "alpha_field": af_best, "p": p_best, "c": c_best, "g_fav": g_fav,
        "mae": mae, "rankcorr": rankcorr, "sim_probs": listed_best,
        "field_share": field_share,
    }
    if verbose:
        print(f"  [calib] FINAL p={p_best:.3f} alpha_field={af_best:.3f} "
              f"MAE={mae:.4f} rankcorr={rankcorr:.3f} field_share={field_share:.3f}")
    return alpha_best, info


# ==========================================================================
# 5. LECTURE DU STATE-TRACKER (CSV buteurs/favoris enrichi)
# ==========================================================================
def load_scorer_players(df_market):
    """
    Extrait la table des joueurs-buteurs depuis le CSV enrichi (State-Tracker).

    Colonnes attendues sur les lignes `category == 'scorer'` :
      selection, nation, cote, gain_mpp, crowd, current_goals, is_eliminated, alpha.
    Un buteur named a un `gain_mpp` ; les ghost scorers (gain_mpp vide) composent la
    catégorie 'autre'. La ligne `selection == 'autre'` (gain_mpp renseigné, sans cote
    ni nation) porte le gain du pick générique.

    Retour : (players, autre_gain, field_alpha).
      players     : DataFrame des joueurs SIMULÉS (named + ghosts, tous avec une
                    nation) + colonne booléenne `is_ghost` (membre de 'autre').
      autre_gain  : gain du pick 'autre' (ligne selection == 'autre'), ou None.
      field_alpha : taux de but commun des buteurs de FOND (ligne category == 'field',
                    colonne alpha ; rempli par calibrate_scorer_alphas), ou None si
                    absent/non calibré.
    La ligne 'autre' et la ligne 'field' sont exclues de la table simulée.
    """
    field_alpha = None
    if "category" in df_market.columns and (df_market["category"] == "field").any():
        fa = df_market.loc[df_market["category"] == "field", "alpha"]
        fa = pd.to_numeric(fa, errors="coerce").dropna()
        if not fa.empty:
            field_alpha = float(fa.values[0])

    df = df_market[df_market["category"] == "scorer"].copy()
    df["selection"] = df["selection"].astype(str).str.strip().str.lower()

    autre_mask = df["selection"] == "autre"
    autre_gain = None
    if autre_mask.any():
        ag = df.loc[autre_mask, "gain_mpp"].values[0]
        autre_gain = None if pd.isna(ag) else int(ag)

    players = df[~autre_mask].copy()
    players["is_ghost"] = players["gain_mpp"].isna()
    players["nation"] = players["nation"].astype(str).str.strip().str.lower()
    players["current_goals"] = players.get("current_goals", 0.0)
    players["current_goals"] = players["current_goals"].fillna(0.0).astype(float)
    if "is_eliminated" in players.columns:
        players["is_eliminated"] = (
            players["is_eliminated"].astype(str).str.strip().str.lower()
            .isin(["1", "true", "vrai", "oui", "yes"])
        )
    else:
        players["is_eliminated"] = False
    if "alpha" in players.columns:
        players["alpha"] = pd.to_numeric(players["alpha"], errors="coerce")
    else:
        players["alpha"] = np.nan

    return players.reset_index(drop=True), autre_gain, field_alpha
