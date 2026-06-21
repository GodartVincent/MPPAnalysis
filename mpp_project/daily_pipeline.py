import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import date
from pathlib import Path

from mpp_project.core import apply_heteroscedastic_noise, apply_temporal_drift, build_exact_score_market, build_horizon_bonus_arrays, calibrate_horizon_bonus_model, calculate_true_outcome_probas_from_odds, estimate_crowd_3D, expected_mpp_points, expected_simple_points, normalize_crowds, validate_match_dataframe
from mpp_project.oracle_dp import compute_alphas_isolement, compute_full_Q_table, evaluate_exact_score_day, forward_propagate_exact, solve_dp_coarse, solve_dp_coarse_exact, solve_dp_coarse_hbonus, GAP_MIN, GAP_MAX, GAP_OFFSET

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

def run_daily_pipeline(
    csv_path: Path,
    match_id_cible: int,
    mon_gap_1: int = 0,
    mon_gap_2: int = 0,
    has_booster: int = 1,
    use_drift: bool = True,
    horizon_nuit: int = 5,
    seuil_isolement: float = 80.0,
    nb_scenarios: int = 3,
    near_horizon: int = 10,
    p_empirique_override=None,
    v_horizon_override=None,
    save_abaques: bool = True,
    validate_input: bool = True,
    exact_score_data=None,
    exact_scores_by_match=None,
    reference_date=None
):
    """
    Exécute la chaîne de décision (Drift + DP) et sauvegarde les abaques des prochains matchs.

    Paramètres d'injection (tests unitaires — cf. notebook 22) :
      p_empirique_override : ndarray (n_matches, 3, max_gain). Si fourni, remplace
                             le chargement de p_empirique_1D.npy (ex. peloton fantôme).
      v_horizon_override   : ndarray (1001, 1001, 2). Si fourni, remplace le chargement
                             de expected_V_phases_finales.npy comme condition limite (horizon
                             des phases finales), utilisé dans les deux chemins (avec/sans drift).
                             Injecter ici une condition terminale signe-de-gap reproduit une
                             partie auto-contenue (poules = fin), cf. tests oracle.
      save_abaques         : si False, n'écrit aucun fichier Abaque_Match_*.npz sur disque
                             (utile pour ne pas polluer data/ pendant les tests).
      validate_input       : si True (défaut), contrôle la cohérence du CSV (phases,
                             gains, équipes, crowds) et émet des warnings. Mettre à
                             False pour les fixtures de test synthétiques.
      reference_date       : date de référence du DRIFT PAR DATE (cf. apply_temporal_drift).
                             None (défaut) -> date du jour (date.today()). Le drift des
                             true_probas lointaines croît alors avec le nombre de jours
                             entre cette date et la date de chaque match (colonne 'date'
                             du CSV) selon un modèle de diffusion. Si le CSV n'a pas de
                             colonne 'date' (fixtures de test), retombe sur le drift par
                             phase historique. À fixer pour préparer l'analyse la veille
                             ou pour des tests déterministes.
      exact_score_data     : dict { "b1-b2": (cote, crowd_pct) } optionnel. Si fourni,
                             la décision du MATCH DU JOUR porte sur les SCORES EXACTS
                             (et non 1N2) : l'agent, Bob et le peloton peuvent décrocher
                             le bonus de score exact (cf. build_exact_score_market /
                             evaluate_exact_score_day). Change le contrat de retour :
                               - mode 1N2 (None)  : (reco, best_wr, ev_actions, Q_table_jour)
                                 ev_actions = ndarray (3,) des EV par issue.
                               - mode exact       : (reco, best_wr, market_df, Q_table_jour)
                                 reco = score exact (+ " x2"), market_df = pandas DataFrame
                                 (score/outcome/proba/cond_crowd/bonus/WR_base/keep/use).
                             Q_table_jour reste la Q 1N2 du jour (projection / fallback).
      exact_scores_by_match : dict { match_id (int) : { "b1-b2": (cote, crowd) } } optionnel.
                             Mode MULTI-MATCHS « nuit » : la DP proche devient exact-aware sur
                             tous les matchs de la nuit présents (Bob/peloton décrochent leur
                             bonus, l'agent optimise son score) -> la décision du match courant
                             en hérite. En plus, une reco score-exact est calculée pour CHAQUE
                             match de la nuit dans l'horizon. Le match courant DOIT être une clé.
                             Retour à 5 éléments :
                               (reco, best_wr, market_df, Q_table_jour, night_markets)
                             night_markets = OrderedDict { match_id : (reco_m, wr_m, market_df_m) }
                             (inclut le match courant). Prioritaire sur exact_score_data.
    """
    match_idx = match_id_cible - 1

    # ==========================================
    # 1. CHARGEMENT DES DONNÉES
    # ==========================================
    df = pd.read_csv(csv_path)
    if validate_input:
        validate_match_dataframe(df)
    match_phases = df['phase'].tolist()
    # Dates des matchs (pour le drift par date). Absentes des fixtures de test
    # synthétiques -> None -> apply_temporal_drift retombe sur le drift par phase.
    match_dates = pd.to_datetime(df['date']).to_numpy() if 'date' in df.columns else None
    n_matches = len(df)

    odds = df[['cote_1', 'cote_N', 'cote_2']].values.astype(np.float64)
    base_true_probas = calculate_true_outcome_probas_from_odds(odds)

    # Normalisation systématique (+ alerte si somme de crowd aberrante, en mode validé)
    base_crowds = normalize_crowds(df[['crowd_1', 'crowd_N', 'crowd_2']].values,
                                   label="poule", warn=validate_input)
    
    gains_1N2 = df[['gain_mpp_1', 'gain_mpp_N', 'gain_mpp_2']].values.astype(np.int32)
    ev_actions = base_true_probas[match_idx] * gains_1N2[match_idx]
    
    if p_empirique_override is not None:
        p_empirique_1D = p_empirique_override[:n_matches]
    else:
        p_empirique_1D_full = np.load(DATA_DIR / "p_empirique_1D.npy")
        p_empirique_1D = p_empirique_1D_full[:n_matches]
    max_gain_dynamique = p_empirique_1D.shape[2]

    # On charge l'horizon des phases finales généré par bracket_simulator.py
    # (expected_V_phases_finales.npy = matrice _full marginalisée sur les favoris vivants).
    # On isole la matrice du premier match des phases finales (index 0)
    # V_next_base : (1001, 1001, 2) → downsamplé en (501, 501, 2) pour solve_dp_coarse
    if v_horizon_override is not None:
        V_next_base = v_horizon_override
    else:
        V_phases_finales_brut = np.load(DATA_DIR / "expected_V_phases_finales.npy")
        V_next_base = V_phases_finales_brut[0]
    V_next_base_coarse = V_next_base[::2, ::2, :].copy()  # indices 0,2,4,...,1000 → 501 pts
    # NB : la valeur APRÈS le dernier match de poule = l'entrée des phases finales,
    # donc l'HORIZON lui-même (V_next_base_coarse), et NON une condition terminale
    # signe-de-gap. Sert de V_next à la Q-table du dernier match si l'horizon des
    # abaques l'atteint (sinon V_near[match_idx + k + 1] déborderait en fin de tournoi).

    # ==========================================
    # 2. DRIFT & DP  (3 phases, grille grossière 501×501)
    # ==========================================
    # Grille coarse (501×501, 2 pts/step) : même range [-600,+400], 4× plus rapide.
    # -----------------------------------------------------------------------
    # Poids du blend bayésien des crowds (confiance CSV vs crowd théorique) : 0.95 au
    # présent, décroissant avec l'éloignement TEMPOREL du match (demi-vie 4 jours via
    # la date du CSV vs reference_date). Rétro-compat : sans colonne 'date' (fixtures),
    # on retombe sur la distance en NOMBRE DE MATCHS (demi-vie 5 indices).
    if match_dates is not None:
        ref_d = pd.Timestamp(reference_date if reference_date is not None else date.today()).normalize()
        days_ahead = np.array([
            0 if pd.isna(d) else max(0, (pd.Timestamp(d).normalize() - ref_d).days)
            for d in match_dates
        ], dtype=np.float64)
        alphas_bayes = 0.95 * (0.5 ** (days_ahead / 4.0))
    else:
        dists = np.arange(n_matches) - match_idx
        dists_positive = np.maximum(0, dists)
        alphas_bayes = 0.95 * (0.5 ** (dists_positive / 5.0))
    alphas_2d = alphas_bayes.astype(np.float32)[:, np.newaxis]

    c1, cN, c2 = estimate_crowd_3D(base_true_probas[:, 0], base_true_probas[:, 1], base_true_probas[:, 2], add_noise=False)
    theo_crowds_pure = np.column_stack((c1, cN, c2)).astype(np.float32)
    blended_mean_crowds = (alphas_2d * base_crowds) + ((1.0 - alphas_2d) * theo_crowds_pure)
    dynamic_rmse = 0.083 * (1.0 - alphas_2d)

    horizon_effectif = min(horizon_nuit, n_matches - match_idx)

    # Alphas sans drift (pour le DP proche et les Q-tables)
    base_alphas = compute_alphas_isolement(base_true_probas, base_crowds, gains_1N2, seuil_isolement=seuil_isolement)

    n_runs = nb_scenarios if use_drift else 1
    split_t = match_idx + near_horizon
    use_split = use_drift and (split_t < n_matches)

    # --- Setup MODE MULTI-MATCHS (scores exacts sur la nuit) ---
    # Construit un ExactScoreMarket par match de la nuit (ancré sur le 1N2 du CSV) et
    # prépare des tableaux de scores paddés pour la DP proche exact-aware. Un match 1N2
    # = 3 pseudo-scores (bonus 0) -> la transition retombe sur le comportement 1N2.
    exact_multi = exact_scores_by_match is not None
    markets_by_idx = {}
    Kmax = 3
    if exact_multi:
        for mid, data in exact_scores_by_match.items():
            idx = int(mid) - 1
            if idx < match_idx or idx >= n_matches:
                continue  # match déjà joué ou hors tournoi
            mkt = build_exact_score_market(data, outcome_probas=base_true_probas[idx],
                                           shape_correction=True, mpp_outcome_crowd=base_crowds[idx])
            markets_by_idx[idx] = mkt
            Kmax = max(Kmax, len(mkt.scores))
        if match_idx not in markets_by_idx:
            raise ValueError(
                f"exact_scores_by_match : le match courant (id={match_id_cible}) doit "
                f"figurer dans le CSV des scores exacts."
            )

    # MODÈLE DE BONUS HORIZON (dé-biais du booster x2) : calibré sur TOUS les marchés
    # renseignés, il donne un bonus de score exact SYNTHÉTIQUE aux matchs de la DP
    # LOINTAINE (1N2 pur sinon) -> E[points] futurs réalistes -> la valeur de garder le
    # x2 n'est plus sous-estimée. Inactif si aucune donnée (horizon laissé en 1N2).
    hbonus_model = None
    hb_q = hb_B = hb_opp = None
    if exact_multi:
        hbonus_model = calibrate_horizon_bonus_model(
            exact_scores_by_match, base_true_probas, base_crowds=base_crowds
        )
        hb_q, hb_B, hb_opp = build_horizon_bonus_arrays(base_true_probas, hbonus_model)

    gains_f = gains_1N2.astype(np.float64) if exact_multi else None
    pemp_f = p_empirique_1D.astype(np.float64) if exact_multi else None

    def _build_sc(true_probas):
        """Tableaux paddés (n_matches, Kmax) : exact si match renseigné, sinon pseudo-1N2."""
        sc_outcome = np.zeros((n_matches, Kmax), dtype=np.int8)
        sc_p = np.zeros((n_matches, Kmax), dtype=np.float64)
        sc_cc = np.zeros((n_matches, Kmax), dtype=np.float64)
        sc_bonus = np.zeros((n_matches, Kmax), dtype=np.float64)
        sc_count = np.zeros(n_matches, dtype=np.int64)
        for t in range(n_matches):
            if t in markets_by_idx:
                m = markets_by_idx[t]
                K = len(m.scores)
                sc_outcome[t, :K] = np.asarray(m.outcomes, dtype=np.int8)
                sc_p[t, :K] = m.p_score
                sc_cc[t, :K] = m.cond_crowd
                sc_bonus[t, :K] = m.bonus
                sc_count[t] = K
            else:
                sc_outcome[t, 1] = 1
                sc_outcome[t, 2] = 2
                sc_p[t, :3] = true_probas[t]
                sc_count[t] = 3
        return sc_outcome, sc_p, sc_cc, sc_bonus, sc_count

    # --- Phase 1 : DP LOINTAIN coarse (avec drift × nb_scenarios) ---
    # Drift fort sur les matchs lointains (alpha_bayes ≈ 0 → rmse ≈ 8%).
    # nb_scenarios réalisations → E[V(split_t)] en grille coarse (501, 501).
    if use_split:
        V_far_sum = np.zeros((501, 501, 2), dtype=np.float32)
        for s in range(n_runs):
            v_true_probas = apply_temporal_drift(base_true_probas, match_phases, current_match_idx=match_idx,
                                                 match_dates=match_dates, reference_date=reference_date)
            v_crowds = apply_heteroscedastic_noise(blended_mean_crowds, rmse=dynamic_rmse)
            v_alphas = compute_alphas_isolement(v_true_probas, v_crowds, gains_1N2, seuil_isolement=seuil_isolement)
            if exact_multi and hbonus_model.active:
                # Horizon lointain AVEC bonus de score exact synthétique (dé-biais x2).
                # start_t=n_matches : chaîne vers l'horizon knockout sans l'écraser.
                V_far_s = solve_dp_coarse_hbonus(
                    v_true_probas, v_crowds, gains_1N2, p_empirique_1D, v_alphas,
                    V_next_base_coarse, hb_q, hb_B, hb_opp,
                    stop_t=split_t, start_t=n_matches,
                )
            else:
                V_far_s, _ = solve_dp_coarse(
                    base_true_probas=v_true_probas,
                    base_crowds=v_crowds,
                    gains_1N2=gains_1N2,
                    p_empirique_1D=p_empirique_1D,
                    alphas_isolement=v_alphas,
                    V_horizon=V_next_base_coarse,
                    stop_t=split_t,
                    horizon_nuit=0,
                    # start_t=n_matches : démarre à n_matches-1 SANS écraser V_horizon par
                    # le terminal interne -> chaîne réellement vers l'horizon des phases finales.
                    start_t=n_matches
                )
            V_far_sum += V_far_s[split_t]
        V_far_avg = V_far_sum / n_runs  # shape (501, 501, 2)
    else:
        # Edge case : near_horizon couvre toute la tournée, ou drift désactivé
        V_far_avg = V_next_base_coarse.copy()

    # --- Phase 2 : DP PROCHE coarse (sans drift, 1 seul run) ---
    # Drift quasi nul sur les matchs proches (alpha_bayes ≈ 0.95 → rmse ≈ 0.4%).
    # Condition limite = V_far_avg (incertitude future déjà encodée).
    # Sous-produits pour la PASSE FORWARD (matchs de nuit, k>=1). Restent None hors du
    # chemin use_split exact-aware -> les autres chemins gardent l'éval au gap courant.
    Q_near = None
    Q_near_floor = None
    sc_fwd = None
    if use_split:
        if exact_multi:
            sc_o, sc_p, sc_cc, sc_b, sc_n = _build_sc(base_true_probas)
            V_near, Q_near, Q_near_floor = solve_dp_coarse_exact(
                sc_o, sc_p, sc_cc, sc_b, sc_n,
                gains_f, base_crowds.astype(np.float64), pemp_f, base_alphas.astype(np.float64),
                V_far_avg, stop_t=match_idx, start_t=split_t,
                store_q_count=horizon_effectif,   # Q par-action stockées pour la forward pass
                store_floor=True,                 # + plancher 'WR outcome' (sans bonus agent)
            )
            # Plancher rangé en float16 (affichage : précision suffisante, RAM /2 ; numba ne
            # sait pas écrire de f16 en njit -> cast côté Python). eval_exact_market_forward
            # ré-upcaste la tranche du match en f32 pour le tensordot (BLAS).
            Q_near_floor = Q_near_floor.astype(np.float16)
            sc_fwd = (sc_o, sc_p, sc_cc, sc_b, sc_n)
        else:
            V_near, _ = solve_dp_coarse(
                base_true_probas=base_true_probas,
                base_crowds=base_crowds,
                gains_1N2=gains_1N2,
                p_empirique_1D=p_empirique_1D,
                alphas_isolement=base_alphas,
                V_horizon=V_far_avg,
                stop_t=match_idx,
                horizon_nuit=0,
                start_t=split_t   # Run partiel : split_t-1 → match_idx
            )
        # V_near shape : (n_matches, 501, 501, 2) — on extrait les tranches d'horizon.
        # Pour le dernier match du tournoi, V[n_matches] = condition terminale.
        V_nexts_avg_coarse = np.stack([
            V_near[match_idx + k + 1] if (match_idx + k + 1) < n_matches else V_next_base_coarse
            for k in range(horizon_effectif)
        ])
    else:
        # Edge case : boucle complète (fin de tournoi ou use_drift=False)
        V_nexts_sum = np.zeros((horizon_effectif, 501, 501, 2), dtype=np.float32)
        for s in range(n_runs):
            if use_drift:
                v_true_probas = apply_temporal_drift(base_true_probas, match_phases, current_match_idx=match_idx,
                                                     match_dates=match_dates, reference_date=reference_date)
                v_crowds = apply_heteroscedastic_noise(blended_mean_crowds, rmse=dynamic_rmse)
            else:
                v_true_probas = base_true_probas.copy()
                v_crowds = base_crowds.copy()
            v_alphas = compute_alphas_isolement(v_true_probas, v_crowds, gains_1N2, seuil_isolement=seuil_isolement)
            if exact_multi:
                sc_o, sc_p, sc_cc, sc_b, sc_n = _build_sc(v_true_probas)
                V_near_s, _, _ = solve_dp_coarse_exact(
                    sc_o, sc_p, sc_cc, sc_b, sc_n,
                    gains_f, v_crowds.astype(np.float64), pemp_f, v_alphas.astype(np.float64),
                    V_next_base_coarse, stop_t=match_idx, start_t=n_matches
                )
            else:
                V_near_s, _ = solve_dp_coarse(
                    base_true_probas=v_true_probas,
                    base_crowds=v_crowds,
                    gains_1N2=gains_1N2,
                    p_empirique_1D=p_empirique_1D,
                    alphas_isolement=v_alphas,
                    V_horizon=V_next_base_coarse,
                    stop_t=match_idx,
                    horizon_nuit=0,
                    start_t=n_matches   # chaîne vers l'horizon knockout (cf. run lointain)
                )
            for k in range(horizon_effectif):
                nxt = match_idx + k + 1
                V_nexts_sum[k] += V_near_s[nxt] if nxt < n_matches else V_next_base_coarse
        V_nexts_avg_coarse = V_nexts_sum / n_runs

    # Upsample coarse (501,501) → fine (1001,1001) pour compute_full_Q_table
    # Nearest-neighbor : index fin g_f → index coarse g_f // 2
    fine_idx = np.arange(1001) // 2   # [0,0,1,1,...,499,499,500]
    V_nexts_avg_fine = V_nexts_avg_coarse[:, fine_idx[:, None], fine_idx[None, :], :]

    # --- Phase 3 : Q-tables pleine résolution (V_next upsamplé) ---
    # L'incertitude future est encodée dans V_nexts_avg_fine via le multiverse lointain.
    # Q-tables à (1001, 1001, 3, 3) — compatibles avec notebook 17.
    Q_tables = np.zeros((horizon_effectif, 1001, 1001, 3, 3), dtype=np.float32)
    for k in range(horizon_effectif):
        t = match_idx + k
        Q_tables[k] = compute_full_Q_table(
            t, base_true_probas[t], base_crowds[t], gains_1N2[t],
            p_empirique_1D, base_alphas[t], V_nexts_avg_fine[k], max_gain_dynamique
        )

    # ==========================================
    # 3. EXPORT DES ABAQUES
    # ==========================================
    Q_table_jour = Q_tables[0]
    if save_abaques:
        for k in range(horizon_effectif):
            t = match_idx + k
            np.savez_compressed(
                DATA_DIR / f"Abaque_Match_{t + 1}.npz",
                q_table=Q_tables[k],
                ev_actions=base_true_probas[t] * gains_1N2[t]
            )
            
    # ==========================================
    # 4. RECOMMANDATION FINALE
    # ==========================================
    g1_idx = max(GAP_MIN, min(GAP_MAX, int(round(mon_gap_1)))) + GAP_OFFSET
    g2_idx = max(GAP_MIN, min(GAP_MAX, int(round(mon_gap_2)))) + GAP_OFFSET
    noms_choix = ["1 (Dom)", "N (Nul)", "2 (Ext)"]

    # --- Mode MULTI-MATCHS (scores exacts sur la nuit) ---
    if exact_multi:
        night_markets = OrderedDict()
        # PASSE FORWARD : distribution d'occupation des états (coarse 501², couches
        # booster), propagée à travers la nuit en suivant la politique optimale. Permet,
        # pour chaque match k>=1, un WR MOYEN par score pondéré par la proba d'atteindre
        # chaque état. Le match courant (k=0) garde l'éval fine EXACTE au gap réel.
        do_forward = Q_near is not None and sc_fwd is not None
        if do_forward:
            sc_o, sc_p, sc_cc, sc_b, sc_n = sc_fwd
            g1c, g2c = g1_idx // 2, g2_idx // 2   # fine -> coarse (offset 600 -> 300)
            P = np.zeros((501, 501, 2), dtype=np.float64)
            P[min(g1c, g2c), max(g1c, g2c), has_booster] = 1.0  # convention g1<=g2

        for k in range(horizon_effectif):
            idx = match_idx + k
            if idx in markets_by_idx:
                if k == 0 or not do_forward:
                    reco_m, wr_m, df_m = eval_exact_market(
                        markets_by_idx[idx], g1_idx, g2_idx, has_booster,
                        gains_1N2[idx], base_crowds[idx], p_empirique_1D[idx],
                        base_alphas[idx], V_nexts_avg_fine[k], noms_choix
                    )
                else:
                    reco_m, wr_m, df_m = eval_exact_market_forward(
                        markets_by_idx[idx], P, Q_near[k], has_booster,
                        gains_1N2[idx], noms_choix, Q_floor=Q_near_floor[k]
                    )
                night_markets[idx + 1] = (reco_m, wr_m, df_m)

            # Avance la distribution à travers le match idx (politique optimale) -> k+1
            if do_forward and (k + 1) < horizon_effectif:
                P = forward_propagate_exact(
                    P, sc_o[idx], sc_p[idx], sc_cc[idx], sc_b[idx], int(sc_n[idx]),
                    gains_f[idx], base_crowds[idx].astype(np.float64),
                    pemp_f[idx], float(base_alphas[idx]), Q_near[k]
                )
                # Garde-fou : si un histogramme peloton est dégénéré (somme < 1 — typique
                # d'un p_empirique_1D.npy périmé généré avec des issues figées), la masse
                # fuit. On renormalise pour que P reste une vraie distribution (no-op sur
                # données saines). NB : pour des projections fiables, régénérer le .npy.
                p_sum = P.sum()
                if p_sum > 0.0:
                    P /= p_sum

        # Marchés renseignés au-delà de l'horizon de nuit -> warn (comportement historique)
        for idx in sorted(markets_by_idx):
            if (idx - match_idx) >= horizon_effectif:
                import warnings
                warnings.warn(
                    f"Match id={idx + 1} hors horizon (k={idx - match_idx} >= horizon_nuit "
                    f"effectif={horizon_effectif}) : reco score-exact ignorée. "
                    f"Augmenter horizon_nuit pour le couvrir.",
                    stacklevel=2,
                )

        reco, best_wr, market_df = night_markets[match_id_cible]
        return reco, best_wr, market_df, Q_table_jour, night_markets

    # --- Mode SCORES EXACTS (match du jour seul) ---
    if exact_score_data is not None:
        return _recommend_exact_score(
            exact_score_data, g1_idx, g2_idx, has_booster,
            gains_1N2[match_idx], base_crowds[match_idx], p_empirique_1D[match_idx],
            base_alphas[match_idx], V_nexts_avg_fine[0], noms_choix, Q_table_jour,
            outcome_probas=base_true_probas[match_idx]
        )

    if has_booster:
        wr_keep = Q_table_jour[g1_idx, g2_idx, :, 1]
        wr_use = Q_table_jour[g1_idx, g2_idx, :, 2]
        best_keep_idx, best_use_idx = np.argmax(wr_keep), np.argmax(wr_use)
        
        if wr_use[best_use_idx] > wr_keep[best_keep_idx]:
            reco = f"{noms_choix[best_use_idx]} + x2"
            best_wr = wr_use[best_use_idx]
        else:
            reco = f"{noms_choix[best_keep_idx]} (Safe)"
            best_wr = wr_keep[best_keep_idx]
    else:
        wr_base = Q_table_jour[g1_idx, g2_idx, :, 0]
        best_action = np.argmax(wr_base)
        reco = f"{noms_choix[best_action]}"
        best_wr = wr_base[best_action]

    return reco, best_wr, ev_actions, Q_table_jour


def eval_exact_market(market, g1_idx, g2_idx, has_booster,
                      gains_match, crowd_match, p_empirique_match, alpha_match,
                      V_next, noms_choix):
    """
    Évalue un ExactScoreMarket déjà construit à l'état (g1_idx, g2_idx), chaîné sur
    V_next (post-match, fine 1001x1001x2). Renvoie (reco, best_wr, market_df).
    Brique partagée : modes mono/multi-matchs du NB10 ET phases finales (NB18), où
    V_next est l'horizon endgame tranché par l'état des favoris.
    """
    gm = gains_match.astype(np.float64)
    cm = crowd_match.astype(np.float64)
    pm = p_empirique_match.astype(np.float64)
    a = float(alpha_match)

    Q_exact = evaluate_exact_score_day(
        g1_idx, g2_idx, market.outcomes, market.p_score, market.cond_crowd, market.bonus,
        gm, cm, pm, a, V_next
    )  # shape (K, 3) : [base, keep, use]
    # WR "si l'agent loupe son score exact" : l'agent ne touche que le gain d'outcome
    # (bonus agent neutralisé) ; Bob/peloton gardent leur bonus. Plancher robuste.
    Q_no_bonus = evaluate_exact_score_day(
        g1_idx, g2_idx, market.outcomes, market.p_score, market.cond_crowd, market.bonus,
        gm, cm, pm, a, V_next, agent_bonus_factor=0.0
    )

    # Choix de l'action (score) et du timing du booster — même logique qu'en 1N2.
    if has_booster:
        keep_idx = int(np.argmax(Q_exact[:, 1]))
        use_idx = int(np.argmax(Q_exact[:, 2]))
        if Q_exact[use_idx, 2] > Q_exact[keep_idx, 1]:
            best_idx, best_wr = use_idx, float(Q_exact[use_idx, 2])
            reco = f"{market.scores[best_idx]} + x2"
        else:
            best_idx, best_wr = keep_idx, float(Q_exact[keep_idx, 1])
            reco = f"{market.scores[best_idx]} (Safe)"
    else:
        best_idx = int(np.argmax(Q_exact[:, 0]))
        best_wr = float(Q_exact[best_idx, 0])
        reco = f"{market.scores[best_idx]}"

    # WR plancher (sans le bonus de l'agent), dans le même régime de booster que la reco
    if has_booster:
        wr_outcome = np.maximum(Q_no_bonus[:, 1], Q_no_bonus[:, 2])
    else:
        wr_outcome = Q_no_bonus[:, 0]

    market_df = pd.DataFrame({
        "Score": market.scores,
        "Outcome": [noms_choix[o] for o in market.outcomes],
        "True Proba (%)": market.p_score * 100.0,
        "Crowd cond. (%)": market.cond_crowd * 100.0,
        "Bonus": market.bonus,
        "E[pts MPP]": expected_mpp_points(market, gains_match),
        "E[pts 1/2/3]": expected_simple_points(market),
        "WR base (%)": Q_exact[:, 0] * 100.0,
        "WR keep (%)": Q_exact[:, 1] * 100.0,
        "WR x2 (%)": Q_exact[:, 2] * 100.0,
        "WR outcome (%)": wr_outcome * 100.0,
    })

    return reco, best_wr, market_df


def eval_exact_market_forward(market, P, Q_m, has_booster, gains_match, noms_choix,
                              Q_floor=None):
    """
    WR MOYEN de chaque score à un match FUTUR de la nuit, pondéré par la distribution
    d'états ATTEIGNABLE (passe forward) :  E[WR(s)] = Σ_état P(état) · Q_m(état, s).

    P (501,501,2) : distribution d'occupation coarse (couche 0 = booster déjà utilisé,
    couche 1 = booster en main), produite par forward_propagate_exact. Q_m (501,501,
    Kmax,3) = valeurs par-action [base, keep, use] stockées par solve_dp_coarse_exact.

    Combinaison booster (par score s) :
      - base : WR sans tenir compte du booster (Σ sur toute la masse) ;
      - keep : on NE pose PAS le x2 ici -> Q keep sur la masse 'booster en main',
               Q base sur la masse 'déjà utilisé' ;
      - x2   : on pose le x2 ici si on l'a encore -> Q use sur la masse 'en main',
               Q base sinon.

    `Q_floor` (optionnel, même shape que Q_m, typiquement float16) : valeurs par-action
    du PLANCHER « WR si l'agent loupe son score exact » (Q_store_floor de
    solve_dp_coarse_exact). Si fourni, ajoute la colonne 'WR outcome (%)' (même
    construction booster que les WR principaux, max keep/use si booster) — pendant
    forward de la colonne du match du jour. La tranche est upcastée en f32 pour le
    tensordot (BLAS). Si None, la colonne est omise (rétro-compat).

    Renvoie (reco, best_wr, market_df), même schéma que eval_exact_market (la colonne
    'WR outcome' n'est présente que si Q_floor est fourni).
    """
    K = len(market.scores)
    Qk = Q_m[:, :, :K, :]                                   # (501,501,K,3)
    P0 = P[:, :, 0]
    P1 = P[:, :, 1]

    base_b0 = np.tensordot(P0, Qk[:, :, :, 0], axes=([0, 1], [0, 1]))  # masse booster utilisé
    base_b1 = np.tensordot(P1, Qk[:, :, :, 0], axes=([0, 1], [0, 1]))  # masse en main, base
    keep_b1 = np.tensordot(P1, Qk[:, :, :, 1], axes=([0, 1], [0, 1]))
    use_b1 = np.tensordot(P1, Qk[:, :, :, 2], axes=([0, 1], [0, 1]))

    wr_base = base_b0 + base_b1
    wr_keep = base_b0 + keep_b1
    wr_use = base_b0 + use_b1

    if has_booster:
        keep_idx = int(np.argmax(wr_keep))
        use_idx = int(np.argmax(wr_use))
        if wr_use[use_idx] > wr_keep[keep_idx]:
            best_idx, best_wr = use_idx, float(wr_use[use_idx])
            reco = f"{market.scores[best_idx]} + x2"
        else:
            best_idx, best_wr = keep_idx, float(wr_keep[keep_idx])
            reco = f"{market.scores[best_idx]} (Safe)"
    else:
        best_idx = int(np.argmax(wr_base))
        best_wr = float(wr_base[best_idx])
        reco = f"{market.scores[best_idx]}"

    market_df = pd.DataFrame({
        "Score": market.scores,
        "Outcome": [noms_choix[o] for o in market.outcomes],
        "True Proba (%)": market.p_score * 100.0,
        "Crowd cond. (%)": market.cond_crowd * 100.0,
        "Bonus": market.bonus,
        "E[pts MPP]": expected_mpp_points(market, gains_match),
        "E[pts 1/2/3]": expected_simple_points(market),
        "WR base (%)": wr_base * 100.0,
        "WR keep (%)": wr_keep * 100.0,
        "WR x2 (%)": wr_use * 100.0,
    })

    # Plancher 'WR outcome' (agent sans bonus de score), même combinaison booster que
    # ci-dessus (base sur la masse booster utilisé, keep/use sur la masse en main).
    if Q_floor is not None:
        Qf = Q_floor[:, :, :K, :].astype(np.float32)        # upcast f16 -> f32 (BLAS)
        f_b0 = np.tensordot(P0, Qf[:, :, :, 0], axes=([0, 1], [0, 1]))
        f_b1 = np.tensordot(P1, Qf[:, :, :, 0], axes=([0, 1], [0, 1]))
        f_keep1 = np.tensordot(P1, Qf[:, :, :, 1], axes=([0, 1], [0, 1]))
        f_use1 = np.tensordot(P1, Qf[:, :, :, 2], axes=([0, 1], [0, 1]))
        floor_keep = f_b0 + f_keep1
        floor_use = f_b0 + f_use1
        if has_booster:
            wr_outcome = np.maximum(floor_keep, floor_use)
        else:
            wr_outcome = f_b0 + f_b1
        market_df["WR outcome (%)"] = wr_outcome * 100.0

    return reco, best_wr, market_df


def _recommend_exact_score(exact_score_data, g1_idx, g2_idx, has_booster,
                           gains_match, crowd_match, p_empirique_match, alpha_match,
                           V_next, noms_choix, Q_table_jour, outcome_probas=None):
    """
    Mode mono-match : construit le marché (ancré sur le 1N2 via outcome_probas) et
    renvoie (reco, best_wr, market_df, Q_table_jour).
    """
    market = build_exact_score_market(exact_score_data, outcome_probas=outcome_probas,
                                      shape_correction=True, mpp_outcome_crowd=crowd_match)
    reco, best_wr, market_df = eval_exact_market(
        market, g1_idx, g2_idx, has_booster,
        gains_match, crowd_match, p_empirique_match, alpha_match, V_next, noms_choix
    )
    return reco, best_wr, market_df, Q_table_jour