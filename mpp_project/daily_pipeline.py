import numpy as np
import pandas as pd
from pathlib import Path

from mpp_project.core import apply_heteroscedastic_noise, apply_temporal_drift, build_exact_score_market, calculate_true_outcome_probas_from_odds, estimate_crowd_3D, expected_simple_points, normalize_crowds, validate_match_dataframe
from mpp_project.oracle_dp import compute_alphas_isolement, compute_full_Q_table, evaluate_exact_score_day, solve_dp_coarse, GAP_MIN, GAP_MAX, GAP_OFFSET

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
    exact_score_data=None
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
    """
    match_idx = match_id_cible - 1

    # ==========================================
    # 1. CHARGEMENT DES DONNÉES
    # ==========================================
    df = pd.read_csv(csv_path)
    if validate_input:
        validate_match_dataframe(df)
    match_phases = df['phase'].tolist()
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

    # --- Phase 1 : DP LOINTAIN coarse (avec drift × nb_scenarios) ---
    # Drift fort sur les matchs lointains (alpha_bayes ≈ 0 → rmse ≈ 8%).
    # nb_scenarios réalisations → E[V(split_t)] en grille coarse (501, 501).
    if use_split:
        V_far_sum = np.zeros((501, 501, 2), dtype=np.float32)
        for s in range(n_runs):
            v_true_probas = apply_temporal_drift(base_true_probas, match_phases, current_match_idx=match_idx)
            v_crowds = apply_heteroscedastic_noise(blended_mean_crowds, rmse=dynamic_rmse)
            v_alphas = compute_alphas_isolement(v_true_probas, v_crowds, gains_1N2, seuil_isolement=seuil_isolement)
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
    if use_split:
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
                v_true_probas = apply_temporal_drift(base_true_probas, match_phases, current_match_idx=match_idx)
                v_crowds = apply_heteroscedastic_noise(blended_mean_crowds, rmse=dynamic_rmse)
            else:
                v_true_probas = base_true_probas.copy()
                v_crowds = base_crowds.copy()
            v_alphas = compute_alphas_isolement(v_true_probas, v_crowds, gains_1N2, seuil_isolement=seuil_isolement)
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

    # --- Mode SCORES EXACTS (match du jour) ---
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


def _recommend_exact_score(exact_score_data, g1_idx, g2_idx, has_booster,
                           gains_match, crowd_match, p_empirique_match, alpha_match,
                           V_next, noms_choix, Q_table_jour, outcome_probas=None):
    """
    Décision du match du jour sur les SCORES EXACTS. Construit le marché (ancré
    sur le 1N2 du CSV via outcome_probas), évalue le WR de chaque score via
    evaluate_exact_score_day, et renvoie (reco, best_wr, market_df, Q_table_jour).
    """
    market = build_exact_score_market(exact_score_data, outcome_probas=outcome_probas)

    Q_exact = evaluate_exact_score_day(
        g1_idx, g2_idx,
        market.outcomes, market.p_score, market.cond_crowd, market.bonus,
        gains_match.astype(np.float64), crowd_match.astype(np.float64),
        p_empirique_match.astype(np.float64), float(alpha_match), V_next
    )  # shape (K, 3) : [base, keep, use]

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

    market_df = pd.DataFrame({
        "Score": market.scores,
        "Outcome": [noms_choix[o] for o in market.outcomes],
        "True Proba (%)": market.p_score * 100.0,
        "Crowd cond. (%)": market.cond_crowd * 100.0,
        "Bonus": market.bonus,
        "E[pts 1/2/3]": expected_simple_points(market),
        "WR base (%)": Q_exact[:, 0] * 100.0,
        "WR keep (%)": Q_exact[:, 1] * 100.0,
        "WR x2 (%)": Q_exact[:, 2] * 100.0,
    })

    return reco, best_wr, market_df, Q_table_jour