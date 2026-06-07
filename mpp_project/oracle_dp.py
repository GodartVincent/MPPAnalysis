import numpy as np
import time
from numba import njit, prange

# --- IMPORTS DE TON PROJET ---
from mpp_project.core import RMSE_GAINS, apply_heteroscedastic_noise, apply_temporal_drift, estimate_crowd_3D, calculate_mpp_gains

# --- 1. CONFIGURATION & CONSTANTES ---
N_MATCHES = 104
N_MATCHES_POULES = 72
N_MATCHES_FINALES = N_MATCHES - N_MATCHES_POULES  # 32 matchs

GAP_MIN = -600
GAP_MAX = 400
GAP_OFFSET = -GAP_MIN  # 600
GRID_SIZE = GAP_MAX - GAP_MIN + 1  # 1001

N_ACTIONS = 3
N_PLAYERS = 12
N_SIMULATIONS = 2000 # Nombre de phases finales à simuler

match_params = {
    'n_matches': N_MATCHES,
    'ev_avg': 35,
    'draw_fact_min': 0.2,
    'draw_fact_max': 0.75,
    'outsider_fact_min': 1/7.5,
    'outsider_fact_max': 1.0,
    'proba_fact_std': 0.025
}

@njit
def compute_alphas_isolement(true_probas, crowds, gains_1N2, seuil_isolement=80.0):
    """
    Calcule le facteur d'isolement (alpha) au sein du peloton de 0.0 (Début/Meute dense)
    à 1.0 (Fin/Joueurs isolés) basé sur l'écart-type cumulé réel des gains des adversaires.
    """
    n_matches = true_probas.shape[0]
    alphas = np.zeros(n_matches, dtype=np.float32)
    variance_cumulee = 0.0
    
    for t in range(n_matches):
        E_X = 0.0
        E_X2 = 0.0
        for out in range(3):
            # Probabilité conjointe que l'issue se réalise ET qu'un joueur l'ait pronostiquée
            p_gain = true_probas[t, out] * crowds[t, out]
            gain = gains_1N2[t, out]
            
            E_X += p_gain * gain
            E_X2 += p_gain * (gain ** 2)
            
        var_t = E_X2 - (E_X ** 2)
        variance_cumulee += var_t
        
        sigma_t = np.sqrt(variance_cumulee)
        
        # Gating par saturation (Michaelis-Menten)
        alphas[t] = sigma_t / (sigma_t + seuil_isolement)
        
    return alphas

# --- MODÉLISATION DU CROWD MPP POUR FAVORIS/BUTEURS (Le Biais Chauvin / Héroïque) ---
def estimate_mpp_crowd(selections, true_probs, gains, base_beta=1.2):
    """
    Part des probabilités true pour estimer la repartition Crowd des favoris/buteurs.
    """
    crowd_weights = true_probs ** base_beta
    
    # Biais de désirabilité / Chauvinisme
    chauvinism_multiplier = np.ones(len(selections))
    
    for i, sel in enumerate(selections):
        sel_lower = str(sel).lower()
        
        # Biais Énorme : La France et ses joueurs
        if sel_lower in ['france', 'kylian_mbappe']:
            chauvinism_multiplier[i] = 6.5
            
        # Biais Fort : Les superstars internationales (Ronaldo, Messi si présents)
        elif sel_lower in ['portugal', 'cristiano_ronaldo', 'messi', 'argentine', 'bresil']:
            chauvinism_multiplier[i] = 1.5
            
    # La catégorie "Autre" est historiquement sous-jouée par les casuals (biais de représentativité)
    for i, sel in enumerate(selections):
        if str(sel).lower() == 'autre':
            chauvinism_multiplier[i] = 0.9
            
    final_weights = crowd_weights * chauvinism_multiplier
    return final_weights / final_weights.sum()

# --- 2. L'ORACLE C++ (NUMBA) ---
@njit(parallel=True)
def solve_final_phases_numba(true_probas, mpp_gains, crowd_repartitions):
    V_all_matches = np.zeros((N_MATCHES_FINALES, GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
    V_next = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
    
    # Condition Terminale
    for g1 in range(GRID_SIZE):
        val_g1 = g1 - GAP_OFFSET
        for g2 in range(GRID_SIZE):
            val_g2 = g2 - GAP_OFFSET
            if val_g1 > 0 and val_g2 > 0:
                win_prob = 1.0
            elif (val_g1 == 0 and val_g2 > 0) or (val_g1 > 0 and val_g2 == 0):
                win_prob = 0.5
            elif val_g1 == 0 and val_g2 == 0:
                win_prob = 1.0 / 3.0
            else:
                win_prob = 0.0
            
            V_next[g1, g2, 0] = win_prob
            V_next[g1, g2, 1] = win_prob
            
    # Backward Induction
    for t in range(N_MATCHES_FINALES - 1, -1, -1):
        V_current = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        
        global_t = N_MATCHES_POULES + t
        time_fraction = 1.0 - (global_t / N_MATCHES)
        N_eff = 1.0 + (N_PLAYERS - 3.0) * (time_fraction ** 5.50)
        
        t_prob = true_probas[t]
        c_rep = crowd_repartitions[t]
        gains = mpp_gains[t]
        
        for g1 in prange(GRID_SIZE):
            val_g1 = g1 - GAP_OFFSET
            for g2 in range(g1, GRID_SIZE): 
                val_g2 = g2 - GAP_OFFSET
                
                best_v_b0 = 0.0 
                best_v_b1 = 0.0 
                
                for a in range(3):
                    expected_v_b0 = 0.0
                    expected_v_keep = 0.0
                    expected_v_use = 0.0
                    
                    for out in range(3):
                        p_out = t_prob[out]
                        
                        # --- CORRECTION NUMBA : CAST STRICT ---
                        gain_val = int(gains[out])
                        
                        a_g = gain_val if a == out else 0
                        a_g_boost = (gain_val * 2) if a == out else 0
                        
                        prob_pack_hits = 1.0 - (1.0 - c_rep[out]) ** N_eff
                        
                        for bob_action in range(3):
                            p_bob = c_rep[bob_action]
                            
                            # On utilise le gain_val (int) pour que les types soient parfaits
                            bob_g = gain_val if bob_action == out else 0
                            pack_g_hit = gain_val
                            pack_g_miss = 0
                            
                            # HIT
                            jp_hit = p_out * p_bob * prob_pack_hits
                            ng_min_norm = max(-600, min(400, min(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_hit)))
                            ng_max_norm = max(-600, min(400, max(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_hit)))
                            ng_min_boost = max(-600, min(400, min(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_hit)))
                            ng_max_boost = max(-600, min(400, max(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_hit)))
                            
                            expected_v_b0   += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 0]
                            expected_v_keep += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 1]
                            expected_v_use  += jp_hit * V_next[ng_min_boost + GAP_OFFSET, ng_max_boost + GAP_OFFSET, 0]
                            
                            # MISS
                            jp_miss = p_out * p_bob * (1.0 - prob_pack_hits)
                            ng_min_norm_m = max(-600, min(400, min(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_miss)))
                            ng_max_norm_m = max(-600, min(400, max(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_miss)))
                            ng_min_boost_m = max(-600, min(400, min(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_miss)))
                            ng_max_boost_m = max(-600, min(400, max(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_miss)))
                            
                            expected_v_b0   += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 0]
                            expected_v_keep += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 1]
                            expected_v_use  += jp_miss * V_next[ng_min_boost_m + GAP_OFFSET, ng_max_boost_m + GAP_OFFSET, 0]
                    
                    if expected_v_b0 > best_v_b0:
                        best_v_b0 = expected_v_b0
                        
                    best_action_b1 = max(expected_v_keep, expected_v_use)
                    if best_action_b1 > best_v_b1:
                        best_v_b1 = best_action_b1
                        
                V_current[g1, g2, 0] = best_v_b0
                V_current[g1, g2, 1] = best_v_b1
                
        # Copie de Symétrie Safe
        for g1 in prange(GRID_SIZE):
            for g2 in range(g1 + 1, GRID_SIZE):
                V_current[g2, g1, 0] = V_current[g1, g2, 0]
                V_current[g2, g1, 1] = V_current[g1, g2, 1]
                
        V_all_matches[t] = V_current
        V_next = V_current
        
    return V_all_matches


# --- 3. FONCTION D'INFÉRENCE LOCALE (Notebook 10) ---
@njit(parallel=True)
def evaluate_current_match(match_idx_to_predict, last_csv_idx, probas, gains, crowds, V_horizon, gap1, gap2, has_booster):
    V_next = np.copy(V_horizon)
    
    for t in range(last_csv_idx - 1, match_idx_to_predict, -1):
        V_current = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        
        time_fraction = 1.0 - (t / N_MATCHES)
        N_eff = 1.0 + (N_PLAYERS - 3.0) * (time_fraction ** 5.50)
        
        t_prob = probas[t]
        c_rep = crowds[t]
        t_gain = gains[t]
        
        for g1 in prange(GRID_SIZE):
            val_g1 = g1 - GAP_OFFSET
            for g2 in range(g1, GRID_SIZE):
                val_g2 = g2 - GAP_OFFSET
                
                best_v_b0 = 0.0
                best_v_b1 = 0.0
                
                for a in range(3):
                    expected_v_b0 = 0.0
                    expected_v_keep = 0.0
                    expected_v_use = 0.0
                    
                    for out in range(3):
                        p_out = t_prob[out]
                        
                        # --- CORRECTION NUMBA : CAST STRICT ---
                        gain_val = int(t_gain[out])
                        
                        a_g = gain_val if a == out else 0
                        a_g_boost = (gain_val * 2) if a == out else 0
                        
                        prob_pack_hits = 1.0 - (1.0 - c_rep[out]) ** N_eff
                        
                        for bob_action in range(3):
                            p_bob = c_rep[bob_action]
                            bob_g = gain_val if bob_action == out else 0
                            pack_g_hit = gain_val
                            pack_g_miss = 0
                            
                            # HIT
                            jp_hit = p_out * p_bob * prob_pack_hits
                            ng_min_norm = max(-600, min(400, min(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_hit)))
                            ng_max_norm = max(-600, min(400, max(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_hit)))
                            ng_min_boost = max(-600, min(400, min(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_hit)))
                            ng_max_boost = max(-600, min(400, max(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_hit)))
                            
                            expected_v_b0   += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 0]
                            expected_v_keep += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 1]
                            expected_v_use  += jp_hit * V_next[ng_min_boost + GAP_OFFSET, ng_max_boost + GAP_OFFSET, 0]
                            
                            # MISS
                            jp_miss = p_out * p_bob * (1.0 - prob_pack_hits)
                            ng_min_norm_m = max(-600, min(400, min(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_miss)))
                            ng_max_norm_m = max(-600, min(400, max(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_miss)))
                            ng_min_boost_m = max(-600, min(400, min(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_miss)))
                            ng_max_boost_m = max(-600, min(400, max(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_miss)))
                            
                            expected_v_b0   += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 0]
                            expected_v_keep += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 1]
                            expected_v_use  += jp_miss * V_next[ng_min_boost_m + GAP_OFFSET, ng_max_boost_m + GAP_OFFSET, 0]
                            
                    if expected_v_b0 > best_v_b0:
                        best_v_b0 = expected_v_b0
                        
                    best_action_b1 = max(expected_v_keep, expected_v_use)
                    if best_action_b1 > best_v_b1:
                        best_v_b1 = best_action_b1
                        
                V_current[g1, g2, 0] = best_v_b0
                V_current[g1, g2, 1] = best_v_b1
                
        for g1 in prange(GRID_SIZE):
            for g2 in range(g1 + 1, GRID_SIZE):
                V_current[g2, g1, 0] = V_current[g1, g2, 0]
                V_current[g2, g1, 1] = V_current[g1, g2, 1]
                
        V_next = V_current
        
    # --- CALCUL FINAL POUR LE MATCH DU JOUR ---
    t = match_idx_to_predict
    time_fraction = 1.0 - (t / N_MATCHES)
    N_eff = 1.0 + (N_PLAYERS - 3.0) * (time_fraction ** 5.50)
    
    t_prob = probas[t]
    c_rep = crowds[t]
    t_gain = gains[t]
    
    wr_keep = np.zeros(3, dtype=np.float32)
    wr_use = np.zeros(3, dtype=np.float32)
    ev_actions = np.zeros(3, dtype=np.float32)
    
    for a in range(3):
        expected_v_b0 = 0.0
        expected_v_keep = 0.0
        expected_v_use = 0.0
        
        # ev_actions calcul purement Python
        ev_actions[a] = t_prob[a] * t_gain[a]
        
        for out in range(3):
            p_out = t_prob[out]
            
            gain_val = int(t_gain[out])
            a_g = gain_val if a == out else 0
            a_g_boost = (gain_val * 2) if a == out else 0
            
            prob_pack_hits = 1.0 - (1.0 - c_rep[out]) ** N_eff
            
            for bob_action in range(3):
                p_bob = c_rep[bob_action]
                bob_g = gain_val if bob_action == out else 0
                pack_g_hit = gain_val
                pack_g_miss = 0
                
                # HIT
                jp_hit = p_out * p_bob * prob_pack_hits
                ng_min_norm = max(-600, min(400, min(gap1 + a_g - bob_g, gap2 + a_g - pack_g_hit)))
                ng_max_norm = max(-600, min(400, max(gap1 + a_g - bob_g, gap2 + a_g - pack_g_hit)))
                ng_min_boost = max(-600, min(400, min(gap1 + a_g_boost - bob_g, gap2 + a_g_boost - pack_g_hit)))
                ng_max_boost = max(-600, min(400, max(gap1 + a_g_boost - bob_g, gap2 + a_g_boost - pack_g_hit)))
                
                expected_v_b0   += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 0]
                expected_v_keep += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 1]
                expected_v_use  += jp_hit * V_next[ng_min_boost + GAP_OFFSET, ng_max_boost + GAP_OFFSET, 0]
                
                # MISS
                jp_miss = p_out * p_bob * (1.0 - prob_pack_hits)
                ng_min_norm_m = max(-600, min(400, min(gap1 + a_g - bob_g, gap2 + a_g - pack_g_miss)))
                ng_max_norm_m = max(-600, min(400, max(gap1 + a_g - bob_g, gap2 + a_g - pack_g_miss)))
                ng_min_boost_m = max(-600, min(400, min(gap1 + a_g_boost - bob_g, gap2 + a_g_boost - pack_g_miss)))
                ng_max_boost_m = max(-600, min(400, max(gap1 + a_g_boost - bob_g, gap2 + a_g_boost - pack_g_miss)))
                
                expected_v_b0   += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 0]
                expected_v_keep += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 1]
                expected_v_use  += jp_miss * V_next[ng_min_boost_m + GAP_OFFSET, ng_max_boost_m + GAP_OFFSET, 0]
                
        if has_booster == 1:
            wr_keep[a] = expected_v_keep
            wr_use[a] = expected_v_use
        else:
            wr_keep[a] = expected_v_b0
            wr_use[a] = 0.0
            
    return wr_keep, wr_use, ev_actions

# --- 4. CALCUL DE L'ESPÉRANCE DE WIN RATE DE L'AGENT A L'HORIZON (MONTE CARLO) ---
def compute_expected_V_phases_finales(n_simulations=5000):
    print(f"--- DÉMARRAGE DU CALCUL MONTE CARLO ({n_simulations} simulations) ---")
    
    # NOUVEAU : La matrice devient une structure 4D (32, 1001, 1001, 2)
    V_sum = np.zeros((N_MATCHES_FINALES, GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
    start_time = time.time()
    
    for i in range(n_simulations):
        final_probas, final_gains, final_crowd = generate_knockout_data()
        
        V_matrix = solve_final_phases_numba(
            final_probas.astype(np.float32), 
            final_gains.astype(np.int32), 
            final_crowd.astype(np.float32)
        )
        
        V_sum += V_matrix
        
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            rem_time = avg_time * (n_simulations - (i + 1))
            print(f"[{i + 1}/{n_simulations}] Temps écoulé : {elapsed:.1f}s | Reste estimé : {rem_time:.1f}s")
            
    expected_V_finals = V_sum / n_simulations
    filename = "expected_V_phases_finales.npy"
    np.save(filename, expected_V_finals)
    
    print(f"\n✅ CALCUL TERMINÉ EN {time.time() - start_time:.1f}s")

# Batching pour calculer plusieurs drift et moyenner les win rates résultants pour lisser les résultats et réduire la variance de l'estimation.
def evaluate_match_ensemble(match_idx, last_csv_match_id, true_probas, match_phases, mpp_gains, crowd_repartitions, V_next_base, gap1, gap2, has_booster, n_ensembles=30):
    """
    Inférence locale Monte Carlo avec Lissage Bayésien Temporel (Alpha)
    et Bruit Hétéroscédastique Dynamique.
    """
    # 1. Extraction et sécurisation des types
    window_probas = true_probas[:last_csv_match_id].astype(np.float32)
    window_gains = mpp_gains[:last_csv_match_id].astype(np.int32) 
    window_csv_crowds = crowd_repartitions[:last_csv_match_id].astype(np.float32)
    
    # 2. Modélisation Continue de la Confiance (Exponential Decay)
    # On crée un tableau des distances [0, 1, 2, ..., N] par rapport au match_idx
    dists = np.arange(last_csv_match_id) - match_idx
    
    # Sécurité : Si un match est dans le passé (dist < 0), on le bloque à 0 pour garder alpha_max
    dists_positive = np.maximum(0, dists)
    
    # Paramètres de la décroissance
    ALPHA_MAX = 0.95
    HALF_LIFE = 5.0  # Au bout de 5 matchs d'écart, l'information perd 50% de son poids
    
    # Calcul vectorisé en une seule ligne
    alphas = ALPHA_MAX * (0.5 ** (dists_positive / HALF_LIFE))
    alphas_2d = alphas.astype(np.float32)[:, np.newaxis]

    # 3. Calcul du Crowd Théorique PUR (Sans bruit)
    c1, cN, c2 = estimate_crowd_3D(
        window_probas[:, 0], 
        window_probas[:, 1], 
        window_probas[:, 2], 
        add_noise=False  # On bloque le bruit ici !
    )
    theo_crowds_pure = np.column_stack((c1, cN, c2)).astype(np.float32)
    
    # 4. Le Lissage Bayésien : La moyenne des Espérances (Sans bruit)
    blended_mean_crowds = (alphas_2d * window_csv_crowds) + ((1.0 - alphas_2d) * theo_crowds_pure)
    
    # Préparation des accumulateurs
    sum_wr_keep = np.zeros(3, dtype=np.float32)
    sum_wr_use = np.zeros(3, dtype=np.float32)
    sum_ev = np.zeros(3, dtype=np.float32)
    
    # L'astuce magique : Le RMSE de 8.3% (0.083) est écrasé si on fait confiance au CSV
    dynamic_rmse = 0.083 * (1.0 - alphas_2d)
    
    # 5. Boucle de Monte Carlo avec Injection du Bruit Dynamique
    for _ in range(n_ensembles):
        
        # --- UTILISATION DE LA NOUVELLE FONCTION ---
        noisy_crowds = apply_heteroscedastic_noise(blended_mean_crowds, rmse=dynamic_rmse)
        
        # 6. Appel à l'Oracle Numba
        wr_k, wr_u, ev = evaluate_current_match(
            match_idx_to_predict=match_idx,
            last_csv_idx=last_csv_match_id,
            probas=window_probas,
            gains=window_gains,
            crowds=noisy_crowds,
            V_horizon=V_next_base,
            gap1=int(gap1),
            gap2=int(gap2),
            has_booster=int(has_booster)
        )
        
        sum_wr_keep += wr_k
        sum_wr_use += wr_u
        sum_ev += ev
        
    return sum_wr_keep / n_ensembles, sum_wr_use / n_ensembles, sum_ev / n_ensembles

def generate_knockout_data():
    """
    Génère les 32 matchs des phases finales.
    - Les gains MPP sont basés sur les cotes 90 min (via calculate_mpp_gains) + un bruit (RMSE).
    - La Foule (pire cas) et l'Oracle se basent sur les vraies probabilités 120 min.
    """
    probas_90 = np.zeros((32, 3), dtype=np.float32)
    
    # --- 1. Génération des Probas sur 90 MIN (Le référentiel Bookmaker/MPP) ---
    # 1/16e (16 matchs)
    probas_90[0:16, 0] = np.random.uniform(0.45, 0.65, 16)
    probas_90[0:16, 1] = np.random.uniform(0.22, 0.28, 16)
    # 1/8e (8 matchs)
    probas_90[16:24, 0] = np.random.uniform(0.40, 0.55, 8)
    probas_90[16:24, 1] = np.random.uniform(0.25, 0.32, 8)
    # Quarts (4 matchs)
    probas_90[24:28, 0] = np.random.uniform(0.38, 0.50, 4)
    probas_90[24:28, 1] = np.random.uniform(0.28, 0.34, 4)
    # Demies (2 matchs)
    probas_90[28:30, 0] = np.random.uniform(0.36, 0.45, 2)
    probas_90[28:30, 1] = np.random.uniform(0.28, 0.35, 2)
    # Petite Finale (1 match)
    probas_90[30, 0] = np.random.uniform(0.40, 0.50)
    probas_90[30, 1] = np.random.uniform(0.25, 0.30)
    # Finale (1 match)
    probas_90[31, 0] = np.random.uniform(0.35, 0.42)
    probas_90[31, 1] = np.random.uniform(0.30, 0.36)
    
    # Outsider (col 2)
    probas_90[:, 2] = 1.0 - probas_90[:, 0] - probas_90[:, 1]
    
    # Mélange Domicile / Extérieur
    for i in range(32):
        if np.random.rand() > 0.5:
            probas_90[i, 0], probas_90[i, 2] = probas_90[i, 2], probas_90[i, 0]
            
    # --- 2. LA VÉRITÉ TERRAIN (Probas 120 MIN pour le moteur DP et la Foule) ---
    true_probas_120 = np.copy(probas_90)
    
    # Tirage d'un ratio dynamique de matchs allant aux Tirs au But (entre 40% et 50%)
    penalties_ratio = np.random.uniform(0.4, 0.5, 32)
    
    # Probabilité qu'un match nul à 90min trouve un vainqueur avant les TAB
    et_resolved_prob = probas_90[:, 1] * (1.0 - penalties_ratio)
    
    # On redistribue ces victoires en prolongation proportionnellement à la force initiale
    team1_strength = probas_90[:, 0] / (probas_90[:, 0] + probas_90[:, 2])
    team2_strength = probas_90[:, 2] / (probas_90[:, 0] + probas_90[:, 2])
    
    true_probas_120[:, 0] += et_resolved_prob * team1_strength
    true_probas_120[:, 2] += et_resolved_prob * team2_strength
    
    # La nouvelle probabilité de match nul (uniquement si Tirs au But)
    true_probas_120[:, 1] = probas_90[:, 1] * penalties_ratio
    
    # --- 3. Génération des Gains MPP ---
    # Utilisation du polynôme d'ordre 3 de core.py sur les probas 90min
    mpp_gains = calculate_mpp_gains(probas_90, add_noise=True)
    
    # --- 4. Réaction de la Foule (Pire Cas : Foule Intelligente sur 120 min) ---
    c1, cN, c2 = estimate_crowd_3D(true_probas_120[:, 0], true_probas_120[:, 1], true_probas_120[:, 2])
    crowds = np.column_stack((c1, cN, c2)).astype(np.float32)
    
    return true_probas_120, mpp_gains, crowds

# Utilisé pour pronostiquer le meilleur buteur et le favori (notebook 16)
@njit(parallel=True)
def backpropagate_group_stages_stochastic(match_du_jour_idx, probas_poules, gains_poules, ensemble_crowds, V_phases_finales, p_empirique_3D):
    
    V_next = np.copy(V_phases_finales)
    
    N_POULES = len(probas_poules)
    N_ENS = ensemble_crowds.shape[0]
    max_gain = p_empirique_3D.shape[2]
    
    # Pré-calcul de la foule moyenne pour la vitesse
    mean_crowds = np.zeros((N_POULES, 3), dtype=np.float32)
    for t in range(N_POULES):
        for out in range(3):
            sum_c = 0.0
            for e in range(N_ENS):
                sum_c += ensemble_crowds[e, t, out]
            mean_crowds[t, out] = sum_c / N_ENS
    
    # On recule jusqu'au match du jour
    for t in range(N_POULES - 1, match_du_jour_idx - 1, -1):
        V_current = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        
        t_prob = probas_poules[t]
        t_gain = gains_poules[t]
        t_p_empirique = p_empirique_3D[t]
        c_rep = mean_crowds[t] 
        
        for g1 in prange(GRID_SIZE):
            val_g1 = g1 - GAP_OFFSET
            # 🚀 OPTIMISATION RESTAURÉE : Grâce au tri, g1 est toujours <= g2. 
            # On ne calcule que le triangle supérieur !
            for g2 in range(g1, GRID_SIZE):
                val_g2 = g2 - GAP_OFFSET
                
                best_v_b0 = 0.0
                best_v_b1 = 0.0
                
                for a in range(3):
                    expected_v_b0 = 0.0
                    expected_v_keep = 0.0
                    expected_v_use = 0.0
                    
                    for out in range(3):
                        p_out = t_prob[out]
                        if p_out == 0.0: continue
                            
                        a_g = t_gain[out] if a == out else 0
                        a_g_boost = (t_gain[out] * 2) if a == out else 0
                        true_gain = t_gain[out]
                        out_p_empirique = t_p_empirique[out]
                        
                        for bob_a in range(3):
                            p_bob = c_rep[bob_a]
                            if p_bob == 0.0: continue
                                
                            bob_g = true_gain if bob_a == out else 0
                            jp_base = p_out * p_bob
                            
                            new_gap_bob_norm = val_g1 + a_g - bob_g
                            new_gap_bob_boost = val_g1 + a_g_boost - bob_g
                            
                            esp_peloton_b0 = 0.0
                            esp_peloton_keep = 0.0
                            esp_peloton_use = 0.0
                            
                            # BOUCLE HISTOGRAMME
                            for delta_gain in range(max_gain):
                                p_peloton = out_p_empirique[delta_gain]
                                if p_peloton == 0.0: continue
                                    
                                new_gap_peloton_norm = val_g2 + a_g - delta_gain
                                new_gap_peloton_boost = val_g2 + a_g_boost - delta_gain
                                
                                # 🚀 RESTAURATION DE TA LOGIQUE (LE NOUVEAU BOB)
                                g1_norm = max(GAP_MIN, min(GAP_MAX, min(new_gap_bob_norm, new_gap_peloton_norm))) + GAP_OFFSET
                                g2_norm = max(GAP_MIN, min(GAP_MAX, max(new_gap_bob_norm, new_gap_peloton_norm))) + GAP_OFFSET
                                
                                g1_boost = max(GAP_MIN, min(GAP_MAX, min(new_gap_bob_boost, new_gap_peloton_boost))) + GAP_OFFSET
                                g2_boost = max(GAP_MIN, min(GAP_MAX, max(new_gap_bob_boost, new_gap_peloton_boost))) + GAP_OFFSET
                                
                                esp_peloton_b0 += p_peloton * V_next[g1_norm, g2_norm, 0]
                                esp_peloton_keep += p_peloton * V_next[g1_norm, g2_norm, 1]
                                esp_peloton_use += p_peloton * V_next[g1_boost, g2_boost, 0]
                                
                            expected_v_b0 += jp_base * esp_peloton_b0
                            expected_v_keep += jp_base * esp_peloton_keep
                            expected_v_use += jp_base * esp_peloton_use
                    
                    if expected_v_b0 > best_v_b0:
                        best_v_b0 = expected_v_b0
                        
                    best_action_b1 = max(expected_v_keep, expected_v_use)
                    if best_action_b1 > best_v_b1:
                        best_v_b1 = best_action_b1
                        
                V_current[g1, g2, 0] = best_v_b0
                V_current[g1, g2, 1] = best_v_b1
                
        # Symétrie pour l'autre moitié de la grille
        for g1 in prange(GRID_SIZE):
            for g2 in range(g1 + 1, GRID_SIZE):
                V_current[g2, g1, 0] = V_current[g1, g2, 0]
                V_current[g2, g1, 1] = V_current[g1, g2, 1]
                
        V_next = V_current
        
    return V_current

@njit
def rescale_histogram_1D(ref_hist, ref_gain, target_gain, target_crowd, max_bins=250):
    """
    Applique une transformation affine sur l'axe X d'une distribution de probabilité.
    Forcé en entiers (int) pour la compatibilité stricte Numba.
    """
    # 🚀 BLINDAGE DES TYPES : On force les gains à être des index entiers
    ref_g = int(round(ref_gain))
    tgt_g = int(round(target_gain))
    
    scaled_hist = np.zeros(max_bins, dtype=np.float32)
    
    if ref_g <= 0 or tgt_g <= 0:
        scaled_hist[0] = 1.0 
        return scaled_hist
        
    scale_factor = tgt_g / ref_g
    sum_scaled = 0.0
    
    for i in range(max_bins):
        if i == ref_g:
            continue # On ignore le pic de référence
            
        p = ref_hist[i]
        if p > 0.0:
            new_idx = int(round(i * scale_factor))
            
            if new_idx >= tgt_g:
                new_idx = max(0, tgt_g - 1)
                
            if new_idx >= max_bins:
                new_idx = max_bins - 1
                
            scaled_hist[new_idx] += p
            sum_scaled += p
            
    if sum_scaled > 0.0:
        norm_factor = (1.0 - target_crowd) / sum_scaled
        for i in range(max_bins):
            scaled_hist[i] *= norm_factor
    else:
        scaled_hist[0] = 1.0 - target_crowd
        
    # Le calcul d'index est maintenant 100% sécurisé avec des entiers
    idx_pic = min(tgt_g, max_bins - 1)
    scaled_hist[idx_pic] = target_crowd
    
    total = np.sum(scaled_hist)
    if total > 0:
        scaled_hist /= total
    else:
        scaled_hist[0] = 1.0
        
    return scaled_hist

@njit
def extract_peloton_full_distribution(true_probas_3d, crowds_3d, gains_1N2, max_gain=250, n_runs=1000000, n_players=11):
    n_universes = true_probas_3d.shape[0]
    n_matches = true_probas_3d.shape[1]
    
    counts = np.zeros((n_matches, 3, max_gain), dtype=np.float64)
    occurrences = np.zeros((n_matches, 3), dtype=np.float64)
    
    for run in range(n_runs):
        scores = np.zeros(n_players, dtype=np.int32)
        u = run % n_universes
        
        for t in range(n_matches):
            
            # 1. Identification de l'actuel 1er (Futur Bob)
            best_score = -1
            bob_idx = 0
            for i in range(n_players):
                if scores[i] > best_score:
                    best_score = scores[i]
                    bob_idx = i
                    
            # 2. Récupération des points du meilleur PARMI LES AUTRES (prev_second_best)
            prev_second_best = -1
            for i in range(n_players):
                if i == bob_idx: # On exclut Bob
                    continue
                if scores[i] > prev_second_best:
                    prev_second_best = scores[i]
            if prev_second_best == -1: prev_second_best = 0
                    
            # 3. Tirage de la réalité du match
            r_true = np.random.rand()
            if r_true < true_probas_3d[u, t, 0]: true_out = 0
            elif r_true < true_probas_3d[u, t, 0] + true_probas_3d[u, t, 1]: true_out = 1
            else: true_out = 2
            
            true_gain_mpp = gains_1N2[t, true_out]
            
            # 4. Pronostics et mise à jour des points de TOUS les adversaires
            for i in range(n_players):
                r_bet = np.random.rand()
                if r_bet < crowds_3d[u, t, 0]: bet = 0
                elif r_bet < crowds_3d[u, t, 0] + crowds_3d[u, t, 1]: bet = 1
                else: bet = 2
                
                if bet == true_out:
                    scores[i] += true_gain_mpp
                    
            # 5. Récupération des points du meilleur HORS ANCIEN BOB (new_second_best)
            new_second_best = -1
            for i in range(n_players):
                if i == bob_idx: # On exclut l'ancien Bob
                    continue
                if scores[i] > new_second_best:
                    new_second_best = scores[i]
            if new_second_best == -1: new_second_best = 0
            
            # 6. Mise à jour de l'histogramme du peloton
            delta_pack = new_second_best - prev_second_best
            
            if delta_pack >= max_gain:
                delta_pack = max_gain - 1 
            elif delta_pack < 0:
                delta_pack = 0
                
            occurrences[t, true_out] += 1
            counts[t, true_out, delta_pack] += 1

    p_empirique = np.zeros_like(counts)
    for t in range(n_matches):
        for out in range(3):
            if occurrences[t, out] > 0:
                p_empirique[t, out, :] = counts[t, out, :] / occurrences[t, out]
                
    return p_empirique

# ==========================================
# 2. LE MOTEUR ORACLE DP
# ==========================================
@njit(parallel=True)
def solve_dp_with_full_empirical_distribution(base_true_probas, base_crowds, gains_1N2, p_empirique_1D, alphas_isolement, V_horizon, stop_t=0, horizon_nuit=0, start_t=-1):
    """
    start_t=-1  → run complet depuis n_matches-1, condition terminale appliquée.
    start_t>=0  → run partiel depuis start_t-1 (phase proche), V_horizon utilisé tel quel
                  comme condition limite fournie par un run lointain précédent.
    """
    n_matches = base_true_probas.shape[0]
    max_gain = p_empirique_1D.shape[2]

    V_all_matches = np.zeros((n_matches, GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
    Q_tables = np.zeros((horizon_nuit, GRID_SIZE, GRID_SIZE, 3, 3), dtype=np.float32)
    V_next = np.copy(V_horizon)

    # 1. Condition Terminale (uniquement pour run complet depuis le début)
    if start_t < 0:
        for g1 in range(GRID_SIZE):
            val_g1 = g1 - GAP_OFFSET
            for g2 in range(g1, GRID_SIZE):
                val_g2 = g2 - GAP_OFFSET

                if val_g1 > 0 and val_g2 > 0: win_prob = 1.0
                elif (val_g1 == 0 and val_g2 > 0) or (val_g1 > 0 and val_g2 == 0): win_prob = 0.5
                elif val_g1 == 0 and val_g2 == 0: win_prob = 1.0 / 3.0
                else: win_prob = 0.0

                V_next[g1, g2, 0] = win_prob
                V_next[g1, g2, 1] = win_prob

    # 2. Point de départ de la rétro-propagation
    if start_t < 0:
        t_debut = n_matches - 1
    else:
        t_debut = start_t - 1

    # 3. Rétro-propagation
    for t in range(t_debut, stop_t - 1, -1):
        V_current = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        t_prob = base_true_probas[t]
        c_rep = base_crowds[t]
        t_gains = gains_1N2[t]
        alpha = alphas_isolement[t]
        store_q = horizon_nuit > 0 and t >= stop_t and t < stop_t + horizon_nuit
        k = t - stop_t

        for g1 in prange(GRID_SIZE):
            val_g1 = g1 - GAP_OFFSET
            for g2 in range(g1, GRID_SIZE):
                val_g2 = g2 - GAP_OFFSET

                best_v_b0 = 0.0
                best_v_b1 = 0.0

                for a in range(3):
                    expected_v_b0 = 0.0
                    expected_v_keep = 0.0
                    expected_v_use = 0.0

                    for out in range(3):
                        p_out = t_prob[out]
                        if p_out == 0.0: continue

                        a_g = t_gains[out] if a == out else 0
                        a_g_boost = (t_gains[out] * 2) if a == out else 0
                        true_gain = t_gains[out]
                        out_p_empirique = p_empirique_1D[t, out]

                        for bob_a in range(3):
                            p_bob = c_rep[bob_a]
                            if p_bob == 0.0: continue

                            bob_g = true_gain if bob_a == out else 0
                            jp_base = p_out * p_bob

                            new_gap_bob_norm = val_g1 + a_g - bob_g
                            new_gap_bob_boost = val_g1 + a_g_boost - bob_g

                            esp_peloton_b0 = 0.0
                            esp_peloton_keep = 0.0
                            esp_peloton_use = 0.0

                            for delta_gain in range(max_gain):
                                p_peloton = out_p_empirique[delta_gain]
                                if p_peloton == 0.0: continue

                                new_gap_peloton_norm = val_g2 + a_g - delta_gain
                                new_gap_peloton_boost = val_g2 + a_g_boost - delta_gain

                                val_g1_final_norm = min(new_gap_bob_norm, new_gap_peloton_norm)
                                val_g2_final_norm = alpha * max(new_gap_bob_norm, new_gap_peloton_norm) + (1.0 - alpha) * new_gap_peloton_norm

                                val_g1_final_boost = min(new_gap_bob_boost, new_gap_peloton_boost)
                                val_g2_final_boost = alpha * max(new_gap_bob_boost, new_gap_peloton_boost) + (1.0 - alpha) * new_gap_peloton_boost

                                g1_norm = max(GAP_MIN, min(GAP_MAX, int(round(val_g1_final_norm)))) + GAP_OFFSET
                                g2_norm = max(GAP_MIN, min(GAP_MAX, int(round(val_g2_final_norm)))) + GAP_OFFSET
                                g1_boost = max(GAP_MIN, min(GAP_MAX, int(round(val_g1_final_boost)))) + GAP_OFFSET
                                g2_boost = max(GAP_MIN, min(GAP_MAX, int(round(val_g2_final_boost)))) + GAP_OFFSET

                                esp_peloton_b0 += p_peloton * V_next[g1_norm, g2_norm, 0]
                                esp_peloton_keep += p_peloton * V_next[g1_norm, g2_norm, 1]
                                esp_peloton_use += p_peloton * V_next[g1_boost, g2_boost, 0]

                            expected_v_b0 += jp_base * esp_peloton_b0
                            expected_v_keep += jp_base * esp_peloton_keep
                            expected_v_use += jp_base * esp_peloton_use

                    if store_q:
                        Q_tables[k, g1, g2, a, 0] = expected_v_b0
                        Q_tables[k, g1, g2, a, 1] = expected_v_keep
                        Q_tables[k, g1, g2, a, 2] = expected_v_use

                    if expected_v_b0 > best_v_b0: best_v_b0 = expected_v_b0
                    best_action_b1 = max(expected_v_keep, expected_v_use)
                    if best_action_b1 > best_v_b1: best_v_b1 = best_action_b1

                V_current[g1, g2, 0] = best_v_b0
                V_current[g1, g2, 1] = best_v_b1

        # Symétrie Safe
        for g1 in prange(GRID_SIZE):
            for g2 in range(g1 + 1, GRID_SIZE):
                V_current[g2, g1, 0] = V_current[g1, g2, 0]
                V_current[g2, g1, 1] = V_current[g1, g2, 1]
                if store_q:
                    for a in range(3):
                        Q_tables[k, g2, g1, a, 0] = Q_tables[k, g1, g2, a, 0]
                        Q_tables[k, g2, g1, a, 1] = Q_tables[k, g1, g2, a, 1]
                        Q_tables[k, g2, g1, a, 2] = Q_tables[k, g1, g2, a, 2]

        V_next = V_current
        V_all_matches[t] = V_current

    return V_all_matches, Q_tables


@njit(parallel=True)
def solve_dp_coarse(base_true_probas, base_crowds, gains_1N2, p_empirique_1D, alphas_isolement, V_horizon, stop_t=0, horizon_nuit=0, start_t=-1):
    """
    Version grille grossière (501×501, 2 pts/step) de solve_dp_with_full_empirical_distribution.
    Couvre le même range [-600, +400] mais avec résolution 2 pts → 4× plus rapide
    (moins d'états + V_next 2 MB au lieu de 8 MB → meilleure localité cache).

    V_horizon doit être de shape (501, 501, 2)  ← downsampler depuis (1001, 1001, 2) avec [::2, ::2].
    Retourne V_all_matches de shape (n_matches, 501, 501, 2).

    Convention : val_g = g - 300  représente le gap en "demi-points".
    Tous les gains sont divisés par 2 pour rester dans cet espace.
    start_t=-1  → run complet depuis n_matches-1, condition terminale appliquée.
    start_t>=0  → run partiel depuis start_t-1, V_horizon utilisé tel quel.
    """
    n_matches = base_true_probas.shape[0]
    max_gain = p_empirique_1D.shape[2]

    V_all_matches = np.zeros((n_matches, 501, 501, 2), dtype=np.float32)
    Q_tables = np.zeros((horizon_nuit, 501, 501, 3, 3), dtype=np.float32)
    V_next = np.copy(V_horizon)

    # 1. Condition Terminale (uniquement pour run complet depuis le début)
    if start_t < 0:
        for g1 in range(501):  # En dur pour la rapidité de Numba
            val_g1 = g1 - 300
            for g2 in range(g1, 501):  # En dur pour la rapidité de Numba
                val_g2 = g2 - 300

                if val_g1 > 0 and val_g2 > 0: win_prob = 1.0
                elif (val_g1 == 0 and val_g2 > 0) or (val_g1 > 0 and val_g2 == 0): win_prob = 0.5
                elif val_g1 == 0 and val_g2 == 0: win_prob = 1.0 / 3.0
                else: win_prob = 0.0

                V_next[g1, g2, 0] = win_prob
                V_next[g1, g2, 1] = win_prob

    # 2. Point de départ de la rétro-propagation
    if start_t < 0:
        t_debut = n_matches - 1
    else:
        t_debut = start_t - 1

    # 3. Rétro-propagation
    for t in range(t_debut, stop_t - 1, -1):
        V_current = np.zeros((501, 501, 2), dtype=np.float32)  # En dur pour la rapidité de Numba
        t_prob = base_true_probas[t]
        c_rep = base_crowds[t]
        t_gains = gains_1N2[t]
        alpha = alphas_isolement[t]
        store_q = horizon_nuit > 0 and t >= stop_t and t < stop_t + horizon_nuit
        k = t - stop_t

        for g1 in prange(501):  # En dur pour la rapidité de Numba
            val_g1 = g1 - 300   # Demi-points (= gap réel / 2)
            for g2 in range(g1, 501):  # En dur pour la rapidité de Numba
                val_g2 = g2 - 300

                best_v_b0 = 0.0
                best_v_b1 = 0.0

                for a in range(3):
                    expected_v_b0 = 0.0
                    expected_v_keep = 0.0
                    expected_v_use = 0.0

                    for out in range(3):
                        p_out = t_prob[out]
                        if p_out == 0.0: continue

                        # Gains divisés par 2 pour rester dans l'espace coarse (demi-points)
                        a_g       = (t_gains[out] / 2.0) if a == out else 0.0
                        a_g_boost = float(t_gains[out])  if a == out else 0.0  # = 2 × a_g
                        true_gain = t_gains[out] / 2.0
                        out_p_empirique = p_empirique_1D[t, out]

                        for bob_a in range(3):
                            p_bob = c_rep[bob_a]
                            if p_bob == 0.0: continue

                            bob_g = true_gain if bob_a == out else 0.0
                            jp_base = p_out * p_bob

                            new_gap_bob_norm  = val_g1 + a_g       - bob_g
                            new_gap_bob_boost = val_g1 + a_g_boost - bob_g

                            esp_peloton_b0   = 0.0
                            esp_peloton_keep = 0.0
                            esp_peloton_use  = 0.0

                            for delta_gain in range(max_gain):
                                p_peloton = out_p_empirique[delta_gain]
                                if p_peloton == 0.0: continue

                                dg_c = delta_gain / 2.0  # delta_gain divisé par 2 (espace coarse)
                                new_gap_peloton_norm  = val_g2 + a_g       - dg_c
                                new_gap_peloton_boost = val_g2 + a_g_boost - dg_c

                                val_g1_final_norm  = min(new_gap_bob_norm,  new_gap_peloton_norm)
                                val_g2_final_norm  = alpha * max(new_gap_bob_norm,  new_gap_peloton_norm) + (1.0 - alpha) * new_gap_peloton_norm

                                val_g1_final_boost = min(new_gap_bob_boost, new_gap_peloton_boost)
                                val_g2_final_boost = alpha * max(new_gap_bob_boost, new_gap_peloton_boost) + (1.0 - alpha) * new_gap_peloton_boost

                                # Clampage dans [0, 500] (grille 501)
                                g1_norm  = max(0, min(500, int(round(val_g1_final_norm))  + 300))
                                g2_norm  = max(0, min(500, int(round(val_g2_final_norm))  + 300))
                                g1_boost = max(0, min(500, int(round(val_g1_final_boost)) + 300))
                                g2_boost = max(0, min(500, int(round(val_g2_final_boost)) + 300))

                                esp_peloton_b0   += p_peloton * V_next[g1_norm,  g2_norm,  0]
                                esp_peloton_keep += p_peloton * V_next[g1_norm,  g2_norm,  1]
                                esp_peloton_use  += p_peloton * V_next[g1_boost, g2_boost, 0]

                            expected_v_b0   += jp_base * esp_peloton_b0
                            expected_v_keep += jp_base * esp_peloton_keep
                            expected_v_use  += jp_base * esp_peloton_use

                    if store_q:
                        Q_tables[k, g1, g2, a, 0] = expected_v_b0
                        Q_tables[k, g1, g2, a, 1] = expected_v_keep
                        Q_tables[k, g1, g2, a, 2] = expected_v_use

                    if expected_v_b0 > best_v_b0: best_v_b0 = expected_v_b0
                    best_action_b1 = max(expected_v_keep, expected_v_use)
                    if best_action_b1 > best_v_b1: best_v_b1 = best_action_b1

                V_current[g1, g2, 0] = best_v_b0
                V_current[g1, g2, 1] = best_v_b1

        # Symétrie Safe (triangle inférieur)
        for g1 in prange(501):  # En dur pour la rapidité de Numba
            for g2 in range(g1 + 1, 501):
                V_current[g2, g1, 0] = V_current[g1, g2, 0]
                V_current[g2, g1, 1] = V_current[g1, g2, 1]
                if store_q:
                    for a in range(3):
                        Q_tables[k, g2, g1, a, 0] = Q_tables[k, g1, g2, a, 0]
                        Q_tables[k, g2, g1, a, 1] = Q_tables[k, g1, g2, a, 1]
                        Q_tables[k, g2, g1, a, 2] = Q_tables[k, g1, g2, a, 2]

        V_next = V_current
        V_all_matches[t] = V_current

    return V_all_matches, Q_tables


@njit(parallel=True)
def compute_full_Q_table(t, t_prob, c_rep, t_gains, p_empirique_1D, alpha, V_next, max_gain):
    
    Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 3, 3), dtype=np.float32)
    
    for g1 in prange(GRID_SIZE):
        val_g1 = g1 - GAP_OFFSET
        for g2 in range(g1, GRID_SIZE):
            val_g2 = g2 - GAP_OFFSET
            
            for a in range(3):
                expected_v_b0 = 0.0
                expected_v_keep = 0.0
                expected_v_use = 0.0
                
                for out in range(3):
                    p_out = t_prob[out]
                    if p_out == 0.0: continue
                    
                    a_g = t_gains[out] if a == out else 0
                    a_g_boost = (t_gains[out] * 2) if a == out else 0
                    true_gain = t_gains[out]
                    
                    # 🚀 CORRECTION ICI : On utilise bien [t, out] pour cibler la matrice complète en 3D
                    out_p_empirique = p_empirique_1D[t, out] 
                    
                    for bob_a in range(3):
                        p_bob = c_rep[bob_a]
                        if p_bob == 0.0: continue
                        
                        bob_g = true_gain if bob_a == out else 0
                        jp_base = p_out * p_bob
                        
                        new_gap_bob_norm = val_g1 + a_g - bob_g
                        new_gap_bob_boost = val_g1 + a_g_boost - bob_g
                        
                        esp_peloton_b0 = 0.0
                        esp_peloton_keep = 0.0
                        esp_peloton_use = 0.0
                        
                        for delta_gain in range(max_gain):
                            p_peloton = out_p_empirique[delta_gain]
                            if p_peloton == 0.0: continue
                            
                            new_gap_peloton_norm = val_g2 + a_g - delta_gain
                            new_gap_peloton_boost = val_g2 + a_g_boost - delta_gain
                            
                            # La formule de Gating Temporel
                            val_g1_final_norm = min(new_gap_bob_norm, new_gap_peloton_norm)
                            val_g2_final_norm = alpha * max(new_gap_bob_norm, new_gap_peloton_norm) + (1.0 - alpha) * new_gap_peloton_norm
                            
                            val_g1_final_boost = min(new_gap_bob_boost, new_gap_peloton_boost)
                            val_g2_final_boost = alpha * max(new_gap_bob_boost, new_gap_peloton_boost) + (1.0 - alpha) * new_gap_peloton_boost
                            
                            g1_norm = max(GAP_MIN, min(GAP_MAX, int(round(val_g1_final_norm)))) + GAP_OFFSET
                            g2_norm = max(GAP_MIN, min(GAP_MAX, int(round(val_g2_final_norm)))) + GAP_OFFSET
                            g1_boost = max(GAP_MIN, min(GAP_MAX, int(round(val_g1_final_boost)))) + GAP_OFFSET
                            g2_boost = max(GAP_MIN, min(GAP_MAX, int(round(val_g2_final_boost)))) + GAP_OFFSET
                            
                            esp_peloton_b0 += p_peloton * V_next[g1_norm, g2_norm, 0]
                            esp_peloton_keep += p_peloton * V_next[g1_norm, g2_norm, 1]
                            esp_peloton_use += p_peloton * V_next[g1_boost, g2_boost, 0]
                            
                        expected_v_b0 += jp_base * esp_peloton_b0
                        expected_v_keep += jp_base * esp_peloton_keep
                        expected_v_use += jp_base * esp_peloton_use
                
                Q_table[g1, g2, a, 0] = expected_v_b0
                Q_table[g1, g2, a, 1] = expected_v_keep
                Q_table[g1, g2, a, 2] = expected_v_use
                
    # Symétrie
    for g1 in prange(GRID_SIZE):
        for g2 in range(g1 + 1, GRID_SIZE):
            for a in range(3):
                Q_table[g2, g1, a, 0] = Q_table[g1, g2, a, 0]
                Q_table[g2, g1, a, 1] = Q_table[g1, g2, a, 1]
                Q_table[g2, g1, a, 2] = Q_table[g1, g2, a, 2]
                
    return Q_table


if __name__ == "__main__":
    compute_expected_V_phases_finales(8000)