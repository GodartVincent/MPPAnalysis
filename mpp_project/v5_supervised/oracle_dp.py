import numpy as np
import time
from numba import njit, prange

# --- IMPORTS DE TON PROJET ---
from mpp_project.mpp_env import MppEnv
from mpp_project.strategies import strat_noisy_typical_opponent
from mpp_project.core import RMSE_GAINS, apply_heteroscedastic_noise, apply_temporal_drift, estimate_crowd_3D, calculate_mpp_gains

# --- 1. CONFIGURATION & CONSTANTES ---
N_MATCHES = 104
N_MATCHES_POULES = 72
N_MATCHES_FINALES = N_MATCHES - N_MATCHES_POULES  # 32 matchs

GAP_MIN = -600
GAP_MAX = 400
GAP_OFFSET = -GAP_MIN  # 600
GRID_SIZE = GAP_MAX - GAP_MIN + 1  # 1001

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


def generate_tournament_environment(env_config=None):
    """
    Génère un tournoi complet. Nous utiliserons les 32 derniers matchs.
    """
    if env_config is None:
        env_config = {}
        
    env = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                 num_random_opponents=0, 
                 use_domain_randomization=False,
                 use_winner_reward=False)
    env.reset()
    env._generate_tournament()
    
    true_probas = env.outcome_probas 
    mpp_gains = env.match_gains
    mpp_probas = env.outcome_probas 
    
    # Appel de ta stratégie pour récupérer la matrice (104, 3) bruitée
    crowd_repartitions = strat_noisy_typical_opponent(
        [], [], 
        mpp_probas=mpp_probas, 
        opp_repartition=env.opp_repartition, 
        player_scores=None, 
        my_idx=0
    )
    
    return true_probas, mpp_probas, mpp_gains, crowd_repartitions

# --- 2. L'ORACLE C++ (NUMBA) ---
# 2. Ajout du paramètre parallel=True
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


# --- 2. FONCTION D'INFÉRENCE LOCALE (Notebook 10) ---
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

# --- 3. CALCUL DE L'ESPÉRANCE (MONTE CARLO) ---
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
def backpropagate_group_stages_stochastic(match_du_jour_idx, probas_poules, gains_poules, ensemble_crowds, V_phases_finales):
    """
    Rétropropage la DP en intégrant l'incertitude temporelle de la foule.
    ensemble_crowds : Tenseur de forme (N_ensembles, N_poules, 3)
    """
    V_next = np.copy(V_phases_finales)
    
    N_POULES = len(probas_poules)
    N_ENS = ensemble_crowds.shape[0]
    
    # On recule jusqu'au match du jour (inutile de calculer le passé)
    for t in range(N_POULES - 1, match_du_jour_idx - 1, -1):
        V_current = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        
        time_fraction = 1.0 - (t / N_MATCHES)
        N_eff = 1.0 + (N_PLAYERS - 3.0) * (time_fraction ** 5.50)
        
        t_prob = probas_poules[t]
        t_gain = gains_poules[t]
        
        for g1 in prange(GRID_SIZE):
            val_g1 = g1 - GAP_OFFSET
            for g2 in range(g1, GRID_SIZE):
                val_g2 = g2 - GAP_OFFSET
                
                best_v_b0 = 0.0
                best_v_b1 = 0.0
                
                for a in range(3):
                    # Les accumulateurs moyennés sur les N_ENS univers de foule
                    sum_expected_v_b0 = 0.0
                    sum_expected_v_keep = 0.0
                    sum_expected_v_use = 0.0
                    
                    # --- LA BOUCLE MONTE CARLO EST ICI (Très rapide en C++) ---
                    for e in range(N_ENS):
                        c_rep = ensemble_crowds[e, t]
                        
                        expected_v_b0_e = 0.0
                        expected_v_keep_e = 0.0
                        expected_v_use_e = 0.0
                        
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
                                ng_min_norm = max(-600, min(400, min(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_hit)))
                                ng_max_norm = max(-600, min(400, max(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_hit)))
                                ng_min_boost = max(-600, min(400, min(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_hit)))
                                ng_max_boost = max(-600, min(400, max(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_hit)))
                                
                                expected_v_b0_e   += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 0]
                                expected_v_keep_e += jp_hit * V_next[ng_min_norm + GAP_OFFSET, ng_max_norm + GAP_OFFSET, 1]
                                expected_v_use_e  += jp_hit * V_next[ng_min_boost + GAP_OFFSET, ng_max_boost + GAP_OFFSET, 0]
                                
                                # MISS
                                jp_miss = p_out * p_bob * (1.0 - prob_pack_hits)
                                ng_min_norm_m = max(-600, min(400, min(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_miss)))
                                ng_max_norm_m = max(-600, min(400, max(val_g1 + a_g - bob_g, val_g2 + a_g - pack_g_miss)))
                                ng_min_boost_m = max(-600, min(400, min(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_miss)))
                                ng_max_boost_m = max(-600, min(400, max(val_g1 + a_g_boost - bob_g, val_g2 + a_g_boost - pack_g_miss)))
                                
                                expected_v_b0_e   += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 0]
                                expected_v_keep_e += jp_miss * V_next[ng_min_norm_m + GAP_OFFSET, ng_max_norm_m + GAP_OFFSET, 1]
                                expected_v_use_e  += jp_miss * V_next[ng_min_boost_m + GAP_OFFSET, ng_max_boost_m + GAP_OFFSET, 0]
                        
                        sum_expected_v_b0 += expected_v_b0_e
                        sum_expected_v_keep += expected_v_keep_e
                        sum_expected_v_use += expected_v_use_e
                        
                    # L'espérance réelle : la moyenne sur les N_ENS univers
                    avg_v_b0 = sum_expected_v_b0 / N_ENS
                    avg_v_keep = sum_expected_v_keep / N_ENS
                    avg_v_use = sum_expected_v_use / N_ENS
                    
                    if avg_v_b0 > best_v_b0:
                        best_v_b0 = avg_v_b0
                        
                    best_action_b1 = max(avg_v_keep, avg_v_use)
                    if best_action_b1 > best_v_b1:
                        best_v_b1 = best_action_b1
                        
                V_current[g1, g2, 0] = best_v_b0
                V_current[g1, g2, 1] = best_v_b1
                
        for g1 in prange(GRID_SIZE):
            for g2 in range(g1 + 1, GRID_SIZE):
                V_current[g2, g1, 0] = V_current[g1, g2, 0]
                V_current[g2, g1, 1] = V_current[g1, g2, 1]
                
        V_next = V_current
        
    return V_current

if __name__ == "__main__":
    compute_expected_V_phases_finales(8000)