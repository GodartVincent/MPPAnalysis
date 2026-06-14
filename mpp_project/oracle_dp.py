import numpy as np
from numba import njit, prange

# --- 1. CONFIGURATION & CONSTANTES ---
N_MATCHES = 104

GAP_MIN = -600
GAP_MAX = 400
GAP_OFFSET = -GAP_MIN  # 600
GRID_SIZE = GAP_MAX - GAP_MIN + 1  # 1001

N_ACTIONS = 3

# --- Effectifs de la ligue (source UNIQUE : tout le code importe ces constantes
#     depuis oracle_dp plutôt que d'écrire des nombres en dur). ---
N_PLAYERS = 9                   # Taille de la ligue MPP.
COEFF_ROBUSTESSE = 1.25         # Gonflement du peloton simulé pour absorber la variance
                                # des bonus (favori/buteur ~ x2). cf. NB15/16.
N_PELOTON = N_PLAYERS - 1                                 # 8 : peloton réel (on retire le leader/Bob).
N_PELOTON_ROBUSTE = round(N_PELOTON * COEFF_ROBUSTESSE)   # 10 : peloton gonflé pour le Monte-Carlo.

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

def extract_peloton_full_distribution(true_probas_3d, crowds_3d, gains_1N2, max_gain=250,
                                      n_runs=1000000, n_players=N_PELOTON, known_outcomes=None):
    """
    Histogrammes de transition de gap_2 (vitesse du peloton) par match et par issue,
    estimés par Monte-Carlo (n_runs tournois). Wrapper autour du cœur Numba.

    known_outcomes : array (n_matches,) optionnel.
      Pour les matchs DÉJÀ JOUÉS (mode horizon glissant en milieu de poules), fournir
      l'issue RÉELLE (0 = '1', 1 = 'N', 2 = '2') ; -1 = match futur -> issue tirée
      aléatoirement (comportement par défaut). Fixer les issues passées rend la
      trajectoire de score du peloton réaliste jusqu'au présent, donc les histogrammes
      des matchs FUTURS plus représentatifs. Les paris des joueurs restent tirés selon
      `crowds` (on ne connaît pas les paris réels de la ligue).
    """
    n_matches = true_probas_3d.shape[1]
    if known_outcomes is None:
        known_outcomes = np.full(n_matches, -1, dtype=np.int64)
    else:
        known_outcomes = np.ascontiguousarray(known_outcomes, dtype=np.int64)
    return _extract_peloton_full_distribution_core(
        true_probas_3d, crowds_3d, gains_1N2, known_outcomes, max_gain, n_runs, n_players
    )


@njit
def _extract_peloton_full_distribution_core(true_probas_3d, crowds_3d, gains_1N2,
                                            known_outcomes, max_gain, n_runs, n_players):
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

            # 3. Issue du match : RÉELLE si connue (match passé), sinon tirée au sort
            ko = known_outcomes[t]
            if ko >= 0:
                true_out = ko
            else:
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

    V_horizon doit être de shape (501, 501, 2)  <- downsampler depuis (1001, 1001, 2) avec [::2, ::2].
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


# ==========================================================================
# DP coarse EXACT-AWARE : scores exacts sur les matchs renseignés, 1N2 sinon
# ==========================================================================
@njit
def _lk_coarse(val_g1, val_g2, ap, bob_g, pel_delta, alpha, V_next, layer):
    """Lookup coarse (demi-points, clamp [0,500], offset 300) — cf. solve_dp_coarse."""
    ngb = val_g1 + ap - bob_g
    ngp = val_g2 + ap - pel_delta
    vg1 = min(ngb, ngp)
    vg2 = alpha * max(ngb, ngp) + (1.0 - alpha) * ngp
    i1 = max(0, min(500, int(round(vg1)) + 300))
    i2 = max(0, min(500, int(round(vg2)) + 300))
    return V_next[i1, i2, layer]


@njit
def _esp_coarse_exact(val_g1, val_g2, ap_norm, ap_boost, o, cc, b_half,
                      true_gain, crowd, hist, alpha, V_next):
    """
    Sommes Bob×peloton (coarse) de la valeur future pour des points agent fixés
    (ap_norm pour base/keep, ap_boost = 2·ap_norm pour use). Bob et le peloton
    peuvent décrocher le bonus du score RÉEL (b_half, déjà /2) avec proba cc.
    Renvoie (esp_b0, esp_keep, esp_use). Avec cc=0 et b_half=0 -> transition 1N2.
    """
    max_gain = hist.shape[0]
    esp_b0 = 0.0
    esp_keep = 0.0
    esp_use = 0.0
    for bob_a in range(3):
        p_bob = crowd[bob_a]
        if p_bob == 0.0:
            continue
        bob_correct = (bob_a == o)
        for bob_branch in range(2):
            if bob_correct:
                if bob_branch == 0:
                    p_bb = p_bob * cc
                    bob_g = true_gain + b_half
                else:
                    p_bb = p_bob * (1.0 - cc)
                    bob_g = true_gain
            else:
                if bob_branch == 1:
                    continue
                p_bb = p_bob
                bob_g = 0.0
            if p_bb == 0.0:
                continue
            for dg in range(max_gain):
                p_pel = hist[dg]
                if p_pel == 0.0:
                    continue
                dgc = dg / 2.0
                if dg != 0:
                    for pel_branch in range(2):
                        if pel_branch == 0:
                            p_pb = p_pel * cc
                            pel_delta = dgc + b_half
                        else:
                            p_pb = p_pel * (1.0 - cc)
                            pel_delta = dgc
                        if p_pb == 0.0:
                            continue
                        w = p_bb * p_pb
                        esp_b0 += w * _lk_coarse(val_g1, val_g2, ap_norm, bob_g, pel_delta, alpha, V_next, 0)
                        esp_keep += w * _lk_coarse(val_g1, val_g2, ap_norm, bob_g, pel_delta, alpha, V_next, 1)
                        esp_use += w * _lk_coarse(val_g1, val_g2, ap_boost, bob_g, pel_delta, alpha, V_next, 0)
                else:
                    w = p_bb * p_pel
                    pel_delta = dgc  # = 0.0
                    esp_b0 += w * _lk_coarse(val_g1, val_g2, ap_norm, bob_g, pel_delta, alpha, V_next, 0)
                    esp_keep += w * _lk_coarse(val_g1, val_g2, ap_norm, bob_g, pel_delta, alpha, V_next, 1)
                    esp_use += w * _lk_coarse(val_g1, val_g2, ap_boost, bob_g, pel_delta, alpha, V_next, 0)
    return esp_b0, esp_keep, esp_use


@njit(parallel=True)
def solve_dp_coarse_exact(sc_outcome, sc_p, sc_cc, sc_bonus, sc_count,
                          gains_1N2, crowd_1N2, p_empirique_1D, alphas_isolement,
                          V_horizon, stop_t=0, start_t=-1):
    """
    Variante EXACT-AWARE de solve_dp_coarse (501², demi-points). Chaque match `t`
    porte K_t « scores » (paddés) : 1N2 -> 3 pseudo-scores (outcomes 0/1/2, bonus 0) ;
    match renseigné -> les scores exacts (outcome, p ancrée, cond_crowd, bonus). La
    transition est celle de evaluate_exact_score_day, sur toute la grille.

    Tableaux paddés (n_matches, Kmax) : sc_outcome (int8), sc_p (f8), sc_cc (f8),
    sc_bonus (f8, points PLEINS — divisés par 2 en interne), sc_count (n,) int.
    gains_1N2 (n,3) PLEINS (/2 interne), crowd_1N2 (n,3), p_empirique_1D (n,3,max_gain),
    alphas_isolement (n,), V_horizon (501,501,2). Conventions start_t IDENTIQUES à
    solve_dp_coarse (PIÈGE : start_t=n_matches pour CHAÎNER un horizon sans l'écraser).

    Optimisation (effondre le facteur K_action, exact) : pour chaque score RÉEL ks,
    les points agent ne prennent que 3 valeurs (mauvais outcome / bon outcome raté /
    score exact) ; on calcule les sommes Bob×peloton pour ces 3 niveaux puis on
    assemble la valeur de chaque action.

    Renvoie V_all_matches (n_matches, 501, 501, 2).
    """
    n_matches = sc_outcome.shape[0]
    Kmax = sc_outcome.shape[1]

    V_all_matches = np.zeros((n_matches, 501, 501, 2), dtype=np.float32)
    V_next = np.copy(V_horizon)

    # 1. Condition terminale (run complet uniquement)
    if start_t < 0:
        for g1 in range(501):
            val_g1 = g1 - 300
            for g2 in range(g1, 501):
                val_g2 = g2 - 300
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

    if start_t < 0:
        t_debut = n_matches - 1
    else:
        t_debut = start_t - 1

    for t in range(t_debut, stop_t - 1, -1):
        V_current = np.zeros((501, 501, 2), dtype=np.float32)
        K = sc_count[t]
        alpha = alphas_isolement[t]
        gains_t = gains_1N2[t]
        crowd_t = crowd_1N2[t]
        hist_t = p_empirique_1D[t]

        for g1 in prange(501):
            val_g1 = g1 - 300
            # Buffers réutilisés sur la ligne g1 (alloués hors boucle g2)
            Ain = np.zeros((3, 3), dtype=np.float64)
            Bin_ = np.zeros((3, 3), dtype=np.float64)
            dC = np.zeros((Kmax, 3), dtype=np.float64)

            for g2 in range(g1, 501):
                val_g2 = g2 - 300

                for ii in range(3):
                    for jj in range(3):
                        Ain[ii, jj] = 0.0
                        Bin_[ii, jj] = 0.0
                totalA0 = 0.0
                totalA1 = 0.0
                totalA2 = 0.0

                for ks in range(K):
                    o = sc_outcome[t, ks]
                    p_s = sc_p[t, ks]
                    cc = sc_cc[t, ks]
                    b_half = sc_bonus[t, ks] / 2.0
                    gh = gains_t[o] / 2.0
                    hist_o = hist_t[o]

                    # case A : agent mauvais outcome -> AP=0
                    a0, ak, au = _esp_coarse_exact(val_g1, val_g2, 0.0, 0.0, o, cc, b_half,
                                                   gh, crowd_t, hist_o, alpha, V_next)
                    # case B : bon outcome, score raté -> AP=gh
                    b0, bk, bu = _esp_coarse_exact(val_g1, val_g2, gh, 2.0 * gh, o, cc, b_half,
                                                   gh, crowd_t, hist_o, alpha, V_next)
                    # case C : score exact -> AP=gh+b_half
                    apc = gh + b_half
                    c0, ck, cu = _esp_coarse_exact(val_g1, val_g2, apc, 2.0 * apc, o, cc, b_half,
                                                   gh, crowd_t, hist_o, alpha, V_next)

                    totalA0 += p_s * a0
                    totalA1 += p_s * ak
                    totalA2 += p_s * au
                    Ain[o, 0] += p_s * a0
                    Ain[o, 1] += p_s * ak
                    Ain[o, 2] += p_s * au
                    Bin_[o, 0] += p_s * b0
                    Bin_[o, 1] += p_s * bk
                    Bin_[o, 2] += p_s * bu
                    dC[ks, 0] = p_s * (c0 - b0)
                    dC[ks, 1] = p_s * (ck - bk)
                    dC[ks, 2] = p_s * (cu - bu)

                best_b0 = 0.0
                best_b1 = 0.0
                for ka in range(K):
                    ao = sc_outcome[t, ka]
                    v_b0 = totalA0 - Ain[ao, 0] + Bin_[ao, 0] + dC[ka, 0]
                    v_keep = totalA1 - Ain[ao, 1] + Bin_[ao, 1] + dC[ka, 1]
                    v_use = totalA2 - Ain[ao, 2] + Bin_[ao, 2] + dC[ka, 2]
                    if v_b0 > best_b0:
                        best_b0 = v_b0
                    bb1 = max(v_keep, v_use)
                    if bb1 > best_b1:
                        best_b1 = bb1

                V_current[g1, g2, 0] = best_b0
                V_current[g1, g2, 1] = best_b1

        # Symétrie (triangle inférieur)
        for g1 in prange(501):
            for g2 in range(g1 + 1, 501):
                V_current[g2, g1, 0] = V_current[g1, g2, 0]
                V_current[g2, g1, 1] = V_current[g1, g2, 1]

        V_next = V_current
        V_all_matches[t] = V_current

    return V_all_matches


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


# ==========================================================================
# 3. SCORES EXACTS (match du jour) — évaluateur DP mono-état
# ==========================================================================
@njit
def _lookup_V(val_g1, val_g2, agent_pts, bob_g, pel_delta, alpha, V_next, layer):
    """
    Gating temporel identique à compute_full_Q_table : à partir des gaps "réels"
    (val_g1 vs Bob, val_g2 vs peloton), applique les gains de l'agent (agent_pts),
    de Bob (bob_g) et le delta peloton (pel_delta), puis lit V_next[.,.,layer].
    """
    new_gap_bob = val_g1 + agent_pts - bob_g
    new_gap_pel = val_g2 + agent_pts - pel_delta

    val_g1_final = min(new_gap_bob, new_gap_pel)
    val_g2_final = alpha * max(new_gap_bob, new_gap_pel) + (1.0 - alpha) * new_gap_pel

    i1 = max(GAP_MIN, min(GAP_MAX, int(round(val_g1_final)))) + GAP_OFFSET
    i2 = max(GAP_MIN, min(GAP_MAX, int(round(val_g2_final)))) + GAP_OFFSET
    return V_next[i1, i2, layer]


@njit
def evaluate_exact_score_day(g1_idx, g2_idx, sc_outcome, sc_p, sc_cc, sc_bonus,
                             gains_1N2, crowd_1N2, p_empirique_day, alpha, V_next,
                             agent_bonus_factor=1.0):
    """
    Évalue, à l'état courant (g1_idx, g2_idx), le win rate de CHAQUE action de
    score exact pour le match du jour, en chaînant vers V_next (post-jour, fine
    1001x1001x2). Renvoie Q de shape (K, 3) : colonnes [base, keep, use].

    `agent_bonus_factor` (défaut 1.0) multiplie le bonus que l'AGENT touche quand
    son score sort. Mettre 0.0 donne le WR « si l'agent loupe son score exact »
    (il ne compte que sur le gain d'outcome) — Bob et le peloton gardent leur
    bonus dans tous les cas. Sert à afficher le plancher robuste du pari.

    Paramètres alignés (K = nb de scores listés) :
      sc_outcome (K,) issue 1N2 du score, sc_p (K,) proba vraie, sc_cc (K,) crowd
      conditionnel, sc_bonus (K,) bonus de points. gains_1N2/crowd_1N2 (3,) du match.
      p_empirique_day (3, max_gain) histogramme de transition gap_2 par issue.

    Modèle du bonus :
      - agent : touche bonus(score parié) avec certitude si son score == score réel ;
      - Bob : s'il a le bon outcome, touche bonus(score réel) avec proba cc (sinon non) ;
      - peloton : pour chaque bin de gain non nul, ajoute bonus(score réel) avec proba cc ;
      - booster x2 : double (gain_outcome + bonus) de l'AGENT uniquement.
    """
    K = sc_outcome.shape[0]
    max_gain = p_empirique_day.shape[1]
    Q = np.zeros((K, 3), dtype=np.float32)

    val_g1 = g1_idx - GAP_OFFSET
    val_g2 = g2_idx - GAP_OFFSET

    for ka in range(K):                       # action agent = parier le score ka
        a_o = sc_outcome[ka]
        a_bonus = sc_bonus[ka]
        v_base = 0.0
        v_keep = 0.0
        v_use = 0.0

        for ks in range(K):                   # score RÉEL = ks
            p_s = sc_p[ks]
            if p_s == 0.0:
                continue
            o = sc_outcome[ks]
            cc = sc_cc[ks]
            b_s = sc_bonus[ks]
            true_gain = gains_1N2[o]
            out_hist = p_empirique_day[o]

            # Points de l'agent : gain d'outcome si bon outcome, + bonus si score exact
            agent_outcome_gain = true_gain if a_o == o else 0.0
            agent_bonus = (a_bonus * agent_bonus_factor) if ka == ks else 0.0   # ka==ks => a_o==o
            agent_pts_norm = agent_outcome_gain + agent_bonus
            agent_pts_boost = 2.0 * agent_pts_norm

            for bob_a in range(3):
                p_bob = crowd_1N2[bob_a]
                if p_bob == 0.0:
                    continue
                bob_correct = (bob_a == o)

                # Branches du bonus de Bob : s'il a le bon outcome, 2 sous-cas
                # (proba cc : +bonus ; proba 1-cc : rien) ; sinon une seule branche.
                for bob_branch in range(2):
                    if bob_correct:
                        if bob_branch == 0:
                            p_bob_b = p_bob * cc
                            bob_g = true_gain + b_s
                        else:
                            p_bob_b = p_bob * (1.0 - cc)
                            bob_g = true_gain
                    else:
                        if bob_branch == 1:
                            continue              # une seule branche si mauvais outcome
                        p_bob_b = p_bob
                        bob_g = 0.0
                    if p_bob_b == 0.0:
                        continue

                    esp_base = 0.0
                    esp_keep = 0.0
                    esp_use = 0.0
                    for dg in range(max_gain):
                        p_pel = out_hist[dg]
                        if p_pel == 0.0:
                            continue

                        if dg != 0:
                            # 2 branches du bonus peloton
                            for pel_branch in range(2):
                                if pel_branch == 0:
                                    p_pb = p_pel * cc
                                    pel_delta = dg + b_s
                                else:
                                    p_pb = p_pel * (1.0 - cc)
                                    pel_delta = float(dg)
                                if p_pb == 0.0:
                                    continue
                                esp_base += p_pb * _lookup_V(val_g1, val_g2, agent_pts_norm,
                                                            bob_g, pel_delta, alpha, V_next, 0)
                                esp_keep += p_pb * _lookup_V(val_g1, val_g2, agent_pts_norm,
                                                            bob_g, pel_delta, alpha, V_next, 1)
                                esp_use += p_pb * _lookup_V(val_g1, val_g2, agent_pts_boost,
                                                            bob_g, pel_delta, alpha, V_next, 0)
                        else:
                            pel_delta = 0.0
                            esp_base += p_pel * _lookup_V(val_g1, val_g2, agent_pts_norm,
                                                         bob_g, pel_delta, alpha, V_next, 0)
                            esp_keep += p_pel * _lookup_V(val_g1, val_g2, agent_pts_norm,
                                                         bob_g, pel_delta, alpha, V_next, 1)
                            esp_use += p_pel * _lookup_V(val_g1, val_g2, agent_pts_boost,
                                                        bob_g, pel_delta, alpha, V_next, 0)

                    jp = p_s * p_bob_b
                    v_base += jp * esp_base
                    v_keep += jp * esp_keep
                    v_use += jp * esp_use

        Q[ka, 0] = v_base
        Q[ka, 1] = v_keep
        Q[ka, 2] = v_use

    return Q


