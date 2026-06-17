import numpy as np
from numba import njit, prange

# ==========================================
# 0. CONSTANTES DU MODÈLE
# ==========================================
GRID_SIZE = 501
GAP_OFFSET = 300
GAP_MIN = -GAP_OFFSET
GAP_MAX = GRID_SIZE - 1 - GAP_OFFSET

# ==========================================
# 1. INITIALISATION DE L'ÉTAT TERMINAL (LA FINALE EST JOUÉE)
# ==========================================
@njit
def build_terminal_state(
    points_victoire_my, points_victoire_bob, points_victoire_pack,
    scorer_gain_my, scorer_gain_bob, scorer_gain_pack, joint_scorer
):
    """
    Construit la matrice V terminale des phases finales (après la finale, t=32).

    Demi-points : la grille (501, offset 300) et solve_endgame_dp travaillent en
    DEMI-POINTS (gains /2). On divise donc TOUS les points (victoire + buteur) par 2
    pour rester dans le même espace (corrige un ancien décalage ×2).

    Bonus FAVORI (axes my_fav/bob_fav/pack_fav) : fav == 1 signifie que l'équipe-favori
    du joueur a gagné la finale -> il touche `points_victoire_*`. Ces axes sont pilotés
    DYNAMIQUEMENT par la DP (corrélés aux issues 1N2).

    Bonus BUTEUR (loi jointe `joint_scorer`, 8 cases) : distribution CONJOINTE des
    indicateurs (mon buteur top ?, buteur de Bob top ?, buteur du peloton top ?),
    estimée par Poisson Monte-Carlo pour l'arbre courant (cf.
    scorer_model.top_scorer_joint_probs). `joint_scorer[combo]` avec
    combo = my_top*4 + bob_top*2 + pack_top ; somme = 1. Les égalités (plusieurs picks
    meilleurs buteurs simultanément) sont nativement portées par la loi jointe.
    `scorer_gain_*` = gain MPP (PLEIN) du buteur parié par chaque joueur.
    """
    V_term = np.zeros((GRID_SIZE, GRID_SIZE, 2, 2, 2, 2), dtype=np.float32)

    sg_my_h = scorer_gain_my / 2.0
    sg_bob_h = scorer_gain_bob / 2.0
    sg_pack_h = scorer_gain_pack / 2.0

    for g1 in range(GRID_SIZE):
        val_g1 = g1 - GAP_OFFSET
        for g2 in range(GRID_SIZE):
            val_g2 = g2 - GAP_OFFSET

            for b in range(2):
                for my_fav in range(2):
                    for bob_fav in range(2):
                        for pack_fav in range(2):

                            # 1. Bonus de victoire finale (demi-points)
                            bonus_my = (points_victoire_my / 2.0) if my_fav == 1 else 0.0
                            bonus_bob = (points_victoire_bob / 2.0) if bob_fav == 1 else 0.0
                            bonus_pack = (points_victoire_pack / 2.0) if pack_fav == 1 else 0.0

                            base_gap1 = val_g1 + bonus_my - bonus_bob
                            base_gap2 = val_g2 + bonus_my - bonus_pack

                            expected_wr = 0.0

                            # 2. CONVOLUTION DE LA LOI JOINTE DU MEILLEUR BUTEUR
                            for combo in range(8):
                                p_c = joint_scorer[combo]
                                if p_c == 0.0:
                                    continue
                                my_top = (combo >> 2) & 1
                                bob_top = (combo >> 1) & 1
                                pack_top = combo & 1

                                add_my = sg_my_h if my_top == 1 else 0.0
                                add_bob = sg_bob_h if bob_top == 1 else 0.0
                                add_pack = sg_pack_h if pack_top == 1 else 0.0

                                final_gap1 = base_gap1 + add_my - add_bob
                                final_gap2 = base_gap2 + add_my - add_pack

                                if final_gap1 > 0 and final_gap2 > 0:
                                    expected_wr += p_c * 1.0
                                elif final_gap1 == 0 and final_gap2 > 0:
                                    expected_wr += p_c * 0.5
                                elif final_gap1 == 0 and final_gap2 == 0:
                                    expected_wr += p_c * (1.0 / 3.0)

                            V_term[g1, g2, b, my_fav, bob_fav, pack_fav] = expected_wr

    return V_term

# ==========================================
# 2. RÉSOLUTION DES CONFLITS (MATCH NUL & TIRS AU BUT)
# ==========================================
@njit
def get_expected_v_outcome(V_next, g1, g2, b, my_fav, bob_fav, pack_fav, my_role, bob_role, pack_role, out):
    """
    Lit la valeur de la grille V_next en mettant à jour la survie des favoris.
    Gère nativement les tirs au but parfaitement corrélés si 2 favoris s'affrontent.
    """
    if out == 0 or out == 2: # Victoire dans le temps réglementaire d'une équipe
        next_my = my_fav
        if my_role != -1: next_my = 1 if out == my_role else 0
            
        next_bob = bob_fav
        if bob_role != -1: next_bob = 1 if out == bob_role else 0
            
        next_pack = pack_fav
        if pack_role != -1: next_pack = 1 if out == pack_role else 0
            
        return V_next[g1, g2, b, next_my, next_bob, next_pack]
    
    else: # MATCH NUL (out == 1) -> Tirs au but
        
        # Qui est à Domicile (0) et qui est à l'Extérieur (2) ?
        home_my = (my_role == 0); away_my = (my_role == 2)
        home_bob = (bob_role == 0); away_bob = (bob_role == 2)
        home_pack = (pack_role == 0); away_pack = (pack_role == 2)
        
        # Scénario A : L'équipe à Domicile gagne les TAB (50%)
        next_my_A = 1 if home_my else (0 if away_my else my_fav)
        next_bob_A = 1 if home_bob else (0 if away_bob else bob_fav)
        next_pack_A = 1 if home_pack else (0 if away_pack else pack_fav)
        
        # Scénario B : L'équipe à l'Extérieur gagne les TAB (50%)
        next_my_B = 1 if away_my else (0 if home_my else my_fav)
        next_bob_B = 1 if away_bob else (0 if home_bob else bob_fav)
        next_pack_B = 1 if away_pack else (0 if home_pack else pack_fav)
        
        vA = V_next[g1, g2, b, next_my_A, next_bob_A, next_pack_A]
        vB = V_next[g1, g2, b, next_my_B, next_bob_B, next_pack_B]
        
        return 0.5 * vA + 0.5 * vB

# ==========================================
# 2. LE MOTEUR DP OPTIMISÉ (VECTORISATION + GRILLE 501)
# ==========================================
@njit(parallel=True)
def solve_endgame_dp(match_probs, crowds, gains_1N2, p_empirique_1D, alphas, 
                     my_fav_roles, bob_fav_roles, pack_fav_roles, V_terminal, stop_t=0):
    
    max_gain = p_empirique_1D.shape[2]
    
    # Historique complet 7D
    V_history = np.zeros((32, 501, 501, 2, 2, 2, 2), dtype=np.float32)
    V_next = V_terminal
    
    for t in range(31, stop_t - 1, -1):
        V_current = np.zeros((501, 501, 2, 2, 2, 2), dtype=np.float32)
        
        t_prob = match_probs[t]
        c_rep = crowds[t]
        t_gains = gains_1N2[t]
        alpha = alphas[t]
        
        my_r = my_fav_roles[t]
        bob_r = bob_fav_roles[t]
        pack_r = pack_fav_roles[t]
        
        for g1 in prange(501): # En dur pour la rapidité de numba
            # ==============================================================
            # OPTIMISATION EXTRÊME : Allocation "Thread-Local" UNE SEULE FOIS
            # ==============================================================
            best_wr_b1 = np.zeros((2, 2, 2), dtype=np.float32)
            best_wr_b0 = np.zeros((2, 2, 2), dtype=np.float32)
            
            exp_keep = np.zeros((2, 2, 2), dtype=np.float32)
            exp_use  = np.zeros((2, 2, 2), dtype=np.float32)
            exp_b0   = np.zeros((2, 2, 2), dtype=np.float32)
            
            next_my_A = np.zeros(2, dtype=np.int32)
            next_my_B = np.zeros(2, dtype=np.int32)
            next_bob_A = np.zeros(2, dtype=np.int32)
            next_bob_B = np.zeros(2, dtype=np.int32)
            next_pack_A = np.zeros(2, dtype=np.int32)
            next_pack_B = np.zeros(2, dtype=np.int32)
            # ==============================================================
            
            val_g1 = g1 - 300 # GAP_OFFSET
            for g2 in range(501): # En dur pour la rapidité de numba
                val_g2 = g2 - 300
                
                # On recycle la mémoire ! (Très rapide)
                best_wr_b1.fill(0.0)
                best_wr_b0.fill(0.0)
                
                for a in range(3):
                    # On recycle les accumulateurs d'espérance
                    exp_keep.fill(0.0)
                    exp_use.fill(0.0)
                    exp_b0.fill(0.0)
                    
                    for out in range(3):
                        p_out = t_prob[out]
                        if p_out == 0.0: continue
                        
                        a_g = (t_gains[out] / 2.0) if a == out else 0
                        a_g_boost = float(t_gains[out]) if a == out else 0
                        true_gain = t_gains[out] / 2.0
                        
                        out_p_empirique = p_empirique_1D[t, out]
                        
                        for f in range(2):
                            if out != 1:
                                next_my_A[f] = 1 if out == my_r else (0 if my_r != -1 else f)
                                next_bob_A[f] = 1 if out == bob_r else (0 if bob_r != -1 else f)
                                next_pack_A[f] = 1 if out == pack_r else (0 if pack_r != -1 else f)
                                next_my_B[f] = next_my_A[f]
                                next_bob_B[f] = next_bob_A[f]
                                next_pack_B[f] = next_pack_A[f]
                            else:
                                next_my_A[f] = 1 if my_r == 0 else (0 if my_r == 2 else f)
                                next_my_B[f] = 1 if my_r == 2 else (0 if my_r == 0 else f)
                                next_bob_A[f] = 1 if bob_r == 0 else (0 if bob_r == 2 else f)
                                next_bob_B[f] = 1 if bob_r == 2 else (0 if bob_r == 0 else f)
                                next_pack_A[f] = 1 if pack_r == 0 else (0 if pack_r == 2 else f)
                                next_pack_B[f] = 1 if pack_r == 2 else (0 if pack_r == 0 else f)
                        
                        for bob_a in range(3):
                            p_bob = c_rep[bob_a]
                            if p_bob == 0.0: continue
                            
                            bob_g = true_gain if bob_a == out else 0
                            jp_base = p_out * p_bob
                            
                            new_g1_k = val_g1 + a_g - bob_g
                            new_g1_u = val_g1 + a_g_boost - bob_g
                            
                            for delta_gain in range(max_gain):
                                p_peloton = out_p_empirique[delta_gain]
                                if p_peloton == 0.0: continue
                                
                                weight = jp_base * p_peloton
                                delta_scaled = delta_gain / 2.0 
                                
                                new_g2_k = val_g2 + a_g - delta_scaled
                                new_g2_u = val_g2 + a_g_boost - delta_scaled
                                
                                val_g1_f_k = min(new_g1_k, new_g2_k)
                                val_g2_f_k = alpha * max(new_g1_k, new_g2_k) + (1.0 - alpha) * new_g2_k
                                val_g1_f_u = min(new_g1_u, new_g2_u)
                                val_g2_f_u = alpha * max(new_g1_u, new_g2_u) + (1.0 - alpha) * new_g2_u
                                
                                idx_g1_k = max(0, min(500, int(round(val_g1_f_k)) + 300))
                                idx_g2_k = max(0, min(500, int(round(val_g2_f_k)) + 300))
                                idx_g1_u = max(0, min(500, int(round(val_g1_f_u)) + 300))
                                idx_g2_u = max(0, min(500, int(round(val_g2_f_u)) + 300))
                                
                                for my_f in range(2):
                                    nmA = next_my_A[my_f]; nmB = next_my_B[my_f]
                                    for bob_f in range(2):
                                        nbA = next_bob_A[bob_f]; nbB = next_bob_B[bob_f]
                                        for pack_f in range(2):
                                            npA = next_pack_A[pack_f]; npB = next_pack_B[pack_f]
                                            
                                            if out != 1:
                                                v_k = V_next[idx_g1_k, idx_g2_k, 1, nmA, nbA, npA]
                                                v_u = V_next[idx_g1_u, idx_g2_u, 0, nmA, nbA, npA]
                                                v_0 = V_next[idx_g1_k, idx_g2_k, 0, nmA, nbA, npA]
                                            else:
                                                v_k = 0.5 * V_next[idx_g1_k, idx_g2_k, 1, nmA, nbA, npA] + 0.5 * V_next[idx_g1_k, idx_g2_k, 1, nmB, nbB, npB]
                                                v_u = 0.5 * V_next[idx_g1_u, idx_g2_u, 0, nmA, nbA, npA] + 0.5 * V_next[idx_g1_u, idx_g2_u, 0, nmB, nbB, npB]
                                                v_0 = 0.5 * V_next[idx_g1_k, idx_g2_k, 0, nmA, nbA, npA] + 0.5 * V_next[idx_g1_k, idx_g2_k, 0, nmB, nbB, npB]
                                                
                                            exp_keep[my_f, bob_f, pack_f] += weight * v_k
                                            exp_use[my_f, bob_f, pack_f]  += weight * v_u
                                            exp_b0[my_f, bob_f, pack_f]   += weight * v_0
                                            
                    for my_f in range(2):
                        for bob_f in range(2):
                            for pack_f in range(2):
                                if exp_b0[my_f, bob_f, pack_f] > best_wr_b0[my_f, bob_f, pack_f]:
                                    best_wr_b0[my_f, bob_f, pack_f] = exp_b0[my_f, bob_f, pack_f]
                                
                                val_b1 = max(exp_keep[my_f, bob_f, pack_f], exp_use[my_f, bob_f, pack_f])
                                if val_b1 > best_wr_b1[my_f, bob_f, pack_f]:
                                    best_wr_b1[my_f, bob_f, pack_f] = val_b1
                                    
                for my_f in range(2):
                    for bob_f in range(2):
                        for pack_f in range(2):
                            V_current[g1, g2, 0, my_f, bob_f, pack_f] = best_wr_b0[my_f, bob_f, pack_f]
                            V_current[g1, g2, 1, my_f, bob_f, pack_f] = best_wr_b1[my_f, bob_f, pack_f]
                            
        V_next = V_current
        V_history[t] = V_current.copy()
        
    return V_history