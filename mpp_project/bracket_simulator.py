import numpy as np
import pandas as pd
import time
import warnings
from numba import njit
from pathlib import Path
import sys

# --- IMPORTS DU PROJET ---
from mpp_project.core import MAX_TRUE_PROBA, MIN_TRUE_PROBA, apply_temporal_drift, estimate_crowd_3D, calculate_mpp_gains, calculate_true_outcome_probas_from_odds, calibrate_qualification_probs, normalize_team_name

# Nombre d'équipes qualifiées en phase à élimination directe (CDM 2026 : 48 -> 32).
N_QUALIFIED = 32


def devig_group_odds(df_odds):
    """
    Probabilités VRAIES (dé-viggées) par équipe, alignées sur l'ordre des lignes de df_odds :
      - victoire finale : Shin sur cote_victoire (marché à vainqueur unique, somme = 1) ;
      - qualification   : calibrate_qualification_probs sur 1/cote_qualif (multi-vainqueurs,
                          somme = N_QUALIFIED).
    Remplace les 1/cote BRUTS (avec marge) utilisés auparavant pour la force conditionnelle
    et la survie (cf. conditional_matchup_prob).
    """
    true_cv = np.asarray(calculate_true_outcome_probas_from_odds(df_odds['cote_victoire'].values), dtype=np.float64)
    true_qualif = calibrate_qualification_probs(1.0 / df_odds['cote_qualif'].values, target_sum=float(N_QUALIFIED))
    return true_cv, true_qualif
# On importe le solveur qu'on a codé précédemment
from mpp_project.end_game_solver import solve_endgame_dp, build_terminal_state
from mpp_project.oracle_dp import estimate_mpp_crowd, rescale_histogram_1D, extract_peloton_full_distribution, N_PELOTON, N_PELOTON_ROBUSTE
from mpp_project.scorer_model import top_scorer_joint_probs, load_scorer_players

# ==========================================
# 1. PARSING ET PRÉPARATION DES DONNÉES
# ==========================================
def load_and_prepare_data(odds_path, market_path):
    """
    Lit les CSV, gère le buteur 'Autre', et prépare les dictionnaires.
    """
    # 1. Lecture des Groupes
    df_odds = pd.read_csv(odds_path)
    
    # 2. Lecture des Buteurs et Favoris (Logique issue du Notebook 16)
    df_market = pd.read_csv(market_path)
    
    df_fav = df_market[df_market['category'] == 'favorite'].copy()
    df_fav['true_proba'] = calculate_true_outcome_probas_from_odds(df_fav['cote'].values)
    
    df_sco = df_market[df_market['category'] == 'scorer'].copy()
    autre_mask = df_sco['selection'].str.lower() == 'autre'
    df_sco_autre = df_sco[autre_mask].copy()
    df_sco_individuals = df_sco[~autre_mask].copy()
    
    df_sco_individuals['true_proba'] = calculate_true_outcome_probas_from_odds(df_sco_individuals['cote'].values)
    
    mpp_scorers = df_sco_individuals[df_sco_individuals['gain_mpp'].notna()].copy()
    ghost_scorers = df_sco_individuals[df_sco_individuals['gain_mpp'].isna()].copy()
    proba_autre = ghost_scorers['true_proba'].sum()
    
    if not df_sco_autre.empty:
        df_sco_autre['true_proba'] = proba_autre
        df_mpp_scorers = pd.concat([mpp_scorers, df_sco_autre], ignore_index=True)
    else:
        df_mpp_scorers = mpp_scorers.copy()
        
    # Calcul du Crowd Biaisé
    df_fav['estimated_crowd'] = estimate_mpp_crowd(df_fav['selection'].str.lower(), df_fav['true_proba'].values, df_fav['gain_mpp'].values)
    df_mpp_scorers['estimated_crowd'] = estimate_mpp_crowd(df_mpp_scorers['selection'].str.lower(), df_mpp_scorers['true_proba'].values, df_mpp_scorers['gain_mpp'].values)
    
    return df_odds, df_fav, df_mpp_scorers

def precompute_diluted_peloton(df_poules, df_odds_finales, n_brackets=10, n_runs=100_000):
    """
    Génère la distribution du peloton de référence en lissant plusieurs arbres.
    Force l'ordre conceptuel [Favori, Nul, Outsider] pour garantir un rescaling parfait.
    """
    print("Pré-calcul de la dilution temporelle du peloton (Historique complet)...")
    
    # 1. Chargement des 72 matchs de poules (Les probas et foules réelles du tournoi)
    # df_poules est le CDM_2026.csv classique utilisé dans le Notebook 10
    cotes_poules = df_poules[['cote_1', 'cote_N', 'cote_2']].values
    true_probas_poules = calculate_true_outcome_probas_from_odds(cotes_poules)
    crowds_poules = df_poules[['crowd_1', 'crowd_N', 'crowd_2']].values
    gains_poules = df_poules[['gain_mpp_1', 'gain_mpp_N', 'gain_mpp_2']].values
    
    # 2. Création de l'accumulateur pour l'Arbre Synthétique
    avg_probs = np.zeros((32, 3), dtype=np.float32)
    avg_gains = np.zeros((32, 3), dtype=np.float32)
    avg_crowds = np.zeros((32, 3), dtype=np.float32)
    
    for _ in range(n_brackets):
        match_probs, mpp_gains, crowds, _, _ = generate_bracket_scenario(df_odds_finales)
        
        for t in range(32):
            # Identification spatiale
            p1, p2 = match_probs[t, 0], match_probs[t, 2]
            idx_fav = 0 if p1 >= p2 else 2
            idx_out = 2 if p1 >= p2 else 0
            
            # On range TOUJOURS dans l'ordre : [0=Fav, 1=Nul, 2=Outsider]
            avg_probs[t, 0] += match_probs[t, idx_fav]
            avg_probs[t, 1] += match_probs[t, 1]
            avg_probs[t, 2] += match_probs[t, idx_out]
            
            avg_gains[t, 0] += mpp_gains[t, idx_fav]
            avg_gains[t, 1] += mpp_gains[t, 1]
            avg_gains[t, 2] += mpp_gains[t, idx_out]
            
            avg_crowds[t, 0] += crowds[t, idx_fav]
            avg_crowds[t, 1] += crowds[t, 1]
            avg_crowds[t, 2] += crowds[t, idx_out]
            
    # Moyenne
    avg_probs /= n_brackets
    avg_gains /= n_brackets
    avg_crowds /= n_brackets
    avg_gains_int = np.round(avg_gains).astype(np.int32)
    
    # 3. Concaténation (Poules + Arbre Synthétique Abstrait)
    full_probas = np.vstack((true_probas_poules, avg_probs))
    full_crowds = np.vstack((crowds_poules, avg_crowds))
    full_gains = np.vstack((gains_poules, avg_gains_int))
    
    full_probas_3d = np.expand_dims(full_probas, axis=0)
    full_crowds_3d = np.expand_dims(full_crowds, axis=0)
    
    # 4. Monte-Carlo Numba
    p_empirique_104 = extract_peloton_full_distribution(
        true_probas_3d=full_probas_3d,
        crowds_3d=full_crowds_3d,
        gains_1N2=full_gains,
        max_gain=250,
        n_runs=n_runs,
        n_players=N_PELOTON_ROBUSTE
    )
    
    # 5. On retourne les histogrammes des 32 derniers matchs, 
    # et les gains de référence correspondants (qui font déjà 32 lignes)
    return p_empirique_104[72:], avg_gains_int

# ==========================================
# 2. SIMULATION DU TABLEAU (MONTE CARLO FORWARD)
# ==========================================
def simulate_group_stages(df_odds, qualif_probs=None):
    """
    Simule la phase de poule avec un proxy Plackett-Luce et l'équation des 3èmes.
    Retourne les équipes qualifiées avec leurs slots.

    qualif_probs : probas de qualification VRAIES (dé-viggées) alignées sur df_odds
        (cf. devig_group_odds). Si None, calculées en interne pour rester cohérent
        avec generate_bracket_scenario. Utilisées pour la force des 3èmes (gamma).
    """
    if qualif_probs is None:
        qualif_probs = calibrate_qualification_probs(
            1.0 / df_odds['cote_qualif'].values, target_sum=float(N_QUALIFIED))

    df = df_odds.copy()
    df['_qualif_true'] = qualif_probs  # aligné sur l'ordre des lignes de df_odds
    groups = df.groupby('group')
    qualifiers_1st = []
    qualifiers_2nd = []
    third_places = []

    for name, group in groups:
        teams = group.to_dict('records')

        # Poids pour le tirage (basé sur la cote de finir 1er) : distribution
        # P(1er du groupe) sur 4 équipes, dé-viggée par Shin puis renormalisée
        # (np.random.choice exige une somme exacte de 1).
        p1_weights = calculate_true_outcome_probas_from_odds(
            np.array([t['cote_1er'] for t in teams]))
        p1_weights = p1_weights / p1_weights.sum()

        # Tirage séquentiel SANS REMISE (Proxy de Plackett-Luce)
        indices = np.random.choice(4, size=3, replace=False, p=p1_weights)

        # 1er et 2ème
        qualifiers_1st.append({'team': teams[indices[0]]['team'], 'slot': teams[indices[0]]['slot_1er']})
        qualifiers_2nd.append({'team': teams[indices[1]]['team'], 'slot': teams[indices[1]]['slot_2e']})

        # Le 3ème avec la force intrinsèque de son groupe (Gamma), sur probas dé-viggées
        p_qual = np.array([t['_qualif_true'] for t in teams])
        gamma_group = max(0.01, p_qual.sum() - 2.0) # Probabilité qu'un 3ème survive ici

        third_places.append({
            'team': teams[indices[2]]['team'],
            'gamma': gamma_group
        })
        
    # Tirage des 8 meilleurs 3èmes basé sur la force de leurs groupes
    third_weights = np.array([t['gamma'] for t in third_places])
    third_weights /= third_weights.sum()
    best_thirds_idx = np.random.choice(12, size=8, replace=False, p=third_weights)
    best_thirds = [third_places[i]['team'] for i in best_thirds_idx]
    
    return qualifiers_1st, qualifiers_2nd, best_thirds

def simulate_knockout_match_math(p_qualif_1, penalties_ratio, phase_str, 
                            known_probas_90=None, known_gains=None, known_crowd=None):
    """
    Cœur thermodynamique d'un match : Applique le bruit, le polynôme, les règles 120min et le drift.
    Accepte des paramètres 'known' pour écraser la simulation avec la réalité (utile en Phases Finales).
    """
    
    # --- 1. ESTIMATION À 90 MINUTES ---
    if known_probas_90 is not None:
        # La réalité dicte les probabilités
        p1_90, p_nul_90, p2_90 = known_probas_90
        p_qualif_1_eff = p1_90 + (p_nul_90 / 2.0)
        p_qualif_2_eff = 1.0 - p_qualif_1_eff
        probas_90 = known_probas_90
    else:
        # Le futur est simulé avec bruit et polynôme
        noise = np.random.normal(0, 0.03)
        p_q1 = np.clip(p_qualif_1 + noise, 10*MIN_TRUE_PROBA, MAX_TRUE_PROBA**4)
        p_q2 = 1.0 - p_q1
        
        delta_qualif = abs(p_q1 - p_q2)
        A, B, C = -0.2637, 0.0634, 0.2645 
        base_p_nul = A * (delta_qualif**2) + B * delta_qualif + C
        
        facteur_bruit = 1.0 + np.random.normal(0, 0.0680)
        p_nul_90 = np.clip(base_p_nul * facteur_bruit, 3*MIN_TRUE_PROBA, 0.4)
        
        p1_90 = max(MIN_TRUE_PROBA, p_q1 - p_nul_90 / 2.0)
        p2_90 = max(MIN_TRUE_PROBA, p_q2 - p_nul_90 / 2.0)
        sum_90 = p1_90 + p_nul_90 + p2_90
        
        # On re-normalise pour être sûr que la somme fasse exactement 1.0
        p1_90 /= sum_90
        p_nul_90 /= sum_90
        p2_90 /= sum_90
        
        probas_90 = np.array([p1_90, p_nul_90, p2_90])
        p_qualif_1_eff, p_qualif_2_eff = p_q1, p_q2

    # --- 2. RÈGLES MPP (120 MINUTES) ---
    p_nul_120 = p_nul_90 * penalties_ratio
    p_prolong_win = p_nul_90 * (1.0 - penalties_ratio)
    p1_120 = p1_90 + p_prolong_win * p_qualif_1_eff
    p2_120 = p2_90 + p_prolong_win * p_qualif_2_eff
    probas_120 = np.array([p1_120, p_nul_120, p2_120])
    
    # --- 3. DRIFT TEMPOREL ---
    # L'Agent possède son edge. On utilise apply_temporal_drift en version 1D.
    probas_120_drifted = apply_temporal_drift(probas_120, match_phases=phase_str)
    
    # --- 4. GAINS ET CROWD ---
    final_gains = known_gains if known_gains is not None else calculate_mpp_gains(probas_90, add_noise=True)
    final_crowd = known_crowd if known_crowd is not None else estimate_crowd_3D(probas_120[0], probas_120[1], probas_120[2], add_noise=True)
    
    return probas_120_drifted, final_gains, final_crowd, p_qualif_1_eff

def get_phase_str_from_idx(m):
    if m < 16: return "16e"
    if m < 24: return "8e"
    if m < 28: return "Quart"
    if m < 30: return "Demi"
    return "Finale"


def poules_horizon_from_full(V_full):
    """
    Dérive l'horizon "poules" (lu par le Notebook 10 / daily_pipeline) à partir de
    la matrice riche des phases finales `expected_V_phases_finales_full.npy`.

    L'horizon poules ne suit pas les favoris : on prend la tranche "favoris tous
    vivants" (index 1 sur my_fav / bob_fav / pack_fav), valide à l'entrée des 16es
    où aucune équipe n'est encore éliminée.

    Entrée  : V_full de forme (..., g1, g2, booster, my_fav, bob_fav, pack_fav),
              soit 7D (32, 1001, 1001, 2, 2, 2, 2) soit 6D (1001, 1001, 2, 2, 2, 2).
    Sortie  : (N, 1001, 1001, 2) avec un axe "match" en tête (N=32 ou 1), compatible
              avec le `[0]` de daily_pipeline.
    """
    V = V_full[..., 1, 1, 1]          # favoris vivants -> supprime les 3 dims favoris
    if V.ndim == 3:                   # cas 6D (matrice unique) -> ajoute l'axe match
        V = V[None, ...]
    return V

def conditional_matchup_prob(cv_inv_a, surv_a, cv_inv_b, surv_b, beta=0.95, eps=1e-6):
    """
    Probabilité que l'équipe A batte B dans un match knockout, à partir de leurs
    FORCES CONDITIONNELLES plutôt que du simple ratio des cotes de victoire finale.

    Force conditionnelle = cv_inv / surv**beta, où
      cv_inv = P(gagner le tournoi) VRAIE (Shin de cote_victoire), a priori avant poules
      surv   = P(atteindre ce tour) = P(qualif) VRAIE (dé-viggée) puis × probas des
               matchs franchis.
    (Les entrées sont désormais des probabilités VRAIES dé-viggées, cf. devig_group_odds ;
    le nom `cv_inv` reste pour compatibilité historique mais ce n'est plus 1/cote brut.)

    Conditionner par la survie fait monter la force d'une équipe qui a déjà éliminé
    de gros morceaux (faible surv -> ratio cv/surv élevé), sous hypothèse
    d'indépendance des matchs.

    beta : exposant d'AMORTISSEMENT du conditionnement (par défaut 0.95, calibré via
           le notebook 23 pour que les fréquences de titre simulées collent à 1/cote_victoire).
      - beta = 1.0 : conditionnement COMPLET = cv_inv / surv (identité bayésienne
                     exacte sous indépendance).
      - beta = 0.0 : AUCUN conditionnement (surv**0 = 1) -> simple ratio des cotes
                     de victoire, comme avant l'introduction de la survie.
      - beta ∈ (0,1) : conditionnement PARTIEL -> limite continûment le boost de
                     l'outsider (utile car l'hypothèse d'indépendance est imparfaite
                     et les cotes proviennent de sources hétérogènes). Plus beta est
                     petit, moins « battre un cador » renforce une équipe.
    """
    s_a = cv_inv_a / (max(surv_a, eps) ** beta)
    s_b = cv_inv_b / (max(surv_b, eps) ** beta)
    total = s_a + s_b
    return 0.5 if total <= 0.0 else s_a / total


def generate_bracket_scenario(df_odds, df_tournoi=None, beta=0.95):
    """
    Simulateur Universel (Avant-tournoi & Horizon Glissant).
    beta : amortissement de la force conditionnelle (cf. conditional_matchup_prob).
    Génère le bracket Monte-Carlo de base, puis l'écrase avec la réalité 
    si un fichier de tournoi en cours (df_tournoi) est fourni.
    """
    # Probabilités VRAIES (dé-viggées) : victoire finale par Shin, qualification par
    # calibration logit-shift (somme = N_QUALIFIED). Remplacent les 1/cote bruts.
    true_cv, true_qualif = devig_group_odds(df_odds)

    # 1. Simulation pure des poules (Le Brouillon Monte-Carlo)
    q1, q2, q3 = simulate_group_stages(df_odds, qualif_probs=true_qualif)

    bracket_teams = np.full((32, 2), -1, dtype=np.int32)
    team_to_id = {str(row['team']).strip().lower(): i for i, row in df_odds.iterrows()}
    id_to_odds = {i: float(true_cv[i]) for i in range(len(df_odds))}

    # Survie : P(atteindre le tour courant). Initialisée à P(qualifié en 16e) = proba VRAIE
    # de qualification (dé-viggée), car cote_victoire est une probabilité AVANT poules
    # (elle inclut la qualification). Mise à jour après chaque match franchi (cf. boucle).
    surv = {i: float(true_qualif[i]) for i in range(len(df_odds))}
    
    # --- PLACEMENT DES QUALIFIÉS MONTE-CARLO ---
    for q in q1 + q2:
        match_idx = int(q['slot'])
        if bracket_teams[match_idx, 0] == -1: bracket_teams[match_idx, 0] = team_to_id[q['team']]
        else: bracket_teams[match_idx, 1] = team_to_id[q['team']]
            
    np.random.shuffle(q3)
    q3_idx = 0
    for m in range(16):
        for s in range(2):
            if bracket_teams[m, s] == -1:
                bracket_teams[m, s] = team_to_id[q3[q3_idx]]
                q3_idx += 1

    # 2. L'ÉCRASEMENT PAR LA RÉALITÉ (The Override)
    df_pf = None
    if df_tournoi is not None:
        # case=False : robuste à la casse (la convention CSV est en minuscules :
        # 16e, 8e, quart, demi, finale) ; aucune phase de poule ne contient ces motifs.
        mask_pf = df_tournoi['phase'].str.contains('16e|8e|quart|demi|finale', case=False, na=False)
        df_pf = df_tournoi[mask_pf].reset_index(drop=True)

    # Initialisation des matrices de l'Oracle
    match_probs = np.zeros((32, 3), dtype=np.float32)
    mpp_gains = np.zeros((32, 3), dtype=np.int32)
    crowds = np.zeros((32, 3), dtype=np.float32)
    match_teams = np.zeros((32, 2), dtype=np.int32)
    
    next_match_map = {0: (16,0), 1: (16,1), 2: (17,0), 3: (17,1), 4: (18,0), 5: (18,1), 6: (19,0), 7: (19,1), 8: (20,0), 9: (20,1), 10: (21,0), 11: (21,1), 12: (22,0), 13: (22,1), 14: (23,0), 15: (23,1), 16: (24,0), 17: (24,1), 18: (25,0), 19: (25,1), 20: (26,0), 21: (26,1), 22: (27,0), 23: (27,1), 24: (28,0), 25: (28,1), 26: (29,0), 27: (29,1)}
    
    all_penalties_ratio = np.random.uniform(0.4, 0.5, 32)
    
    # 3. LA BOUCLE DE SIMULATION DES 32 MATCHS
    for m in range(32):
        result_str = ''
        k_p90, k_gains, k_crowd = None, None, None
        
        # --- INJECTION DU CSV (Si disponible) ---
        if df_pf is not None and m < len(df_pf):
            row = df_pf.iloc[m]
            
            # 3A. L'affiche est-elle déjà connue dans la vraie vie ?
            tA_name = str(row.get('team_A', '')).strip().lower()
            tB_name = str(row.get('team_B', '')).strip().lower()
            
            if tA_name != 'nan' and tA_name != '':
                bracket_teams[m, 0] = team_to_id.get(tA_name, bracket_teams[m, 0])
            if tB_name != 'nan' and tB_name != '':
                bracket_teams[m, 1] = team_to_id.get(tB_name, bracket_teams[m, 1])
                
            # 3B. Les cotes du match sont-elles ouvertes ?
            if pd.notna(row.get('cote_1')):
                k_p90 = calculate_true_outcome_probas_from_odds(
                    np.array([row['cote_1'], row['cote_N'], row['cote_2']]))
                k_gains = [row['gain_mpp_1'], row['gain_mpp_N'], row['gain_mpp_2']]
                c_brut = np.array([row['crowd_1'], row['crowd_N'], row['crowd_2']], dtype=np.float32)
                k_crowd = c_brut / np.sum(c_brut)
                
            # 3C. Le match est-il déjà terminé ?
            result_str = str(row.get('result', '')).strip().lower()

        # Lecture des équipes finalisées pour ce match
        t1, t2 = bracket_teams[m, 0], bracket_teams[m, 1]
        match_teams[m, 0], match_teams[m, 1] = t1, t2
        
        # Force CONDITIONNELLE (cv_inv / surv) au lieu du simple ratio des cotes de
        # victoire : une équipe ayant déjà battu de gros adversaires est renforcée.
        # NB : si les cotes 1N2 du match sont connues (known_probas_90), cette valeur
        # est de toute façon écrasée par p1+pN/2 dans simulate_knockout_match_math.
        base_p_qualif_1 = conditional_matchup_prob(
            id_to_odds.get(t1, 50.0), surv.get(t1, 1.0),
            id_to_odds.get(t2, 50.0), surv.get(t2, 1.0),
            beta=beta,
        )
        
        # --- APPEL DU MOTEUR MATHÉMATIQUE (Factorisé) ---
        p_120_drift, g, c, p_qualif_final = simulate_knockout_match_math(
            base_p_qualif_1, all_penalties_ratio[m], get_phase_str_from_idx(m),
            known_probas_90=k_p90, known_gains=k_gains, known_crowd=k_crowd
        )
        
        match_probs[m] = p_120_drift
        mpp_gains[m] = g
        crowds[m] = c
        
        # --- AVANCEMENT DANS L'ARBRE ---
        if result_str != 'nan' and result_str != '':
            # Forçage par la réalité
            winner = team_to_id.get(result_str, t1)
            loser = t2 if winner == t1 else t1
        else:
            # Tirage Monte-Carlo
            r = np.random.rand()
            winner = t1 if r < p_qualif_final else t2
            loser = t2 if winner == t1 else t1

        # Mise à jour de la survie pour le tour suivant : on multiplie par la prob
        # A PRIORI de victoire du gagnant à ce match (cotes 1N2 si connues, sinon
        # ratio conditionnel) — MÊME si le résultat est forcé par la réalité : c'est
        # ce prior qui encode "a-t-il battu un cador ?" et fait grimper sa force.
        p_adv_winner = p_qualif_final if winner == t1 else (1.0 - p_qualif_final)
        surv[winner] = surv.get(winner, 1.0) * p_adv_winner

        if m in next_match_map:
            nm, slot = next_match_map[m]
            bracket_teams[nm, slot] = winner
        elif m == 28: 
            bracket_teams[31, 0], bracket_teams[30, 0] = winner, loser
        elif m == 29: 
            bracket_teams[31, 1], bracket_teams[30, 1] = winner, loser
            
    return match_probs, mpp_gains, crowds, match_teams, team_to_id

# ==========================================
# 3. L'INTÉGRATEUR GÉANT (VERSION FINALE HYBRIDE)
# ==========================================
def _resolve_known_pick(known, idx, valid_names, kind):
    """Résout le choix RÉEL du joueur d'index `idx` dans la liste `known`.

    Renvoie le nom canonique (tel qu'il figure dans `valid_names`, robuste à la
    casse / aux espaces) si l'entrée est renseignée, sinon None -> pick à tirer
    selon la foule. `known[i]` = choix du joueur `i` du roster (favoris et buteurs
    sont des listes alignées sur les mêmes joueurs). La liste peut être incomplète
    (plus courte que le roster) ou contenir des None (joueur dont le pick est inconnu).
    """
    if known is None or idx >= len(known):
        return None
    pick = known[idx]
    if pick is None:
        return None
    pick = str(pick).strip()
    if pick == '':
        return None
    if pick in valid_names:
        return pick
    lower_map = {str(n).strip().lower(): n for n in valid_names}
    key = pick.lower()
    if key in lower_map:
        return lower_map[key]
    raise ValueError(
        f"{kind} imposé '{pick}' (index {idx}) introuvable parmi les choix "
        f"disponibles : {sorted(valid_names)}"
    )


def compute_robust_endgame_horizon(my_fav, my_scorer, path_poules="data/CDM_2026.csv",
                                   n_simulations=15, save=True, verbose=True, beta=0.95,
                                   known_favorites=None, known_scorers=None, n_poisson=2000,
                                   market_path="data/CDM_2026_goal_scorer_and_favorite.csv"):
    """
    Calcule l'horizon des phases finales pour un portefeuille (favori, buteur) donné.

    Retourne toujours V_start_1001 (7D : 32, 1001, 1001, 2, 2, 2, 2).
    save=True   : écrit aussi les .npy sur disque (usage de production).
    save=False  : ne touche pas au disque (usage validation, ex. recherche du meilleur
                  combo favori/buteur dans le Notebook 16).
    verbose     : journalise la progression (couper pour les boucles sur plusieurs combos).

    known_favorites / known_scorers : listes optionnelles et ALIGNÉES des choix RÉELS du
        roster d'adversaires — `known_favorites[i]`/`known_scorers[i]` sont le favori et le
        buteur du joueur `i`. Le roster complet compte `N_PELOTON = N_PLAYERS-1` adversaires.
        Les listes peuvent être incomplètes (taille < N_PELOTON) ou contenir des None/vides :
        ces slots inconnus sont complétés par un tirage selon la foule, RE-tiré à chaque run.
        À chaque run, Bob (gap_1) et le peloton (gap_2) sont DEUX indices de joueurs distincts
        tirés sans remise dans le roster ; leurs picks viennent du roster ainsi complété.
        Les noms sont résolus à la casse/aux espaces près (typo -> ValueError).
    """
    if verbose:
        print(f"--- DÉMARRAGE DU CALCUL HORIZON ({n_simulations} arbres) ---")
        print(f"Portfolio Agent -> Favori: {my_fav.upper()} | Buteur: {my_scorer.upper()}")
    
    # 1. Chargement des données
    df_odds, df_fav, df_sco = load_and_prepare_data("data/CDM_2026_group_stage_odds.csv", market_path)
    
    try:
        # Bien que nommée df_poules historiquement, elle contient tout le tournoi (CDM_2026.csv)
        df_poules = pd.read_csv(path_poules)
    except FileNotFoundError:
        print(f"Fichier {path_poules} introuvable. L'inertie du peloton risque d'être faussée.")
        return
        
    # --- DÉTECTION DU MODE (Le "GPS" du tournoi) ---
    # case=False : robuste à la casse (convention CSV en minuscules : quart/demi/finale)
    mask_pf = df_poules['phase'].str.contains('16e|8e|quart|demi|finale', case=False, na=False)
    df_pf = df_poules[mask_pf].reset_index(drop=True)
    
    # 1. Combien de résultats de PF sont déjà connus ?
    if 'result' in df_pf.columns:
        # On sécurise la lecture pour éviter les 'nan' ou espaces vides
        matchs_joues = df_pf[df_pf['result'].notna() & 
                             (df_pf['result'].astype(str).str.strip() != '') & 
                             (df_pf['result'].astype(str).str.lower() != 'nan')]
        match_idx_pf = len(matchs_joues)
    else:
        match_idx_pf = 0

    # 2. Les affiches des 16èmes sont-elles renseignées ?
    affiches_16e_connues = False
    if not df_pf.empty:
        premier_16e_team_A = str(df_pf.iloc[0].get('team_A', '')).strip().lower()
        if premier_16e_team_A != 'nan' and premier_16e_team_A != '':
            affiches_16e_connues = True

    # 3. Affichage du vrai statut
    if verbose:
        if match_idx_pf > 0:
            print(f"MODE HORIZON GLISSANT : {match_idx_pf} matchs de phases finales déjà validés.")
        elif affiches_16e_connues:
            print(f"MODE HORIZON GLISSANT : Début des 16èmes de finale (Affiches connues, 0 match terminé).")
        else:
            print(f"MODE AVANT-TOURNOI : Pré-calcul complet depuis la fin des poules.")
    
    fav_names = df_fav['selection'].tolist()
    fav_probs = df_fav['estimated_crowd'].values
    
    sco_names = df_sco['selection'].tolist()
    sco_probs = df_sco['estimated_crowd'].values
    n_scorers = len(sco_names)

    # Roster des adversaires : picks RÉELS figés par joueur (None -> tiré selon la foule
    # à chaque run). known_*[i] = choix du joueur i ; roster complet = N_PELOTON joueurs.
    if N_PELOTON < 2:
        raise ValueError(
            f"N_PELOTON={N_PELOTON} < 2 : impossible de tirer Bob et le peloton distincts. "
            f"Augmenter N_PLAYERS dans oracle_dp.")
    for kind, known in (("favorites", known_favorites), ("scorers", known_scorers)):
        if known is not None and len(known) > N_PELOTON:
            warnings.warn(
                f"known_{kind} : {len(known)} entrées > N_PELOTON={N_PELOTON} ; "
                f"les entrées au-delà du roster sont ignorées.",
                stacklevel=2,
            )
    fav_fixed = [_resolve_known_pick(known_favorites, i, fav_names, "Favori") for i in range(N_PELOTON)]
    sco_fixed = [_resolve_known_pick(known_scorers, i, sco_names, "Buteur") for i in range(N_PELOTON)]
    if verbose:
        n_fav_known = sum(f is not None for f in fav_fixed)
        n_sco_known = sum(s is not None for s in sco_fixed)
        print(f"Roster adversaires : {N_PELOTON} joueurs ; favoris connus {n_fav_known}/{N_PELOTON}, "
              f"buteurs connus {n_sco_known}/{N_PELOTON}. Bob & peloton = 2 indices distincts "
              f"tirés sans remise/run (slots inconnus complétés selon la foule à chaque run).")

    # --- MOTEUR POISSON BUTEUR (loi conjointe Favori/Buteur, in-play) ---
    # Table des joueurs SIMULÉS (named + ghosts) avec nation / current_goals / alpha.
    # Remplace la convolution buteur STATIQUE : pour chaque arbre, on tire
    # Y_final,i = current_goals_i + Poisson(alpha_i * M_restants,i) et le meilleur
    # buteur en découle (cf. scorer_model). M_restants vient de l'arbre réalisé.
    df_market_full = pd.read_csv(market_path)
    players, autre_gain, field_alpha = load_scorer_players(df_market_full)
    team_to_id_static = {str(t).strip().lower(): i for i, t in enumerate(df_odds['team'])}
    n_teams = len(df_odds)

    if players['alpha'].isna().any():
        manquants = players.loc[players['alpha'].isna(), 'selection'].tolist()
        raise ValueError(
            f"Colonne 'alpha' non calibrée pour les buteurs {manquants} : lancer la "
            f"calibration Phase A (scorer_model.calibrate_scorer_alphas) et renseigner "
            f"la colonne 'alpha' dans {market_path} avant de calculer l'horizon.")
    if field_alpha is None:
        raise ValueError(
            f"Taux de but du CHAMP DE FOND non calibré : renseigner la colonne 'alpha' "
            f"de la ligne category='field' dans {market_path} (sortie de "
            f"calibrate_scorer_alphas : info['alpha_field']).")

    player_nation_id = np.empty(len(players), dtype=np.int64)
    for i, nat in enumerate(players['nation']):
        if nat not in team_to_id_static:
            raise ValueError(
                f"Buteur '{players['selection'].iloc[i]}' : nation '{nat}' introuvable "
                f"dans le fichier des groupes. Vérifier la colonne 'nation' du CSV.")
        player_nation_id[i] = team_to_id_static[nat]

    player_alpha = players['alpha'].values.astype(np.float64)
    player_cg = players['current_goals'].values.astype(np.float64)
    player_is_ghost = players['is_ghost'].values.astype(np.bool_)
    player_is_elim = players['is_eliminated'].values.astype(np.bool_)
    player_names = players['selection'].values

    # Champ de fond : un buteur GÉNÉRIQUE par équipe (taux commun field_alpha).
    # Il hérite du M_restants de son équipe dans l'arbre -> sa part croît quand les
    # stars sont éliminées (M -> 0). N'appartient à aucun pick (membres = False) :
    # quand il décroche le titre, c'est le combo "aucun de mes 3 buteurs".
    field_nation_id = np.arange(n_teams, dtype=np.int64)
    ext_alpha = np.concatenate([player_alpha, np.full(n_teams, float(field_alpha))])
    ext_cg = np.concatenate([player_cg, np.zeros(n_teams)])
    ext_is_elim = np.concatenate([player_is_elim, np.zeros(n_teams, dtype=np.bool_)])
    ext_nation_id = np.concatenate([player_nation_id, field_nation_id])

    def _extend_member(mask):
        return np.concatenate([mask, np.zeros(n_teams, dtype=np.bool_)])

    # Matchs de POULE restants par équipe (depuis CDM_2026.csv) : composante de
    # M_restants pendant la phase de groupes (0 une fois les poules terminées).
    group_remaining = np.zeros(n_teams, dtype=np.float64)
    if {'team_A', 'team_B', 'phase'} <= set(df_poules.columns):
        for _, r in df_poules.iterrows():
            if not str(r['phase']).lower().startswith('poule'):
                continue
            res = str(r.get('result', '')).strip().lower()
            if res not in ('', 'nan', 'none'):
                continue  # match de poule déjà joué -> buts déjà dans current_goals
            for col in ('team_A', 'team_B'):
                nm = normalize_team_name(r[col])
                if nm in team_to_id_static:
                    group_remaining[team_to_id_static[nm]] += 1.0

    def _pick_mask_gain(pick_name):
        """(masque d'appartenance sur les joueurs simulés, gain MPP plein) d'un pick.
        'autre' -> tous les ghost scorers + gain du pick générique ; sinon le buteur
        named singleton + son gain_mpp."""
        pn = str(pick_name).strip().lower()
        if pn == 'autre':
            return player_is_ghost.copy(), float(autre_gain if autre_gain else 0)
        mask = (player_names == pn)
        gain = 0.0
        if mask.any():
            g = players.loc[mask, 'gain_mpp'].values[0]
            gain = float(g) if not pd.isna(g) else 0.0
        return mask, gain

    my_member, my_scorer_gain = _pick_mask_gain(my_scorer)
    my_member_ext = _extend_member(my_member)

    # 2. PRÉ-CALCUL DU PELOTON (Historique Complet 104 matchs)
    p_emp_ref, gains_ref = precompute_diluted_peloton(df_poules, df_odds)
    
    # 3. Initialisation de l'Accumulateur (Dynamique pour éviter les erreurs de dimension)
    V_sum = None 
    alphas_mock = np.ones(32, dtype=np.float32) # Pas de Gating artificiel, dilution naturelle
    
    # Points de victoire finale de MON favori (my_member / my_scorer_gain déjà
    # préparés ci-dessus pour le moteur Poisson buteur).
    my_pts = int(df_fav.loc[df_fav['selection'] == my_fav, 'gain_mpp'].values[0])
    
    start_time = time.time()
    
    for i in range(n_simulations):
        # A. Tirage du scénario de l'Arbre (Auto-conscient de la réalité)
        match_probs, gains_arbre_actuel, crowds, teams, team_to_id = generate_bracket_scenario(
            df_odds, df_tournoi=df_poules, beta=beta
        )
        
        # B. Bob (gap_1) et le peloton (gap_2) = 2 joueurs DISTINCTS du roster, tirés sans
        #    remise. Leur (favori, buteur) = pick réel si connu, sinon tiré selon la foule
        #    (re-tiré à chaque run pour les slots inconnus).
        bob_idx, pack_idx = np.random.choice(N_PELOTON, size=2, replace=False)
        bob_fav = fav_fixed[bob_idx] if fav_fixed[bob_idx] is not None else np.random.choice(fav_names, p=fav_probs)
        pack_fav = fav_fixed[pack_idx] if fav_fixed[pack_idx] is not None else np.random.choice(fav_names, p=fav_probs)

        bob_sco = sco_fixed[bob_idx] if sco_fixed[bob_idx] is not None else np.random.choice(sco_names, p=sco_probs)
        pack_sco = sco_fixed[pack_idx] if sco_fixed[pack_idx] is not None else np.random.choice(sco_names, p=sco_probs)
        
        bob_member, bob_scorer_gain = _pick_mask_gain(bob_sco)
        pack_member, pack_scorer_gain = _pick_mask_gain(pack_sco)
        bob_member_ext = _extend_member(bob_member)
        pack_member_ext = _extend_member(pack_member)

        # C. Traduction en vecteurs temporels (Les 32 matchs)
        my_id = team_to_id.get(my_fav.lower(), -1)
        bob_id = team_to_id.get(bob_fav.lower(), -1)
        pack_id = team_to_id.get(pack_fav.lower(), -1)
        
        my_roles = np.where(teams[:, 0] == my_id, 0, np.where(teams[:, 1] == my_id, 2, -1)).astype(np.int32)
        bob_roles = np.where(teams[:, 0] == bob_id, 0, np.where(teams[:, 1] == bob_id, 2, -1)).astype(np.int32)
        pack_roles = np.where(teams[:, 0] == pack_id, 0, np.where(teams[:, 1] == pack_id, 2, -1)).astype(np.int32)
        
        # D. État terminal : moteur Poisson buteur (loi conjointe) pour CET arbre.
        bob_pts = int(df_fav.loc[df_fav['selection'] == bob_fav, 'gain_mpp'].values[0])
        pack_pts = int(df_fav.loc[df_fav['selection'] == pack_fav, 'gain_mpp'].values[0])

        # M_restants par équipe dans l'arbre RÉALISÉ = matchs de poule restants
        # + matchs knockout FUTURS (apparitions dans teams[match_idx_pf:]). Les
        # joueurs explicitement éliminés (CSV) sont gelés à M=0.
        M_team = group_remaining.copy()
        for m in range(match_idx_pf, 32):
            for s in range(2):
                tid = teams[m, s]
                if tid >= 0:
                    M_team[tid] += 1.0
        # M_restants par joueur SIMULÉ (listés + champ de fond), gelé si éliminé.
        M_player_ext = np.where(ext_is_elim, 0.0, M_team[ext_nation_id])

        joint_scorer = top_scorer_joint_probs(
            ext_alpha, ext_cg, M_player_ext,
            my_member_ext, bob_member_ext, pack_member_ext, int(n_poisson)
        )

        V_term = build_terminal_state(
            float(my_pts), float(bob_pts), float(pack_pts),
            float(my_scorer_gain), float(bob_scorer_gain), float(pack_scorer_gain),
            joint_scorer,
        )
        
        # E. LE RESCALING MAGIQUE (Transfer Learning Affine)
        p_empirique_scenario = np.zeros((32, 3, 250), dtype=np.float32)
        for t in range(match_idx_pf, 32): # On ne rescale que depuis le match actuel
            p1, p2 = match_probs[t, 0], match_probs[t, 2]
            idx_fav = 0 if p1 >= p2 else 2
            idx_out = 2 if p1 >= p2 else 0
            
            mapping = {idx_fav: 0, 1: 1, idx_out: 2}
            
            for out in range(3):
                ref_idx = mapping[out]
                p_empirique_scenario[t, out] = rescale_histogram_1D(
                    ref_hist=p_emp_ref[t, ref_idx],
                    ref_gain=gains_ref[t, ref_idx],
                    target_gain=gains_arbre_actuel[t, out],
                    target_crowd=crowds[t, out],
                    max_bins=250
                )
        
        # F. Résolution avec Numba (S'arrête intelligemment au présent)
        V_scenario = solve_endgame_dp(
            match_probs, crowds, gains_arbre_actuel, p_empirique_scenario, alphas_mock,
            my_roles, bob_roles, pack_roles, V_term
        )
        
        if V_sum is None:
            V_sum = np.zeros_like(V_scenario)
            
        V_sum += V_scenario
        
        if verbose and ((i + 1) % 5 == 0 or (i + 1) == n_simulations):
            elapsed = time.time() - start_time
            print(f"[{i + 1}/{n_simulations}] Arbres traités. Temps : {elapsed:.1f}s")

    # Moyenne de l'incertitude
    V_final = V_sum / n_simulations

    # 4. EXPORT INTELLIGENT DE L'HORIZON (AVEC UPSAMPLING)
    if verbose:
        print("\nÉlargissement de la grille (Upsampling vers 1001x1001)...")
    
    # Le code s'adapte selon que V_final est en 6D (une seule matrice) ou 7D (32 matrices)
    if V_final.ndim == 6:
        V_start_1001 = np.zeros((1001, 1001, 2, 2, 2, 2), dtype=np.float32)
        for g1 in range(1001):
            idx1 = min(500, int(round(g1 / 2.0)))
            for g2 in range(1001):
                idx2 = min(500, int(round(g2 / 2.0)))
                V_start_1001[g1, g2] = V_final[idx1, idx2]
    else:
        # V_final est 7D : e.g., (32, 501, 501, 2, 2, 2, 2)
        V_start_1001 = np.zeros((32, 1001, 1001, 2, 2, 2, 2), dtype=np.float32)
        for g1 in range(1001):
            idx1 = min(500, int(round(g1 / 2.0)))
            for g2 in range(1001):
                idx2 = min(500, int(round(g2 / 2.0)))
                V_start_1001[:, g1, g2] = V_final[:, idx1, idx2]
                
    if save:
        np.save("data/expected_V_phases_finales_full.npy", V_start_1001)
        print(f"EXPORT RÉUSSI : data/expected_V_phases_finales_full.npy (Forme: {V_start_1001.shape})")

        # --- HORIZON POULES (pour le Notebook 10 / daily_pipeline) ---
        # On ne le sauve QU'EN MODE AVANT-TOURNOI : dès que les affiches des 16es sont
        # renseignées, les matchs des 16es ont été écrasés par la réalité dans le bracket,
        # ce qui biaiserait cet horizon — et de toute façon les poules sont derrière nous.
        if affiches_16e_connues:
            print("16es renseignés -> horizon poules (expected_V_phases_finales.npy) NON sauvé "
                  "(poules terminées, horizon biaisé/inutile).")
        else:
            V_poules = poules_horizon_from_full(V_start_1001)
            np.save("data/expected_V_phases_finales.npy", V_poules)
            print(f"EXPORT RÉUSSI : data/expected_V_phases_finales.npy (Forme: {V_poules.shape}) "
                  f"[horizon poules, favoris vivants]")

    return V_start_1001


def simulate_champion_distribution(df_odds, n_runs=200_000, beta=0.95, verbose=True, seed=None):
    """
    Simule n_runs tournois complets (poules + bracket) et compte combien de fois
    chaque équipe est CHAMPIONNE (gagnante de la finale, match 31).

    Reproduit FIDÈLEMENT le modèle d'avancement de generate_bracket_scenario :
      - poules : proxy Plackett-Luce sur cote_1er + équation des 3èmes sur la proba de
                 qualification VRAIE dé-viggée (mêmes appels np.random.choice sans remise) ;
      - knockout : force conditionnelle cv_true / surv**beta (probas VRAIES dé-viggées :
                 Shin pour la victoire, logit-shift pour la qualif), + bruit N(0, 0.03),
                 + clip [10·MIN_TRUE_PROBA, MAX_TRUE_PROBA**4], comme la branche
                 simulée de simulate_knockout_match_math.
    (Les probas 1N2 / gains / crowd du DP ne sont pas calculées : inutiles ici.)

    Sert à CALIBRER beta : comparer la fréquence empirique de titre de chaque équipe
    à la probabilité de victoire VRAIE (Shin de cote_victoire) (cf. notebook 23).

    Retour : (champion_counts (n_teams,), team_names (n_teams,)).
    """
    if seed is not None:
        np.random.seed(seed)

    team_names = df_odds["team"].astype(str).str.strip().str.lower().values
    n_teams = len(team_names)
    # Probabilités VRAIES (dé-viggées) : Shin pour la victoire, logit-shift pour la qualif.
    cv_true, qualif_true = devig_group_odds(df_odds)
    cv_true = cv_true.astype(np.float64)
    name_to_id = {n: i for i, n in enumerate(team_names)}

    # Structures de groupes pré-calculées (évite tout pandas dans la boucle)
    groups = []
    for _, grp in df_odds.groupby("group"):
        ids = np.array([name_to_id[str(t).strip().lower()] for t in grp["team"]], dtype=np.int64)
        w = calculate_true_outcome_probas_from_odds(grp["cote_1er"].values)
        w = w / w.sum()
        s1 = grp["slot_1er"].values.astype(np.int64)
        s2 = grp["slot_2e"].values.astype(np.int64)
        gamma = max(0.01, float(qualif_true[ids].sum()) - 2.0)
        groups.append((ids, w, s1, s2, gamma))

    next_match_map = {0: (16, 0), 1: (16, 1), 2: (17, 0), 3: (17, 1), 4: (18, 0), 5: (18, 1),
                      6: (19, 0), 7: (19, 1), 8: (20, 0), 9: (20, 1), 10: (21, 0), 11: (21, 1),
                      12: (22, 0), 13: (22, 1), 14: (23, 0), 15: (23, 1), 16: (24, 0), 17: (24, 1),
                      18: (25, 0), 19: (25, 1), 20: (26, 0), 21: (26, 1), 22: (27, 0), 23: (27, 1),
                      24: (28, 0), 25: (28, 1), 26: (29, 0), 27: (29, 1)}

    LO, HI, SIG = 10.0 * MIN_TRUE_PROBA, MAX_TRUE_PROBA ** 4, 0.03
    counts = np.zeros(n_teams, dtype=np.int64)
    champion = -1

    for run in range(n_runs):
        bracket = np.full((32, 2), -1, dtype=np.int64)
        surv = qualif_true.copy()  # P(qualifié en 16e) VRAIE ; réinitialisé chaque tournoi

        # --- Poules ---
        thirds, thirds_gamma = [], []
        for (ids, w, s1, s2, gamma) in groups:
            pick = np.random.choice(4, size=3, replace=False, p=w)
            first, second, third = ids[pick[0]], ids[pick[1]], ids[pick[2]]
            for team, slot in ((first, int(s1[pick[0]])), (second, int(s2[pick[1]]))):
                if bracket[slot, 0] == -1:
                    bracket[slot, 0] = team
                else:
                    bracket[slot, 1] = team
            thirds.append(third)
            thirds_gamma.append(gamma)
        tg = np.array(thirds_gamma)
        tg = tg / tg.sum()
        best8 = np.random.choice(12, size=8, replace=False, p=tg)
        q3 = [thirds[i] for i in best8]
        np.random.shuffle(q3)
        qi = 0
        for m in range(16):
            for s in range(2):
                if bracket[m, s] == -1:
                    bracket[m, s] = q3[qi]
                    qi += 1

        # --- Knockout ---
        for m in range(32):
            t1, t2 = bracket[m, 0], bracket[m, 1]
            s_a = cv_true[t1] / surv[t1] ** beta
            s_b = cv_true[t2] / surv[t2] ** beta
            p1 = s_a / (s_a + s_b)
            p1 = min(HI, max(LO, p1 + np.random.normal(0, SIG)))
            if np.random.rand() < p1:
                winner, pw = t1, p1
            else:
                winner, pw = t2, 1.0 - p1
            surv[winner] *= pw
            if m in next_match_map:
                nm, slot = next_match_map[m]
                bracket[nm, slot] = winner
            elif m == 28:
                bracket[31, 0] = winner
                bracket[30, 0] = t2 if winner == t1 else t1
            elif m == 29:
                bracket[31, 1] = winner
                bracket[30, 1] = t2 if winner == t1 else t1
        champion = winner  # vainqueur de la finale (m = 31, dernier traité)
        counts[champion] += 1

        if verbose and (run + 1) % max(1, n_runs // 10) == 0:
            print(f"  {run + 1:,}/{n_runs:,} tournois simulés...")

    return counts, team_names


if __name__ == "__main__":
    # Test avec 15 arbres pour être rapide au quotidien (Modifiable le Jour J)
    # ~486 s/itération sur CPU (i5-12400) ; 15 itérations = ~1h15 ; 2 itérations = ~10 min.
    compute_robust_endgame_horizon(my_fav="espagne", my_scorer="autre", n_simulations=67,
                                   known_favorites=["france", "france", "france", "france", "espagne"],
                                   known_scorers=  ["autre", "autre", "autre", "harry_kane", "harry_kane"])