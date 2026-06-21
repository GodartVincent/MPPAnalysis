"""
Core logic and utility functions for the MonPetitProno (MPP) analysis project.
"""

import warnings

import pandas as pd
import numpy as np
import shin
import unicodedata
from collections import namedtuple
from scipy.stats import binom, poisson
from scipy.optimize import root_scalar, minimize
from typing import Tuple

# --- CONSTANTES DU MODÈLE MPP (À mettre à jour avec les résultats du Notebook 12_) ---
# Constantes permettant de convertir un gain MPP en probabilité vraie d'outcome via une courbe cubique.
CUBIC_A = -244.3   # Paramètre d'ordre 3
CUBIC_B = 497.7    # Paramètre d'ordre 2
CUBIC_C = -460.5   # Paramètre d'ordre 1
CUBIC_D = 206.3    # Paramètre d'ordre 0
RMSE_GAINS = 9.75  # RMSE du fit : ecart-type du bruit gaussien à ajouter au gain pour simuler des probas réalistes.

# --- CONSTANTES DU MODÈLE DE FOULE SOFTMAX (À mettre à jour avec les résultats du Notebook 13_) ---
CROWD_BETA = 2.24       # Paramètre de sur-réaction aux favoris
CROWD_EPSILON = 0.009   # Plancher incompressible (0.9%) pour les trolls/clics fous
RMSE_CROWD = 0.083      # RMSE du fit : ecart-type du bruit gaussien à ajouter au crowd pour les simuler de manière réaliste.

MIN_MPP_GAIN = 15.0  # Gain MPP minimum réaliste (allemagne - curacao : 15.0)
MAX_MPP_GAIN = 225.0  # Gain MPP maximum réaliste (allemagne - curacao : 222.0)
MIN_TRUE_PROBA = 0.02  # Probabilité minimale réaliste d'une issue (allemagne - curacao : 0.02)
MAX_TRUE_PROBA = 0.95  # Probabilité maximale réaliste d'une issue (allemagne - curacao : 0.95)

# --- CORRECTION DU CROWD SCORE EXACT Winamax -> MPP (calibré au Notebook 25_) ---
# Le crowd Winamax conditionnel approxime mal le crowd MPP. Modèle retenu (NB25), un
# BLEND JOINT d'une surface Winamax et d'une surface génératrice ANCRÉE sur le crowd
# MPP 1N2, plus une pénalité sur le 0-0 (sur-pondéré par Winamax) :
#   cc_mpp(s) ∝ [ a·crowd_wmx(s) + (1-a)·surf_MPP(s) ] · exp(delta·1[s==0-0])
# où surf_MPP = Poisson(i;λ1)·Poisson(j;λ2) avec (λ1,λ2) ajustés pour que les masses
# des 3 régions (1/N/2) de la surface égalent le crowd MPP 1N2 (meilleure info de
# véracité disponible). Le blend est au niveau JOINT (composantes normalisées
# globalement) PUIS renormalisé PAR OUTCOME -> respecte le 1N2 comme contrainte dure
# et reshape la conditionnelle intra-outcome (le crowd MPP 1N2 déplace le centre de la
# surface -> capte le glissement de marge des matchs déséquilibrés).
# LOO par match : exact ~70 %, ±1cat 94 %, |err| points de bonus 6.6 (vs 8.4 sans
# correction, -21 %). Répare la famille « issue peu jouée -> petite marge » + les
# scores à crowd Winamax NUL (injection de masse). Si le crowd MPP 1N2 n'est pas fourni,
# on retombe sur la SEULE pénalité 0-0 sur le crowd Winamax conditionnel (rétro-compat).
# LIMITE connue : blowout extrême (gros favori) -> tendance à SUR-estimer le nombre de
# buts (le Poisson confond P(victoire) élevée et beaucoup de buts) ; cf. NB25 m31 (3-0).
EXACT_SCORE_BLEND_WMX = 0.90           # poids du crowd Winamax dans le blend (1-a sur la surface)
EXACT_SCORE_ZERO_ZERO_LOG_PENALTY = -1.8  # log-pénalité du 0-0 (exp(-1.8) ≈ 17 %)
EXACT_SCORE_GOAL_GRID = 10             # buts max par équipe pour la surface Poisson

def calculate_win_probability(min_successes: int, num_matches: int, success_prob: float) -> float:
    """
    Calculates the probability of achieving at least 'min_successes' 
    in 'num_matches' independent trials, each with 'success_prob'.
    """
    if min_successes > num_matches:
        return 0.0
    if min_successes <= 0:
        return 1.0
    return binom.sf(min_successes - 1, num_matches, success_prob)

def get_simple_ev(probability: float, points: float) -> float:
    """Calculates the simple expected value of one outcome."""
    return probability * points

def get_ev_with_bonus(outcome_prob: float, perfect_prob_given_outcome: float, points: float) -> float:
    """Calculates the EV including the 'perfect score' bonus."""
    expected_points_given_outcome = points * (1.0 + perfect_prob_given_outcome)
    return outcome_prob * expected_points_given_outcome

def calculate_true_outcome_probas_from_odds(odds: np.ndarray) -> np.ndarray:
    """
    Calcule les probabilités vraies (sans marge bookmaker) à partir des cotes,
    via la méthode de Shin (`shin.calculate_implied_probabilities`) qui corrige
    le biais favori/outsider plutôt qu'une simple normalisation 1/cote.

    Comportement selon la forme de l'entrée (inchangé historiquement) :
      - tableau (N, 3) : chaque ligne est un match indépendant à 3 issues ->
        Shin appliqué ligne par ligne, sortie (N, 3) sommant à 1 par ligne.
      - tableau 1D de taille K : K issues d'UN SEUL événement (ex : marché du
        vainqueur du tournoi, du meilleur buteur) -> Shin appliqué une fois,
        sortie (K,) sommant à 1.

    Invariant : si les cotes sont "justes" (somme des 1/cote == 1, aucune marge),
    Shin restitue exactement les probabilités implicites normalisées.
    """
    # Conversion en float pour éviter les divisions entières ou les types mixtes
    odds_array = np.asarray(odds, dtype=float)

    # Garde-fou : la somme des probabilités implicites (1/cote) vaut 1 + marge
    # bookmaker. Une marge NÉGATIVE (somme < 1) est impossible pour un vrai book
    # (ce serait une opportunité d'arbitrage) -> faute de saisie probable. Le
    # plafond > 1.15 ne vaut que pour le 1N2 (<= 3 issues) : un marché outright
    # à K issues a un overround naturellement bien plus élevé.
    if odds_array.ndim == 2:
        events = odds_array
    else:
        events = odds_array.reshape(1, -1)
    # Le contrôle de marge ne vaut que pour un marché 1N2 COMPLET (<= 3 issues) :
    # une somme(1/cote) < 1 y est impossible (arbitrage), > 1.15 trahit une saisie
    # douteuse. Un marché outright partiel (favoris, buteurs) ne liste qu'un sous-
    # ensemble d'issues -> sa somme peut légitimement tomber sous 1 ou dépasser 1.15.
    for i, row in enumerate(events):
        if row.size > 3:
            continue
        with np.errstate(divide="ignore"):
            inv = 1.0 / row
        s = float(np.sum(inv[np.isfinite(inv)]))
        if s <= 0.0:
            continue
        if s < 1.0 - 1e-9:
            warnings.warn(
                f"Cotes match index {i}: somme(1/cote)={s:.4f} < 1 "
                f"(marge bookmaker négative, impossible). Vérifier la saisie CSV.",
                stacklevel=2,
            )
        elif s > 1.15:
            warnings.warn(
                f"Cotes match index {i}: somme(1/cote)={s:.4f} > 1.15 "
                f"(marge bookmaker excessive, cote probablement mal saisie).",
                stacklevel=2,
            )

    # Application de Shin par événement. Une ligne de cotes invalides
    # (non finies ou <= 0) renvoie des probabilités nulles (cf. division par
    # zéro historique).
    def _shin_row(row: np.ndarray) -> np.ndarray:
        if row.size == 0 or not np.all(np.isfinite(row)) or np.any(row <= 0.0):
            return np.zeros_like(row)
        return np.asarray(shin.calculate_implied_probabilities(row.tolist()), dtype=float)

    if odds_array.ndim == 2:
        return np.vstack([_shin_row(row) for row in odds_array])
    return _shin_row(odds_array)


def calibrate_qualification_probs(raw_probs, target_sum=32.0):
    """
    Dé-vig d'un marché MULTI-VAINQUEURS (qualification) : ajuste les probabilités
    implicites brutes `raw_probs` (= 1/cote_qualif) par un SHIFT GLOBAL en log-odds
    (logit) pour que leur somme atteigne `target_sum` (= nombre d'équipes qualifiées),
    sans qu'aucune ne sorte de ]0, 1[ et en préservant l'ordre des équipes.

    Pourquoi pas Shin : Shin force la somme à 1 (marché à VAINQUEUR UNIQUE). La
    qualification est multi-vainqueurs (~32 équipes sur 48 en CDM 2026), donc on retire
    la marge bookmaker de façon homogène en log-odds plutôt que de normaliser à 1.

    Étapes : p -> logit z=ln(p/(1-p)) -> on cherche c tel que Σ sigmoïde(z+c) = target_sum.
    Retourne un array (N,) de probabilités vraies sommant à `target_sum`.
    """
    raw = np.clip(np.asarray(raw_probs, dtype=float), 1e-5, 1.0 - 1e-5)
    logits = np.log(raw / (1.0 - raw))

    def objective(c):
        return np.sum(1.0 / (1.0 + np.exp(-(logits + c)))) - target_sum

    res = root_scalar(objective, bracket=[-10.0, 10.0])
    c = res.root
    return 1.0 / (1.0 + np.exp(-(logits + c)))

def get_observation(
    match_probas: np.ndarray,      # (3,) Raw probabilities
    match_gains: np.ndarray,       # (3,) Raw gains
    opp_repartition: np.ndarray,   # (3,) Raw repartition
    player_scores: np.ndarray,     # (N,) All player scores
    agent_idx: int,                # Index of the agent/strategy
    future_max_points: float,      # Scalar: Sum of max gains available (incl. current)
    matches_remaining_fraction: float, # Scalar: matches_remaining / total_matches
    ev_avg: float                  # Scalar: Normalization factor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centralized function to build the observation vector.
    Used by both the Environment (training) and Strategies (inference/simulation).
    
    Returns:
        obs (np.ndarray): The flat observation vector for the Neural Network.
        sort_idx (np.ndarray): The sorting indices used (to map action back to outcome).
    """
    
    # 1. Sort Match Data by Probability (Descending)
    # We sort so the Network always sees [Best, Middle, Worst] probas,
    # making the input invariant to the specific outcome order (Home/Draw/Away).
    sort_idx = np.argsort(match_probas)[::-1]
    
    sorted_probas = match_probas[sort_idx]
    sorted_repart = opp_repartition[sort_idx]
    
    # 2. Normalize Gains
    sorted_gains = match_gains[sort_idx] / ev_avg
    
    # 3. Process Scores (Relative & Sorted)
    agent_score = player_scores[agent_idx]
    relative_scores = player_scores - agent_score
    
    # Remove agent's own score (always 0 relative)
    opp_relative_scores = np.delete(relative_scores, agent_idx)
    
    # Sort opponents from Leader (highest relative score) to Loser
    # This makes the input invariant to player indices.
    sorted_opp_scores = np.sort(opp_relative_scores)[::-1]
    normalized_opp_scores = sorted_opp_scores / ev_avg

    # 4. Desperation Ratio
    # Logic: (Agent - Leader) / Max_Remaining
    if future_max_points < 1.0: 
        future_max_points = 1.0
    
    # Leader relative score is sorted_opp_scores[0] (biggest positive gap if losing)
    gap_to_leader = -sorted_opp_scores[0] * ev_avg 
    gap_ratio = gap_to_leader / future_max_points
    gap_ratio = np.clip(gap_ratio, -1.0, 1.0)
    
    # 5. Matches Remaining
    matches_rem_arr = np.array([matches_remaining_fraction])

    # 6. Simple EV Feature (Value Detector)
    # This explicitly gives the agent (P * G) to spot mispriced bets.
    # Values >> 1.0 indicate "Value Bets".
    sorted_simple_ev = sorted_probas * sorted_gains
    
    # Concatenate everything
    # Size: 3(P) + 3(G) + 3(Repart) + (N-1)(Scores) + 1(Gap) + 1(Time) + 3(EV)
    obs = np.concatenate([
        sorted_probas,
        sorted_gains,
        sorted_repart,
        normalized_opp_scores,
        np.array([gap_ratio]),
        matches_rem_arr,
        sorted_simple_ev
    ]).astype(np.float32)
    
    return obs, sort_idx

def normalize_team_name(name):
    """
    Standardise les noms d'équipes : 
    'Corée du Sud' -> 'coree_du_sud'
    'République Tchèque' -> 'republique_tcheque'
    """
    if pd.isna(name):
        return name
        
    # 1. Convertir en chaîne de caractères
    name = str(name)
    
    # 2. Supprimer les accents (décomposition Unicode puis filtrage)
    name = ''.join(c for c in unicodedata.normalize('NFD', name)
                   if unicodedata.category(c) != 'Mn')
                   
    # 3. Mettre en minuscules
    name = name.lower()
    
    # 4. Remplacer les espaces et les tirets par des underscores
    name = name.replace(' ', '_').replace('-', '_')
    
    return name

def load_tournament_data(csv_path):
    """
    Charge le CSV du tournoi, normalise le texte, et calcule les probabilités réelles.
    Gère dynamiquement les phases sans constantes codées en dur.
    """
    # Chargement brut
    df = pd.read_csv(csv_path)
    
    # --- 1. NORMALISATION DU TEXTE ---
    df['team_A'] = df['team_A'].apply(normalize_team_name)
    df['team_B'] = df['team_B'].apply(normalize_team_name)
    
    # Optionnel mais propre : convertir la colonne date en vrai format Date Python
    df['date'] = pd.to_datetime(df['date'])
    
    # --- 2. CALCUL DES PROBABILITÉS RÉELLES (Overround Normalization) ---
    cotes = df[['cote_1', 'cote_N', 'cote_2']].values
    
    # Formule : P(X) = (1 / cote_X) / Somme(1 / cotes)
    true_probas = calculate_true_outcome_probas_from_odds(cotes)
    
    # Ajout au DataFrame pour visualisation dans les notebooks
    df['true_proba_1'] = true_probas[:, 0]
    df['true_proba_N'] = true_probas[:, 1]
    df['true_proba_2'] = true_probas[:, 2]
    
    # --- 3. EXTRACTION DES MATRICES POUR L'ORACLE ---
    mpp_gains = df[['gain_mpp_1', 'gain_mpp_N', 'gain_mpp_2']].values
    
    # Extraction et Normalisation des Foules
    raw_crowds = df[['crowd_1', 'crowd_N', 'crowd_2']].values
    crowd_sums = raw_crowds.sum(axis=1, keepdims=True)
    
    # On divise par la somme pour garantir un total strict de 1.0
    # (Le 'where' et 'out' protègent contre une éventuelle ligne vide qui causerait une division par 0)
    crowd_repartitions = np.divide(
        raw_crowds, 
        crowd_sums, 
        out=np.zeros_like(raw_crowds), 
        where=crowd_sums!=0
    )
    
    return df, true_probas, mpp_gains, crowd_repartitions

def estimate_proba_from_gain_cubic(gain, a=CUBIC_A, b=CUBIC_B, c=CUBIC_C, d=CUBIC_D):
    """
    Retrouve la probabilitévraie 'p' en cherchant les racines du polynôme cubique :
    a*p³ + b*p² + c*p + (d - Gain) = 0
    """
    coeffs = [a, b, c, d - gain]
    racines = np.roots(coeffs)
    racines_reelles = racines.real[abs(racines.imag) < 1e-5]
    valid_p = [r for r in racines_reelles if 0.0 <= r <= 1.0]
    
    if len(valid_p) > 0:
        return max(MIN_TRUE_PROBA, min(MAX_TRUE_PROBA, valid_p[0]))
    else:
        # Fallback de sécurité si le gain demandé est totalement hors courbe
        return MAX_TRUE_PROBA if gain < MIN_MPP_GAIN else MIN_TRUE_PROBA

def simulate_true_proba_from_gain(gain_mpp, rmse=RMSE_GAINS, a=CUBIC_A, b=CUBIC_B, c=CUBIC_C, d=CUBIC_D):
    """
    Simule une true_proba réaliste à partir d'un gain MPP connu.
    Intègre le bruit statistique (RMSE) découvert lors de la modélisation de gain à partir de p (voir notebook 12_).
    """
    # 1. Ajout du bruit gaussien
    bruit = np.random.normal(0, rmse)
    noisy_gain = gain_mpp + bruit
    
    # Sécurité : le gain MPP ne descend jamais sous un certain plancher (souvent 10 ou 12)
    # On empêche le bruit de créer des gains aberrants comme -5 pts.
    noisy_gain = max(MIN_MPP_GAIN, noisy_gain)
    
    # 2. Inversion de la courbe avec le gain bruité
    proba_est = estimate_proba_from_gain_cubic(noisy_gain, a, b, c, d)

    # 3. securité : on s'assure que la proba est dans [MIN_TRUE_PROBA, MAX_TRUE_PROBA].
    proba_est = max(MIN_TRUE_PROBA, min(MAX_TRUE_PROBA, proba_est))

    return proba_est

def apply_temporal_drift(true_probas: np.ndarray, match_phases, current_match_idx: int = 0,
                         match_dates=None, reference_date=None,
                         sigma_diff_ref: float = 0.025, days_diff_ref: float = 7.0,
                         sigma_phase_same: float = 0.0, sigma_phase_1: float = 0.0325,
                         sigma_phase_2: float = 0.055) -> np.ndarray:
    """
    Simule la variation des probabilités réelles au fil du temps via un modèle de 'Drift'.
    Gère à la fois une matrice de matchs (N, 3) et un match unique vectorisé (3,).

    Le drift a DEUX composantes INDÉPENDANTES (variances additives) :

      1. DIFFUSION (gain d'info calendaire) : blessures à l'entraînement, fuites sur
         le moral / le vestiaire, etc. Amplitude FAIBLE mais croissant avec le temps,
         modèle de marche aléatoire :
             sigma_diff(d) = sigma_diff_ref * sqrt(d / days_diff_ref)
         où d = (date_match - reference_date) en jours (clampé >= 0). Calibré à
         sigma_diff_ref ≈ 0.025 sur la fenêtre ouverture->fermeture d'un match de
         Premier League (durée days_diff_ref).

      2. PHASE / « l'équipe joue » (saut d'info discret) : quand l'équipe dispute des
         matchs intermédiaires, BEAUCOUP d'info sort d'un coup (l'équipe ne joue pas
         comme prévu, suspensions, scénarios de qualification...). Amplitude PLUS FORTE,
         pilotée par le nombre de « shocks » de phase entre le match courant et le
         match cible (J1->J2->J3->finales) :
             même phase (0 shock) -> sigma_phase_same (def 0.0)
             +1 phase            -> sigma_phase_1    (def 0.0325)
             +2 phases ou plus   -> sigma_phase_2    (def 0.055)
         `sigma_phase_same = 0` car « même phase » = 0 match intermédiaire joué : il n'y
         a aucun saut d'info « l'équipe joue ». Le seul drift d'un match de même phase
         vient du TEMPS qui passe, déjà capté par la composante DIFFUSION (ne pas
         remettre un plancher ici sous peine de DOUBLE COMPTAGE avec sigma_diff).

    Combinaison (cas matriciel, si `match_dates` fourni) :
        std(match) = sqrt( sigma_diff(d)**2 + sigma_phase(shocks)**2 )

    `reference_date` : date de référence comparée aux `match_dates`. None -> date du
    jour (date.today()). Accepte date/datetime/str/Timestamp. À injecter dans les tests
    pour un résultat déterministe. `match_dates` : itérable (N,) de dates ; NaT ->
    composante diffusion nulle (seule la phase joue).

    RÉTRO-COMPAT : si `match_dates` est None (chemin 1D de bracket_simulator sans dates,
    NB15, fixtures de test sans colonne 'date'), seule la composante PHASE s'applique
    (modèle historique). ATTENTION : sans dates il n'y a PAS de diffusion, donc les
    matchs de MÊME PHASE ne sont plus driftés du tout (sigma_phase_same=0). Pour
    retrouver un drift calendaire sur ces chemins (ex. NB15), passer `match_dates`.
    Le cas 1D (true_probas de forme (3,)) reste inchangé (logique propre 0.025/0.035).
    """
    # Fonction utilitaire
    def get_phase_level(p_str):
        if not isinstance(p_str, str): return 4
        if "J1" in p_str: return 1
        if "J2" in p_str: return 2
        if "J3" in p_str: return 3
        return 4 # Phases Finales
        
    # ========================================================
    # CAS 1 : UN SEUL MATCH (Appelé par bracket_simulator.py)
    # true_probas est de forme (3,)
    # ========================================================
    if true_probas.ndim == 1:
        # Ici, match_phases est une simple chaîne (ex: "16e")
        phase_str = match_phases if isinstance(match_phases, str) else "Phase Finale"
        target_level = get_phase_level(phase_str)
        
        # En phases finales, le drift est un bruit de fond standard
        std_dev = 0.025 if target_level == 4 else 0.035
            
        noise = np.random.normal(1.0, std_dev, 3)
        drifted_probas = true_probas * noise
        
        # Bornes et normalisation vectorielle directe (pas de boucle)
        drifted_probas = np.clip(drifted_probas, MIN_TRUE_PROBA, MAX_TRUE_PROBA)
        return drifted_probas / np.sum(drifted_probas)
        
    # ========================================================
    # CAS 2 : MATRICE DE MATCHS (Appelé par oracle_dp.py)
    # true_probas est de forme (N, 3)
    # ========================================================
    n_matches = len(true_probas)
    drifted_probas = np.copy(true_probas)

    use_dates = match_dates is not None

    # Niveau de phase du match courant (composante "l'équipe joue", présente dans les
    # DEUX régimes : seule en rétro-compat, additionnée à la diffusion sinon).
    if isinstance(match_phases, list) and current_match_idx < len(match_phases):
        current_phase_str = match_phases[current_match_idx]
    else:
        current_phase_str = "Phase Finale"
    current_level = get_phase_level(current_phase_str)

    def _sigma_phase(target_level):
        """Composante PHASE : amplitude du saut d'info quand l'équipe dispute des matchs."""
        shocks = target_level - current_level
        if target_level == 4 or shocks <= 0:
            return sigma_phase_same
        if shocks == 1:
            return sigma_phase_1
        return sigma_phase_2

    if use_dates:
        from datetime import date as _date
        ref = pd.Timestamp(reference_date if reference_date is not None else _date.today()).normalize()
        # sigma_diff(d) = diff_rate * sqrt(d), avec diff_rate tel que sigma_diff(days_diff_ref) = sigma_diff_ref
        diff_rate = sigma_diff_ref / np.sqrt(days_diff_ref)

    for i in range(n_matches):
        # Le passé ou le match du jour est connu avec certitude
        if i <= current_match_idx:
            continue

        phase = match_phases[i] if (isinstance(match_phases, list) and i < len(match_phases)) else "Phase Finale"
        sigma_phase = _sigma_phase(get_phase_level(phase))

        if use_dates:
            di = match_dates[i]
            d = 0 if pd.isna(di) else max(0, (pd.Timestamp(di).normalize() - ref).days)
            sigma_diff = diff_rate * np.sqrt(d)
            # Deux sources de bruit INDÉPENDANTES -> les variances s'additionnent.
            std_dev = np.sqrt(sigma_diff ** 2 + sigma_phase ** 2)
        else:
            std_dev = sigma_phase

        noise = np.random.normal(1.0, std_dev, 3)

        # Ici true_probas[i] est un tableau (3,) et noise est (3,), la multiplication est valide
        drifted_probas[i] = true_probas[i] * noise

        drifted_probas[i] = np.clip(drifted_probas[i], MIN_TRUE_PROBA, MAX_TRUE_PROBA)
        drifted_probas[i] = drifted_probas[i] / np.sum(drifted_probas[i])

    return drifted_probas

def calculate_mpp_gains(
    true_probas: np.ndarray, 
    a=CUBIC_A, b=CUBIC_B, c=CUBIC_C, d=CUBIC_D, 
    add_noise: bool = False, 
    rmse: float = RMSE_GAINS  # La valeur issue du Notebook 12
) -> np.ndarray:
    """
    Calcul des gains MPP à partir des vraies probabilités.
    Utilise le modèle polynomial d'ordre 3 (f(p) = a*p³ + b*p² + c*p + d).
    
    Si add_noise=True, injecte l'erreur de prédiction historique (RMSE) des bookmakers.
    """
    # 1. Calcul de base du polynôme
    gains = a * (true_probas**3) + b * (true_probas**2) + c * true_probas + d
    
    # 2. Ajout du bruit stochastique si demandé
    if add_noise:
        noise = np.random.normal(0, rmse, true_probas.shape)
        gains += noise
        
    # 3. Sécurité des bornes MPP
    gains = np.clip(gains, MIN_MPP_GAIN, MAX_MPP_GAIN)
    
    # 4. Arrondi mathématique correct avant conversion en entier
    return np.round(gains).astype(int)

# --- FONCTIONS LIÉES AU MODÈLE DE FOULE (CROWD) ---
def normalize_crowds(raw_crowds, tol=0.01, label="match", warn=True):
    """
    Normalise les répartitions de foule par ligne (somme = 1).

    Deux usages :
      - warn=True (lecture CSV) : alerte si la somme brute sort de [1-tol, 1+tol].
        Les crowds du CSV sont saisis à 1e-2 près, donc une somme de 0.99/1.01 par
        arrondi est tolérée ; au-delà c'est probablement une faute de saisie.
      - warn=False (mode silencieux) : pour renormaliser des crowds dérivés en
        interne (formule true_probas -> crowd, bruit hétéroscédastique), où une
        somme éloignée de 1 avant renormalisation est NORMALE et non une anomalie.

    NB : apply_heteroscedastic_noise et estimate_crowd_3D renormalisent déjà
    en interne (sans passer par cette fonction), donc le chemin drift est couvert.

    Accepte un tableau (N, 3) ou (3,) ; renvoie toujours du (N, 3) float64.
    """
    raw = np.asarray(raw_crowds, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    sums = raw.sum(axis=1)
    if warn:
        for i in np.where((sums < 1.0 - tol) | (sums > 1.0 + tol))[0]:
            warnings.warn(
                f"Crowd {label} index {i}: somme={sums[i]:.4f} hors "
                f"[{1.0 - tol:.2f}, {1.0 + tol:.2f}] (valeurs={raw[i]}). "
                f"Vérifier la saisie CSV.",
                stacklevel=2,
            )
    safe = np.where(sums == 0.0, 1.0, sums)
    return raw / safe[:, None]


EXPECTED_PHASES = {
    "Poule_J1", "Poule_J2", "Poule_J3", "16e", "8e", "quart", "demi", "finale",
}


def validate_match_dataframe(df, gain_min=14.0, gain_max=250.0):
    """
    Émet des warnings pour les anomalies de saisie probables dans le CSV des matchs.
    N'altère pas le DataFrame ; les colonnes absentes sont simplement ignorées.

    Contrôles :
      - 'phase' hors de l'ensemble attendu (EXPECTED_PHASES) ;
      - gain MPP aberrant (< gain_min ou > gain_max) ;
      - équipe (team_A / team_B) n'apparaissant qu'une seule fois dans tout le
        fichier (faute d'orthographe probable).
    """
    # 1. Phases inattendues
    if "phase" in df.columns:
        unknown = set(df["phase"].dropna().unique()) - EXPECTED_PHASES
        if unknown:
            warnings.warn(
                f"Phase(s) inattendue(s) dans le CSV : {sorted(unknown)}. "
                f"Attendu un sous-ensemble de {sorted(EXPECTED_PHASES)}.",
                stacklevel=2,
            )

    # 2. Gains MPP aberrants
    gain_cols = [c for c in ("gain_mpp_1", "gain_mpp_N", "gain_mpp_2") if c in df.columns]
    if gain_cols:
        g = df[gain_cols].to_numpy(dtype=float)
        for r, c in np.argwhere((g < gain_min) | (g > gain_max)):
            warnings.warn(
                f"Gain MPP aberrant ligne {int(r)}, '{gain_cols[c]}'={g[r, c]:.0f} "
                f"(hors [{gain_min:.0f}, {gain_max:.0f}]).",
                stacklevel=2,
            )

    # 3. Équipe n'apparaissant qu'une seule fois (faute d'orthographe probable)
    team_cols = [c for c in ("team_A", "team_B") if c in df.columns]
    if team_cols:
        noms = pd.concat([df[c] for c in team_cols], ignore_index=True).dropna()
        counts = noms.value_counts()
        for nom in counts[counts == 1].index.tolist():
            warnings.warn(
                f"Équipe '{nom}' n'apparaît qu'une seule fois dans le CSV "
                f"(faute d'orthographe ?).",
                stacklevel=2,
            )


def validate_team_consistency(df_matches, df_group_odds=None, df_market=None, min_apparitions=3):
    """
    Vérifie la cohérence des noms d'équipes ENTRE fichiers (faute d'orthographe).

    Comparaison insensible à la casse / aux espaces (comme bracket_simulator).

    - Chaque équipe de df_group_odds['team'] doit apparaître >= min_apparitions
      fois dans df_matches (team_A/team_B) : en poules chaque équipe joue 3 matchs.
    - Chaque 'favorite' de df_market (category == 'favorite') doit aussi y figurer.
      Les 'scorer' sont des JOUEURS -> ignorés. Il est NORMAL que toutes les équipes
      ne soient pas dans df_market : les gros outsiders sont volontairement omis
      (cotes abaissées par les bookmakers -> sur-normalisation). Le contrôle est
      donc à sens unique (market -> matchs), jamais l'inverse.
    """
    if df_matches is None or not {"team_A", "team_B"} <= set(df_matches.columns):
        return
    counts = (
        pd.concat([df_matches["team_A"], df_matches["team_B"]], ignore_index=True)
        .dropna().astype(str).str.strip().str.lower().value_counts()
    )

    def _check(names, source):
        for nom in names:
            n = int(counts.get(str(nom).strip().lower(), 0))
            if n < min_apparitions:
                warnings.warn(
                    f"Équipe '{nom}' ({source}) apparaît {n}x dans le CSV des matchs "
                    f"(attendu >= {min_apparitions}). Incohérence / faute d'orthographe ?",
                    stacklevel=2,
                )

    if df_group_odds is not None and "team" in df_group_odds.columns:
        _check(df_group_odds["team"].dropna(), "group_stage_odds")
    if df_market is not None and {"category", "selection"} <= set(df_market.columns):
        favs = df_market.loc[df_market["category"] == "favorite", "selection"].dropna()
        _check(favs, "favorite")


def result_to_outcome(result, team_a, team_b):
    """
    Convertit une valeur de la colonne 'result' d'un match en indice d'issue 1N2 :
      - nom de l'équipe à domicile (team_a)  -> 0  ('1')
      - 'nul' (ou n / x / draw)              -> 1  ('N')
      - nom de l'équipe à l'extérieur (team_b) -> 2  ('2')
      - vide / NaN / non reconnu             -> -1 (match futur ou à ignorer)

    Convention alignée sur le knockout (on renseigne le nom du vainqueur, ou 'nul').
    Comparaison insensible à la casse et aux espaces.
    """
    res = str(result).strip().lower()
    if res in ("", "nan", "none"):
        return -1
    if res in ("nul", "n", "x", "draw", "match nul"):
        return 1
    if res == str(team_a).strip().lower():
        return 0
    if res == str(team_b).strip().lower():
        return 2
    return -1


def apply_heteroscedastic_noise(crowds, rmse=0.083):
    """
    Applique un bruit hétéroscédastique (variance max à 0.5) sur une matrice de probabilités.
    Accepte un RMSE scalaire (float) ou dynamique (array 2D).
    """
    # Le bruit est max à 0.5, et tend vers 0 aux extrémités
    scale_factor = np.minimum(crowds, 1.0 - crowds)
    
    # Normalisation pour que l'amplitude du bruit corresponde au RMSE demandé
    mean_scale = np.mean(scale_factor) + 1e-9
    normalized_scale = scale_factor / mean_scale
    
    # Tirage du bruit gaussien (rmse s'adaptera que ce soit un float ou un array)
    noise = np.random.normal(0, 1, crowds.shape) * rmse * normalized_scale
    
    noisy_crowds = crowds + noise
    
    # Sécurité : on empêche les probabilités de devenir aberrantes
    noisy_crowds = np.clip(noisy_crowds, 0.005, 0.995)
    
    # Renormalisation stricte pour que la somme par ligne fasse toujours 100%
    noisy_crowds = noisy_crowds / noisy_crowds.sum(axis=1, keepdims=True)
    
    return noisy_crowds.astype(np.float32)

def estimate_crowd_3D(p1, pN, p2, beta=CROWD_BETA, eps=CROWD_EPSILON, add_noise=False, rmse=0.083):
    # 1. Calcul des "utilités" (attractivité brute de chaque issue)
    u1 = p1 ** beta
    uN = pN ** beta
    u2 = p2 ** beta
    
    # 2. Somme des utilités pour la normalisation (Softmax)
    somme_u = u1 + uN + u2
    
    # 3. Application de la formule : Plancher + (Reste à distribuer * Part Softmax)
    c1 = eps + (1.0 - 3.0 * eps) * (u1 / somme_u)
    cN = eps + (1.0 - 3.0 * eps) * (uN / somme_u)
    c2 = eps + (1.0 - 3.0 * eps) * (u2 / somme_u)
    
    # 4. Injection du Bruit via la fonction centralisée
    if add_noise:
        crowds = np.column_stack((np.atleast_1d(c1), np.atleast_1d(cN), np.atleast_1d(c2)))
        noisy_crowds = apply_heteroscedastic_noise(crowds, rmse=rmse)
        
        # Restitution du bon format (scalaire ou tableau)
        if np.isscalar(p1):
            return noisy_crowds[0, 0], noisy_crowds[0, 1], noisy_crowds[0, 2]
        else:
            return noisy_crowds[:, 0], noisy_crowds[:, 1], noisy_crowds[:, 2]
            
    return c1, cN, c2

# Utilisée pour le pronostic des favoris et meilleurs buteurs (notebook 16)
def generate_drifted_ensemble_crowds(match_du_jour_idx, probas_poules, csv_crowds_poules,
                                     n_ensembles=10, match_dates=None, reference_date=None):
    """
    Ensemble de répartitions de foule bruitées (drift) pour NB16. Le poids du blend
    bayésien (confiance crowd CSV vs crowd théorique) décroît avec l'éloignement du
    match, demi-vie 4 :
      - PAR DATE (si `match_dates` fourni) : éloignement en JOURS depuis `reference_date`
        (def = date de `match_du_jour_idx`), cohérent avec daily_pipeline / NB15 ;
      - PAR INDICE (rétro-compat) : éloignement en nombre de matchs depuis match_du_jour_idx.
    """
    n_poules = len(probas_poules)

    # 1. Calcul des distances et des Alphas (Demi-vie de 4 : jours si dates, sinon matchs)
    if match_dates is not None:
        ref = pd.Timestamp(reference_date if reference_date is not None
                           else match_dates[match_du_jour_idx]).normalize()
        dist_units = np.array([
            0.0 if pd.isna(d) else max(0, (pd.Timestamp(d).normalize() - ref).days)
            for d in match_dates
        ], dtype=np.float64)
    else:
        dist_units = np.maximum(0, np.arange(n_poules) - match_du_jour_idx).astype(np.float64)
    alphas = 0.95 * (0.5 ** (dist_units / 4.0))
    alphas_2d = alphas.astype(np.float32)[:, np.newaxis]
    
    # 2. Crowd Théorique Pur
    c1, cN, c2 = estimate_crowd_3D(
        probas_poules[:, 0], probas_poules[:, 1], probas_poules[:, 2], 
        add_noise=False
    )
    theo_crowds_pure = np.column_stack((c1, cN, c2)).astype(np.float32)
    
    # 3. Le Lissage Bayésien
    blended_mean_crowds = (alphas_2d * csv_crowds_poules) + ((1.0 - alphas_2d) * theo_crowds_pure)
    
    # 4. RMSE Dynamique
    dynamic_rmse = 0.083 * (1.0 - alphas_2d)
    
    # 5. Génération du tenseur (N_ensembles, N_poules, 3)
    ensemble_crowds = np.zeros((n_ensembles, n_poules, 3), dtype=np.float32)
    for e in range(n_ensembles):
        ensemble_crowds[e] = apply_heteroscedastic_noise(blended_mean_crowds, rmse=dynamic_rmse)

    return ensemble_crowds


# ==========================================================================
# SCORES EXACTS (match du jour) — barème de bonus + construction du marché
# ==========================================================================
# En plus du gain d'outcome (gain_mpp[1N2]), MPP attribue un BONUS si le score
# exact est trouvé. Le bonus est d'autant plus gros que le score est RARE parmi
# les joueurs qui ont déjà trouvé le bon outcome (crowd conditionnel `cc`).

def exact_score_bonus(cond_crowd: float) -> int:
    """
    Bonus de points du score exact en fonction du crowd CONDITIONNEL `cc`
    (= part des joueurs ayant trouvé le bon outcome qui ont aussi le bon score).

      cc > 30%        -> 20
      20% < cc <= 30% -> 30
      5%  < cc <= 20% -> 50
      0.5%< cc <= 5%  -> 70
      cc <= 0.5%      -> 100
    """
    if cond_crowd > 0.30:
        return 20
    if cond_crowd > 0.20:
        return 30
    if cond_crowd > 0.05:
        return 50
    if cond_crowd > 0.005:
        return 70
    return 100


def _score_to_outcome(score) -> int:
    """Parse un score "b1-b2" en indice d'issue 1N2 : b1>b2 -> 0, b1==b2 -> 1, b1<b2 -> 2."""
    b1, b2 = (int(x) for x in str(score).split("-"))
    if b1 > b2:
        return 0
    if b1 == b2:
        return 1
    return 2


# Structure alignée décrivant le marché des scores exacts d'un match.
#   scores      : list[str]            noms des scores listés (cote valide)
#   outcomes    : ndarray int8 (K,)    issue 1N2 de chaque score (0/1/2)
#   p_score     : ndarray float64 (K,) proba vraie (Shin), somme = 1
#   cond_crowd  : ndarray float64 (K,) crowd normalisé PAR OUTCOME (somme = 1 par issue)
#   bonus       : ndarray int64 (K,)   bonus de points (barème exact_score_bonus)
ExactScoreMarket = namedtuple(
    "ExactScoreMarket", ["scores", "outcomes", "p_score", "cond_crowd", "bonus"]
)


def _fit_poisson_lambdas(mpp_outcome_crowd, goal_grid=EXACT_SCORE_GOAL_GRID):
    """
    Ajuste (λ1, λ2) de deux Poisson indépendants (buts dom / ext) pour que les masses
    des 3 régions de la grille de scores — victoire dom (i>j), nul (i=j), victoire ext
    (i<j) — collent au crowd MPP 1N2 `mpp_outcome_crowd` = [q1, qN, q2]. On matche q1
    (P(i>j)) et q2 (P(i<j)) ; le nul suit. Grille grossière + raffinement local
    (déterministe). Renvoie (λ1, λ2).
    """
    q = np.asarray(mpp_outcome_crowd, dtype=float)
    q = q / q.sum()
    gi = np.arange(goal_grid + 1)

    def regions(l1, l2):
        J = np.outer(poisson.pmf(gi, max(l1, 1e-6)), poisson.pmf(gi, max(l2, 1e-6)))
        return float(np.tril(J, -1).sum()), float(np.triu(J, 1).sum())  # (win_dom, win_ext)

    def err(x):
        w, lo = regions(x[0], x[1])
        return (w - q[0]) ** 2 + (lo - q[2]) ** 2

    best, bl = None, (1.0, 1.0)
    for l1 in np.linspace(0.1, 5.0, 50):
        for l2 in np.linspace(0.05, 3.0, 40):
            e = err((l1, l2))
            if best is None or e < best:
                best, bl = e, (l1, l2)
    # Raffinement local (Nelder-Mead borné, départ = meilleur point de grille)
    res = minimize(err, np.array(bl), method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-8, maxiter=400))
    l1, l2 = res.x if res.success or err(res.x) < best else bl
    return max(l1, 1e-6), max(l2, 1e-6)


def correct_cond_crowd(crowds_raw, scores, outcomes, mpp_outcome_crowd=None,
                       delta00=EXACT_SCORE_ZERO_ZERO_LOG_PENALTY,
                       blend_wmx=EXACT_SCORE_BLEND_WMX,
                       goal_grid=EXACT_SCORE_GOAL_GRID):
    """
    Corrige le crowd des scores exacts Winamax vers une estimation du crowd
    conditionnel MPP (modèle calibré au Notebook 25_). Renvoie le crowd CONDITIONNEL
    corrigé (normalisé PAR OUTCOME), de même forme que `crowds_raw`.

    `crowds_raw` : crowd Winamax BRUT par score (non normalisé ; 0 admis).
    `scores`     : liste des scores "b1-b2" alignée.
    `outcomes`   : issue 1N2 (0/1/2) de chaque score.
    `mpp_outcome_crowd` : crowd MPP 1N2 du match [q1, qN, q2] (= la 1N2 du CSV principal).

    DEUX RÉGIMES :
      - `mpp_outcome_crowd` fourni (chemin nominal du pipeline) : BLEND JOINT
            cc(s) ∝ [ a·wmx_glob(s) + (1-a)·surf_glob(s) ] · exp(delta00·1[0-0])
        renormalisé par outcome, où `wmx_glob` = crowd Winamax normalisé GLOBALEMENT,
        `surf_glob` = surface Poisson(λ1,λ2) (ajustée sur le crowd MPP 1N2 via
        `_fit_poisson_lambdas`) normalisée globalement, `a` = `blend_wmx`. La surface
        INJECTE de la masse là où le crowd Winamax est nul (scores plausibles à 0 %).
      - `mpp_outcome_crowd` None (rétro-compat / tests sans 1N2) : SEULE la pénalité
        0-0 est appliquée sur le crowd Winamax conditionnel (le 0-0 d'un outcome nul
        multi-scores est dévalué de exp(delta00) ; no-op sinon ; crowd nul reste nul).

    Le bonus de score exact et la proba de bonus de Bob/peloton (DP) consomment ce
    crowd corrigé. LIMITE : blowout extrême -> sur-estimation du nombre de buts (NB25).
    """
    cr = np.asarray(crowds_raw, dtype=float)
    outc = np.asarray(outcomes)
    is00 = np.array([
        1.0 if str(s).strip() in ("0-0", "0 - 0") else 0.0 for s in scores
    ], dtype=float)

    # Crowd Winamax CONDITIONNEL (normalisé par outcome) — composante / base du fallback.
    wmx_cond = np.zeros_like(cr)
    for o in np.unique(outc):
        mask = outc == o
        tot = cr[mask].sum()
        if tot > 0.0:
            wmx_cond[mask] = cr[mask] / tot

    # --- Régime rétro-compat : pénalité 0-0 seule sur le conditionnel Winamax ---
    if mpp_outcome_crowd is None:
        out = wmx_cond.copy()
        for o in np.unique(outc):
            mask = outc == o
            if not is00[mask].any():
                continue  # no-op : pas de 0-0 dans cet outcome
            w = np.where(wmx_cond[mask] > 0.0,
                         wmx_cond[mask] * np.exp(delta00 * is00[mask]), 0.0)
            tot = w.sum()
            if tot > 0.0:
                out[mask] = w / tot
        return out

    # --- Régime nominal : BLEND JOINT (Winamax + surface Poisson ancrée 1N2 MPP) ---
    l1, l2 = _fit_poisson_lambdas(mpp_outcome_crowd, goal_grid)
    ij = np.array([[int(x) for x in str(s).split("-")] for s in scores])
    gp1 = poisson.pmf(np.minimum(ij[:, 0], goal_grid), l1)
    gp2 = poisson.pmf(np.minimum(ij[:, 1], goal_grid), l2)
    surf = gp1 * gp2

    # Composantes normalisées GLOBALEMENT (-> masses de région signifiantes), blend joint.
    wmx_glob = cr / cr.sum() if cr.sum() > 0.0 else np.zeros_like(cr)
    surf_glob = surf / surf.sum() if surf.sum() > 0.0 else surf
    a = blend_wmx if cr.sum() > 0.0 else 0.0   # pas de crowd Winamax -> surface seule
    blended = a * wmx_glob + (1.0 - a) * surf_glob
    blended = blended * np.exp(delta00 * is00)

    out = np.zeros_like(cr)
    for o in np.unique(outc):
        mask = outc == o
        tot = blended[mask].sum()
        if tot > 0.0:
            out[mask] = blended[mask] / tot
    return out


def build_exact_score_market(exact_score_data, outcome_probas=None,
                             outcome_tol=0.08, shape_correction=True,
                             mpp_outcome_crowd=None) -> ExactScoreMarket:
    """
    Construit le marché des scores exacts à partir d'un dict
    `{ "b1-b2": (cote, crowd_pct), ... }`. Les slots à cote absente
    (None / <= 0) sont ignorés ; `crowd_pct` None est traité comme 0.

    - probas vraies via `calculate_true_outcome_probas_from_odds` (forme 1D ->
      méthode de Shin sur K issues d'UN événement) ;
    - crowd conditionnel : crowd(s) normalisé par la somme des crowds des scores
      du MÊME outcome (cf. barème de bonus) ;
    - bonus : `exact_score_bonus(cond_crowd)`.

    Lève ValueError si aucun score n'a de cote valide.

    ANCRAGE 1N2 (`outcome_probas`, array (3,) optionnel) :
      Les cotes de scores exacts (souvent issues d'un autre book, ex. Winamax) sont
      MOINS fiables que le 1N2 du CSV principal. Si `outcome_probas` est fourni
      (typiquement `base_true_probas[match_idx]`), on ANCRE : à l'intérieur de
      chaque outcome o, les probas des scores listés sont remises à l'échelle pour
      sommer EXACTEMENT à `outcome_probas[o]`. Les cotes de scores ne fixent alors
      que la répartition RELATIVE intra-outcome, jamais le total de l'outcome.
      Conséquence : `Σ_{out(s)=o} p_score(s) == P(o)` par construction (le total
      ne peut être ni inférieur ni supérieur à P(o)).

      Si `outcome_probas` est None (rétro-compat / tests), on RENORMALISE simplement
      le vecteur global à 1 (outright partiel : masse des scores non listés repliée
      sur les listés).

    CORRECTION DU CROWD (`shape_correction`, défaut True) : le crowd Winamax est corrigé
    vers le crowd MPP via `correct_cond_crowd` (modèle NB25) AVANT le barème. Le `cond_crowd`
    renvoyé est donc déjà corrigé (et alimente aussi la proba de bonus de Bob/peloton dans la
    DP). Si `mpp_outcome_crowd` (= 1N2 MPP du match, ex. `base_crowds[match_idx]`) est fourni,
    la correction applique le BLEND JOINT Winamax + surface Poisson ancrée sur ce 1N2 (capte
    le glissement de marge des matchs déséquilibrés + injecte de la masse sur les scores à
    crowd Winamax nul) ; sinon, repli sur la seule pénalité 0-0. Mettre `shape_correction=False`
    pour récupérer le crowd Winamax brut conditionnel (cc_wmx) sans aucune correction.
    """
    scores, odds, crowds, outcomes = [], [], [], []
    for score, data in exact_score_data.items():
        cote, crowd = data
        if cote is None or cote <= 0:
            continue
        scores.append(str(score))
        odds.append(float(cote))
        crowds.append(float(crowd) if crowd else 0.0)
        outcomes.append(_score_to_outcome(score))

    if not scores:
        raise ValueError("build_exact_score_market : aucun score avec une cote valide.")

    odds = np.asarray(odds, dtype=float)
    outcomes = np.asarray(outcomes, dtype=np.int8)
    crowds = np.asarray(crowds, dtype=float)

    # Probas vraies relatives (Shin sur les K cotes)
    p_score = np.asarray(calculate_true_outcome_probas_from_odds(odds), dtype=float)

    if outcome_probas is not None:
        # ANCRAGE : chaque bloc d'outcome est remis à l'échelle à P(o) du CSV 1N2.
        op = np.asarray(outcome_probas, dtype=float)

        # VALIDATION anti-faute de saisie : comparer le 1N2 BRUT agrégé des scores
        # exacts (Shin normalisé à 1) au 1N2 du CSV. Un gros écart trahit soit une
        # coquille de cote, soit une liste de scores trop incomplète pour un outcome.
        raw_total = float(p_score.sum())
        if raw_total > 0.0:
            raw_norm = p_score / raw_total
            for o in (0, 1, 2):
                agg_o = float(raw_norm[outcomes == o].sum())
                if abs(agg_o - op[o]) > outcome_tol:
                    warnings.warn(
                        f"Scores exacts : 1N2 agrégé de l'outcome {o} = {agg_o:.1%} "
                        f"vs {op[o]:.1%} (CDM_2026) — écart > {outcome_tol:.0%}. "
                        f"Vérifier la saisie des cotes (ou liste de scores incomplète).",
                        stacklevel=2,
                    )

        p_anchored = np.zeros_like(p_score)
        for o in (0, 1, 2):
            mask = outcomes == o
            s = float(p_score[mask].sum())
            if s > 0.0:
                p_anchored[mask] = p_score[mask] / s * op[o]
            # si aucun score listé pour o, la masse P(o) est perdue (renorm. ci-dessous)
        p_score = p_anchored

    # Renormalisation finale à 1 (no-op si ancrage et 3 outcomes représentés ;
    # rattrape la masse perdue d'un outcome sans score listé, ou un overround creux).
    total = float(p_score.sum())
    if total > 0.0:
        p_score = p_score / total

    # Crowd conditionnel : normalisation PAR OUTCOME (utilisée telle quelle si
    # shape_correction=False ; sinon recalculée par correct_cond_crowd à partir du brut).
    cond_crowd = np.zeros_like(crowds)
    for o in (0, 1, 2):
        mask = outcomes == o
        tot = float(crowds[mask].sum())
        if tot > 0.0:
            cond_crowd[mask] = crowds[mask] / tot

    # Correction du crowd Winamax -> MPP (NB25), AVANT le barème -> le cc corrigé pilote le
    # bonus ET sert de proba de bonus de Bob/peloton dans la DP (evaluate_exact_score_day).
    # Si `mpp_outcome_crowd` (1N2 MPP du match) est fourni : blend joint Winamax + surface
    # Poisson ancrée 1N2 ; sinon repli sur la seule pénalité 0-0. On passe le crowd BRUT
    # (correct_cond_crowd gère les normalisations par-outcome et globale).
    if shape_correction:
        cond_crowd = correct_cond_crowd(crowds, scores, outcomes,
                                        mpp_outcome_crowd=mpp_outcome_crowd)

    bonus = np.asarray([exact_score_bonus(cc) for cc in cond_crowd], dtype=np.int64)

    return ExactScoreMarket(scores, outcomes, p_score, cond_crowd, bonus)


def expected_simple_points(market: ExactScoreMarket) -> np.ndarray:
    """
    Espérance de points de chaque score listé sous un barème SIMPLE (indépendant
    du modèle MPP/bonus), utile pour repérer les paris robustes :
      3 pts  si score exact ;
      2 pts  si bon outcome ET bonne différence de buts (mais pas le score exact) ;
      1 pt   si bon outcome mais mauvaise différence de buts ;
      0 pt   sinon.

    L'espérance est prise sur la distribution `market.p_score` des scores RÉELS
    listés (les scores non listés sont ignorés — approximation d'affichage).

    Retourne un array (K,) aligné sur `market.scores`.
    """
    goals = [tuple(int(x) for x in s.split("-")) for s in market.scores]
    diffs = np.array([g[0] - g[1] for g in goals], dtype=np.int64)
    outcomes = np.asarray(market.outcomes)
    p = np.asarray(market.p_score, dtype=float)
    K = len(market.scores)

    E = np.zeros(K, dtype=float)
    for kb in range(K):                      # pari = score kb
        ob, db = outcomes[kb], diffs[kb]
        same_out = outcomes == ob
        same_diff = same_out & (diffs == db)
        # points par score réel ks : 1 (bon outcome) + 1 (bonne diff) + 1 (exact)
        pts = same_out.astype(float) + same_diff.astype(float)
        pts[kb] += 1.0                       # le score exact ajoute le 3e point
        E[kb] = float(np.dot(p, pts))
    return E


def expected_mpp_points(market: ExactScoreMarket, gains_1N2) -> np.ndarray:
    """
    Espérance de points MPP RÉELS de chaque score listé (hors booster) :
        E[pts] = P(bon outcome) * gain_mpp[outcome] + P(score exact) * bonus

    P(bon outcome) = somme des probas des scores du même outcome (= P(o) ancré) ;
    P(score exact) = `market.p_score` du score ; `bonus` = barème exact_score_bonus.
    Le booster ×2 doublerait cette valeur (cf. WR x2). Array (K,) aligné sur scores.
    """
    outcomes = np.asarray(market.outcomes)
    p = np.asarray(market.p_score, dtype=float)
    bonus = np.asarray(market.bonus, dtype=float)
    gains = np.asarray(gains_1N2, dtype=float)

    p_outcome = np.array([p[outcomes == o].sum() for o in (0, 1, 2)])
    E_gain = p_outcome[outcomes] * gains[outcomes]
    return E_gain + p * bonus


# ==========================================================================
# MODÈLE CHEAP DE BONUS DE SCORE EXACT POUR L'HORIZON (dé-biais du booster x2)
# ==========================================================================
# La DP lointaine ne simule que le 1N2 (bonus 0), alors que la DP proche modélise le
# vrai bonus de score exact des matchs renseignés. Du coup les matchs futurs ont une
# E[points] sous-estimée -> la valeur de GARDER le x2 pour plus tard est sous-estimée
# -> la DP sur-recommande de poser le x2 sur le match courant exact-aware. On corrige
# en injectant dans l'horizon un bonus SYNTHÉTIQUE par issue, calibré sur les marchés
# réellement renseignés (cf. solve_dp_coarse_hbonus). Modèle HYBRIDE : l'agent décroche
# le bonus de façon STOCHASTIQUE (la queue +50/+100 compte, le x2 la double) ; Bob et le
# peloton le touchent en MOYENNE (déterministe, moins cher, leur variance pèse peu).

# active        : bool — False si aucune donnée (horizon = 1N2 pur).
# draw          : (q_agent, B_agent, mean_opp) pour les nuls (issue 1).
# dec_thresholds: (2,) seuils de P(o) séparant outsider / médian / favori fort.
# dec_bins      : tuple de 3 (q_agent, B_agent, mean_opp) pour les issues décisives (0/2).
HorizonBonusModel = namedtuple(
    "HorizonBonusModel", ["active", "draw", "dec_thresholds", "dec_bins"]
)


def calibrate_horizon_bonus_model(exact_scores_by_match, base_true_probas,
                                  base_crowds=None, dec_thresholds=(0.35, 0.55)):
    """
    Calibre un HorizonBonusModel à partir des marchés de scores exacts renseignés.

    Pour chaque issue o d'un match renseigné (marché ancré sur le 1N2 du CSV) :
      - q_agent  = proba conditionnelle du MEILLEUR score de o (l'agent parie le score
                   modal -> taux où il décroche le bonus quand o sort) ; B_agent = son bonus ;
      - mean_opp = E[bonus | o sort, bon outcome] = Σ_s cond(s|o)·cc(s)·bonus(s)
                   (Bob/peloton, atténué par le crowd conditionnel cc < 1).
    Les échantillons sont moyennés par classe : un bucket NUL (o==1, dominé par 0-0/1-1),
    et 3 buckets DÉCISIFS (o∈{0,2}) selon la force du favori P(o) (seuils dec_thresholds).
    Buckets vides -> repli sur la moyenne décisive globale, puis globale toutes issues.

    `base_true_probas` (n,3) : 1N2 du CSV (ancrage + alignement des index de match).
    `base_crowds` (n,3, optionnel) : 1N2 MPP du CSV -> passé en `mpp_outcome_crowd` à
    `build_exact_score_market` pour que les bonus calibrés correspondent EXACTEMENT aux
    marchés que la DP consomme (blend joint NB25). None -> repli pénalité 0-0 seule.
    Renvoie active=False si aucun échantillon exploitable (horizon laissé en 1N2 pur).
    """
    thr = np.asarray(dec_thresholds, dtype=float)
    draw_samples = []
    dec_samples = [[], [], []]
    all_samples = []

    for mid, data in exact_scores_by_match.items():
        idx = int(mid) - 1
        if idx < 0 or idx >= len(base_true_probas):
            continue
        try:
            m = build_exact_score_market(
                data, outcome_probas=base_true_probas[idx], shape_correction=True,
                mpp_outcome_crowd=None if base_crowds is None else base_crowds[idx],
            )
        except ValueError:
            continue
        outc = np.asarray(m.outcomes)
        p = np.asarray(m.p_score, dtype=float)
        cc = np.asarray(m.cond_crowd, dtype=float)
        bonus = np.asarray(m.bonus, dtype=float)
        for o in (0, 1, 2):
            mask = outc == o
            if not mask.any():
                continue
            P_o = float(p[mask].sum())
            if P_o <= 0.0:
                continue
            cond = p[mask] / P_o
            j = int(np.argmax(cond))
            sample = (
                float(cond[j]),                                  # q_agent
                float(bonus[mask][j]),                           # B_agent
                float(np.sum(cond * cc[mask] * bonus[mask])),    # mean_opp
            )
            all_samples.append(sample)
            if o == 1:
                draw_samples.append(sample)
            else:
                dec_samples[int(np.digitize(P_o, thr))].append(sample)

    if not all_samples:
        zero = (0.0, 0.0, 0.0)
        return HorizonBonusModel(False, zero, thr, (zero, zero, zero))

    def _mean(samples, fallback):
        if not samples:
            return fallback
        return tuple(np.asarray(samples, dtype=float).mean(axis=0))

    glob = _mean(all_samples, (0.0, 0.0, 0.0))
    dec_glob = _mean([s for b in dec_samples for s in b], glob)
    draw = _mean(draw_samples, glob)
    dec_bins = tuple(_mean(dec_samples[b], dec_glob) for b in range(3))
    return HorizonBonusModel(True, draw, thr, dec_bins)


def build_horizon_bonus_arrays(true_probas, model: HorizonBonusModel):
    """
    Construit les tableaux (n,3) à injecter dans solve_dp_coarse_hbonus à partir d'un
    HorizonBonusModel et d'une matrice de probas 1N2 (n,3) :
      q_agent  : proba que l'agent décroche le bonus s'il parie cette issue et qu'elle sort ;
      B_agent  : bonus agent associé (points PLEINS) ;
      mean_opp : E[bonus adverse | issue, bon outcome] (points PLEINS, Bob/peloton).
    Issue nulle -> bucket draw ; issue décisive -> bucket selon P(o) (seuils du modèle).
    `model.active == False` -> tableaux de zéros (horizon 1N2 pur, no-op dans la DP).
    """
    n = len(true_probas)
    q_agent = np.zeros((n, 3), dtype=np.float64)
    B_agent = np.zeros((n, 3), dtype=np.float64)
    mean_opp = np.zeros((n, 3), dtype=np.float64)
    if not model.active:
        return q_agent, B_agent, mean_opp
    thr = np.asarray(model.dec_thresholds, dtype=float)
    for t in range(n):
        for o in (0, 1, 2):
            if o == 1:
                q, B, mo = model.draw
            else:
                q, B, mo = model.dec_bins[int(np.digitize(true_probas[t, o], thr))]
            q_agent[t, o] = q
            B_agent[t, o] = B
            mean_opp[t, o] = mo
    return q_agent, B_agent, mean_opp


def load_exact_scores(csv_path, match_id) -> dict:
    """
    Charge les scores exacts d'un match depuis un CSV (colonnes `match_id, score,
    cote, crowd`) et renvoie le dict `{ "b1-b2": (cote, crowd) }` attendu par
    `build_exact_score_market` / `run_daily_pipeline(exact_score_data=...)`.

    Avantage vs un dict codé dans le notebook : données scopées par `match_id`
    (pas de mise à zéro défensive, pas de risque de syntaxe Python, éditable au
    tableur, historique conservé).

    Cote vide (NaN) -> None (score ignoré en aval) ; crowd vide -> None (= 0).
    Lève ValueError si colonnes manquantes ou aucun score pour ce `match_id`.
    """
    df = pd.read_csv(csv_path)
    required = {"match_id", "score", "cote", "crowd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"load_exact_scores : colonnes manquantes {sorted(missing)} dans {csv_path}."
        )
    sub = df[df["match_id"] == match_id]
    if sub.empty:
        raise ValueError(
            f"load_exact_scores : aucun score pour match_id={match_id} dans {csv_path}."
        )
    data = {}
    for _, r in sub.iterrows():
        cote = None if pd.isna(r["cote"]) else float(r["cote"])
        crowd = None if pd.isna(r["crowd"]) else float(r["crowd"])
        data[str(r["score"]).strip()] = (cote, crowd)
    return data


def load_exact_scores_by_match(csv_path) -> dict:
    """
    Charge TOUS les matchs du CSV des scores exacts et renvoie un dict-of-dicts
    `{ match_id (int): { "b1-b2": (cote, crowd) } }`. Sert au mode multi-matchs
    (`run_daily_pipeline(exact_scores_by_match=...)`) : la décision du match courant
    devient « exact-aware » sur tous les matchs de la nuit présents dans le CSV, et
    on obtient une reco par match.

    Même conventions que `load_exact_scores` (NaN -> None). Lève ValueError si
    colonnes manquantes.
    """
    df = pd.read_csv(csv_path)
    required = {"match_id", "score", "cote", "crowd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"load_exact_scores_by_match : colonnes manquantes {sorted(missing)} dans {csv_path}."
        )
    by_match = {}
    for _, r in df.iterrows():
        mid = int(r["match_id"])
        cote = None if pd.isna(r["cote"]) else float(r["cote"])
        crowd = None if pd.isna(r["crowd"]) else float(r["crowd"])
        by_match.setdefault(mid, {})[str(r["score"]).strip()] = (cote, crowd)
    return by_match