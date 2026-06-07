"""
Core logic and utility functions for the MonPetitProno (MPP) analysis project.
"""

import warnings

import pandas as pd
import numpy as np
import unicodedata
from scipy.stats import binom
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

MIN_MPP_GAIN = 15.0  # Gain MPP minimum réaliste (allemaagne - curacao : 15.0)
MAX_MPP_GAIN = 225.0  # Gain MPP maximum réaliste (allemaagne - curacao : 222.0)
MIN_TRUE_PROBA = 0.02  # Probabilité minimale réaliste d'une issue (allemaagne - curacao : 0.02)
MAX_TRUE_PROBA = 0.95  # Probabilité maximale réaliste d'une issue (allemaagne - curacao : 0.95)

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
    Calculates normalized ground truth probabilities from bookmaker odds.
    Expects odds to be a NumPy array of shape (N, 3) or (3,).
    """
    # Conversion en float pour éviter les divisions entières ou les types mixtes
    odds_array = np.asarray(odds, dtype=float)
    
    # 1. Calcul de l'inverse des cotes
    inv_odds = 1.0 / odds_array
    
    # 2. Somme par ligne (par match)
    # Si le tableau a une seule dimension (ex: un seul match [2.0, 3.0, 3.0]),
    # sum() s'applique normalement. Si c'est 2D (N, 3), on utilise axis=1.
    if inv_odds.ndim == 2:
        total_inv_odds = np.sum(inv_odds, axis=1, keepdims=True)
    else:
        total_inv_odds = np.sum(inv_odds)

    # 2bis. Garde-fou : la somme des probabilités implicites (1/cote) vaut
    # 1 + marge bookmaker. Une marge NÉGATIVE (somme < 1) est impossible pour un
    # vrai book (ce serait une opportunité d'arbitrage) -> faute de saisie probable.
    _sums = np.atleast_1d(total_inv_odds).ravel()
    for i, s in enumerate(_sums):
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

    # 3. Sécurité contre la division par zéro
    # np.where est très puissant pour faire des conditions sans boucle for
    safe_total = np.where(total_inv_odds == 0, 1.0, total_inv_odds)
    
    # 4. Normalisation
    true_probas = inv_odds / safe_total
    
    # On remet à zéro les lignes où le total était nul
    if inv_odds.ndim == 2:
        true_probas = np.where(total_inv_odds == 0, 0.0, true_probas)
    else:
        true_probas = np.where(total_inv_odds == 0, 0.0, true_probas)
        
    return true_probas

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

def apply_temporal_drift(true_probas: np.ndarray, match_phases, current_match_idx: int = 0) -> np.ndarray:
    """
    Simule la variation des probabilités réelles au fil du temps via un modèle de 'Drift Relatif'.
    Gère à la fois une matrice de matchs (N, 3) et un match unique vectorisé (3,).
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
    
    # Récupération sécurisée (évite les IndexError)
    if isinstance(match_phases, list) and current_match_idx < len(match_phases):
        current_phase_str = match_phases[current_match_idx]
    else:
        current_phase_str = "Phase Finale"
        
    current_level = get_phase_level(current_phase_str)
    
    for i in range(n_matches):
        phase = match_phases[i] if (isinstance(match_phases, list) and i < len(match_phases)) else "Phase Finale"
        
        # Le passé ou le match du jour est connu avec certitude
        if i <= current_match_idx:
            continue
            
        target_level = get_phase_level(phase)
        shocks_remaining = target_level - current_level
        
        # --- LOGIQUE DE DRIFT RELATIF ---
        if target_level == 4:
            std_dev = 0.025
        elif shocks_remaining <= 0:
            std_dev = 0.025 
        elif shocks_remaining == 1:
            std_dev = 0.035 
        elif shocks_remaining >= 2:
            std_dev = 0.06 
        else:
            std_dev = 0.025 
            
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
def generate_drifted_ensemble_crowds(match_du_jour_idx, probas_poules, csv_crowds_poules, n_ensembles=10):
    n_poules = len(probas_poules)
    
    # 1. Calcul des distances et des Alphas (Demi-vie de 4 matchs)
    dists = np.arange(n_poules) - match_du_jour_idx
    dists_positive = np.maximum(0, dists)
    alphas = 0.95 * (0.5 ** (dists_positive / 4.0))
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