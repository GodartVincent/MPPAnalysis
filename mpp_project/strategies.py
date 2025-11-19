"""
Defines the different betting strategies to be compared in the simulation.

Each function takes the current match data and player scores and returns
the chosen bet (0 for Fav, 1 for Draw, 2 for Outsider).
"""

import numpy as np
import os
from stable_baselines3 import PPO

# --- Model Loading Setup (Refactored) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Cache for loaded models to prevent reloading them 5000 times
_LOADED_MODELS = {}

def get_model(model_filename):
    """Lazy loads a model by filename."""
    global _LOADED_MODELS
    if model_filename not in _LOADED_MODELS:
        model_path = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(model_path):
            print(f"Loading RL Model from {model_path}...")
            _LOADED_MODELS[model_filename] = PPO.load(model_path)
        else:
            print(f"WARNING: Model not found at {model_path}")
            _LOADED_MODELS[model_filename] = None
            
    return _LOADED_MODELS[model_filename]

# --- Strategy Definitions ---

def strat_typical_opponent(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """Strategy 0: Bet on the favorite with a proba of 1 - (1-p_favorite)**2 (empircal formula)."""
    return np.random.choice([0, 1, 2], p=opp_repartition)

def strat_random(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """Strategy 1: Bet randomly."""
    return np.random.randint(0, 3)

def strat_best_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """Strategy 2: Bet on the outcome with the highest simple EV (gtp * g)."""
    evs = match_probas * match_gains
    return np.argmax(evs)

def strat_best_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """Strategy 3: Bet on the outcome with the best * Simple Relative* EV."""
    # Formula: E[Simple_Rel_EV_i] = gtpi*gi*(1-pi)
    evs = match_probas * match_gains
    rel_evs = evs * (1 - opp_repartition)
    return np.argmax(rel_evs)

def strat_favorite(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """Strategy 4: Most Probable Outcome."""
    # It bets on the outcome with the highest probability
    return np.argmax(match_probas)

def strat_safe(match_probas, evs):
    # If the best bet is not much better than the 2nd best,
    # check the simple probability (a proxy for risk).
    sorted_indices = np.argsort(evs)
    best_rel_ev = evs[sorted_indices[-1]]
    second_best_rel_ev = evs[sorted_indices[-2]]
    
    # If the EV is too close, pick the one with the highest simple proba
    # among the top two EV bets.
    if best_rel_ev < (second_best_rel_ev + 1) and match_probas[sorted_indices[-1]] < match_probas[sorted_indices[-2]]:
        return sorted_indices[-2]
            
    return sorted_indices[-1]

def strat_safe_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 5: Safe Simple Relative EV.
    Chooses best Simple Rel EV, but if it's close, checks the "risk".
    """
    evs = match_probas * match_gains
    rel_evs = evs * (1 - opp_repartition)
    
    return strat_safe(match_probas, rel_evs)

def strat_adaptive(match_probas, match_gains, opp_repartition, player_scores, my_idx, default_strat, matches_remaining):
    """
    If leading, does blocking bets. If far behind, plays it aggressively.
    If 100+ points ahead of 2nd place, bet on the most popular bet.
    If 150+ points below the 2nd place, bet on the highest gain.
    Otherwise, follow default_strat.
    """
    my_score = player_scores[my_idx]
    
    # Find the leader's score (excluding myself)
    other_scores = np.delete(player_scores, my_idx)
    leader_score = np.max(other_scores)
    
    if (my_score - leader_score) > 100:
        # I'm in the lead, blocking bet.
        return np.argmax(opp_repartition)
    if (leader_score - my_score) > 150:
        # I'm far behind, play aggressively (best on highest gain)
        return np.argmax(match_gains)
    # I'm not in the lead, but not far behind either, play for best Simple Relative EV
    return default_strat(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)

def strat_adaptive_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 6: If leading, does blocking bets. If far behind, plays it aggressively.
    If 100+ points ahead of 2nd place, bet on the most popular bet.
    If 150+ points below the 2nd place, bet on the highest gain.
    Otherwise, bet on the best relative EV.
    """
    return strat_adaptive(
        match_probas, match_gains, opp_repartition, player_scores, my_idx,
        strat_best_simple_rel_ev, matches_remaining
    )

def strat_safe_simple_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 7: Safe Simple EV.
    Chooses best Simple EV, but if it's close, checks the "risk".
    """
    evs = match_probas * match_gains
    
    return strat_safe(match_probas, evs)

def strat_adaptive_simple_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 8: If leading, does blocking bets. If far behind, plays it aggressively.
    If 100+ points ahead of 2nd place, bet on the most popular bet.
    If 150+ points below the 2nd place, bet on the highest gain.
    Otherwise, bet on the best EV.
    """    
    return strat_adaptive(
        match_probas, match_gains, opp_repartition, player_scores, my_idx,
        strat_best_ev, matches_remaining
    )

def strat_highest_variance(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """Strategy 9: Highest variance outcome."""
    # Variance = g^2 * p * (1-p)
    variances = (match_gains ** 2) * match_probas * (1 - match_probas)
    return np.argmax(variances)

# --- RL AGENTS ---

def _predict_with_rl(model_filename, match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining):
    """Helper to run inference for both Phase 1 and Phase 2 agents."""
    model = get_model(model_filename)
    if model is None:
        # Fallback if model fails to load
        return strat_best_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)

    # --- Construct Observation ---
    # The agent expects: [Probas(3), Gains(3), Repart(3), Scores(12), MatchesRemaining(1)]
    
    # Correct Score Ordering: The agent always thinks it is Player 0.
    # We must put 'my_score' first in the score vector.
    my_score = player_scores[my_idx]
    other_scores = np.delete(player_scores, my_idx)
    ordered_scores = np.concatenate(([my_score], other_scores))

    # Ensure matches_remaining is an array
    matches_rem_arr = np.array([float(matches_remaining)])

    obs = np.concatenate([
        match_probas,
        match_gains,
        opp_repartition,
        ordered_scores,
        matches_rem_arr
    ]).astype(np.float32)

    # --- Predict ---
    action, _ = model.predict(obs, deterministic=True)
    return int(action)

def strat_rl_phase1(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 10: The Trained PPO Agent (Phase 1 - Pure EV).
    """
    return _predict_with_rl("ppo_phase1_complete.zip", match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)

def strat_rl_phase2(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 11: The Trained PPO Agent (Phase 2 - Champions League).
    """
    return _predict_with_rl("ppo_phase2_mixed_opps.zip", match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)

def strat_rl_phase3_tanh(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 12: The Trained PPO Agent (Phase 3 - Champions League + tanh reward + 2M iteration training).
    """
    return _predict_with_rl("ppo_phase3_random_opps_2M_training.zip", match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)

def strat_rl_phase3_tanh_4M(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25):
    """
    Strategy 12: The Trained PPO Agent (Phase 3 - Champions League + log reward + 4M iteration training).
    """
    return _predict_with_rl("ppo_phase3_random_opps_4M_training.zip", match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)


# --- Strategy List ---
# This list maps strategy names to their functions
# It's imported by the main script to run the simulation
STRATEGY_FUNCTIONS = [
    strat_typical_opponent,
    strat_random,
    strat_best_ev,
    strat_best_simple_rel_ev,
    strat_favorite,
    strat_safe_simple_rel_ev,
    strat_adaptive_simple_rel_ev,
    strat_safe_simple_ev,
    strat_adaptive_simple_ev,
    strat_highest_variance,
    strat_rl_phase1,
    strat_rl_phase2,
    strat_rl_phase3_tanh,
    strat_rl_phase3_tanh_4M
]

STRATEGY_NAMES = [
    "Typical Opponent",
    "Random",
    "Best Simple EV",
    "Best Simple Relative EV",
    "Favorite",
    "Safe Simple Rel EV",
    "Adaptive Simple Rel EV",
    "Safe Simple EV",
    "Adaptive Simple EV",
    "Highest variance",
    "RL Agent (Phase 1)",
    "RL Agent (Phase 2)",
    "RL Agent (P3 - Tanh 2M)",
    "RL Agent (P3 - Tanh 4M)"
]