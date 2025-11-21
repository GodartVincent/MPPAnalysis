"""
Defines the different betting strategies to be compared in the simulation.
Updated to handle 2D inputs (Current + Future matches).
"""

import numpy as np
import os
from stable_baselines3 import PPO

# --- Paths Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")           # V1 Models
MODELS_V2_DIR = os.path.join(ROOT_DIR, "models_v2")     # V2 Models

# ==========================================
# 1. HELPERS: V2 Observation & Loading
# ==========================================

def _get_v2_obs(outcome_probas, match_gains, opp_repartition, player_scores, agent_idx, ev_avg, total_matches):
    """
    Constructs the V2 Observation vector from 2D match data.
    """
    # --- Current Match Data (Row 0) ---
    m_probas = outcome_probas[0]
    m_gains = match_gains[0]
    m_repart = opp_repartition[0]
    
    # 1. Sort Match Data by Probability (Descending)
    sort_idx = np.argsort(m_probas)[::-1]
    sorted_probas = m_probas[sort_idx]
    sorted_repart = m_repart[sort_idx]
    
    # 2. Normalize Gains
    sorted_gains = m_gains[sort_idx] / ev_avg
    
    # 3. Process Scores (Relative & Sorted)
    agent_score = player_scores[agent_idx]
    relative_scores = player_scores - agent_score
    opp_relative_scores = np.delete(relative_scores, agent_idx)
    
    # Sort opponents from Leader (highest relative score) to Loser
    sorted_opp_scores = np.sort(opp_relative_scores)[::-1]
    normalized_opp_scores = sorted_opp_scores / ev_avg

    # 4. Desperation Ratio
    # Calculate Max Points Remaining (Sum of max of each future row)
    max_points_per_match = np.max(match_gains, axis=1)
    future_max_points = np.sum(max_points_per_match)
    
    if future_max_points < 1.0: future_max_points = 1.0
    
    # Gap to Leader (Leader is sorted_opp_scores[0])
    # (Agent - Leader) = -sorted_opp_scores[0] (since sorted is Opp-Agent)
    gap_to_leader = -sorted_opp_scores[0] * ev_avg
    gap_ratio = np.clip(gap_to_leader / future_max_points, -1.0, 1.0)
    
    # 5. Matches Remaining (Normalized)
    matches_remaining_count = len(outcome_probas)
    matches_rem_norm = np.array([matches_remaining_count / total_matches])
    
    # Concatenate
    obs = np.concatenate([
        sorted_probas,
        sorted_gains,
        sorted_repart,
        normalized_opp_scores,
        np.array([gap_ratio]),
        matches_rem_norm
    ]).astype(np.float32)
    
    return obs, sort_idx

class RLStrategyWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
    def predict(self, outcome_probas, match_gains, opp_repartition, player_scores, agent_idx, **kwargs):
        if self.model is None:
            if not os.path.exists(self.model_path):
                print(f"Warning: Model not found at {self.model_path}. returning 0.")
                return 0
            self.model = PPO.load(self.model_path)
        
        ev_avg = kwargs.get('ev_avg', 35.0)
        total_matches = kwargs.get('n_matches', 51)
        
        obs, sort_idx = _get_v2_obs(outcome_probas, match_gains, opp_repartition, 
                                    player_scores, agent_idx, ev_avg, total_matches)
        
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Map Action (0=Fav, 1=Mid, 2=Outsider) back to original indices
        return int(sort_idx[action])

# Initialize V2 Agents
agent_v2_phase1 = RLStrategyWrapper(os.path.join(MODELS_V2_DIR, "ppo_v2_phase1_deterministic.zip"))
agent_v2_phase2 = RLStrategyWrapper(os.path.join(MODELS_V2_DIR, "ppo_v2_phase2_mixed_opps.zip"))
agent_v2_phase3 = RLStrategyWrapper(os.path.join(MODELS_V2_DIR, "ppo_v2_phase3_more_rand_opps.zip"))
agent_v2_phase4 = RLStrategyWrapper(os.path.join(MODELS_V2_DIR, "ppo_v2_phase4_full_rand_opps.zip"))
agent_v2_phase5 = RLStrategyWrapper(os.path.join(MODELS_V2_DIR, "ppo_v2_phase5_domain_rand.zip"))


# ==========================================
# 2. LEGACY HELPERS (V1)
# ==========================================
_LOADED_LEGACY_MODELS = {}

def get_legacy_model(model_filename):
    global _LOADED_LEGACY_MODELS
    if model_filename not in _LOADED_LEGACY_MODELS:
        model_path = os.path.join(MODELS_DIR, model_filename)
        if os.path.exists(model_path):
            print(f"Loading Legacy RL Model from {model_path}...")
            _LOADED_LEGACY_MODELS[model_filename] = PPO.load(model_path)
        else:
            print(f"WARNING: Legacy Model not found at {model_path}")
            _LOADED_LEGACY_MODELS[model_filename] = None
    return _LOADED_LEGACY_MODELS[model_filename]

def _predict_with_rl_legacy(model_filename, match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining):
    """Helper for Legacy Agents. Handles 2D input slicing."""
    model = get_legacy_model(model_filename)
    
    # --- SLICE INPUTS FOR LEGACY AGENT ---
    # Legacy agent only understands 1D arrays (Current Match)
    p = match_probas[0]
    g = match_gains[0]
    r = opp_repartition[0]
    
    if model is None:
        # Fallback
        evs = p * g
        return np.argmax(evs)

    # --- Construct Observation (Legacy Format) ---
    # [Probas(3), Gains(3), Repart(3), Scores(N), MatchesRemaining(1)]
    
    my_score = player_scores[my_idx]
    other_scores = np.delete(player_scores, my_idx)
    ordered_scores = np.concatenate(([my_score], other_scores))

    matches_rem_arr = np.array([float(matches_remaining)])

    obs = np.concatenate([
        p, g, r,
        ordered_scores,
        matches_rem_arr
    ]).astype(np.float32)

    action, _ = model.predict(obs, deterministic=True)
    return int(action)


# ==========================================
# 3. STRATEGY DEFINITIONS
# ==========================================

def strat_typical_opponent(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Bet on the favorite with a proba of 1 - (1-p_favorite)**2."""
    # Slice [0] to use current match
    return np.random.choice([0, 1, 2], p=opp_repartition[0])

def strat_random(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Bet randomly."""
    return np.random.randint(0, 3)

def strat_best_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Bet on the outcome with the highest simple EV."""
    # Slice [0]
    evs = match_probas[0] * match_gains[0]
    return np.argmax(evs)

def strat_best_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Bet on the outcome with the best Simple Relative EV."""
    evs = match_probas[0] * match_gains[0]
    rel_evs = evs * (1 - opp_repartition[0])
    return np.argmax(rel_evs)

def strat_favorite(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Most Probable Outcome."""
    return np.argmax(match_probas[0])

def strat_safe(match_probas, evs):
    # Helper for safe strategies (Logic unchanged)
    sorted_indices = np.argsort(evs)
    best_rel_ev = evs[sorted_indices[-1]]
    second_best_rel_ev = evs[sorted_indices[-2]]
    
    if best_rel_ev < (second_best_rel_ev + 1) and match_probas[sorted_indices[-1]] < match_probas[sorted_indices[-2]]:
        return sorted_indices[-2]
    return sorted_indices[-1]

def strat_safe_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Safe Simple Relative EV."""
    evs = match_probas[0] * match_gains[0]
    rel_evs = evs * (1 - opp_repartition[0])
    return strat_safe(match_probas[0], rel_evs)

def strat_adaptive(match_probas, match_gains, opp_repartition, player_scores, my_idx, default_strat, matches_remaining, **kwargs):
    """Adaptive Logic Helper."""
    my_score = player_scores[my_idx]
    other_scores = np.delete(player_scores, my_idx)
    leader_score = np.max(other_scores)
    
    if (my_score - leader_score) > 100:
        # Leading: Blocking bet (Most popular)
        return np.argmax(opp_repartition[0])
    if (leader_score - my_score) > 150:
        # Behind: Aggressive (Highest gain)
        return np.argmax(match_gains[0])
    
    # Default
    return default_strat(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining, **kwargs)

def strat_adaptive_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Adaptive Simple Rel EV."""
    return strat_adaptive(
        match_probas, match_gains, opp_repartition, player_scores, my_idx,
        strat_best_simple_rel_ev, matches_remaining, **kwargs
    )

def strat_safe_simple_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Safe Simple EV."""
    evs = match_probas[0] * match_gains[0]
    return strat_safe(match_probas[0], evs)

def strat_adaptive_simple_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Adaptive Simple EV."""
    return strat_adaptive(
        match_probas, match_gains, opp_repartition, player_scores, my_idx,
        strat_best_ev, matches_remaining, **kwargs
    )

def strat_highest_variance(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Highest variance outcome."""
    variances = (match_gains[0] ** 2) * match_probas[0] * (1 - match_probas[0])
    return np.argmax(variances)


# --- AGENT WRAPPERS ---

# Legacy (V1)
def strat_rl_v1(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Legacy Agent (Phase 3 - 4M)."""
    return _predict_with_rl_legacy("ppo_phase3_random_opps_4M_training.zip", match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)

# New (V2)
def strat_rl_v2_phase1(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    return agent_v2_phase1.predict(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs)

def strat_rl_v2_phase2(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    return agent_v2_phase2.predict(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs)

def strat_rl_v2_phase3(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    return agent_v2_phase3.predict(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs)

def strat_rl_v2_phase4(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    return agent_v2_phase4.predict(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs)

def strat_rl_v2_phase5(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    return agent_v2_phase5.predict(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs)


# --- Strategy Lists ---

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
    strat_rl_v1,
    strat_rl_v2_phase1,
    strat_rl_v2_phase2,
    strat_rl_v2_phase3,
    strat_rl_v2_phase4,
    strat_rl_v2_phase5,
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
    "Highest Variance",
    "RL Legacy (V1 - 4M)",
    "RL V2 (P1 - PPO)",
    "RL V2 (P2 - Mixed)",
    "RL V2 (P3 - MoreRand)",
    "RL V2 (P4 - FullRand)",
    "RL V2 (P5 - DomainRnd)"
]