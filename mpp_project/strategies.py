"""
Defines the different betting strategies to be compared in the simulation.
"""

import numpy as np
import os
from stable_baselines3 import PPO
from .core import get_observation

# Paths Setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Default Directories
MODELS_V1_DIR = os.path.join(ROOT_DIR, "models")
MODELS_V2_DIR = os.path.join(ROOT_DIR, "models_v2")
MODELS_V3_DIR = os.path.join(ROOT_DIR, "models_v3")

# Model Caching
_LOADED_MODELS = {}

def get_model(model_path, version="v3"):
    """Loads a model to avoid reloading it 5000 times."""
    global _LOADED_MODELS
    
    # Resolve path
    if not os.path.isabs(model_path):
        # Try finding it in known dirs
        if version == "v3":
            model_path = os.path.join(MODELS_V3_DIR, model_path)
        elif version == "v2":
            model_path = os.path.join(MODELS_V2_DIR, model_path)
        else:
            model_path = os.path.join(MODELS_V1_DIR, model_path)
    
    if model_path not in _LOADED_MODELS:
        if os.path.exists(model_path):
            _LOADED_MODELS[model_path] = PPO.load(model_path)
        else:
            print(f"WARNING: Model not found at {model_path}")
            _LOADED_MODELS[model_path] = None
    return _LOADED_MODELS[model_path]


# ==========================================
# 1. OBSERVATION BUILDERS
# ==========================================

def _get_legacy_obs_v1(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining):
    """V1 Observation (Legacy): Raw data, no sorting."""
    p = match_probas[0]
    g = match_gains[0]
    r = opp_repartition[0]
    
    my_score = player_scores[my_idx]
    other_scores = np.delete(player_scores, my_idx)
    ordered_scores = np.concatenate(([my_score], other_scores))

    return np.concatenate([p, g, r, ordered_scores, np.array([float(matches_remaining)])]).astype(np.float32)

def _get_modern_obs(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs):
    """V3 Core Builder (25 features)."""
    ev_avg = kwargs.get('ev_avg', 35.0)
    total_matches = kwargs.get('n_matches', 51)

    max_points_per_match = np.max(match_gains, axis=1)
    future_max_points = np.sum(max_points_per_match)
    matches_remaining_count = len(match_probas)
    matches_rem_fraction = matches_remaining_count / total_matches

    obs, sort_idx = get_observation(
        match_probas=match_probas[0],       
        match_gains=match_gains[0],         
        opp_repartition=opp_repartition[0], 
        player_scores=player_scores,
        agent_idx=my_idx,
        future_max_points=future_max_points,
        matches_remaining_fraction=matches_rem_fraction,
        ev_avg=ev_avg
    )
    return obs, sort_idx


# ==========================================
# 2. STRATEGY FACTORY
# ==========================================

def create_strategy_from_model(model_filename, version="v3", name=None):
    """
    Creates a strategy function from a model file.
    
    Args:
        model_filename (str): Path or filename of the zip model.
        version (str): 'v1', 'v2', or 'v3' to determine observation format.
        name (str): Optional custom name for the strategy.
    """
    
    def strategy_func(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
        model = get_model(model_filename, version)
        if model is None: return 0 # Fallback to Fav

        # V1 Legacy
        if version == "v1":
            obs = _get_legacy_obs_v1(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining)
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        # V2 (22 features)
        elif version == "v2":
            obs, sort_idx = _get_modern_obs(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs)
            obs_v2 = obs[:22] # Slice off Simple EV
            action, _ = model.predict(obs_v2, deterministic=True)
            return int(sort_idx[action]) # Map back to real index

        # V3 (25 features)
        elif version == "v3":
            obs, sort_idx = _get_modern_obs(match_probas, match_gains, opp_repartition, player_scores, my_idx, **kwargs)
            action, _ = model.predict(obs, deterministic=True)
            return int(sort_idx[action])
            
        else:
            raise ValueError(f"Unknown agent version: {version}")

    # Assign a name for the simulation report
    strategy_func.__name__ = name if name else f"RL_{version}_{os.path.basename(model_filename)}"
    return strategy_func

# ==========================================
# 3. HEURISTICS (Standard)
# ==========================================

def strat_typical_opponent(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Random bet following opp_repartition."""
    # Slice [0] to use current match
    return np.random.choice([0, 1, 2], p=opp_repartition[0])

def strat_random(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Bet randomly."""
    return np.random.randint(0, 3)

def strat_best_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx, matches_remaining=25, **kwargs):
    """Bet on the outcome with the highest simple EV."""
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
    # Helper for safe strategies : if best EV is too close to second best, pick the more probable
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


# ==========================================
# 4. RL AGENTS
# ==========================================

# V1 Family (Legacy)
strat_v1_legacy = create_strategy_from_model("ppo_phase3_random_opps_4M_training.zip", "v1", "RL V1 (Legacy)")

# V2 Family (Affine Reward, Domain Rand, 22 Feats)
strat_v2_p1 = create_strategy_from_model("ppo_v2_phase1_deterministic.zip", "v2", "V2 P1 (Det)")
strat_v2_p2 = create_strategy_from_model("ppo_v2_phase2_mixed_opps.zip", "v2", "V2 P2 (Mixed)")
strat_v2_p3 = create_strategy_from_model("ppo_v2_phase3_more_rand_opps.zip", "v2", "V3 P3 (MoreRand)")
strat_v2_p4 = create_strategy_from_model("ppo_v2_phase4_full_rand_opps.zip", "v2", "V3 P3 (FullRand)")
strat_v2_p5 = create_strategy_from_model("ppo_v2_phase5_domain_rand.zip", "v2", "V2 Final (DomRand)")

# V3 Family (Power Law, Value Injection, 25 Feats)
strat_v3_p1 = create_strategy_from_model("ppo_v3_phase1_deterministic.zip", "v3", "V3 P1 (Det)")
strat_v3_p2 = create_strategy_from_model("ppo_v3_phase2_mixed_opps.zip", "v3", "V3 P2 (Mixed)")
strat_v3_p3 = create_strategy_from_model("ppo_v3_phase3_full_rand_opps.zip", "v3", "V3 P3 (FullRand)")
strat_v3_p4 = create_strategy_from_model("ppo_v3_phase4_domain_rand.zip", "v3", "V3 P4 (DomRand)")

# ==========================================
# 5. EXPORT LISTS
# ==========================================

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
    # V1 Agent
    strat_v1_legacy,
    # V2 Agents
    strat_v2_p1,
    strat_v2_p2,
    strat_v2_p3,
    strat_v2_p4,
    strat_v2_p5,
    # V3 Agents
    strat_v3_p1,
    strat_v3_p2,
    strat_v3_p3,
    strat_v3_p4
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
    "RL V2 (P5 - DomainRnd)",
    "RL V3 (P1 - PPO)",
    "RL V3 (P2 - Mixed)",
    "RL V3 (P3 - FullRand)",
    "RL V3 (P4 - DomainRnd)"
]