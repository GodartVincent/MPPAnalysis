"""
Defines the different betting strategies to be compared in the simulation.

Each function takes the current match data and player scores and returns
the chosen bet (0 for Fav, 1 for Draw, 2 for Outsider).
"""

import numpy as np

# --- Strategy Definitions ---

def strat_typical_opponent(match_probas, match_gains, opp_repartition, player_scores, my_idx):
    """Strategy 0: Bet on the favorite with a proba of 1 - (1-p_favorite)**2 (empircal formula)."""
    return np.random.choice([0, 1, 2], p=opp_repartition)

def strat_random(match_probas, match_gains, opp_repartition, player_scores, my_idx):
    """Strategy 1: Bet randomly."""
    return np.random.randint(0, 3)

def strat_best_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx):
    """Strategy 2: Bet on the outcome with the highest simple EV (gtp * g)."""
    evs = match_probas * match_gains
    return np.argmax(evs)

def strat_best_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx):
    """Strategy 3: Bet on the outcome with the best * Simple Relative* EV."""
    # Formula: E[Simple_Rel_EV_i] = gtpi*gi*(1-pi)
    evs = match_probas * match_gains
    rel_evs = evs * (1 - opp_repartition)
    return np.argmax(rel_evs)

def strat_favorite(match_probas, match_gains, opp_repartition, player_scores, my_idx):
    """Strategy 4: Most Probable Outcome."""
    # It bets on the outcome with the highest probability
    return np.argmax(match_probas)

def strat_safe_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx):
    """
    Strategy 5: Safe Simple Relative EV.
    Chooses best Simple Rel EV, but if it's close, checks the "risk".
    """
    evs = match_probas * match_gains
    rel_evs = evs * (1 - opp_repartition)
    
    # If the best bet is not much better than the 2nd best,
    # check the simple probability (a proxy for risk).
    sorted_indices = np.argsort(rel_evs)
    best_rel_ev = rel_evs[sorted_indices[-1]]
    second_best_rel_ev = rel_evs[sorted_indices[-2]]
    
    # If the EV is too close, pick the one with the highest simple proba
    # among the top two EV bets.
    if best_rel_ev < (second_best_rel_ev + 1) and match_probas[sorted_indices[-1]] < match_probas[sorted_indices[-2]]:
        return sorted_indices[-2]
            
    return sorted_indices[-1]

def strat_adaptive_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx):
    """
    Strategy 6: If leading, does blocking bets. If far behind, plays it aggressively.
    If 100+ points ahead of 2nd place, bet on the most popular bet.
    If 150+ points below the 2nd place, bet on the highest gain.
    Otherwise, bet on the best relative EV.
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
    return strat_best_simple_rel_ev(match_probas, match_gains, opp_repartition, player_scores, my_idx)

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
    strat_adaptive_simple_rel_ev
    # TODO: Add your genetic algorithm strategy here!
]

STRATEGY_NAMES = [
    "Typical Opponent",
    "Random",
    "Best Simple EV",
    "Best Simple Relative EV",
    "Favorite",
    "Safe Simple Rel EV",
    "Adaptive Simple Rel EV"
]