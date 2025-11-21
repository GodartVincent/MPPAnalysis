"""
Core logic and utility functions for the MonPetitProno (MPP) analysis project.
"""

import numpy as np
from scipy.stats import binom
from typing import Tuple

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

def calculate_true_outcome_probas_from_odds(odds: list) -> np.ndarray:
    """Calculates normalized ground truth probabilities from bookmaker odds."""
    inv_odds = 1.0 / np.array(odds)
    total_inv_odds = np.sum(inv_odds)
    if total_inv_odds == 0:
        return np.zeros_like(odds)
    return inv_odds / total_inv_odds

# --- NEW SHARED OBSERVATION BUILDER ---

def get_v2_observation(
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
    Centralized function to build the V2 observation vector.
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
    # If Agent is Leader, gap is negative (Agent > Leader is impossible in relative array logic?)
    # Wait: sorted_opp_scores contains (Opponent - Agent).
    # So the "Leader" relative score is sorted_opp_scores[0] (The biggest positive number if losing).
    # Gap (Agent - Leader) = -sorted_opp_scores[0].
    
    # Safety clip for denominator
    if future_max_points < 1.0: 
        future_max_points = 1.0
    
    gap_to_leader = -sorted_opp_scores[0] * ev_avg # Denormalize to get raw points
    gap_ratio = gap_to_leader / future_max_points
    gap_ratio = np.clip(gap_ratio, -1.0, 1.0)
    
    # 5. Matches Remaining
    matches_rem_arr = np.array([matches_remaining_fraction])
    
    # Concatenate everything
    obs = np.concatenate([
        sorted_probas,
        sorted_gains,
        sorted_repart,
        normalized_opp_scores,
        np.array([gap_ratio]),
        matches_rem_arr
    ]).astype(np.float32)
    
    return obs, sort_idx