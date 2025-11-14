"""
Core logic and utility functions for the MonPetitProno (MPP) analysis project.
"""

import numpy as np
from scipy.stats import binom

def calculate_win_probability(min_successes: int, num_matches: int, success_prob: float) -> float:
    """
    Calculates the probability of achieving at least 'min_successes' 
    in 'num_matches' independent trials, each with 'success_prob'.

    This function calculates P(X >= min_successes).

    Args:
        min_successes: The minimum number of successes needed (k).
        num_matches: The total number of trials (n).
        success_prob: The probability of success for a single trial (p).

    Returns:
        The probability of achieving k or more successes.
    """
    # Handle edge cases
    if min_successes > num_matches:
        return 0.0
    if min_successes <= 0:
        return 1.0
    
    # We want P(X >= k).
    # The survival function (sf) is P(X > k).
    # Therefore, P(X >= k) = P(X > k-1) = binom.sf(k-1, n, p)
    return binom.sf(min_successes - 1, num_matches, success_prob)

def get_simple_ev(probability: float, points: float) -> float:
    """Calculates the simple expected value of one outcome."""
    return probability * points

def get_ev_with_bonus(outcome_prob: float, perfect_prob_given_outcome: float, points: float) -> float:
    """
    Calculates the expected value (EV) of an outcome, including
    the "perfect score" bonus (which doubles the points).
    
    Formula: EV = P(outcome) * E[Points | outcome]
    E[Points | outcome] = points * (1 + P(perfect | outcome))
    
    Args:
        outcome_prob: P(Win), P(Draw), etc.
        perfect_prob_given_outcome: P(Perfect Score | Correct Outcome)
        points: Base MPP points for the outcome.
        
    Returns:
        The total expected value for that bet.
    """
    expected_points_given_outcome = points * (1.0 + perfect_prob_given_outcome)
    return outcome_prob * expected_points_given_outcome

def calculate_true_outcome_probas_from_odds(odds: list) -> np.ndarray:
    """
    Calculates the ground truth probabilities from bookmaker odds,
    normalized to remove the bookmaker's margin ("vig").
    
    This formula is detailled in README.md.
    Example: P(A) = (1/Odds(A)) / ( (1/Odds(A)) + (1/Odds(B)) + (1/Odds(C)) )

    Args:
        odds: A list or array of bookmaker odds [odds1, odds2, odds3, ...].

    Returns:
        A numpy array of the normalized ground truth probabilities.
    """
    inv_odds = 1.0 / np.array(odds)
    total_inv_odds = np.sum(inv_odds)
    
    if total_inv_odds == 0:
        return np.zeros_like(odds)
        
    true_probas = inv_odds / total_inv_odds
    return true_probas
