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

def get_ev(probability: float, points: float) -> float:
    """Calculates the expected value of one outcome."""
    return probability * points
