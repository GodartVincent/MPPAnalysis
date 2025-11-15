"""
Generates realistic match data (probabilities, gains, and opponent bets)
for the MPP simulation.
"""

import numpy as np
import random

def generate_outcome_probas(
    n_matches: int, 
    draw_fact_min: float = 0.2,
    draw_fact_max: float = 0.75,
    outsider_fact_min: float = 1/7.5,
    outsider_fact_max: float = 1.0
) -> np.ndarray:
    """
    Generates realistic 3-outcome probabilities (Fav, Draw, Outsider) for n_matches.
    
    Args:
        n_matches: Number of matches to simulate.
        draw_fact_min/max: Controls draw probability.
        outsider_fact_min/max: Controls outsider win probability.

    Returns:
        A (n_matches, 3) numpy array where each row is [p_fav, p_draw, p_outsider].
    """
    outcome_probas = np.ones((n_matches, 3), dtype=float)
    
    for i in range(n_matches):
        # The first proba is for the favorite, then draw, then outsider.
        # We calculate multiplicative factors relative to the favorite's proba
        # and then normalize to sum to 1.
        
        # fact=0: very unbalanced match, fact=1: very balanced match
        fact = random.uniform(0, 1)
        outcome_probas[i, 1] = fact * (draw_fact_max - draw_fact_min) + draw_fact_min
        outcome_probas[i, 2] = fact * (outsider_fact_max - outsider_fact_min) + outsider_fact_min
    
    # Normalize each row (match) so that probabilities sum to 1
    row_sums = outcome_probas.sum(axis=1)
    outcome_probas = outcome_probas / row_sums[:, np.newaxis]
    
    return outcome_probas

# Here are some actual (from MPP Euro 2024) values for outcome_proba => gain :
#     if p is 0.8  =>  48pts (ev is 38.5pts = 1.1ev_ref) ;
#     if p is 0.35 => 100pts (ev is 35  pts = 1  ev_ref) ;
#     if p is 0.1  => 175pts (ev is 17.5pts = 0.5ev_ref).
# We are looking for a function f(p) such that :
#     f(0.8 ) = 1.1 ;
#     f(0.35) = 1   ;
#     f(0.1 ) = 0.25.
# We can solve this by fitting a 2nd order polynomial through these 3 points :
# a*x^2 + b*x + c => -2.54x^2 + 3.14x + 0.21
def generate_gains(outcome_probas: np.ndarray, ev_ref: float = 35.0, p_rand_fact_std: float = 0.06) -> np.ndarray:
    """
    Generates the corresponding MPP points (gains) for each outcome,
    based on the formula above. Since MPP gains are set few days before the match,
    we introduce a small randomness in the probabilities to simulate this.
    
    outcome_probas = true current probabilities
    MPP_probas = probabilities *WHEN* the gains were set = outcome_probas + small_noise
    G_i = EV / MPP_proba_i
    
    Args:
        outcome_probas: The (n_matches, 3) array from generate_outcome_probas e.g. the current true probability.
        ev_ref: The reference EV for a bet.

    Returns:
        A (n_matches, 3) numpy array of MPP points for each outcome.
    """
    # Introduce small noise to simulate MPP's earlier probability estimates
    mpp_probas = outcome_probas * np.random.normal(1, p_rand_fact_std, outcome_probas.shape)
    mpp_probas[mpp_probas <= 0] = 1e-6  # Prevent negative or zero probas
    mpp_probas = mpp_probas / mpp_probas.sum(axis=1)[:, np.newaxis]  # Re-normalize
    
    ev_factor = -2.54 * np.power(mpp_probas, 2) + 3.14 * mpp_probas + 0.21
    ev = ev_factor * ev_ref
    gains = ev/mpp_probas
    return gains.astype(int)

def generate_opponent_repartition(outcome_probas: np.ndarray) -> np.ndarray:
    """
    Simulates the repartition of bets from all other "lambda" players.
    
    Uses your empirical formula based on the favorite's win probability.
    
    Args:
        outcome_probas: The (n_matches, 3) array.

    Returns:
        A (n_matches, 3) numpy array of opponent bet repartitions.
    """
    repartition = np.zeros_like(outcome_probas)
    
    # Get the favorite's probability for each match
    p_fav = outcome_probas[:, 0]
    
    # Calculate repartition for the favorite using your formula
    # P(bet_fav) = 1 - (1 - p_fav)^2
    repartition_fav = 1 - (1 - p_fav)**2
    repartition[:, 0] = repartition_fav
    
    # For the other two outcomes, same mechanism
    p_remaining = 1 - repartition_fav
    p_second_fav = outcome_probas[:, 1]
    # Proba of betting on second favorite if favorite not bet on
    repartition_second_fav_if_no_fav = 1 - (1 - p_second_fav)**2
    # P(bet second fav) = P(no fav) * P(bet second fav | no fav)
    repartition[:, 1] = p_remaining * repartition_second_fav_if_no_fav
    
    # The last outcome gets the rest
    repartition[:, 2] = 1 - repartition[:, 0] - repartition[:, 1]
    
    return repartition