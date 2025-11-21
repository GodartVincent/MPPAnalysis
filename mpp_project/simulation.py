"""
Core simulation engine for running full tournaments.

This module uses the realistic match generators from 'match_simulator.py'
to simulate entire tournaments for one or more players.
"""

import numpy as np
from typing import Callable, List, Dict, Optional

# Import our refactored project modules
from mpp_project.match_simulator import (
    generate_outcome_probas, 
    generate_gains, 
    generate_opponent_repartition
)
from mpp_project.strategies import STRATEGY_FUNCTIONS, STRATEGY_NAMES

def simulate_full_league(
    n_players: int,
    n_matches: int, 
    match_params: Dict,
    player_strategies: List[Callable],
    precomputed_tournament: Optional[Dict] = None
) -> np.ndarray:
    """
    Simulates an entire league of n_players over n_matches.

    Args:
        n_players: Number of players in the league.
        n_matches: Number of matches.
        match_params: Dictionary of parameters for match generation.
        player_strategies: A list of strategy functions, one for each player.
        precomputed_tournament: Optional dict containing 'probas', 'gains', 'repart'.
                                If provided, ensures all strategies play the exact same matches.

    Returns:
        A numpy array of shape (n_players,) containing the final score for each player.
    """
    
    if len(player_strategies) != n_players:
        raise ValueError(f"Length of player_strategies ({len(player_strategies)}) must equal n_players ({n_players})")

    # --- 1. Get Tournament Data (Generated or Precomputed) ---
    if precomputed_tournament:
        outcome_probas = precomputed_tournament['probas']
        match_gains = precomputed_tournament['gains']
        opp_repartition = precomputed_tournament['repart']
    else:
        # Default behavior: Generate new randomness
        outcome_probas = generate_outcome_probas(
            n_matches, 
            match_params['draw_fact_min'], match_params['draw_fact_max'],
            match_params['outsider_fact_min'], match_params['outsider_fact_max']
        )
        match_gains = generate_gains(outcome_probas, match_params['ev_avg'], match_params['proba_fact_std'])
        opp_repartition = generate_opponent_repartition(outcome_probas)
    
    player_scores = np.zeros(n_players)
    
    # --- 2. Run the Matches ---
    for i in range(n_matches):
        # Slicing for future visibility (V2 Architecture Requirement)
        # [i:] passes the current match AND all future matches as a 2D array
        probas_slice = outcome_probas[i:]
        gains_slice = match_gains[i:]
        repart_slice = opp_repartition[i:]
        
        # Simulate the "true" outcome using the CURRENT match (index 0 of the slice)
        true_outcome = np.random.choice([0, 1, 2], p=probas_slice[0])
        
        # Each player makes their bet
        for p_idx in range(n_players):
            strat_func = player_strategies[p_idx]
            
            # PASS 2D SLICES + CONTEXT KWARGS
            # Strategies updated in V2 expect (probas, gains, repart, scores, idx, **kwargs)
            # They will handle the slicing internally (e.g., taking row [0] for current match)
            bet = strat_func(
                match_probas=probas_slice, 
                match_gains=gains_slice, 
                opp_repartition=repart_slice, 
                player_scores=player_scores, 
                my_idx=p_idx, 
                matches_remaining=n_matches - i, # Legacy support, though len(slice) is better
                ev_avg=match_params['ev_avg'],   # Needed for normalization in V2
                n_matches=n_matches              # Needed for time fraction in V2
            )
            
            if bet == true_outcome:
                # Gains for current match are at row 0 of the slice
                player_scores[p_idx] += gains_slice[0, true_outcome]

    return player_scores