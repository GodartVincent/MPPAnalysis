"""
Core simulation engine for running full tournaments.

This module uses the realistic match generators from 'match_simulator.py'
to simulate entire tournaments for one or more players.
"""

import numpy as np
from typing import Callable, List, Dict

# Import our refactored project modules
from mpp_project.match_simulator import (
    generate_outcome_probas, 
    generate_gains, 
    generate_opponent_repartition
)
from mpp_project.strategies import STRATEGY_FUNCTIONS, STRATEGY_NAMES

def simulate_single_player_score(
    n_matches: int,
    match_params: Dict,
    strategy_func: Callable
) -> float:
    """
    Simulates a single player's score over a full tournament.

    Args:
        n_matches: Number of matches in the tournament.
        match_params: Dictionary of parameters for match generation.
        strategy_func: The betting strategy function to use (e.g., strat_best_ev).

    Returns:
        The final total score for this single simulated player.
    """
    
    # --- 1. Generate the full tournament's matches ---
    outcome_probas = generate_outcome_probas(
        n_matches, 
        match_params['draw_fact_min'], match_params['draw_fact_max'],
        match_params['outsider_fact_min'], match_params['outsider_fact_max']
    )
    match_gains = generate_gains(outcome_probas, match_params['ev_avg'])
    opp_repartition = generate_opponent_repartition(outcome_probas)

    player_score = 0.0
    
    for i in range(n_matches):
        # Get data for the current match
        m_probas = outcome_probas[i, :]
        m_gains = match_gains[i, :]
        m_repart = opp_repartition[i, :]
        
        # Simulate the "true" outcome of the match
        true_outcome = np.random.choice([0, 1, 2], p=m_probas)
        
        # Player makes their bet based on their strategy
        # We pass a score of 0 and idx 0, as this is a single-player sim
        bet = strategy_func(m_probas, m_gains, m_repart, [player_score], 0)
        
        # Check if the bet was correct and update score
        if bet == true_outcome:
            player_score += m_gains[true_outcome]
            
    return player_score

def simulate_full_league(
    n_players: int,
    n_matches: int, 
    match_params: Dict,
    player_strategies: List[Callable]
) -> np.ndarray:
    """
    Simulates an entire league of n_players over n_matches.

    Args:
        n_players: Number of players in the league.
        n_matches: Number of matches.
        match_params: Dictionary of parameters for match generation.
        player_strategies: A list of strategy functions, one for each player.

    Returns:
        A numpy array of shape (n_players,) containing the final score for each player.
    """
    
    if len(player_strategies) != n_players:
        raise ValueError(f"Length of player_strategies ({len(player_strategies)}) must equal n_players ({n_players})")

    # --- 1. Generate the full tournament's matches ---
    outcome_probas = generate_outcome_probas(
        n_matches, 
        match_params['draw_fact_min'], match_params['draw_fact_max'],
        match_params['outsider_fact_min'], match_params['outsider_fact_max']
    )
    match_gains = generate_gains(outcome_probas, match_params['ev_avg'])
    opp_repartition = generate_opponent_repartition(outcome_probas)
    
    player_scores = np.zeros(n_players)
    
    for i in range(n_matches):
        # Get data for the current match
        m_probas = outcome_probas[i, :]
        m_gains = match_gains[i, :]
        m_repart = opp_repartition[i, :]
        
        # Simulate the "true" outcome of the match
        true_outcome = np.random.choice([0, 1, 2], p=m_probas)
        
        # Each player makes their bet
        for p_idx in range(n_players):
            strat_func = player_strategies[p_idx]
            bet = strat_func(m_probas, m_gains, m_repart, player_scores, p_idx)
            
            if bet == true_outcome:
                player_scores[p_idx] += m_gains[true_outcome]

    return player_scores