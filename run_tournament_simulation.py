"""
Main script to run the MonPetitProno (MPP) tournament simulation.

This script simulates a full competition (e.g., World Cup) multiple times
to compare the performance of different betting strategies.

Each strategy ("challenger") is tested independently in different leagues
filled with "Lambda" (typical) players. We find which challenger wins
most often against the standard pool.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import List, Callable, Dict

# Import our refactored project modules
from mpp_project.simulation import simulate_full_league
from mpp_project.strategies import STRATEGY_FUNCTIONS, STRATEGY_NAMES, strat_typical_opponent

# --- Simulation Configuration ---
N_SAMPLES = 5000      # Number of full tournaments to simulate *per strategy*
N_PLAYERS = 12        # Number of players in the league

# --- Match Simulation Parameters ---
# We group these into a dict to pass to the simulation functions
match_params = {
    'n_matches': 51,
    'ev_avg': 35,
    'draw_fact_min': 0.2,
    'draw_fact_max': 0.75,
    'outsider_fact_min': 1/7.5,
    'outsider_fact_max': 1.0
}


# --- Helper Function for Printing Results ---
def print_results(names: List[str], first_place_counts: np.ndarray, n_simulations: int, scores_by_strats: List[List[float]]):
    """Prints a formatted table of the simulation results."""
    
    print(f"\n---------- Simulation Results ({n_simulations} tournaments per strategy) -----------")
    print("Strategy                 | Win Rate | Avg Points |   Median   |  Std Dev  ")
    print("-------------------------------------------------------------------------")
    
    # Sort strategies by win rate
    win_rates = first_place_counts / n_simulations
    sorted_indices = np.argsort(win_rates)[::-1] # Descending
    
    for idx in sorted_indices:
        # Don't print the lambda player's "win rate" against itself (expected 1/N_PLAYERS)
        if names[idx] == "Typical Opponent":
            continue
        win_rate_pct = win_rates[idx] * 100
        avg_points = np.mean(scores_by_strats[idx])
        med_points = np.median(scores_by_strats[idx])
        std_points = np.std(scores_by_strats[idx])
        print(f"{names[idx]:<25}| {win_rate_pct:>7.2f}% | {avg_points:>7.1f}pts | {med_points:>7.1f}pts | {std_points:>6.2f}pts")

# --- Main Simulation Loop ---
def run_simulation():
    
    # We will test all strategies *except* the lambda player itself
    challenger_indices = [i for i, name in enumerate(STRATEGY_NAMES) if name != "Typical Opponent"]
    n_challengers = len(challenger_indices)
    
    # --- Statistics Trackers ---
    # n_times_first[i] = how many times strategy 'i' finished first
    n_times_first = np.zeros(len(STRATEGY_NAMES), dtype=int)
    
    # Histograms for plotting final score distributions
    final_scores_all_players = [] # All scores from all lambda players
    final_scores_by_strat = [[] for _ in range(len(STRATEGY_NAMES))]

    print(f"Running {N_SAMPLES} tournaments for each of {n_challengers} challenger strategies...")
    print(f"Each league has 1 Challenger + {N_PLAYERS - 1} Lambda Players.")
    
    for i_sample in tqdm(range(N_SAMPLES)):
        
        # In this loop, we simulate N_SAMPLES *different* tournaments (match schedules)
        
        # --- 1. Generate the Tournament ---
        # We need to generate the tournament *inside* this loop
        # so that each sample is a different set of matches.
        # This is handled by simulate_full_league.
        
        # --- 2. Simulate one league for each challenger ---
        for i_challenger_idx in challenger_indices:
            
            challenger_func = STRATEGY_FUNCTIONS[i_challenger_idx]
            
            # Create the league: 1 Challenger + (N-1) Lambda Players
            player_strategies: List[Callable] = [challenger_func] + \
                                                [strat_typical_opponent] * (N_PLAYERS - 1)
            
            final_league_scores = simulate_full_league(
                n_players=N_PLAYERS,
                n_matches=match_params['n_matches'],
                match_params=match_params,
                player_strategies=player_strategies
            )
            
            # --- 3. Tally Results for This League ---
            challenger_score = final_league_scores[0]
            max_score = np.max(final_league_scores)
            
            # Store scores for plotting
            final_scores_by_strat[i_challenger_idx].append(challenger_score)
            final_scores_all_players.extend(final_league_scores[1:]) # Add the lambda scores
            
            # Check if the challenger won (can be a tie)
            if challenger_score == max_score:
                n_times_first[i_challenger_idx] += 1

        # Optional: Print progress
        if (i_sample + 1) % 500 == 0:
            print_results(STRATEGY_NAMES, n_times_first, i_sample + 1, final_scores_by_strat)
            
    # --- 4. Final Results ---
    print("\n" + "#" * 30)
    print("  Final Simulation Results")
    print("#" * 30)
    print_results(STRATEGY_NAMES, n_times_first, N_SAMPLES, final_scores_by_strat)
    
    # Store the lambda player scores from the last challenger run for context
    final_scores_by_strat[STRATEGY_NAMES.index("Typical Opponent")] = final_scores_all_players
    
    return final_scores_all_players, final_scores_by_strat

# --- Plotting Functions ---
def plot_score_distributions(all_scores, strat_scores, strat_names):
    """
    Plots the final score distributions.
    """
    plt.figure(figsize=(14, 8))
    
    # Plot all players' scores
    plt.hist(all_scores, bins=100, density=True, alpha=0.5,
             label=f"All Lambda Players (n={(len(all_scores))})")
    
    # Plot distributions for our key strategies
    key_indices = [i for i, name in enumerate(strat_names) if name in [
        "Best Simple EV",
        "Best Simple Relative EV",
        "Safe Simple Rel EV",
        "Adaptive Simple Rel EV"
    ]]
    
    for i in key_indices:
        plt.hist(strat_scores[i], bins=100, density=True, alpha=0.8,
                 histtype='step', linewidth=2,
                 label=f"Strategy: {strat_names[i]}")
        
    plt.title("Final Score Distributions by Strategy", fontsize=16)
    plt.xlabel("Final Score (Points)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    
    # Run the simulation
    all_scores, strat_scores = run_simulation()
    
    # Plot the results
    plot_score_distributions(all_scores, strat_scores, STRATEGY_NAMES)