"""
Main script to run the MonPetitProno (MPP) tournament simulation.

This script simulates a full competition (e.g., World Cup) multiple times
to compare the performance of different betting strategies.

UPDATED: Now uses Paired Evaluation and tracks detailed Ranking statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import List, Callable, Dict

# Import our refactored project modules
from mpp_project.simulation import simulate_full_league
from mpp_project.strategies import STRATEGY_FUNCTIONS, STRATEGY_NAMES, strat_typical_opponent
from mpp_project.match_simulator import (
    generate_outcome_probas, 
    generate_gains, 
    generate_opponent_repartition
)

# --- Simulation Configuration ---
N_SAMPLES = 10000
N_PLAYERS = 12        

# --- Match Simulation Parameters ---
match_params = {
    'n_matches': 51,
    'ev_avg': 35,
    'draw_fact_min': 0.2,
    'draw_fact_max': 0.75,
    'outsider_fact_min': 1/7.5,
    'outsider_fact_max': 1.0,
    'proba_fact_std': 0.06 
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
        if names[idx] == "Typical Opponent":
            continue
        win_rate_pct = win_rates[idx] * 100
        avg_points = np.mean(scores_by_strats[idx])
        med_points = np.median(scores_by_strats[idx])
        std_points = np.std(scores_by_strats[idx])
        print(f"{names[idx]:<25}| {win_rate_pct:>7.2f}% | {avg_points:>7.1f}pts | {med_points:>7.1f}pts | {std_points:>6.2f}pts")

# --- Main Simulation Loop ---
def run_simulation():
    
    challenger_indices = [i for i, name in enumerate(STRATEGY_NAMES) if name != "Typical Opponent"]
    n_challengers = len(challenger_indices)
    
    # --- Statistics Trackers ---
    n_times_first = np.zeros(len(STRATEGY_NAMES), dtype=int)
    final_scores_all_players = [] 
    final_scores_by_strat = [[] for _ in range(len(STRATEGY_NAMES))]
    
    # Rank Tracker: [Strategy Index][Rank (0=1st, 1=2nd, ...)]
    rank_counts = np.zeros((len(STRATEGY_NAMES), N_PLAYERS), dtype=int)

    print(f"Running {N_SAMPLES} paired tournaments...")
    print(f"Each league has 1 Challenger + {N_PLAYERS - 1} Lambda Players.")
    
    for i_sample in tqdm(range(N_SAMPLES)):
        
        # --- 1. Generate the Tournament ONCE for this iteration ---
        outcome_probas = generate_outcome_probas(
            match_params['n_matches'], 
            match_params['draw_fact_min'], match_params['draw_fact_max'],
            match_params['outsider_fact_min'], match_params['outsider_fact_max']
        )
        match_gains = generate_gains(outcome_probas, match_params['ev_avg'], match_params['proba_fact_std'])
        opp_repartition = generate_opponent_repartition(outcome_probas)
        
        tournament_data = {
            'probas': outcome_probas,
            'gains': match_gains,
            'repart': opp_repartition
        }
        
        # --- 2. Run simulations for each challenger ---
        for i_challenger_idx in challenger_indices:
            
            challenger_func = STRATEGY_FUNCTIONS[i_challenger_idx]
            
            player_strategies: List[Callable] = [challenger_func] + \
                                                [strat_typical_opponent] * (N_PLAYERS - 1)
            
            final_league_scores = simulate_full_league(
                n_players=N_PLAYERS,
                n_matches=match_params['n_matches'],
                match_params=match_params,
                player_strategies=player_strategies,
                precomputed_tournament=tournament_data 
            )
            
            # --- 3. Tally Results ---
            challenger_score = final_league_scores[0]
            max_score = np.max(final_league_scores)
            
            final_scores_by_strat[i_challenger_idx].append(challenger_score)
            final_scores_all_players.extend(final_league_scores[1:]) 
            
            # Check if challenger won
            if challenger_score == max_score:
                n_times_first[i_challenger_idx] += 1
                
            # Calculate Specific Rank
            # Rank = number of players strictly better than challenger
            # (Standard competition ranking: 1st, 2nd, 3rd...)
            # 0-indexed: 0 = 1st Place
            better_scores_count = np.sum(final_league_scores > challenger_score)
            rank_counts[i_challenger_idx][better_scores_count] += 1

        if (i_sample + 1) % 1000 == 0:
            print_results(STRATEGY_NAMES, n_times_first, i_sample + 1, final_scores_by_strat)
            
    # --- 4. Final Results ---
    print("\n" + "#" * 30)
    print("  Final Simulation Results")
    print("#" * 30)
    print_results(STRATEGY_NAMES, n_times_first, N_SAMPLES, final_scores_by_strat)
    
    return final_scores_all_players, final_scores_by_strat, rank_counts

# --- Plotting Functions ---
def plot_score_distributions(all_scores, strat_scores, strat_names):
    plt.figure(figsize=(14, 8))
    
    plt.hist(all_scores, bins=100, density=True, alpha=0.5,
             label=f"All Lambda Players (n={(len(all_scores))})")
    
    # Define subset of strategies to plot to avoid clutter
    # Plotting RL agents + Key Heuristics
    indices_to_plot = []
    for i, name in enumerate(strat_names):
        if "RL" in name or "Adaptive" in name or "Best" in name:
            indices_to_plot.append(i)
            
    for i in indices_to_plot:
        if len(strat_scores[i]) > 0: 
             plt.hist(strat_scores[i], bins=100, density=True, alpha=0.8,
                 histtype='step', linewidth=2,
                 label=f"{strat_names[i]}")

    plt.title("Final Score Distributions by Strategy", fontsize=16)
    plt.xlabel("Final Score (Points)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_rank_heatmap(rank_counts, strat_names, n_samples):
    """
    Plots a heatmap showing the frequency of each rank for each strategy.
    """
    # Filter out "Typical Opponent" or unused strategies (rows with all zeros)
    valid_indices = [i for i in range(len(strat_names)) if np.sum(rank_counts[i]) > 0]
    
    filtered_counts = rank_counts[valid_indices]
    filtered_names = [strat_names[i] for i in valid_indices]
    
    # Convert to percentages
    rank_pcts = (filtered_counts / n_samples) * 100
    
    # Sort strategies by "Win Rate" (Rank 0 percentage) for cleaner visualization
    sort_order = np.argsort(rank_pcts[:, 0]) # Sort by 1st place freq
    rank_pcts = rank_pcts[sort_order]
    filtered_names = [filtered_names[i] for i in sort_order]

    plt.figure(figsize=(12, len(filtered_names) * 0.6 + 2))
    
    # Create Heatmap
    plt.imshow(rank_pcts, cmap='viridis', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('Frequency (%)', rotation=270, labelpad=15)
    
    # Axis Labels
    plt.yticks(range(len(filtered_names)), filtered_names)
    plt.xticks(range(rank_pcts.shape[1]), [f"{i+1}st" if i==0 else f"{i+1}nd" if i==1 else f"{i+1}rd" if i==2 else f"{i+1}th" for i in range(rank_pcts.shape[1])])
    
    plt.xlabel("Rank in Tournament")
    plt.title("Strategy Ranking Distribution (Sorted by Win Rate)")
    
    # Annotate values
    for i in range(rank_pcts.shape[0]):
        for j in range(rank_pcts.shape[1]):
            val = rank_pcts[i, j]
            color = "white" if val < 50 else "black" # dynamic text color
            if val > 1.0: # Only show > 1% to reduce clutter
                plt.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9)
                
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    all_scores, strat_scores, rank_counts = run_simulation()
    
    print("\nDisplaying Score Distributions...")
    plot_score_distributions(all_scores, strat_scores, STRATEGY_NAMES)
    
    print("\nDisplaying Rank Heatmap...")
    plot_rank_heatmap(rank_counts, STRATEGY_NAMES, N_SAMPLES)