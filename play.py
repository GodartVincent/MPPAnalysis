import os
import numpy as np
from stable_baselines3 import PPO
from mpp_project.core import get_v2_observation
from mpp_project.match_simulator import generate_gains, generate_opponent_repartition

# --- CONFIGURATION ---
# Point this to the best model
MODEL_PATH = "../models_v2/ppo_v2_phase5_domain_rand.zip" 

# Match Constants (Standard MPP)
EV_AVG = 35.0
TOTAL_MATCHES = 51

def get_agent_decision(model, p_fav, p_draw, p_out, my_score, leader_score, matches_rem):
    """
    Feeds real-world data into the agent's brain.
    """
    # 1. Format Probabilities
    probas = np.array([p_fav, p_draw, p_out])
    
    # 2. Calculate MPP Gains (The "Physics")
    # We use the simulator's logic to generate the exact gains the app would give
    # based on these probabilities.
    # Note: In real life, you might just input the actual app gains if known.
    # Here we simulate them to be consistent with training.
    gains = generate_gains(np.expand_dims(probas, 0), EV_AVG, 0.0)[0]
    
    # 3. Generate Repartition (Crowd Model)
    # Phase 5 agent assumes 1-(1-p)^2. 
    # If using V3 later, this function in match_simulator will be the Power Law one.
    repart = generate_opponent_repartition(np.expand_dims(probas, 0))[0]
    
    # 4. Construct Scores Vector
    # Agent is always index 0.
    # We need to construct a "Relative Score" picture.
    # If Agent < Leader, we set scores[1] to represent that gap.
    # We pad the rest with "average" scores or -9999 if strictly following training padding.
    # Let's use a simplified 12-player vector: [Agent, Leader, ...others...]
    scores = np.zeros(12)
    scores[0] = my_score
    scores[1] = leader_score
    # Fill others with something harmless (e.g., slightly below leader)
    scores[2:] = leader_score - 10 
    
    # 5. Context
    matches_remaining_fraction = matches_rem / TOTAL_MATCHES
    
    # Future Potential
    # Estimate remaining points based on average EV
    future_max_points = np.max(gains) + (matches_rem - 1) * (2.0 * EV_AVG)
    
    # 6. Build Observation
    obs, sort_idx = get_v2_observation(
        match_probas=probas,
        match_gains=gains,
        opp_repartition=repart,
        player_scores=scores,
        agent_idx=0,
        future_max_points=future_max_points,
        matches_remaining_fraction=matches_remaining_fraction,
        ev_avg=EV_AVG
    )
    
    # 7. Predict
    action_sorted, _ = model.predict(obs, deterministic=True)
    real_action_idx = sort_idx[action_sorted]
    
    return real_action_idx, gains

def main():
    print("\n" + "="*40)
    print(" MPP AGENT COMPANION - v2.0")
    print("="*40)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Loading Brain...")
    model = PPO.load(MODEL_PATH)
    print("Ready!\n")
    
    while True:
        print("\n--- NEW MATCH ---")
        try:
            # Inputs
            raw_odds = input("Enter Odds (Fav Draw Out) separated by space (e.g., '1.5 3.5 5.0'): ")
            if raw_odds.lower() in ['q', 'quit', 'exit']: break
            
            odds = [float(x) for x in raw_odds.split()]
            if len(odds) != 3:
                print("Error: Please enter exactly 3 numbers.")
                continue
                
            scores_input = input("Enter [MyScore] [LeaderScore] (e.g., '1200 1250'): ")
            my_score, leader_score = map(float, scores_input.split())
            
            matches_rem = int(input("Matches Remaining (e.g., 10): "))
            
            # Process Odds -> Probabilities (normalized)
            inv_odds = [1/o for o in odds]
            sum_inv = sum(inv_odds)
            probas = [x/sum_inv for x in inv_odds]
            
            # Identify Favorite/Outsider for display
            labels = ['Outcome A', 'Outcome B', 'Outcome C']
            # Sort by probability to identify Fav (0), Draw (1), Out (2) logic internally
            # The function handles the mapping, we just pass raw probas
            
            # Get Decision
            choice_idx, estimated_gains = get_agent_decision(
                model, probas[0], probas[1], probas[2], 
                my_score, leader_score, matches_rem
            )
            
            # Output
            outcomes = ["HOME", "DRAW", "AWAY"]
            print("\n" + "-"*30)
            print(f"ANALYSIS:")
            print(f"Probabilities: {probas[0]:.1%} / {probas[1]:.1%} / {probas[2]:.1%}")
            print(f"Est. MPP Gains: {int(estimated_gains[0])} / {int(estimated_gains[1])} / {int(estimated_gains[2])} pts")
            print(f"Gap to Leader: {my_score - leader_score} pts")
            print("-" * 30)
            print(f">>> AGENT RECOMMENDS: {outcomes[choice_idx]} <<<")
            print("-" * 30)
            
        except ValueError:
            print("Invalid input format. Try again.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()