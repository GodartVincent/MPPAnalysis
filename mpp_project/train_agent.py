import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import numpy as np

# Import custom environment
from .mpp_env import MppEnv 

# --- 1. Define Absolute Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "log")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "ppo_mpp_tensorboard")
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models dir exists

# Define model paths
PHASE1_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_phase1_complete.zip")
PHASE2_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_phase2_mixed_opps.zip")
PHASE3_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_phase3_tanh_reward.zip")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_final_agent.zip")

# --- 2. Simulation Configuration ---
match_params = {
    'n_matches': 51,
    'ev_avg': 35,
    'draw_fact_min': 0.2,
    'draw_fact_max': 0.75,
    'outsider_fact_min': 1/7.5,
    'outsider_fact_max': 1.0,
    'proba_fact_std': 0.06
}
N_PLAYERS = 12
N_MATCHES = 51

# --- 3. Hyperparameters ---
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
PHASE1_HYPERPARAMS = {
    "n_steps": 8192,
    "learning_rate": 1e-4,
    "vf_coef": 0.5,
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.001
}
PHASE2_HYPERPARAMS = {
    "n_steps": 16384,
    "learning_rate": 1e-5,
    "vf_coef": 1.0,
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.01
}
PHASE3_HYPERPARAMS = {
    "n_steps": 16384,
    "learning_rate": 1e-5,
    "vf_coef": 1.0,
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.01
}
PHASE4_HYPERPARAMS = {
    "n_steps": 32768,
    "learning_rate": 1e-5,
    "vf_coef": 1.0,
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.001
}
# Define timesteps for each phase
PHASE1_STEPS = 1_100_000
PHASE2_STEPS = 3_000_000
PHASE3_STEPS = 2_000_000
PHASE4_STEPS = 4_000_000

# --- 4. Helper to create a new logger ---
def create_logger(phase_name: str):
    phase_log_dir = os.path.join(TENSORBOARD_LOG_DIR, phase_name)
    os.makedirs(phase_log_dir, exist_ok=True)
    return configure(phase_log_dir, ["tensorboard"])

# --- 5. Main Training Curriculum ---
def run_training_curriculum():
    
    # --- PHASE 1: Simple, Deterministic Environment ---
    if not os.path.exists(PHASE1_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 1")
        print("Opponents: Deterministic | Domain: Fixed")
        print("="*30 + "\n")
        env_phase1 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=0,
                            use_domain_randomization=False)
        
        model = PPO("MlpPolicy", env_phase1, verbose=0, **PHASE1_HYPERPARAMS)
        
        logger_phase1 = create_logger("Phase1_Deterministic")
        model.set_logger(logger_phase1)
        
        model.learn(total_timesteps=PHASE1_STEPS, progress_bar=True)
        model.save(PHASE1_MODEL_PATH)
        print(f"\n--- Phase 1 Complete. Model saved to {PHASE1_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 1. Model already exists at {PHASE1_MODEL_PATH} ---")


    # --- PHASE 2: Strong Deterministic Opponents ---
    if not os.path.exists(PHASE2_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 2")
        print("Opponents: Mixed Strong Adaptative | Domain: Fixed")
        print("="*30 + "\n")
        env_phase2 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=2,
                            use_advanced_opponents=True,
                            use_domain_randomization=False)

        model = PPO.load(PHASE1_MODEL_PATH, env=env_phase2, **PHASE2_HYPERPARAMS)
        
        logger_phase2 = create_logger("Phase2_MixedAdaptiveOpponents")
        model.set_logger(logger_phase2)

        model.learn(total_timesteps=PHASE2_STEPS, progress_bar=True, reset_num_timesteps=False)
        model.save(PHASE2_MODEL_PATH)
        print(f"\n--- Phase 2 Complete. Model saved to {PHASE2_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 2. Model already exists at {PHASE2_MODEL_PATH} ---")

    # --- PHASE 3: tanh reward ---
    if not os.path.exists(PHASE3_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 3 : looking for the win !")
        print("Opponents: Mixed Strong Adaptative | Domain: Fixed")
        print("="*30 + "\n")
        
        env_phase3 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=2,
                            use_advanced_opponents=True,
                            use_domain_randomization=False,
                            use_winner_reward=True)

        model = PPO.load(PHASE2_MODEL_PATH, env=env_phase3, **PHASE3_HYPERPARAMS)
        logger_phase3 = create_logger("Phase3_WinnerReward")
        model.set_logger(logger_phase3)

        model.learn(total_timesteps=PHASE3_STEPS, progress_bar=True, reset_num_timesteps=False)
        model.save(PHASE3_MODEL_PATH)
        print(f"\n--- Phase 3 Complete. Model saved to {PHASE3_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 3 ---")

    # --- PHASE 4: Full Domain Randomization ---
    if not os.path.exists(FINAL_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 4")
        print("Opponents: All Random | Domain: Random")
        print("="*30 + "\n")
        
        env_phase4 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=N_PLAYERS-1,
                            use_advanced_opponents=True,
                            use_domain_randomization=True,
                            use_winner_reward=True)

        model = PPO.load(PHASE3_MODEL_PATH, env=env_phase4, **PHASE4_HYPERPARAMS)
        
        logger_phase4 = create_logger("Phase4_FullRandomization")
        model.set_logger(logger_phase4)

        model.learn(total_timesteps=PHASE4_STEPS, progress_bar=True, reset_num_timesteps=False)
        model.save(FINAL_MODEL_PATH)
        print(f"\n--- Phase 4 Complete. Final robust model saved to {FINAL_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 4. Model already exists at {FINAL_MODEL_PATH} ---")

    
    # --- 6. Final Evaluation ---
    print("\n" + "="*30)
    print("--- FINAL MODEL EVALUATION ---")
    print("="*30 + "\n")
    
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"*** ERROR: Final model not found at {FINAL_MODEL_PATH}. Cannot run evaluation. ***")
        print("Please run the training script to generate the model.")
        return

    print(f"--- Loading final model from {FINAL_MODEL_PATH} for evaluation ---")
    model = PPO.load(FINAL_MODEL_PATH)
    
    # Evaluate on the hardest environment
    eval_env = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                      use_random_opponents=True, 
                      use_domain_randomization=True)
    
    n_eval_tournaments = 500
    agent_wins = 0

    print(f"Running {n_eval_tournaments} evaluation tournaments...")
    for _ in range(n_eval_tournaments):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
        
        final_scores = eval_env.player_scores
        if np.argmax(final_scores) == eval_env.agent_idx:
            agent_wins += 1

    win_rate = (agent_wins / n_eval_tournaments) * 100
    print(f"\n--- Final Agent Win Rate: {win_rate:.2f}% ---")

# --- 7. Run the script ---
if __name__ == "__main__":
    run_training_curriculum()