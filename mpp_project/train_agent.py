import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import numpy as np

# Import custom environment
from .mpp_env import MppEnv 

# --- 1. Define Absolute Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "log")
MODELS_DIR = os.path.join(ROOT_DIR, "models_v2") # Changed directory for V2
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "ppo_mpp_tensorboard_v2")
os.makedirs(MODELS_DIR, exist_ok=True)

# Define model paths
PHASE1_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v2_phase1_deterministic.zip")
PHASE2_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v2_phase2_mixed_opps.zip")
PHASE3_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v2_phase3_more_rand_opps.zip")
PHASE4_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v2_phase4_full_rand_opps.zip")
PHASE5_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v2_phase5_domain_rand.zip")

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
    "gamma": 0.995,          # Slight discount helps early stability
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.001
}
PHASE2_HYPERPARAMS = {
    "n_steps": 16384,
    "learning_rate": 1e-5,
    "gamma": 1.0,            # No haste to win
    "gae_lambda": 0.95,
    "vf_coef": 1.0,
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.01
}
PHASE3_HYPERPARAMS = {
    "n_steps": 16384,
    "learning_rate": 1e-5,
    "gamma": 1.0,            # No haste to win
    "gae_lambda": 0.95,
    "vf_coef": 1.0,
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.01
}
PHASE4_HYPERPARAMS = {
    "n_steps": 32768,        # Huge batch size to smooth out variance
    "learning_rate": 1e-5,   # Tiny steps to avoid overfitting to noise
    "gamma": 1.0,            # Long-term horizon
    "gae_lambda": 0.95,      # Smoothing
    "vf_coef": 1.0,          # Value function is critical here
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.005        # Reduced entropy: Find the exploit and stick to it
}
PHASE5_HYPERPARAMS = {
    "n_steps": 65536,        # Huge batch size to smooth out variance
    "learning_rate": 1e-5,   # Tiny steps to avoid overfitting to noise
    "gamma": 1.0,            # Long-term horizon
    "gae_lambda": 0.95,      # Smoothing
    "vf_coef": 1.0,          # Value function is critical here
    "policy_kwargs": policy_kwargs,
    "ent_coef": 0.005        # Reduced entropy: Find the exploit and stick to it
}

PHASE1_STEPS = 1_500_000
PHASE2_STEPS = 3_000_000
PHASE3_STEPS = 4_000_000
PHASE4_STEPS = 5_000_000
PHASE5_STEPS = 12_000_000

# --- Helper for Callbacks ---
def get_callbacks(phase_name, eval_env):
    """
    Creates a list of callbacks:
    1. Checkpoint: Save every 200k steps (Safety).
    2. Eval: Save 'best_model' if reward improves (Performance).
    """
    
    # 1. Checkpoint (Safety Net)
    # Saves model_200000_steps.zip, model_400000_steps.zip, etc.
    ckpt_callback = CheckpointCallback(
        save_freq=200_000, 
        save_path=os.path.join(MODELS_DIR, f"checkpoints_{phase_name}"),
        name_prefix=f"ckpt_{phase_name}"
    )
    
    # 2. Evaluation (Best Model)
    # Every 50k steps, plays 20 episodes on a separate env.
    # If average reward is the highest seen so far, saves 'best_model.zip'
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, f"best_{phase_name}"),
        log_path=os.path.join(MODELS_DIR, f"best_{phase_name}"),
        eval_freq=50_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    return CallbackList([ckpt_callback, eval_callback])

# --- 4. Helper to create a new logger ---
def create_logger(phase_name: str):
    phase_log_dir = os.path.join(TENSORBOARD_LOG_DIR, phase_name)
    os.makedirs(phase_log_dir, exist_ok=True)
    return configure(phase_log_dir, ["tensorboard"])

# --- 5. Main Training Curriculum ---
def run_training_curriculum():
    
    # --- PHASE 1: Simple, Deterministic Environment + Tanh Reward ---
    if not os.path.exists(PHASE1_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 1")
        print("Opponents: Deterministic | Domain: Fixed | Reward: Tanh Winner")
        print("="*30 + "\n")
        env_phase1 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=0,
                            use_domain_randomization=False,
                            use_advanced_opponents=False,
                            use_winner_reward=True)) # V2 change: Enable winner reward early
        
        model = PPO("MlpPolicy", env_phase1, verbose=0, **PHASE1_HYPERPARAMS)
        
        logger_phase1 = create_logger("Phase1_Deterministic")
        model.set_logger(logger_phase1)
        
        model.learn(total_timesteps=PHASE1_STEPS, progress_bar=True, 
                        callback=get_callbacks("phase1", env_phase1))
        model.save(PHASE1_MODEL_PATH)
        print(f"\n--- Phase 1 Complete. Model saved to {PHASE1_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 1. Model already exists at {PHASE1_MODEL_PATH} ---")


    # --- PHASE 2: Strong Deterministic + 2 Typical Random Opponents ---
    if not os.path.exists(PHASE2_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 2")
        print("Opponents: Mixed Strong Adaptative | Domain: Fixed | Reward: Tanh Winner")
        print("="*30 + "\n")
        env_phase2 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=2,
                            use_advanced_opponents=True,
                            use_domain_randomization=False,
                            use_winner_reward=True))

        model = PPO.load(PHASE1_MODEL_PATH, env=env_phase2, **PHASE2_HYPERPARAMS)
        
        logger_phase2 = create_logger("Phase2_MixedAdaptiveOpponents")
        model.set_logger(logger_phase2)

        model.learn(total_timesteps=PHASE2_STEPS, progress_bar=True, reset_num_timesteps=False, 
                    callback=get_callbacks("phase2", env_phase2))
        model.save(PHASE2_MODEL_PATH)
        print(f"\n--- Phase 2 Complete. Model saved to {PHASE2_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 2. Model already exists at {PHASE2_MODEL_PATH} ---")

    # --- PHASE 3: More Randomness ---
    if not os.path.exists(PHASE3_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 3")
        print("Opponents: Random Opponents Increase | Domain: Fixed | Reward: Tanh Winner")
        print("="*30 + "\n")
        
        # Slightly more chaos in opponents before full domain randomization
        env_phase3 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=4, 
                            use_advanced_opponents=True,
                            use_domain_randomization=False,
                            use_winner_reward=True))

        model = PPO.load(PHASE2_MODEL_PATH, env=env_phase3, **PHASE3_HYPERPARAMS)
        logger_phase3 = create_logger("Phase3_MoreRandomOpps")
        model.set_logger(logger_phase3)

        model.learn(total_timesteps=PHASE3_STEPS, progress_bar=True, reset_num_timesteps=False, 
                    callback=get_callbacks("phase3", env_phase3))
        model.save(PHASE3_MODEL_PATH)
        print(f"\n--- Phase 3 Complete. Model saved to {PHASE3_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 3. Model already exists at {PHASE3_MODEL_PATH} ---")

    # --- PHASE 4: Full Typical Random Opponents ---
    if not os.path.exists(PHASE4_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 4")
        print("Opponents: All Random | Domain: Fixed | Reward: Tanh Winner")
        print("="*30 + "\n")
        
        env_phase4 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=N_PLAYERS-1,
                            use_domain_randomization=False,
                            use_winner_reward=True))

        model = PPO.load(PHASE3_MODEL_PATH, env=env_phase4, **PHASE4_HYPERPARAMS)
        
        logger_phase4 = create_logger("Phase4_FullRandomOpps")
        model.set_logger(logger_phase4)

        model.learn(total_timesteps=PHASE4_STEPS, progress_bar=True, reset_num_timesteps=False, 
                    callback=get_callbacks("phase4", env_phase4))
        model.save(PHASE4_MODEL_PATH)
        print(f"\n--- Phase 4 Complete. Model saved to {PHASE4_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 4. Model already exists at {PHASE4_MODEL_PATH} ---")

    # --- PHASE 5: Full Domain Randomization ---
    if not os.path.exists(PHASE5_MODEL_PATH):
        print("\n" + "="*30)
        print("STARTING CURRICULUM: PHASE 5")
        print("Opponents: All Random | Domain: Random | Reward: Tanh Winner")
        print("="*30 + "\n")
        
        env_phase5 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                            num_random_opponents=N_PLAYERS-1,
                            use_domain_randomization=True,
                            use_winner_reward=True))

        model = PPO.load(PHASE4_MODEL_PATH, env=env_phase5, **PHASE5_HYPERPARAMS)
        
        logger_phase4 = create_logger("Phase5_DomainRandomization")
        model.set_logger(logger_phase4)

        model.learn(total_timesteps=PHASE5_STEPS, progress_bar=True, reset_num_timesteps=False, 
                    callback=get_callbacks("phase5", env_phase5))
        model.save(PHASE5_MODEL_PATH)
        print(f"\n--- Phase 5 Complete. Model saved to {PHASE5_MODEL_PATH} ---")
    else:
        print(f"--- SKIPPING Phase 5. Model already exists at {PHASE5_MODEL_PATH} ---")

    
    # --- 6. Final Evaluation ---
    print("\n" + "="*30)
    print("--- FINAL MODEL EVALUATION ---")
    print("="*30 + "\n")
    
    if not os.path.exists(PHASE5_MODEL_PATH):
        print(f"*** ERROR: Final model not found. ***")
        return

    print(f"--- Loading final model from {PHASE5_MODEL_PATH} for evaluation ---")
    model = PPO.load(PHASE5_MODEL_PATH)
    
    # Evaluate on the hardest environment
    eval_env = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                      num_random_opponents=N_PLAYERS-1, # Max randomness
                      use_advanced_opponents=True,
                      use_domain_randomization=True,
                      use_winner_reward=True)
    
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