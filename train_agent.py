import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import numpy as np

# Import custom environment
from mpp_project.mpp_env import MppEnv 

# Define Absolute Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "log")
MODELS_DIR = os.path.join(ROOT_DIR, "models_v3")
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "ppo_mpp_tensorboard_v3")
os.makedirs(MODELS_DIR, exist_ok=True)

# Define model paths
PHASE1_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v3_phase1_deterministic_final.zip")
PHASE2_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v3_phase2_mixed_opps_final.zip")
PHASE3_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v3_phase3_full_rand_opps_final.zip")
PHASE4_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_v3_phase4_domain_rand_final.zip")

# Default Simulation Configuration
match_params = {
    'n_matches': 51,
    'ev_avg': 35,
    'draw_fact_min': 0.2,
    'draw_fact_max': 0.75,
    'outsider_fact_min': 1/7.5,
    'outsider_fact_max': 1.0,
    'proba_fact_std': 0.2 # Since v2 seemed to rely on gain to estimate probas, we decorrelate them more
}
N_PLAYERS = 12
N_MATCHES = 51

# Hyperparameters
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

PHASE1_STEPS = 2_000_000
PHASE2_STEPS = 6_000_000
PHASE3_STEPS = 9_000_000
PHASE4_STEPS = 13_000_000

# Helper for Callbacks
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

# Helper to load previous phase model
def get_transition_model_path(phase_name, fallback_path):
    """
    Smart Loader: Tries to find the 'best_model.zip' from the EvalCallback.
    If found, returns that. Otherwise, returns the standard 'fallback_path' (Last).
    """
    best_model_path = os.path.join(MODELS_DIR, f"best_{phase_name}", "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"   >>> Loading BEST model from: {best_model_path}")
        return best_model_path
    else:
        print(f"   >>> Loading LAST model from: {fallback_path}")
        return fallback_path

# Helper to create a new logger
def create_logger(phase_name: str):
    phase_log_dir = os.path.join(TENSORBOARD_LOG_DIR, phase_name)
    os.makedirs(phase_log_dir, exist_ok=True)
    return configure(phase_log_dir, ["tensorboard"])

# Main Training Curriculum
def run_training_curriculum():
    
    try:
        # --- PHASE 1: Deterministic Warmup ---
        if not os.path.exists(PHASE1_MODEL_PATH):
            print(f"\n=== PHASE 1: Deterministic Warmup ===")
            
            env_phase1 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                                num_random_opponents=0, 
                                use_domain_randomization=False,
                                use_winner_reward=True)
            
            eval_env1 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                               num_random_opponents=0, 
                               use_domain_randomization=False,
                               use_winner_reward=True))

            model = PPO("MlpPolicy", env_phase1, verbose=0, **PHASE1_HYPERPARAMS)
            model.set_logger(create_logger("Phase1_Deterministic"))
            
            model.learn(total_timesteps=PHASE1_STEPS, progress_bar=True, 
                        callback=get_callbacks("phase1", eval_env1))
            model.save(PHASE1_MODEL_PATH)
            print(f"--- Phase 1 Saved ---")
        else:
            print(f"--- SKIPPING Phase 1 (Found) ---")


        # --- PHASE 2: Calibration (Mixed Opponents) ---
        if not os.path.exists(PHASE2_MODEL_PATH):
            print(f"\n=== PHASE 2: Calibration (Mixed Opponents) ===")
            
            env_phase2 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                                num_random_opponents=2,
                                use_advanced_opponents=True,
                                use_domain_randomization=False,
                                use_winner_reward=True)
            
            eval_env2 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                                num_random_opponents=2,
                                use_advanced_opponents=True,
                                use_domain_randomization=False,
                                use_winner_reward=True))

            # Load Previous Model: For Phase 1, Last is usually fine
            prev_model = PHASE1_MODEL_PATH
            # To load best PHASE 1 model, use:
            # prev_model = get_transition_model_path("phase1", PHASE1_MODEL_PATH)
            
            model = PPO.load(prev_model, env=env_phase2, **PHASE2_HYPERPARAMS)
            model.set_logger(create_logger("Phase2_Mixed"))
            
            model.learn(total_timesteps=PHASE2_STEPS, progress_bar=True, reset_num_timesteps=False,
                        callback=get_callbacks("phase2", eval_env2))
            model.save(PHASE2_MODEL_PATH)
            print(f"--- Phase 2 Saved ---")
        else:
            print(f"--- SKIPPING Phase 2 (Found) ---")


        # --- PHASE 3: The Real Game (11 Randoms) ---
        if not os.path.exists(PHASE3_MODEL_PATH):
            print(f"\n=== PHASE 3: The Real Game (11 Power-Law Randoms) ===")
            
            env_phase3 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                                num_random_opponents=11,
                                use_advanced_opponents=True,
                                use_domain_randomization=False,
                                use_winner_reward=True)
            
            eval_env3 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                                num_random_opponents=11, 
                                use_advanced_opponents=True,
                                use_domain_randomization=False,
                                use_winner_reward=True))

            # Load Previous Model: Last Phase 2 model
            prev_model = PHASE2_MODEL_PATH
            # To load best PHASE 2 model, use:
            #prev_model = get_transition_model_path("phase2", PHASE2_MODEL_PATH)

            model = PPO.load(prev_model, env=env_phase3, **PHASE3_HYPERPARAMS)
            model.set_logger(create_logger("Phase3_11Randoms"))
            
            model.learn(total_timesteps=PHASE3_STEPS, progress_bar=True, reset_num_timesteps=False,
                        callback=get_callbacks("phase3", eval_env3))
            model.save(PHASE3_MODEL_PATH)
            print(f"--- Phase 3 Saved ---")
        else:
            print(f"--- SKIPPING Phase 3 ---")


        # --- PHASE 4: Robustness (Domain Randomization) ---
        if not os.path.exists(PHASE4_MODEL_PATH):
            print(f"\n=== PHASE 4: Robustness (Domain Randomization) ===")
            
            env_phase4 = MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                                num_random_opponents=11,
                                use_advanced_opponents=True,
                                use_domain_randomization=True,
                                use_winner_reward=True)
            
            eval_env4 = Monitor(MppEnv(n_players=N_PLAYERS, n_matches=N_MATCHES, match_params=match_params,
                                num_random_opponents=11,
                                use_advanced_opponents=True,
                                use_domain_randomization=True,
                                use_winner_reward=True))

            # Load Previous Model: Last Phase 3 model
            prev_model = PHASE3_MODEL_PATH
            # To load best PHASE 2 model, use:
            #prev_model = get_transition_model_path("phase3", PHASE3_MODEL_PATH)

            model = PPO.load(prev_model, env=env_phase4, **PHASE4_HYPERPARAMS)
            model.set_logger(create_logger("Phase4_DomainRand"))
            
            model.learn(total_timesteps=PHASE4_STEPS, progress_bar=True, reset_num_timesteps=False,
                        callback=get_callbacks("phase4", eval_env4))
            model.save(PHASE4_MODEL_PATH)
            print(f"--- Phase 4 Complete ---")
        else:
            print(f"--- SKIPPING Phase 4 (Found) ---")
            
    except KeyboardInterrupt:
        print("\n\nINTERRUPTED BY USER (Ctrl+C)")
        if 'model' in locals():
            model.save(os.path.join(MODELS_DIR, "model_INTERRUPTED.zip"))
            print("Saved backup.")

# Run the script
if __name__ == "__main__":
    run_training_curriculum()