import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Callable, Dict

# Strategies
from .strategies import (
    strat_typical_opponent, strat_best_ev, strat_highest_variance, 
    strat_favorite, strat_adaptive_simple_ev, 
    strat_safe_simple_ev
)
# Simulators
from .match_simulator import (
    generate_outcome_probas, generate_gains, generate_opponent_repartition
)
# Core (Shared Logic)
from .core import get_observation

class MppEnv(gym.Env):
    """
    Custom Environment for the MonPetitProno Tournament.
    Supports Domain Randomization, Power Law Opponents, and Variable League Sizes.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, 
                 n_players: int = 12,  # Max size for player_scores in observation
                 n_matches: int = 51, 
                 match_params: Dict = None,
                 # --- Curriculum Arguments ---
                 num_random_opponents: int = 0,
                 use_domain_randomization: bool = False,
                 use_advanced_opponents: bool = False,
                 use_winner_reward: bool = False
                 # ----------------------------
                ):
        super(MppEnv, self).__init__()
        
        # The fixed input size for the Neural Network (Padding / Top-K Logic)
        self.obs_n_players = n_players 
        self.n_matches = n_matches
        self.match_params = match_params

        # Store the curriculum settings
        self.use_domain_randomization = use_domain_randomization
        self.use_winner_reward = use_winner_reward
        self.base_num_random = num_random_opponents
        
        # Action Space: Bet on Favorite (0), Draw (1), or Outsider (2) (Mapped by probability sorting)
        self.action_space = spaces.Discrete(3)

        # Observation Space (Fixed Size based on obs_n_players)
        # 1. Match Probas (3)
        # 2. Match Gains (3)
        # 3. Opponent Repartition (3)
        # 4. All Player Scores (n_players - 1 relative to agent)
        # 5. Gap Ratio (1)
        # 6. Time Remaining (1)
        # 7. Simple EV (3)
        obs_size = 3 + 3 + 3 + (self.obs_n_players - 1) + 1 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # Internal State Variables
        self.current_match_idx = 0
        self.agent_idx = 0 
        self.n_active_players = n_players # Will vary per episode in Domain Rand
        self.player_scores = np.zeros(self.n_active_players)
        self.max_points_per_match = np.zeros(n_matches)
        
        # Domain Parameters (Initialized in reset)
        self.repart_noise_factor = 0.0
        self.future_estimation_error = 1.0
        self.repartition_gamma = 2.0

        # Strategy Pool Initialization
        self._init_strategy_pool(use_advanced_opponents)

    def _init_strategy_pool(self, use_advanced):
        """Initialize the pool of potential opponent strategies."""
        if use_advanced:
            self.strat_pool_random = [strat_typical_opponent]
            self.strat_pool_elite = [strat_adaptive_simple_ev, strat_safe_simple_ev, strat_best_ev]
        else:
            self.strat_pool_random = [strat_typical_opponent]
            self.strat_pool_elite = [strat_best_ev, strat_highest_variance, strat_favorite]

    def _generate_tournament(self):
        """Generates match data."""
        self.outcome_probas = generate_outcome_probas(
            self.n_matches,
            self.match_params['draw_fact_min'], self.match_params['draw_fact_max'],
            self.match_params['outsider_fact_min'], self.match_params['outsider_fact_max']
        )
        self.match_gains = generate_gains(self.outcome_probas,
                                          self.match_params['ev_avg'],
                                          self.match_params['proba_fact_std'])
        
        # Use the Power Law Repartition, (gamma randomized in reset)
        self.opp_repartition = generate_opponent_repartition(
            self.outcome_probas, 
            gamma=self.repartition_gamma
        )
        self.max_points_per_match = np.max(self.match_gains, axis=1)

    def _get_obs(self) -> np.ndarray:
        """Constructs the observation with Domain Randomization noise."""
        
        # 1. Future Points (Noisy Estimation)
        if self.current_match_idx < self.n_matches:
            true_future_max = np.sum(self.max_points_per_match[self.current_match_idx:])
        else:
            true_future_max = 1.0
            
        # Apply estimation noise
        observed_future_max = true_future_max * self.future_estimation_error

        matches_rem_fraction = (self.n_matches - self.current_match_idx) / self.n_matches

        # 2. Repartition Noise
        raw_repart = self.opp_repartition[self.current_match_idx]
        noisy_repart = raw_repart + np.random.normal(0, self.repart_noise_factor, 3)
        noisy_repart = np.clip(noisy_repart, 0.01, 0.99)
        noisy_repart /= np.sum(noisy_repart)
        
        # 3. Handle Variable Player Count (Top-K Logic & Padding)
        agent_score = self.player_scores[self.agent_idx]
        opp_scores = np.delete(self.player_scores, self.agent_idx)
        opp_scores_sorted = np.sort(opp_scores)[::-1]
        
        target_n_opps = self.obs_n_players - 1
        
        if len(opp_scores_sorted) >= target_n_opps:
            # Too many players: Take Top K
            obs_opp_scores = opp_scores_sorted[:target_n_opps]
        else:
            # Too few players: Pad with "Dead" scores (-99999.0)
            padding = np.full(target_n_opps - len(opp_scores_sorted), -99999.0)
            obs_opp_scores = np.concatenate([opp_scores_sorted, padding])

        # Reconstruct fixed-size score array for the core function
        fixed_size_scores = np.concatenate([[agent_score], obs_opp_scores])

        # Call Shared Core
        obs, _ = get_observation(
            match_probas=self.outcome_probas[self.current_match_idx],
            match_gains=self.match_gains[self.current_match_idx],
            opp_repartition=noisy_repart,
            player_scores=fixed_size_scores,
            agent_idx=0, 
            future_max_points=observed_future_max,
            matches_remaining_fraction=matches_rem_fraction,
            ev_avg=self.match_params['ev_avg']
        )
        return obs

    def _get_info(self) -> Dict:
        """Helper function to return diagnostic info."""
        return {
            "current_match": self.current_match_idx,
            "agent_score": self.player_scores[self.agent_idx],
            "leader_score": np.max(self.player_scores),
            "n_players": self.n_active_players
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Domain Randomization
        if self.use_domain_randomization:
            # 1. Match Rules
            self.match_params['ev_avg'] = np.random.uniform(25, 45)
            self.match_params['draw_fact_min'] = np.random.uniform(0.15, 0.25)
            self.match_params['draw_fact_max'] = np.random.uniform(0.7, 0.8)
            self.match_params['outsider_fact_min'] = np.random.uniform(1/11, 1/6)
            self.match_params['outsider_fact_max'] = np.random.uniform(0.95, 1.0)
            self.match_params['proba_fact_std'] = np.random.uniform(0.03, 0.20)
            
            # 2. Noise Factors
            self.repart_noise_factor = np.random.uniform(0.0, 0.15)
            self.future_estimation_error = np.random.uniform(0.8, 1.2)
            
            # 3. Variable Player Count
            self.n_active_players = np.random.randint(6, 26)
            
            # 4. Crowd Herding (Gamma)
            self.repartition_gamma = np.random.uniform(1.5, 3.0)
            
        else:
            # Defaults
            self.repart_noise_factor = 0.0
            self.future_estimation_error = 1.0
            self.n_active_players = self.obs_n_players
            self.repartition_gamma = 2.0

        # --- Init Opponents for this Episode ---
        self.player_scores = np.zeros(self.n_active_players)
        current_opponents = []
        n_opps_needed = self.n_active_players - 1
        
        if self.use_domain_randomization:
            # In Domain Rand: 80% Randoms (Crowd), 20% Elite
            n_random = int(n_opps_needed * 0.8)
            n_elite = n_opps_needed - n_random
        else:
            # Standard logic
            n_random = self.base_num_random
            n_elite = max(0, n_opps_needed - n_random)

        for _ in range(n_random):
            current_opponents.append(self.strat_pool_random[0])
        for i in range(n_elite):
            current_opponents.append(self.strat_pool_elite[i % len(self.strat_pool_elite)])
            
        np.random.shuffle(current_opponents)
        self.active_opponent_strats = current_opponents

        self.current_match_idx = 0
        self.agent_idx = 0
        self._generate_tournament()
        
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """
        The agent takes an 'action'.
        The environment simulates one match and returns the result.
        """
        m_probas = self.outcome_probas[self.current_match_idx]
        m_gains = self.match_gains[self.current_match_idx]
        
        # Simulate the match
        true_outcome = np.random.choice([0, 1, 2], p=m_probas)
        
        # Map Agent Action
        # Reconstruct the sort order to know what the agent picked
        sort_idx = np.argsort(m_probas)[::-1]
        agent_bet = sort_idx[action]
        
        # Capture State before Update
        previous_scores = np.copy(self.player_scores)
        
        # Update Agent
        if agent_bet == true_outcome:
            self.player_scores[self.agent_idx] += m_gains[true_outcome]
            
        # Update Opponents
        for i, strat_func in enumerate(self.active_opponent_strats):
            p_idx = i + 1
            bet = strat_func(
                match_probas=self.outcome_probas[self.current_match_idx:], 
                match_gains=self.match_gains[self.current_match_idx:], 
                opp_repartition=self.opp_repartition[self.current_match_idx:], 
                player_scores=self.player_scores, 
                my_idx=p_idx, 
                matches_remaining=self.n_matches - self.current_match_idx,
                ev_avg=self.match_params['ev_avg'],
                n_matches=self.n_matches
            )
            if bet == true_outcome:
                self.player_scores[p_idx] += m_gains[true_outcome]

        # Reward (Affine Decay)
        opp_scores_prev = np.delete(previous_scores, self.agent_idx)
        leader_score_prev = np.max(opp_scores_prev)
        agent_score_prev = previous_scores[self.agent_idx]

        opp_scores_new = np.delete(self.player_scores, self.agent_idx)
        leader_score_new = np.max(opp_scores_new)
        agent_score_new = self.player_scores[self.agent_idx]

        if not self.use_winner_reward:
             # Legacy Dense Reward
            gap_change = (agent_score_new - leader_score_new) - (agent_score_prev - leader_score_prev)
            reward = np.sign(gap_change)
        else:
            # Adaptive Temperature Tanh Reward
            # Dynamic Base Temperature: Scales with this tournament's ev_avg
            T_base = 3.0 * self.match_params['ev_avg']
            
            matches_rem_prev = self.n_matches - self.current_match_idx
            matches_rem_new = matches_rem_prev - 1
            
            # Formula: T = T_base * (0.2 + 0.8 * (Rem/Total))
            decay_prev = 0.2 + 0.8 * (matches_rem_prev / self.n_matches)
            T_prev = max(0.1, T_base * decay_prev)
            
            decay_new = 0.2 + 0.8 * (matches_rem_new / self.n_matches)
            T_new = max(0.1, T_base * decay_new)

            def get_potential(gap, temp): 
                return np.tanh(gap / temp)
            
            phi_prev = get_potential(agent_score_prev - leader_score_prev, T_prev)
            phi_new  = get_potential(agent_score_new - leader_score_new, T_new)
            
            reward = (phi_new - phi_prev) * 10.0

        self.current_match_idx += 1
        done = (self.current_match_idx >= self.n_matches)
        
        if done:
            # Terminal Win Bonus
            if np.argmax(self.player_scores) == self.agent_idx:
                reward += 10.0
        
        obs = self._get_obs() if not done else self.observation_space.sample()
        return obs, reward, done, False, self._get_info()