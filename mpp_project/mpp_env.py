import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Callable, Dict

# Import all strategies we might need
from .strategies import (
    strat_typical_opponent, 
    strat_best_ev, 
    strat_highest_variance, 
    strat_best_simple_rel_ev
)
from .match_simulator import (
    generate_outcome_probas, 
    generate_gains, 
    generate_opponent_repartition
)

class MppEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, 
                 n_players: int = 12, 
                 n_matches: int = 51, 
                 match_params: Dict = None,
                 # --- Curriculum Arguments ---
                 num_random_opponents: int = 0,
                 use_domain_randomization: bool = False
                 # ----------------------------
                ):
        super(MppEnv, self).__init__()

        self.n_players = n_players
        self.n_matches = n_matches
        self.match_params = match_params

        # --- Store the curriculum settings ---
        self.num_random_opponents = num_random_opponents
        self.use_domain_randomization = use_domain_randomization

        # --- Define Action and Observation Spaces ---
        
        # Action: Bet on Favorite (0), Draw (1), or Outsider (2)
        self.action_space = spaces.Discrete(3)

        # Observation: the state
        # Simple Box (a flat vector of numbers)
        # 1. Match Probas (3)
        # 2. Match Gains (3)
        # 3. Opponent Repartition (3)
        # 4. All Player Scores (n_players)
        # 5. Matches Remaining (1)
        obs_size = 3 + 3 + 3 + n_players + 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # --- Internal State Variables ---
        self.player_scores = np.zeros(n_players)
        self.current_match_idx = 0
        self.agent_idx = 0 
        
        # --- Curriculum-aware opponents ---
        num_deterministic_opps = (n_players - 1) - self.num_random_opponents
        
        deterministic_strats = [strat_best_ev] * num_deterministic_opps
        random_strats = [strat_typical_opponent] * self.num_random_opponents
        
        self.other_player_strategies = deterministic_strats + random_strats
        np.random.shuffle(self.other_player_strategies)
        
        self._generate_tournament()

    def _generate_tournament(self):
        """Generates all match data for the entire tournament."""
        self.outcome_probas = generate_outcome_probas(
            self.n_matches,
            self.match_params['draw_fact_min'], self.match_params['draw_fact_max'],
            self.match_params['outsider_fact_min'], self.match_params['outsider_fact_max']
        )
        self.match_gains = generate_gains(self.outcome_probas,
                                          self.match_params['ev_avg'],
                                          self.match_params['proba_fact_std'])
        self.opp_repartition = generate_opponent_repartition(self.outcome_probas)

    def _get_obs(self) -> np.ndarray:
        """Helper function to create the observation vector."""
        m_probas = self.outcome_probas[self.current_match_idx]
        m_gains = self.match_gains[self.current_match_idx]
        m_repart = self.opp_repartition[self.current_match_idx]
        matches_remaining = np.array([self.n_matches - self.current_match_idx])
        
        # Concatenate everything into a single, flat vector
        obs = np.concatenate([
            m_probas,
            m_gains,
            m_repart,
            self.player_scores,
            matches_remaining
        ]).astype(np.float32)
        
        return obs

    def _get_info(self) -> Dict:
        """Helper function to return diagnostic info."""
        return {
            "current_match": self.current_match_idx,
            "agent_score": self.player_scores[self.agent_idx],
            "leader_score": np.max(self.player_scores)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Curriculum-Aware Domain Randomization ---
        if self.use_domain_randomization:
            # On each new tournament, pick new "rules"
            self.match_params['ev_avg'] = np.random.uniform(25, 45)
            self.match_params['draw_fact_min'] = np.random.uniform(0.15, 0.25)
            self.match_params['draw_fact_max'] = np.random.uniform(0.7, 0.8)
            self.match_params['outsider_fact_min'] = np.random.uniform(1/11, 1/6)
            # Small randomness : competition with no balanced matches are rare
            self.match_params['outsider_fact_max'] = np.random.uniform(0.95, 1.0)
            self.match_params['proba_fact_std'] = np.random.uniform(0.03, 0.09)
        
        self.player_scores = np.zeros(self.n_players)
        self.current_match_idx = 0
        self._generate_tournament()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action: int):
        """
        The agent takes an 'action' (the bet).
        The environment simulates one match and returns the result.
        """
        
        # Get data for the current match
        m_probas = self.outcome_probas[self.current_match_idx]
        m_gains = self.match_gains[self.current_match_idx]
        m_repart = self.opp_repartition[self.current_match_idx]
        
        # --- 1. Simulate the match ---
        true_outcome = np.random.choice([0, 1, 2], p=m_probas)
        
        # --- 2. Get bets from all players ---
        
        # The agent's bet is the 'action'
        agent_bet = action
        
        # Get bets for all other players
        other_bets = []
        for i, strat_func in enumerate(self.other_player_strategies):
            # We pass p_idx = i+1 because agent is 0
            bet = strat_func(m_probas, m_gains, m_repart, self.player_scores, i + 1)
            other_bets.append(bet)
            
        # --- 3. Update scores ---
        previous_scores = np.copy(self.player_scores)

        # Agent (Player 0)
        if agent_bet == true_outcome:
            self.player_scores[self.agent_idx] += m_gains[true_outcome]
            
        # Other Players (Player 1 to N-1)
        for i, bet in enumerate(other_bets):
            if bet == true_outcome:
                self.player_scores[i + 1] += m_gains[true_outcome]

        # --- 4. Define the Reward ---
        # /!\ We want to reward winning, not just scoring.
        # We'll use a multi-part reward:
        # 1. A dense "guide" for gaining/losing ground to the best opponent.
        # 2. A large "event" bonus for taking the lead.
        # 3. A large "event" penalty for losing the lead.
        # 4. A massive "terminal" bonus for winning the tournament.

        # Get the agent's and leader's rank/score *before* this match
        agent_score_prev = previous_scores[self.agent_idx]
        opp_scores_prev = np.delete(previous_scores, self.agent_idx)
        leader_score_prev = np.max(opp_scores_prev)
        was_agent_leading = agent_score_prev > leader_score_prev

        # Get the agent's and leader's rank/score *after* this match
        agent_score_new = self.player_scores[self.agent_idx]
        opp_scores_new = np.delete(self.player_scores, self.agent_idx)
        leader_score_new = np.max(opp_scores_new)
        is_agent_leading = agent_score_new > leader_score_new

        # 1. The Dense "Guide" Reward
        # Use np.sign() on the *point gap* change, just as a guide.
        previous_gap_pts = agent_score_prev - leader_score_prev
        current_gap_pts = agent_score_new - leader_score_new
        gap_change_pts = current_gap_pts - previous_gap_pts
        reward = np.sign(gap_change_pts)  # +1, -1, or 0

        # 2. "Took the Lead" Event Bonus
        if (not was_agent_leading) and is_agent_leading:
            reward += 25  # Big, stable bonus for a critical event

        # 3. "Lost the Lead" Event Penalty
        if was_agent_leading and (not is_agent_leading):
            reward -= 25  # Big, stable penalty

        # 5. Prepare for next step
        self.current_match_idx += 1
        done = (self.current_match_idx >= self.n_matches)

        # 4. The Terminal Win Bonus
        if done:
            # Check if agent won (is #1)
            if np.argmax(self.player_scores) == self.agent_idx:
                # A bonus larger than all others
                reward += 100
        
        # Get new observation and info
        observation = self._get_obs() if not done else self.observation_space.sample() # Return dummy obs if done
        info = self._get_info()
        
        # Gym API requires a 5-tuple: (obs, reward, terminated, truncated, info)
        # 'terminated' means the episode ended naturally (we finished the tournament)
        # 'truncated' means it ended artificially (e.g., time limit)
        return observation, reward, done, False, info