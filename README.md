# MonPetitProno (MPP) Strategy Simulator - V3

This repository contains a sophisticated **Reinforcement Learning (RL)** system designed to solve the "MonPetitProno" (MPP) football prediction tournament.

Unlike standard betting models that simply maximize Expected Value (EV), this agent optimizes for **Tournament Win Probability**. It learns to navigate a noisy, adversarial environment populated by "Lambda" (casual) players who exhibit realistic betting biases (herding).

**Current Version:** `v3.0` (Power Law Opponents & Domain Randomization)

---

## 🧠 The "V3" Intelligence: The Cynical Detective

Through **Domain Randomization** and **Curriculum Learning**, the V3 agent has evolved beyond simple statistics. It has developed a distinct psychological profile we call the **"Cynical Detective"**:

1.  **Rational Aggression:**
    * **Leading:** Plays conservatively (bets Favorite) to protect its lead.
    * **Trailing:** Takes calculated risks, but *never* blindly.
2.  **The "Trap" Detector:**
    * It refuses to bet on high-paying Outsiders if the gain implies a near-zero probability. It perceives "too good to be true" odds as traps.
3.  **Mispriced Safety:**
    * When desperate, it targets **"Low-Value Outsiders"**—bets that pay poorly but are statistically more likely to hit than the odds suggest. It prefers **consistency over lottery tickets**.

---

## 🔬 Key Technical Features

### 1. Realistic Opponent Modeling (Power Law)
Casual players do not bet randomly. They "herd" onto favorites. V3 models this using a **Power Law** distribution:
$$\text{Share}_i = \frac{P_i^\gamma}{\sum P_k^\gamma}$$
* $\gamma \approx 2.0$: Standard herding behavior.
* The agent learns to exploit this by finding value in the under-bet Draw/Outsider markets.

### 2. Domain Randomization
To prevent overfitting to a specific tournament structure, the agent is trained in a chaotic environment where every episode has different rules:
* **Variable EV:** Average match scores range from 25 to 45 points.
* **Variable League Size:** Competes against 6 to 25 opponents.
* **Noisy Information:** The agent receives imperfect estimates of future match points and crowd behavior.

### 3. Value Injection Training
The simulator artificially "injects" value (pricing errors) into the training data to force the agent to decouple **Gain** from **Probability**. This teaches the agent to read the board like a human analyst rather than relying on simple heuristics (like "High Gain = Low Probability").

---

## 📂 Project Structure

```

mppanalysis/
│
├── mpp_project/          # The core Python package
│   ├── core.py               # Core math and utility functions
│   ├── match_simulator.py    # Generates realistic match data (probas, gains, etc.)
│   ├── simulation.py         # The main tournament simulation engine
│   ├── mpp_env.py            # Custom Gymnasium Environment (The Dojo)
│   └── strategies.py         # Defines all "agent" strategies to be tested
│
├── notebooks/              # Jupyter notebooks for exploration and analysis
│   ├── 00_analysis_single_match_ev.ipynb         # Explores the expected value calculations for individual match bets to establish a baseline for betting efficiency.
│   ├── 01_analysis_by_match_count.ipynb          # Analyzes how strategy performance and variance evolve as the number of matches in a tournament increases.
│   ├── 02_analysis_by_success_prob.ipynb         # Investigates the impact of varying prediction success probabilities on the overall tournament outcome.
│   ├── 03_analysis_variable_p_bernoulli.ipynb    # Models match outcomes using Bernoulli trials with variable probabilities to simulate realistic tournament dynamics.
│   ├── 04_analysis_final_score_distribution.ipynb # Compares the final score distributions of different strategies to visualize their risk/reward profiles.
│   ├── 05_analysis_agent_psychology.ipynb        # Performs an "MRI scan" of the trained agent to map its decision-making boundaries relative to score gaps and match certainty.
│   ├── 06_analysis_value_sensitivity.ipynb       # Tests the agent's ability to identify and exploit "value bets" (mispriced odds) under different tournament conditions.
│   └── 07_analysis_crowd_psychology.ipynb        # Tests the agent's dependency to crowd betting repartition. 
│
├── .gitignore
├── play.py                       # Script to get the Agent decision for a real world match
├── requirements.txt
├── run_tournament_simulation.py  # MAIN SCRIPT: Runs the full simulation and prints results
├── train_agent.py                # Script to train the RL agent through the curriculum
└── README.md                     # Project documentation

````

## 🚀 How to Run the Simulation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/godartvincent/mppanalysis.git](https://github.com/godartvincent/mppanalysis.git)
    cd mppanalysis
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Agent:**

    The training uses a 4-Phase Curriculum to guide the agent from "Novice" to "Grandmaster":

    ```bash
    # Run from the root directory
    python -m mpp_project.train_agent
    ```
    Phase 1: Deterministic Warmup (Learn the rules).

    Phase 2: Calibration (Learn to beat Mixed opponents).

    Phase 3: The Real Game (Beat 11 Power-Law Randoms).

    Phase 4: Robustness (Full Domain Randomization).

4.  **Run the simulation:**
    ```bash
    python run_tournament_simulation.py
    ```
    This will run `N_SAMPLES` (e.g., 20000) full tournaments for each strategy, pitting each "challenger" against a league of `N_PLAYERS - 1` "typical" opponents. It will then print the final results table and display a plot of the score distributions.

5.  **Ask the Agent for Advice (Real World):**

    Input real match odds to get a recommendation:

    ```bash
    python play.py
    ```

## 📊 Performance

The simulation was run with `N_SAMPLES = 20000` tournaments in which each strategy were evaluated against 11 other players.

### Final Simulation Results (20000 tournaments per strategy)

Strategy                 | Win Rate | Avg Points |   Median   |  Std Dev
| :--- | :---: | :---: | :---: | :---: |
RL V3 (P4 - DomainRnd)   |   13.49% |  1661.1pts |  1629.0pts | 459.76pts
RL V2 (P3 - MoreRand)    |   13.41% |  1746.0pts |  1709.0pts | 374.86pts
Adaptive Simple EV       |   12.98% |  1635.6pts |  1590.0pts | 470.97pts
RL V3 (P3 - FullRand)    |   12.93% |  1763.2pts |  1790.0pts | 439.37pts
RL V2 (P4 - FullRand)    |   12.68% |  1715.3pts |  1697.0pts | 406.49pts
RL V2 (P5 - DomainRnd)   |   12.46% |  1682.5pts |  1675.0pts | 412.93pts
RL V3 (P2 - Mixed)       |   12.39% |  1776.8pts |  1835.0pts | 447.23pts
RL V2 (P2 - Mixed)       |   12.39% |  1722.1pts |  1688.0pts | 372.96pts
RL V3 (P1 - PPO)         |   12.34% |  1829.0pts |  1893.0pts | 420.59pts
RL Legacy (V1 - 4M)      |   11.59% |  1857.6pts |  1894.0pts | 388.94pts
RL V2 (P1 - PPO)         |   11.45% |  1773.3pts |  1875.0pts | 478.31pts
Best Simple EV           |   11.21% |  1995.1pts |  1995.0pts | 282.08pts
Favorite                 |   10.79% |  1991.8pts |  1991.0pts | 280.69pts
Safe Simple EV           |   10.71% |  1991.6pts |  1990.0pts | 280.99pts
Adaptive Simple Rel EV   |    7.20% |  1436.1pts |  1397.0pts | 421.79pts
Random                   |    5.18% |  1631.1pts |  1621.0pts | 338.68pts
Safe Simple Rel EV       |    4.25% |  1488.0pts |  1484.0pts | 365.45pts
Best Simple Relative EV  |    3.86% |  1460.5pts |  1445.0pts | 362.36pts
Highest Variance         |    3.70% |  1435.1pts |  1426.0pts | 369.73pts

*(Note: The expected win rate for a "lambda" (typical) player is 1/12 = 8.33%)*

### 📊 Analysis of Key Insights

This table reveals several powerful (and sometimes paradoxical) insights into tournament strategy.

#### 1. The "EV Edge" (Maximizing Points)
The **`Best Simple EV`** strategy is a massive success for maximizing points. It achieves the highest **average score** (1995.1 pts) by a huge margin. This validates the "Value" hypothesis: by calculating $EV = P_{\text{true}} \cdot G_{\text{frozen}}$, this strategy effectively identifies "mispriced" value bets, significantly outperforming baselines like `Best Simple Relative EV` or `Random`.

#### 2. The "Game Theory Edge" (Maximizing Wins)
Among the heuristic strategies, **`Adaptive Simple EV`** is the clear winner. Despite having a *lower* average score (1635.6 pts) than the pure EV strategies, its **win rate is 12.98%**, making it the best heuristic for consistently beating the "lambda" baseline (8.33%). This confirms the "Game Theory" hypothesis:
* **Sacrificing EV for Position:** It wins *because* it does not purely maximize points.
* **Leading:** It plays defensively with "safe" blocking bets (`np.argmax(opp_repartition)`) to lock in the win.
* **Trailing:** It plays aggressively with high-risk bets (`np.argmax(match_gains)`) to create variance and catch up.
* This strategy correctly identifies that maximizing *points* is fundamentally different from maximizing *wins*.

#### 3. The Failure of "Relative EV"
Strategies based on "Relative EV" (`strat_best_simple_rel_ev`) performed poorly. The opponent model (`opp_repartition`) assumes heavy herding toward the favorite. Consequently, the `(1 - opp_repartition)` factor dominates the decision-making, turning the strategy into a "Contrarian" that systematically selects high-risk, low-EV outcomes just to be different, resulting in a poor average score.

#### 4. The Variance Paradox: Static vs. Dynamic Risk
`Adaptive Simple EV` exhibits one of the **highest tournament variances (Std Dev = 470.97)**. Interestingly, a `Highest Variance` strategy that *statically* picks the most volatile bet per match had a *lower* final variance (369.73).

**The Paradox:**
* A **static policy** (like `Highest Variance`) is subject to the Central Limit Theorem. Summing 51 independent high-variance outcomes still produces a normal, bell-shaped distribution where most final scores cluster near the mean.
* A **dynamic/adaptive policy** effectively *fights* the Central Limit Theorem. It creates a **"barbell" distribution**:
    * **If Ahead:** The agent plays safely, keeping its simulation in the **"high score" pole**.
    * **If Behind:** The agent takes "all-or-nothing" bets. Failure keeps it in the **"low score" pole**, while success rockets it toward the "high score" pole.

This dynamic switching manufactures a distribution with two extremes and a hollowed-out middle, increasing the probability of landing in the top percentile (winning). This principle is also exploited by the RL agents, with `RL V2 (P1 - PPO)` showing the highest variance in the tournament.

#### 5. RL Agents Outperform "Best EV"
As shown in the final score distribution graph, `Best Simple EV` clearly produces the most optimal Gaussian-shaped distribution. However, all RL agents manage to beat this baseline by cleverly sacrificing some EV to access the "tail" of the distribution (higher scores) when necessary. This adaptive behavior creates heavy-tailed distributions that are statistically more likely to win a tournament.

#### 6. The Legacy Dense Reward (`RL Legacy (V1 - 4M)`)
The "legacy" reward function encouraged the agent to balance frequent point-scoring (betting Favorite/EV) with winning. As a result, it achieves the highest average score among all RL agents and the lowest variance, making it a consistent but less "aggressive" winner.

#### 7. The Tanh Reward Shift
From V2 onwards, the reward function was changed to a **Tanh Potential** based on the score gap. This created a fundamental shift in mentality: the **only** way to get a high reward is to keep pace with or beat the leader. There is no reward for being "good on average," only for being "excellent relative to the leader."

Consequently, V2 and V3 agents score worse on average than the Legacy agent but achieve (mostly) higher win rates, proving they are better optimized for the specific goal of taking 1st place.

#### 8. Key Results for RL Agents
Despite the heavy stochasticity of the environment, clear trends emerge:
* **Variance as a Weapon:** To beat increasingly tough opponents, agents learn to convert average score into variance. `RL V2 (P3 - MoreRand)` is unique in maintaining a high win rate, high average score, and low variance simultaneously.
* **Progression:** As the training phases progress, the agents become more sophisticated, eventually surpassing the `Adaptive Simple EV` heuristic.
* **Generalization:** The shift to Power Law opponents in V3 did not degrade the performance of V2 agents (trained on different opponents). Their win rates remained stable, indicating that the strategies learned in V2 generalize well to different crowd behaviors.

#### 9. The Best Strategy
The top-performing strategy is **`RL V3 (P4 - DomainRnd)`**.
The introduction of Domain Randomization did not hurt its performance in standard tournaments. Instead, it produced a robust agent that plays safely but takes calculated risks when necessary, achieving a strong **13.49% win rate** against 11 lambda players—approximately **1.6x better than random chance**.

## 🔮 Future Work

* **Perfect Score Bonus:** Integrate the "Exact Score" mechanic into the environment. Simulating representative perfect score probabilities would add complexity but likely push the strategy slightly back towards `Best Simple EV` (since RL agents would still have an edge when they make the same bet as opponents).
* **2026 World Cup Format:** Update the `n_matches` and tournament structure to reflect the new expanded format.