# MonPetitProno (MPP) Strategy Simulator

This repository contains a Monte Carlo simulation engine built in Python to analyze and compare betting strategies for the popular "MonPetitProno" (MPP) football prediction game.

The primary goal is to use a quantitative approach to find strategies that maximize a player's probability of winning a league.

This project serves as a portfolio piece demonstrating skills in:
* Python (NumPy, Matplotlib)
* Simulation & Modeling (Monte Carlo)
* Quantitative Analysis & Game Theory
* Software Engineering (Modular Code, OOP principles)

## 🎯 The Core Hypothesis

The project is designed to test two main hypotheses:

1.  **The "Value" Edge:** MPP "gains" (points) are frozen weeks before a tournament. My simulation models this by first generating "initial" probabilities to set the gains, then generating "true" probabilities (the evolving bookmaker odds) for the match outcome. Can a player find an edge by identifying "value bets" where the simple Expected Value ($EV = P_{\text{true}} \cdot G_{\text{frozen}}$) is high?
2.  **The "Game Theory" Edge:** In a tournament with a fixed number of players, is maximizing simple EV the optimal strategy? Or does a *dynamic* strategy—which adapts to the player's current rank (e.g., "play safe" when leading, "play risky" when trailing)—yield a higher *win rate*?

## 📂 Repository Structure

The project is structured as a clean, modular Python application:

```

mppanalysis/
│
├── mpp\_project/          \# The core Python package
│   ├── core.py               \# Core math and utility functions
│   ├── match\_simulator.py    \# Generates realistic match data (probas, gains, etc.)
│   ├── simulation.py         \# The main tournament simulation engine
│   └── strategies.py         \# Defines all "agent" strategies to be tested
│
├── notebooks/              \# Jupyter notebooks for exploration and analysis
│   ├── 00\_...ipynb         \# Initial EDA on EV and probabilities
│   └── 04\_...ipynb         \# Analysis of final score distributions
│
├── .gitignore
├── run\_tournament\_simulation.py  \# MAIN SCRIPT: Runs the full simulation and prints results
└── README.md                     \# You are here

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
    *(You should create this file by running `pip freeze > requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the simulation:**
    ```bash
    python run_tournament_simulation.py
    ```
    This will run `N_SAMPLES` (e.g., 5000) full tournaments for each strategy, pitting one "challenger" at a time against a league of `N_PLAYERS - 1` "typical" opponents. It will then print the final results table and display a plot of the score distributions.

## 📊 Key Findings & Results

The simulation was run with `N_SAMPLES = 5000` tournaments for each strategy, each in a league of 12 players.

### Final Simulation Results (5000 tournaments per strategy)

| Strategy | Win Rate | Avg Points | Median | Std Dev |
| :--- | :---: | :---: | :---: | :---: |
| **Adaptive Simple EV** | **12.70%** | 1658.4pts | 1632.5pts | 468.34pts|
| **Adaptive Simple Rel EV** | **9.80%** | 1462.3pts | 1412.0pts | 434.99pts|
| Best Simple EV | 8.40% | 2002.8pts | 2007.0pts | 281.30pts|
| Safe Simple EV | 7.02% | 1984.1pts | 1981.0pts | 281.54pts|
| Safe Simple Rel EV | 6.28% | 1529.2pts | 1518.0pts | 361.31pts|
| Best Simple Relative EV | 6.24% | 1522.5pts | 1512.0pts | 358.10pts|
| Favorite | 6.16% | 1990.7pts | 1993.0pts | 280.85pts|
| Random | 5.72% | 1638.2pts | 1629.0pts | 337.40pts|
| Highest Variance | 4.58% | 1432.2pts | 1427.0pts | 372.03pts|

*(Note: The expected win rate for a "lambda" (typical) player is 1/12 = 8.33%)*

### Analysis of Key Insights

This table reveals several powerful (and sometimes paradoxical) insights into tournament strategy.

#### 1. The "EV Edge" (Maximizing Points)
The `Best Simple EV` strategy is a massive success. It achieves the highest **average score** (2002.8 pts) by a huge margin. This proves the "Value" hypothesis: by calculating $EV = P_{\text{true}} \cdot G_{\text{frozen}}$, this strategy successfully exploits the "inefficient market" to find value bets, performing far better than `Favorite` or `Random`.

#### 2. The "Game Theory Edge" (Maximizing Wins)
The **`Adaptive Simple EV`** strategy is the clear winner. Despite having a *worse* average score (1658.4 pts) than the "EV" strategies, its **win rate is 12.70%**, making it the *only* strategy to consistently beat the "lambda" baseline. This proves the "Game Theory" hypothesis:
* It wins *because* it sacrifices EV.
* When leading, it makes "safe" blocking bets (`np.argmax(opp_repartition)`) to secure the win.
* When trailing, it makes "Hail Mary" high-risk bets (`np.argmax(match_gains)`) to have a chance at catching up.
* It correctly identifies that maximizing *points* is not the same as maximizing *wins*.

#### 3. The Failure of "Relative EV"
The "Relative EV" strategies (`strat_best_simple_rel_ev`) performed very poorly. My model for opponent behavior (`opp_repartition`) is heavily skewed toward the favorite. The `(1 - opp_repartition)` factor becomes the dominant term, turning the strategy into an "unpopular outcome seeker" that often selects high-risk, low-EV bets, resulting in a poor average score.

#### 4. The Variance Paradox: Static vs. Dynamic Risk
This is the most critical insight. `Adaptive Simple EV` has the **highest tournament variance (Std Dev = 468.34)**. To test this, I created a `Highest Variance` strategy that *statically* picks the bet with the highest *per-match* variance ($G^2 \cdot P \cdot (1-P)$).

**The Paradox:** The `Highest Variance` strategy had a *lower* final variance (370.44) than the adaptive one. Why?

* A **static policy** (like `Highest Variance`), by the Central Limit Theorem, produces a *normal, bell-shaped* distribution of final scores. It just adds 51 high-variance outcomes, which still results in most scores clustering near the mean.
* A **dynamic/adaptive policy** *fights* the Central Limit Theorem. It is a *mixture model* that intentionally creates a **"barbell" distribution** of final scores:
    * **If Ahead:** The agent plays to "lock in" its score, keeping its simulation into the **"high score" pole** of the barbell.
    * **If Behind:** The agent takes an "all-or-nothing" bet. The *failure mode* (getting 0 points) keeps it in the **"low score" pole**, while the *success mode* (hitting the `argmax(G)` bet) rockets it toward the "high score" pole.

This dynamic switching *manufactures* a final distribution with two extremes and a hollowed-out middle, which is why it has a higher standard deviation and a higher chance of landing on the "high score" pole (i.e., winning the tournament).

## 🧠 Next Steps: The Reinforcement Learning Agent

The success of the heuristic-based `Adaptive` strategy proves that a state-dependent policy is optimal. The clear next step is to replace these hand-crafted "if-then" rules with a fully optimized agent.

**Part 2 of this project is to build an RL agent:**

* **Environment:** Wrap the `simulation.py` engine into a custom `gymnasium.Env`.
* **State:** The observation space will be the full state: `(match_probas, match_gains, opp_repartition, all_player_scores, my_idx, matches_remaining)`.
* **Agent:** Use `stable-baselines3` (PPO) to train a neural network to learn the optimal betting policy.

This agent will *learn* the optimal, non-linear function for *when* to switch between "safe" and "risky" behaviors, creating a far more robust and powerful strategy than any hard-coded heuristic.