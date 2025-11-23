# Dota 2 Drafting Model (AlphaZero Style)

A deep learning system that learns the "Energy Landscape" of Dota 2 drafting. It moves beyond simple imitation learning by building a comprehensive World Model of the game's drafting phase, capable of understanding roles, lane dynamics, and win conditions.

## Data Source

The model is trained on high-level professional matches fetched from the **OpenDota API**.
*   **Input:** Captain's Mode draft sequences (Picks & Bans).
*   **Rich Features:**
    *   Hero IDs and Draft Order.
    *   Player Roles (1-5) derived from in-game lane presence and farm.
    *   Lane Outcomes (Gold Difference at 10 minutes).
    *   Match Winner.

## The World Model

The core of the system is a **Multi-Task Transformer** that maps the complex interactions of heroes into a high-dimensional vector space. It doesn't just memorize picks; it learns the underlying mechanics of the draft:

1.  **Policy Head (The Prior):** Predicts the most likely next move based on professional trends.
2.  **Value Head (The Evaluator):** Estimates the win probability of any given draft state.
3.  **Role Head (The Context):** Explicitly predicts which hero is playing which role (Carry, Mid, Offlane, Support), giving the model a structural understanding of the lineup.
4.  **Synergy Head (The Mechanics):** Predicts the **Gold Difference** at 10 minutes for each lane.
    *   *Architecture:* Fuses the Draft Representation with the Predicted Roles to calculate precise, role-aware lane matchups.

## MCTS: Charting the Probability Space (Under Construction)

While the World Model captures the static rules and correlations of the game, **Monte Carlo Tree Search (MCTS)** captures the dynamic consequences.

*   **Adversarial Search:** Instead of assuming a cooperative environment, MCTS simulates the opponent's best responses (Minimax).
*   **Probability Charting:** It explores the branching future of the draft, using the Policy Head to guide exploration and the Value Head to evaluate leaf nodes.
*   **Result:** It finds the "Path of Least Resistance" to victory, effectively solving for the Nash Equilibrium of the draft rather than just copying human biases.

## Performance

*   **Top-5 Accuracy:** ~40% (Predicting human pro picks).
*   **Top-1 Accuracy:** ~16%.
*   **Loss:** Converged to ~4.5 (Multi-task weighted loss).

*Note: Accuracy is limited by the "Aleatoric Uncertainty" of drafting—there are often many valid picks in any given situation.*

## Features

### Interactive Draft Assistant (`play.py`)
A CLI tool to draft against the AI or use it as a companion.
*   **Real-time Suggestions:** Top 5 recommended picks/bans.
*   **Win Rate Estimation:** Live update of your winning chances.
*   **Lane Forecasting:** Predicts specific gold advantages/disadvantages for Safe, Mid, and Off lanes based on the current lineup.

## 🛠️ Setup & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Collect Data:**
    ```bash
    python scripts/01_collect_data.py
    ```

3.  **Process Data:**
    ```bash
    python scripts/02_process_data.py
    ```

4.  **Train:**
    ```bash
    python scripts/03_train.py
    ```

5.  **Play/Test:**
    ```bash
    python play.py
    ```
