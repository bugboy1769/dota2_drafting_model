# Dota 2 Drafting Model (AlphaZero Style)

A deep learning model designed to draft Dota 2 lineups, predicting picks, bans, roles, and lane outcomes. It uses a Transformer-based architecture and is evolving towards an AlphaZero-style MCTS inference engine.

## 🧠 Model Architecture

The model is a **Multi-Task Transformer** that learns to understand the draft state and predict multiple objectives simultaneously:

1.  **Draft Representation:** A Transformer Encoder processes the sequence of picks and bans (with Positional, Team, and Type embeddings).
2.  **Policy Head:** Predicts the next hero to pick or ban (Imitation Learning).
3.  **Value Head:** Predicts the win probability of the current draft state.
4.  **Role Head:** Predicts the role (1-5) of each hero in the draft.
5.  **Synergy Head:** Predicts the **Gold Difference** at 10 minutes for Safe, Mid, and Off lanes.
    *   *Innovation:* This head receives both the draft context AND the predicted roles to calculate precise lane matchups.

## 📊 Performance

*   **Top-5 Accuracy:** ~40% (Predicting human pro picks).
*   **Top-1 Accuracy:** ~16%.
*   **Loss:** Converged to ~4.5 (Multi-task weighted loss).

*Note: Accuracy is limited by the "Aleatoric Uncertainty" of drafting—there are often many valid picks in any given situation.*

## 🚀 Features

### 1. Interactive Draft Assistant (`play.py`)
Draft against the AI or use it as a companion tool.
*   Suggests top 5 picks.
*   Predicts Win Probability.
*   **Predicts Lane Outcomes:** "Safe Lane: +200 Gold", "Mid Lane: -500 Gold".

### 2. Training Visualization
Generate training curves from your logs:
```bash
python scripts/plot_training.py
```

### 3. (Upcoming) MCTS Inference
We are implementing **Monte Carlo Tree Search (MCTS)** to move beyond simple imitation.
*   Instead of just predicting what a human *would* pick, MCTS simulates future counter-picks to find what you *should* pick.
*   It uses the **Value Head** to evaluate leaf nodes and the **Policy Head** to guide the search.

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
