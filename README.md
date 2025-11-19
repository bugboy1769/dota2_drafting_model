# Dota 2 Draft Model

ML model that predicts optimal hero picks/bans in Dota 2 drafts using Transformer architecture trained on professional matches.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Collect data (takes 4-6 hours)
python scripts/01_collect_data.py --num-matches 5000

# Process data
python scripts/02_process_data.py

# Train model
python scripts/03_train.py

# Evaluate
python scripts/04_evaluate.py

# Demo predictions
python scripts/05_demo_prediction.py
```

## Project Structure

```
dota2-draft-model/
├── src/                    # Source code
│   ├── data.py            # Data collection from OpenDota API
│   ├── dataset.py         # PyTorch Dataset
│   ├── model.py           # Transformer model
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation
│   ├── predict.py         # Inference
│   └── utils.py           # Helpers
├── scripts/               # Executable scripts
├── data/                  # Data (gitignored)
├── models/                # Saved models (gitignored)
└── config.yaml           # Configuration
```

## Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
model:
  embedding_dim: 128        # Hero embedding size
  num_layers: 4             # Transformer layers
  num_heads: 8              # Attention heads

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 50
```

## Usage

### CLI Prediction

```python
from src.predict import DraftPredictor

predictor = DraftPredictor('config.yaml', 'models/best_model.pt')

draft = [
    {'hero_id': 1, 'is_pick': False, 'team': 0},  # Ban
    {'hero_id': 8, 'is_pick': True, 'team': 0},   # Pick
]

result = predictor.predict(draft, top_k=5)
print(f"Win Prob: {result['win_probability']:.1%}")
for s in result['suggestions']:
    print(f"  {s['hero_name']}: {s['confidence']:.1%}")
```

## Expected Performance

- **Training Time**: 2-4 hours (GPU) / 8-12 hours (CPU)
- **Accuracy**: 25-35% (5,000 matches), 35-45% (10,000+ matches)
- **Baseline**: 0.8% (random guess among 124 heroes)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM
- GPU recommended (optional)

## Model Architecture

```
Input → Hero Embeddings → Positional Encoding → 
Transformer (4 layers, 8 heads) → 
  ├─ Policy Head → Hero prediction
  └─ Value Head → Win probability
```

Total parameters: ~900K

## Troubleshooting

**CUDA out of memory**: Reduce `batch_size` to 16 or 8

**Training too slow**: Use Google Colab or reduce model size:
```yaml
model:
  embedding_dim: 64
  num_layers: 2
```

**Hero ID out of bounds**: Delete `data/processed/*.pkl` and re-run `02_process_data.py`

## License

MIT
