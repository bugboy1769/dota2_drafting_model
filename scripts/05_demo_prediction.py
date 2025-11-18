import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.predict import DraftPredictor

def main():
    #Initialize predictor
    predictor=DraftPredictor(
        config_path='config.yaml',
        model_path='models/best_model.pt'
    )

    # Example draft history
    draft_history = [
        {'hero_id': 1, 'is_pick': False, 'team': 0},   # Radiant bans Anti-Mage
        {'hero_id': 5, 'is_pick': False, 'team': 1},   # Dire bans Crystal Maiden
        {'hero_id': 8, 'is_pick': True, 'team': 0},    # Radiant picks Juggernaut
        {'hero_id': 14, 'is_pick': True, 'team': 1},   # Dire picks Pudge
    ]

    #Get predictions
    result=predictor.predict(draft_history, top_k=5)

    print(f"\n=== Draft Assistant Demo ===")
    print(f"Win Probability: {result['win_probability']:.1%}\n")
    print("Top 5 Suggested Picks:")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"{i}. {suggestion['hero_name']} ({suggestion['confidence']:.1%})")
    
    print(f"\n Demo complete!")

if __name__=='__main__':
    main()