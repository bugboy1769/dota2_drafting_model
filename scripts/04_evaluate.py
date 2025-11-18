import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluate import evaluate_model

def main():
    accuracy=evaluate_model(
        config_path='config.yaml',
        model_path='models/best_model.pt'
    )

    print(f"\nEvaluation complete!")
    print(f"Test Accuracy: {accuracy:.2%}")

if __name__=='__main__':
    main()