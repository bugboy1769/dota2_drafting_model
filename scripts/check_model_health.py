import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(".")
from src.model import DraftModel

MODEL_PATH = "models/best_model.pt"

def check_model():
    print(f"Checking {MODEL_PATH}...")
    try:
        # Initialize with 150 heroes (as per recent fix)
        model = DraftModel(num_heroes=150, embedding_dim=128)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Check Embeddings
        weights = model.hero_embedding.weight.detach()
        print(f"Embedding Shape: {weights.shape}")
        print(f"Min: {weights.min()}, Max: {weights.max()}")
        
        if torch.isnan(weights).any():
            print("❌ CRITICAL: Embeddings contain NaNs!")
            return False
        
        if torch.isinf(weights).any():
            print("❌ CRITICAL: Embeddings contain Infs!")
            return False
            
        print("✅ Model seems healthy (No NaNs/Infs in embeddings).")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

if __name__ == "__main__":
    check_model()
