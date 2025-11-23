import torch
import pickle
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DraftModel
from src.utils import load_checkpoint
from tqdm import tqdm

def evaluate_model(model_path, data_path, device='cpu'):
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        val_examples = pickle.load(f)
    
    print(f"Loading model from {model_path}...")
    model = DraftModel(num_heroes=150)
    load_checkpoint(model, model_path)
    model.to(device)
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    print(f"Evaluating on {len(val_examples)} examples...")
    
    with torch.no_grad():
        for ex in tqdm(val_examples):
            # Prepare inputs
            hero_seq = torch.tensor([ex['hero_sequence']], dtype=torch.long).to(device)
            type_seq = torch.tensor([ex['type_sequence']], dtype=torch.long).to(device)
            team_seq = torch.tensor([ex['team_sequence']], dtype=torch.long).to(device)
            
            # Valid actions mask
            valid_actions = torch.tensor([ex['valid_actions']], dtype=torch.bool).to(device)
            
            # Skip empty sequences (all padding) to avoid Transformer crash
            if (hero_seq == 0).all():
                continue

            target = ex['target_actions'] # 0-149 index
            
            # Forward pass
            action_logits, _, _, _ = model(hero_seq, type_seq, team_seq, valid_actions)
            
            # Get predictions
            probs = torch.softmax(action_logits[0], dim=0)
            
            # Top 1
            top1_prob, top1_idx = torch.topk(probs, 1)
            if top1_idx.item() == target:
                correct_top1 += 1
                
            # Top 5
            top5_prob, top5_indices = torch.topk(probs, 5)
            if target in top5_indices.tolist():
                correct_top5 += 1
            
            total += 1
            
    acc_top1 = (correct_top1 / total) * 100
    acc_top5 = (correct_top5 / total) * 100
    
    print(f"\nResults:")
    print(f"Total Examples: {total}")
    print(f"Top-1 Accuracy: {acc_top1:.2f}%")
    print(f"Top-5 Accuracy: {acc_top5:.2f}%")
    
    return acc_top1, acc_top5

if __name__ == "__main__":
    MODEL_PATH = "models/best_model.pt"
    VAL_DATA_PATH = "data/processed/val.pkl"
    
    if torch.backends.mps.is_available():
        device = "cpu" # Force CPU due to MPS nested tensor issues
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    evaluate_model(MODEL_PATH, VAL_DATA_PATH, device)
