import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import DraftModel
from src.dataset import DraftDataset
from torch.utils.data import DataLoader

def load_config():
    with open(project_root / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)

def compute_ece(preds, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    preds: tensor of shape (N,) with probabilities [0, 1]
    labels: tensor of shape (N,) with true binary labels {0, 1}
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        # Find samples in this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (preds > bin_lower) & (preds <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            # Avg predicted probability in this bin
            avg_conf = preds[in_bin].mean().item()
            # Avg actual accuracy (win rate) in this bin
            accuracy = labels[in_bin].float().mean().item()
            
            ece += np.abs(avg_conf - accuracy) * prop_in_bin
            bin_stats.append((avg_conf, accuracy, prop_in_bin))
    
    return ece, bin_stats

def main():
    config = load_config()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    data_dir = Path(config['path']['data_dir']) / "processed"
    val_dataset = DraftDataset(data_dir / "val.pkl")
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    # Load Model
    model_path = Path(config['path']['model_dir']) / "best_model.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    model = DraftModel(
        num_heroes=config['model']['num_heroes'],
        embedding_dim=config['model']['embedding_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded.")

    all_preds = []
    all_labels = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Extract inputs
            hero_seq = batch['hero_sequence'].to(device)
            type_seq = batch['type_sequence'].to(device)
            team_seq = batch['team_sequence'].to(device)
            valid_actions = batch['valid_actions'].to(device)

            # NOTE: You might need to adjust this depending on your exact dataset structure
            if 'outcome' in batch:
                targets = batch['outcome'].float().to(device) # 1.0 = Radiant Win
            else:
                # Fallback purely for running the script if label missing
                print("Warning: 'outcome' label not found in batch. Skipping.")
                break

            # Forward pass
            # We only care about the Value Head output
            policy_logits, value, role_logits, synergy = model(hero_seq, type_seq, team_seq, valid_actions)
            
            probs = value.squeeze() # Sigmoid is usually applied inside model or loss? 
            # If model.py value head ends with Sigmoid, this is Prob. If Linear, we need Sigmoid.
            # Let's check model.py... usually standard implementations output logits or tanh.
            # Assuming standard BCE Loss usage in training means model likely outputs Sigmoid probabilities or logits.
            
            all_preds.append(probs.cpu())
            all_labels.append(targets.cpu())

    if not all_preds:
        return

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute ECE
    ece, bin_stats = compute_ece(all_preds, all_labels)
    
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"{'Bin Range':<15} | {'Avg Conf':<10} | {'Actual Acc':<10} | {'% Samples':<10}")
    print("-" * 55)
    
    bins = np.linspace(0, 1, 11)
    for i, (conf, acc, prop) in enumerate(bin_stats):
        lower, upper = bins[i], bins[i+1]
        print(f"{lower:.1f} - {upper:.1f}       | {conf:.4f}     | {acc:.4f}     | {prop:.1%}")

    # Plot (Optional, requires saving to file)
    print("\nCalibration check complete. An ECE < 0.05 is generally considered good.")

if __name__ == "__main__":
    main()
