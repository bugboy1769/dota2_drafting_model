import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm

from src.model import DraftModel
from .dataset import DraftDataset
from .utils import load_checkpoint

def evaluate_model(config_path:str, model_path:str):
    """Evaluate trained model"""
    #Load config
    with open(config_path, 'r') as f:
        config=yaml.safe_load(f)
    
    device=config['training']['device']

    #Load model
    model=DraftModel(
        num_heroes=config['model']['num_heroes'],
        embedding_dim=config['model']['embedding_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    ).to(device)

    load_checkpoint(model, model_path)
    model.eval()

    #Load test data
    data_dir=Path(config['paths']['data_dir'])/"processed"
    test_dataset=DraftDataset(data_dir/"test.pkl")
    test_loader=DataLoader(test_dataset, batch_size=32, shuffle=True)

    #Evaluate
    correct=0
    total=0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            hero_seq=batch['hero_sequence'].to(device)
            valid_actions=batch['valid_actions'].to(device)
            target_action=batch['target_action'].to(device)

            action_logits, _=model(hero_seq, valid_actions)
            predictions=torch.argmax(action_logits, dim=1)

            correct+=(predictions==target_action).sum().item()
            total+=target_action.size(0)

    accuracy=correct/total
    print(f"\nTest Accuracy: {accuracy:.2%}")
    return accuracy
