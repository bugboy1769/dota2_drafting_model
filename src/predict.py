import torch
import yaml
import json
from pathlib import Path
from typing import List, Dict

from .model import DraftModel
from .utils import load_checkpoint

class DraftPredictor:
    """Interface for draft predictions"""

    def __init__(self, config_path:str, model_path:str):
        #load config
        with open(config_path, 'r') as f:
            self.config=yaml.safe_load(f)
        
        #load hero info
        data_dir=Path(self.config['paths']['data_dir'])/"raw"
        with open(data_dir/"heroes.json", 'r') as f:
            heroes=json.load(f)
            self.hero_id_to_name={h['id']: h['localized_name'] for h in heroes}
        
        #load model
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=DraftModel(
            num_heroes=self.config['model']['num_heroes'],
            embedding_dim=self.config['model']['embedding_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
        ).to(self.device)

        load_checkpoint(self.model, model_path)
        self.model.eval()
    
    def predict(self, draft_history: List[Dict], top_k:int=5)->Dict:
        """Predict next best heroes
        Args:
            draft_history: List of {'hero_id':int, 'is_pick':bool, 'team':int}
            top_k: number of suggestions to return
            
        Returns:
            Dictionary with suggestions and win probability
        """
        #Prepare input
        hero_sequence=[0]*24
        for i, action in enumerate(draft_history):
            hero_sequence[i]=action['hero_id']
        
        picked_banned=set(action['hero_id'] for action in draft_history)
        valid_actions=[hero_id not in picked_banned for hero_id in range(1, 125)]

        #Convert to tensors
        hero_seq_tensor=torch.tensor([hero_sequence], dtype=torch.long.to(self.device))
        valid_actions_tensor=torch.tensor([valid_actions], dtype=torch.bool.to(self.device))

        #Predict
        with torch.no_grad():
            action_logits, win_prob=self.model(hero_seq_tensor, valid_actions_tensor)
        
        #Get top k predictions
        probabilities=torch.softmax(action_logits[0], dim=0)
        top_probs, top_indices=torch.topk(probabilities, top_k)

        #Format results
        suggestions=[]
        for prob, idx in zip(top_probs, top_indices):
            hero_id=idx.item()+1
            suggestions.append({
                'hero_id':hero_id,
                'hero_name':self.hero_id_to_name.get(hero_id, f"Hero {hero_id}"),
                'confidence':prob.item()
            })
        
        return {
            'suggestions':suggestions,
            'win_probability':win_prob.item()
        }