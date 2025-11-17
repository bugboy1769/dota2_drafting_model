import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path

class DraftDataset(Dataset):
    """Pytorch Dataset for draft examples"""

    def __init__(self, examples_path:Path):
        with open(examples_path, 'rb') as f:
            self.examples=pickle.load(f)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example=self.examples[index]
        return {
            'hero_sequence': torch.tensor(example['hero_sequence'], dtype=torch.long),
            'valid_actions': torch.tensor(example['valid_actions'], dtype=torch.bool),
            'target_action': torch.tensor(example['target_action'], dtype=torch.long),
            'outcome': torch.tensor(example['target_action'], dtype=torch.float32),
            'team': torch.tensor(example['team'], dtype=torch.long)
        }