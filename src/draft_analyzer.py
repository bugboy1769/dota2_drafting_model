"""
Draft Probability Space Analysis.
Our win probability landscape evolves as we move through the hero embedding space.
We will attempt to visualize this.
Probability is funny because state space evolution does not make any sense,
and that is the only thing a human might do in the face of absurdity.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict

class DraftAnalyzer:

    def __init__(self, model, device='cpu'):
        self.model=model
        self.device=device
        self.model.eval()
    
    def _build_input_tensors(self, partial_draft: List[int]) -> Tuple:
        """Basic draft state tensor builder from hero_id, exactly like model internals"""
        seq_len=24
        hero_ids=torch.zeros(1, seq_len, dtype=torch.long)
        type_seq=torch.zeros(1, seq_len, dtype=torch.long)
        team_seq=torch.zeros(1, seq_len, dtype=torch.long)

        # Valid actions mask: 1 for available, 0 for taken
        # Assuming 150 heroes (adjust if needed based on model config)
        num_heroes = 150 
        valid_actions = torch.ones(1, num_heroes, dtype=torch.bool)

        for i, hero_id in enumerate(partial_draft):
            hero_ids[0, i]=hero_id
            type_seq[0, i]=1 if i<8 else 0 #First 8 are bans Is this not too simple?
            team_seq[0, i]=i%2 #Alternates

            # Mark this hero as invalid for future picks
            if hero_id < num_heroes:
                valid_actions[0, hero_id] = 0

        return hero_ids.to(self.device), type_seq.to(self.device), team_seq.to(self.device), valid_actions.to(self.device)
    
    def _compute_entropy(self, policy_logits: torch.Tensor) -> float:
        """Shannon Entropy: H = -sum(p * log(p))"""
        probs=torch.softmax(policy_logits, dim=-1).squeeze()
        probs=probs + 1e-10 #Avoid log(0)
        entropy=-torch.sum(probs * torch.log(probs)).item()
        return entropy
    
    def analyze_draft_sequence(self, draft_sequence: List[int])->Dict:
        """Returns a Dict of 'win probs', 'entropies', 'turns'"""

        win_probs=[]
        entropies=[]

        with torch.no_grad():
            for turn in range(1, len(draft_sequence)+1):
                partial_draft=draft_sequence[:turn]
                hero_ids, type_seq, team_seq, valid_actions=self._build_input_tensors(partial_draft)

                #Forward pass
                policy, value, roles, synergy=self.model(hero_ids, type_seq, team_seq, valid_actions)

                win_probs.append(value.item())
                entropies.append(self._compute_entropy(policy))
        
        return {
            'win_probs':np.array(win_probs),
            'entropies':np.array(entropies),
            'turns': np.arange(1, len(draft_sequence)+1)
        }
    
    def find_turning_points(self, win_probs:np.ndarray, entropies:np.ndarray, turns:np.ndarray)->Dict:
        """Identify key moments shaping the probability landscape
            All of this stems from our win_prob module, would be nice to make it nicer.
        """

        #Calculate delta, prepend 0 for shape consistency
        deltas=np.diff(win_probs, prepend=win_probs[0])

        #find indices
        drop_idx=np.argmin(deltas)
        spike_idx=np.argmax(deltas)
        forced_idx=np.argmin(entropies)

        return {
            'biggest_drop': {
                'turn': int(turns[drop_idx]),
                'delta': float(deltas[drop_idx]),
                'win_prob_after': float(win_probs[drop_idx]),
                'description': f"Win% dropeed by {abs(deltas[drop_idx]):.1%}"
            },
            'biggest_spike': {
                'turn': int(turns[spike_idx]),
                'delta': float(deltas[spike_idx]),
                'win_prob_after': float(win_probs[spike_idx]),
                'description': f"Win% spiked by {abs(deltas[spike_idx]):.1%}"
            },
            'most_forced': {
                'turn': int(turns[forced_idx]),
                'delta': float(deltas[forced_idx]),
                'win_prob_after': float(win_probs[forced_idx]),
                'description': f"Lowest Entropy {entropies[forced_idx]:.2f}"
            }
        }

def load_sample_draft() -> List[int]:
    """Returns a sample 18-hero draft sequence for testing."""
    # A classic TI-style draft
    return [
        # Bans (4 each)
        1, 2, 5, 10, 20, 25, 30, 35,
        # Picks (5 each)
        8, 12, 15, 18, 22, 28, 33, 40, 45, 50
    ]