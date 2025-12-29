"""
Draft Assistant Backend
Manages the state of a live draft and interfaces with the model for predictions.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.dota_mcts import DotaMCTS

class DraftAssistant:
    def __init__(self, model, hero_map: Dict[int, dict], device='cpu'):
        self.model = model
        self.hero_map = hero_map
        self.device = device
        self.model.eval()
        
        # Invert map for name -> id lookups
        # hero_map structure: {id: {'name': 'Anti-Mage', ...}}
        # Ensure ID is int, as JSON keys might be strings
        self.name_to_id = {v['name']: int(k) for k, v in hero_map.items()}

        # Initialize MCTS
        # Assuming first_pick_team is 0 (Radiant) or handled by MCTS context if needed later
        self.mcts = DotaMCTS(model, device=device, c_puct=1.0, first_pick_team=0)
        
    def get_turn_info(self, step: int) -> Tuple[str, str]:
        """
        Returns (Team, Type) for a given step (0-indexed).
        Based on standard Dota 2 Captain's Mode (approximate for this model).
        
        Model assumes:
        - Steps 0-7: Bans (Alternating R, D, R, D...)
        - Steps 8-23: Picks (Alternating R, D, R, D...)
        """
        # Team: Even = Radiant, Odd = Dire
        team = "Radiant" if step % 2 == 0 else "Dire"
        
        # Type: First 8 steps are Bans, rest are Picks
        # Note: This must match how you trained the model!
        # In draft_analyzer.py you used: type=1 if i<8 else 0
        action_type = "Ban" if step < 8 else "Pick"
        
        return team, action_type

    def predict_next_step(self, draft_sequence: List[int], use_mcts: bool = False) -> Dict:
        """
        Get model suggestions for the NEXT step.
        """
        step = len(draft_sequence)
        if step >= 24:
            return {"error": "Draft Complete"}
            
        # 1. Build Tensor Inputs
        # We reuse the logic from DraftAnalyzer but optimized for single step
        seq_len = 24
        hero_ids = torch.zeros(1, seq_len, dtype=torch.long)
        type_seq = torch.zeros(1, seq_len, dtype=torch.long)
        team_seq = torch.zeros(1, seq_len, dtype=torch.long)
        valid_actions = torch.ones(1, 150, dtype=torch.bool) # 150 heroes
        
        for i, h_id in enumerate(draft_sequence):
            if i >= seq_len: break # Safety break
            h_id = int(h_id) # Safety cast
            hero_ids[0, i] = h_id
            type_seq[0, i] = 1 if i < 8 else 0
            team_seq[0, i] = i % 2
            if h_id < 150:
                valid_actions[0, h_id] = 0 # Mark taken
        
        # Move to device
        hero_ids = hero_ids.to(self.device)
        type_seq = type_seq.to(self.device)
        team_seq = team_seq.to(self.device)
        valid_actions = valid_actions.to(self.device)
        
        # 2. Forward Pass (Standard)
        with torch.no_grad():
            policy, value, roles, synergy = self.model(
                hero_ids, type_seq, team_seq, valid_actions
            )
            
        # 3. Process Outputs
        
        # A. Win Probability (for Radiant)
        win_prob = value.item()
        
        # B. Suggestions
        suggestions = []
        
        if use_mcts:
            # --- MCTS LOGIC ---
            # Run simulation
            root = self.mcts.search(draft_sequence, num_simulations=50)
            mcts_probs = self.mcts.get_action_probs(root)
            
            # Sort by visit count (prob)
            # mcts_probs is list of (action, prob)
            mcts_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 5
            for action, prob in mcts_probs[:5]:
                h_name = self.hero_map[action]['name'] if action in self.hero_map else "Unknown"
                suggestions.append({
                    "hero_id": int(action),
                    "name": h_name,
                    "logit": float(prob) # It's a prob, not logit, but we use same key for UI
                })
                
        else:
            # --- STANDARD LOGIC ---
            # Apply mask manually to logits to ensure we don't suggest taken heroes
            policy_logits = policy.squeeze().cpu().numpy()
            # Mask taken heroes (-inf)
            for h_id in draft_sequence:
                if h_id < len(policy_logits):
                    policy_logits[h_id] = -float('inf')
                    
            # Get Top 5
            top_indices = np.argsort(policy_logits)[-5:][::-1]
            suggestions = []
            for idx in top_indices:
                if idx in self.hero_map:
                    # Softmax approx for UI "confidence"
                    prob = np.exp(policy_logits[idx]) / np.sum(np.exp(policy_logits)) 
                    suggestions.append({
                        "hero_id": int(idx),
                        "name": self.hero_map[idx]['name'],
                        "logit": float(policy_logits[idx])
                    })
                
        # C. Lane Synergy (if available)
        # synergy output is [Safe, Mid, Off] win probabilities (0-1)
        lane_probs = torch.sigmoid(synergy).squeeze().cpu().numpy()
        
        return {
            "win_prob": win_prob,
            "suggestions": suggestions,
            "lanes": {
                "Safe": float(lane_probs[0]),
                "Mid": float(lane_probs[1]),
                "Off": float(lane_probs[2])
            }
        }
