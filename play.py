import yaml
import torch
import json
import numpy as np
from pathlib import Path
from src.model import DraftModel
from src.utils import load_checkpoint, prepare_model_input
from src.dota_mcts import DotaMCTS

class DraftSession:
    def __init__(self, config_path, model_path):
        #Load config
        with open(config_path, 'r') as f:
            self.config=yaml.safe_load(f)
        
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Draft Assitant Initializing on {self.device}")

        #Load Hero Data and translate for model
        data_dir=Path(self.config['path']['data_dir'])/"raw"
        with open(data_dir/"heroes.json", 'r') as f:
            heroes=json.load(f)
        
        #Create lookup tables
        self.id_to_name={h['id']: h['localized_name'] for h in heroes}
        #Normalise names
        self.name_to_id={
            h['localized_name'].lower().replace(" ", ""): h['id'] for h in heroes
        }

        #Initialise the model
        self.model=DraftModel(
            num_heroes=self.config['model']['num_heroes'],
            embedding_dim=self.config['model']['embedding_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads']
        ).to(self.device)

        #Load trained weights
        load_checkpoint(self.model, model_path)
        self.model.eval()
        
        #Initialize MCTS
        self.mcts = DotaMCTS(self.model, device=self.device, c_puct=1.0)

        #Initialize the state
        self.history=[]
        self.picked_set=set()
    
    def update_draft(self, hero_name):
        """Update state with human readable name"""
        clean_name=hero_name.lower().replace(" ", "")

        if clean_name not in self.name_to_id:
            print(f"Error: Could not find hero '{hero_name}'")
            return False
        
        hero_id=self.name_to_id[clean_name]

        if hero_id in self.picked_set:
            print(f"Error: {hero_name} is already in the draft!")
            return False
        
        #Update internal state
        self.history.append(hero_id)
        self.picked_set.add(hero_id)
        return True

    def get_suggestion(self):
        """Get best next move"""
        #Stop condition
        if len(self.history)>=26: # Fixed to 22 (length of DRAFT_ORDER)
            return None, 0.0, [0,0,0]
        
        #A. Prepare Input using Helper
        seq_tensor, type_tensor, team_tensor, valid_tensor = prepare_model_input(self.history, self.device)

        #C. Forward Pass
        #C. Forward Pass (for Synergy & Initial Win Rate)
        with torch.no_grad():
            #action_logits -> raw scores for next pick
            #win_prob -> value heads prediction for winning with predicted state
            action_logits, win_prob, role_logits, synergy_preds = self.model(seq_tensor, type_tensor, team_tensor, valid_tensor)
        
        #D. Run MCTS for Better Suggestions
        print("Thinking (MCTS)...")
        root = self.mcts.search(self.history, num_simulations=50) # 50 sims for speed
        mcts_probs = self.mcts.get_action_probs(root)
        
        # Extract top 5 from MCTS
        suggestions = []
        for action, prob in mcts_probs[:5]:
            name = self.id_to_name.get(action, "Unknown")
            suggestions.append((name, prob))
        
        # If MCTS fails (e.g. terminal state), fallback to raw model
        if not suggestions:
             probs = torch.softmax(action_logits[0], dim=0)
             top_probs, top_indices = torch.topk(probs, 5)
             for p, idx in zip(top_probs, top_indices):
                h_id = idx.item() + 1
                name = self.id_to_name.get(h_id, "Unknown")
                suggestions.append((name, p.item()))

        return suggestions, win_prob.item(), synergy_preds[0].tolist()
    
#INTERACTIVE LOOP

def main():
    CONFIG_PATH="config.yaml"
    MODEL_PATH="models/best_model.pt"

    session=DraftSession(CONFIG_PATH, MODEL_PATH)

    print("\n=== DOTA 2 DRAFT ASSISTANT ===")
    print("Type a hero name to add them to the draft.")
    print("Type 'suggest' to get AI recommendations.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input=input(f"\nDraft ({len(session.history)}/24) > ").strip()

        if user_input.lower()=='quit':
            break
        elif user_input.lower()=='suggest':
            suggestions, win_rate, synergy=session.get_suggestion()
            print(f"\nEstimated Win Probability: {win_rate:.1%}")
            print(f"Predicted Lane Advantage (Radiant): Safe={synergy[0]:+.2f}, Mid={synergy[1]:+.2f}, Off={synergy[2]:+.2f}")
            print("Recommended Picks:")
            for name, conf in suggestions:
                print(f"    - {name}: {conf:.1%}")
        else:
            #Assume its a hero pick
            success=session.update_draft(user_input)
            if success:
                print(f"Added -> {session.id_to_name[session.history[-1]]}")
                #Auto suggest after pick
                if len(session.history) < 22:
                    suggestions, win_rate, synergy=session.get_suggestion()
                    print(f"    (Win Probability: {win_rate:.1%})")

if __name__=='__main__':
    main()

