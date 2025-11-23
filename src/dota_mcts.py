from src.mcts import MCTS
import torch

class DotaMCTS(MCTS):
    def __init__(self, model, c_puct=1.0):
        super().__init__(model, c_puct)
        self.device=device
        self.model=model.to(device)
        self.c_puct=c_puct
    
    def _evaluate_state(self, state):
        
        seq_len=24

        #A. Hero Sequence (Pad with 0)
        hero_seq=[0]*seq_len
        for i, h_id in enumerate(state):
            hero_seq[i]=h_id
        
        #B. Team and Type Sequence (From DRAFT_ORDER)
        


        self.state=state
        seq_tensor=torch.tensor([state], dtype=torch.long).to(self.device)

        with torch.no_grad():
            action_logits, win_prob, _, _ = self.model(self.state)


