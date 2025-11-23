from src.mcts import MCTS
import torch
from src.utils import prepare_model_input

class DotaMCTS(MCTS):
    def __init__(self, model, c_puct=1.0):
        super().__init__(model, c_puct)
        self.device=device
        self.model=model.to(device)
        self.c_puct=c_puct
    
    def _evaluate_state(self, state):
        self.state=state
        seq_tensor, type_tensor, team_tensor, valid_tensor=prepare_model_input(state, self.device)

        with torch.no_grad():
            action_logits, win_prob, _, _ = self.model(seq_tensor, type_tensor, team_tensor, valid_tensor)

        #Process model output
        valid_moves=[]
        valid_mask=valid_tensor[0].cpu().numpy()
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                valid_moves.append(i+1)
        
        #Softmax for policy
        probs=torch.softmax(action_logits[0], dim=0).cpu().numpy()

        #Create policy dictionary
        policy={a: probs[a-1] for a in valid_moves}

        value=win_prob.item()

        return policy, value, valid_moves