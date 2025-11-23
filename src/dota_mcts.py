from src.utils import prepare_model_input, get_draft_order
from src.mcts import MCTS
import torch

class DotaMCTS(MCTS):
    def __init__(self, model, device='cpu', c_puct=1.0, first_pick_team=0):
        super().__init__(model, c_puct)
        self.device=device
        self.model=model.to(device)
        self.first_pick_team = first_pick_team
        self.draft_order = get_draft_order(first_pick_team)
    
    def _evaluate_state(self, state):
        self.state=state

        #Prepare model input
        seq_tensor, type_tensor, team_tensor, valid_tensor=prepare_model_input(state, self.device, self.first_pick_team)
        
        #Run the model
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

    def _apply_action(self, state, action):
        return state + [action]
    
    def _is_terminal(self, state):
        return len(state)==24
    
    def _get_terminal_reward(self, state):
        _, value, _ = self._evaluate_state(state)
        return value
    
    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count +=1

            step=len(node.state)
            if step<len(self.draft_order):
                _, team=self.draft_order[step]
                #Team: 0=Radiant, 1=Dire
                if team==1:
                    node.value_sum+=(1.0-value)
                else:
                    node.value_sum+=value
            else: #Terminal State
                node.value_sum+=value
            
            node=node.parent