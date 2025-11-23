import math
import torch
import numpy as np

class Node:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {} # {action: Node}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior # P(s, a) from the neural net
        
    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, model, c_puct=1.0):
        self.model = model # The Neural Network (The "Consultant")
        self.c_puct = c_puct # Exploration constant (The "Curiosity")

    def search(self, root_state, num_simulations=100):
        # 1. Create Root
        root = Node(root_state, prior=1.0)
        
        # 2. Expand Root (Get initial predictions)
        self._expand(root)
        
        # 3. Run Simulations
        for _ in range(num_simulations):
            node = root
            
            # A. SELECT: Traverse down until we hit a leaf
            while node.is_expanded():
                node = self._select_child(node)
            
            # B. EXPAND & EVALUATE: If not terminal, ask the model
            value = 0.0
            if not self._is_terminal(node.state):
                value = self._expand(node)
            else:
                value = self._get_terminal_reward(node.state)
            
            # C. BACKPROPAGATE: Update stats up the tree
            self._backpropagate(node, value)
            
        # 4. Return best action (usually most visited)
        # 4. Return the root node so the caller can extract stats
        return root

    def get_action_probs(self, root, temperature=1.0):
        """
        Get the probability distribution over actions based on visit counts.
        Returns: [(action, probability), ...] sorted by probability.
        """
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        
        if not visit_counts:
            return []
            
        # Apply temperature
        # If temp -> 0, argmax. If temp -> inf, uniform.
        # For now, simple normalization (temp=1)
        total_visits = sum(vc for _, vc in visit_counts)
        
        probs = []
        for action, count in visit_counts:
            p = count / total_visits
            probs.append((action, p))
            
        # Sort by probability descending
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs

    def _select_child(self, node):
        """Select the child with the highest UCB score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            # UCB Formula: Q + U
            # Q = Exploitation (Average Value)
            # U = Exploration (Prior * sqrt(ParentVisits) / (1 + ChildVisits))
            
            u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.value + u
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_child

    def _expand(self, node):
        """
        Ask the Neural Net for Policy (P) and Value (V).
        Create children for all valid actions.
        Return V.
        """
        # This part is specific to your specific problem/model
        # In a generic implementation, you'd call self.game.evaluate(node.state)
        
        # Mocking the model call for demonstration:
        # policy_logits, value = self.model(node.state)
        # valid_actions = self.game.get_valid_actions(node.state)
        
        # For now, let's assume we have a helper to get these
        policy_probs, value, valid_moves = self._evaluate_state(node.state)
        
        for action in valid_moves:
            if action not in node.children:
                node.children[action] = Node(
                    state=self._apply_action(node.state, action),
                    parent=node,
                    prior=policy_probs[action]
                )
        
        return value

    def _backpropagate(self, node, value):
        """Update the node and its ancestors."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            
            # Important: In 2-player games (Zero-Sum), the value flips!
            # If it's good for me (Radiant), it's bad for parent (Dire).
            # value = -value 
            # (We will handle this in the specific implementation)
            
            node = node.parent

    def _get_best_action(self, root):
        """Select the action with the most visits (Robustness)."""
        best_count = -1
        best_action = -1
        
        for action, child in root.children.items():
            if child.visit_count > best_count:
                best_count = child.visit_count
                best_action = action
                
        return best_action

    # --- Abstract Methods (You implement these for Dota) ---
    def _evaluate_state(self, state):
        raise NotImplementedError
    
    def _apply_action(self, state, action):
        raise NotImplementedError

    def _is_terminal(self, state):
        raise NotImplementedError

    def _get_terminal_reward(self, state):
        raise NotImplementedError
