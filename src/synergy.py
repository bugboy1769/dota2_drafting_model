import json
import os
import numpy as np
from collections import defaultdict

class SynergyMatrix:
    def __init__(self, matches_dir="data/raw/matches", heroes_file="data/raw/heroes.json"):
        self.matches_dir = matches_dir
        self.heroes_file = heroes_file
        self.matrix = None # 150x150 matrix
        self.hero_ids = [] # Mapping index -> hero_id
        self.id_to_idx = {} # Mapping hero_id -> index
        
    def build(self):
        """
        Constructs the synergy/counter matrix.
        Entry [i, j] = Score of Hero i vs Hero j.
        Positive = Hero i is good against Hero j.
        """
        print("Building Synergy Matrix...")
        
        # 1. Load Heroes to establish dimensions
        with open(self.heroes_file, 'r') as f:
            heroes_data = json.load(f)
        
        if isinstance(heroes_data, list):
            heroes = {str(h['id']): h for h in heroes_data}
        else:
            heroes = heroes_data
            
        # Create dense mapping (0 to N-1)
        self.hero_ids = sorted([int(k) for k in heroes.keys()])
        self.id_to_idx = {hid: i for i, hid in enumerate(self.hero_ids)}
        n_heroes = len(self.hero_ids)
        
        # Matrix: [Hero A, Hero B] -> Win Rate Delta
        # We use a raw count matrix first
        # wins[i, j] = times i beat j
        # games[i, j] = times i played against j
        wins = np.zeros((n_heroes, n_heroes))
        games = np.zeros((n_heroes, n_heroes))
        
        # 2. Scan Matches
        files = [f for f in os.listdir(self.matches_dir) if f.endswith('.json')]
        
        for fname in files:
            try:
                with open(os.path.join(self.matches_dir, fname), 'r') as f:
                    data = json.load(f)
                
                if 'radiant_win' not in data: continue
                
                radiant_win = data['radiant_win']
                r_team = []
                d_team = []
                
                for pick in data['draft_timings']:
                    if pick['pick']:
                        hid = pick['hero_id']
                        if hid not in self.id_to_idx: continue # Skip unknown heroes
                        idx = self.id_to_idx[hid]
                        
                        if pick['active_team'] == 2: r_team.append(idx)
                        else: d_team.append(idx)
                
                # Update Counters (Opponents)
                for r_idx in r_team:
                    for d_idx in d_team:
                        games[r_idx, d_idx] += 1
                        games[d_idx, r_idx] += 1
                        
                        if radiant_win:
                            wins[r_idx, d_idx] += 1 # R beat D
                        else:
                            wins[d_idx, r_idx] += 1 # D beat R
                            
            except: continue
            
        # 3. Compute Scores (Win Rate - 0.5)
        # Avoid division by zero and filter low-sample matchups
        MIN_GAMES = 20  # Require at least 20 games for reliable stats
        
        with np.errstate(divide='ignore', invalid='ignore'):
            win_rates = wins / games
            self.matrix = win_rates - 0.5
            # Zero out entries with insufficient data or NaN
            self.matrix[games < MIN_GAMES] = 0.0
            self.matrix[np.isnan(self.matrix)] = 0.0
            
        print("Matrix built successfully.")
        return self.matrix

    def get_advantage_vector(self, current_draft):
        """
        Returns a vector of size (150,) representing the advantage of picking
        each hero against the current enemy team.
        """
        if self.matrix is None:
            self.build()
            
        # Identify enemy heroes in the current draft
        # This requires knowing which team we are drafting for
        # For now, let's assume we pass in the *enemy* hero IDs directly
        pass 
        # (This logic will move to MCTS, this class just holds the data)

    def save(self, path="data/processed/synergy_matrix.npy"):
        np.save(path, self.matrix)
        print(f"Saved matrix to {path}")

if __name__ == "__main__":
    sm = SynergyMatrix()
    sm.build()
    sm.save()
