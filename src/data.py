import requests
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

class OpenDotaAPI:
    BASE_URL = "https://api.opendota.com/api"

    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay=rate_limit_delay
        self.logger=logging.getLogger(__name__)

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params)
                if response.status_code == 429: # Rate limit hit
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Rate limit hit. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                return response
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Connection error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
        return None

    def get_pro_matches(self, less_than_match_id: Optional[int]=None)->List[Dict]:
        url = f"{self.BASE_URL}/proMatches"
        params={'less_than_match_id':less_than_match_id} if less_than_match_id else None
        response = self._make_request(url, params=params)
        if response and response.status_code == 200:
            return response.json()
        return []
    
    def get_match(self, match_id: int)->Dict:
        url=f"{self.BASE_URL}/matches/{match_id}"
        response = self._make_request(url)
        if response and response.status_code==200:
            return response.json()
        return None

    def get_heroes(self)->Dict:
        url=f"{self.BASE_URL}/heroes"
        response = self._make_request(url)
        if response and response.status_code == 200:
            return response.json()
        return []

class DataCollector:
    
    def __init__(self, data_dir: Path):
        self.data_dir=Path(data_dir)
        self.raw_dir=self.data_dir/"raw"
        self.processed_dir=self.data_dir/"processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.api=OpenDotaAPI()
        self.logger=logging.getLogger(__name__)
    
    def collect_matches(self, num_matches: int =1000):
        self.logger.info(f"Collecting {num_matches} matches...")

        #Fetch hero data
        heroes=self.api.get_heroes()
        with open(self.raw_dir/"heroes.json", 'w') as f:
            json.dump(heroes, f)
        #Collect matches
        matches_dir=self.raw_dir/"matches"
        matches_dir.mkdir(exist_ok=True)

        collected=0
        less_than_id=None

        with tqdm(total=num_matches, desc="Collecting matches") as pbar:
            while collected<num_matches:
                #Get batch of match IDs
                pro_matches=self.api.get_pro_matches(less_than_match_id=less_than_id)

                if not pro_matches:
                    break
                for match_info in pro_matches:
                    if collected>=num_matches:
                        break
                    match_id=match_info['match_id']
                    match_file=matches_dir/f"{match_id}.json"

                    if match_file.exists():
                        continue
                    #Fetch detailed match data
                    match_data=self.api.get_match(match_id)

                    if match_data and match_data.get('game_mode')==2:
                        with open(match_file, 'w') as f:
                            json.dump(match_data, f)
                        collected+=1
                        pbar.update()
                less_than_id=pro_matches[-1]['match_id']
        
        self.logger.info(f"Collected {collected} matches")
    
    def process_matches(self, val_split:float=0.15, test_split: float=0.15):
        """Process raw matches into training examples"""
        self.logger.info("Processing matches into training examples ...")
        matches_dir=self.raw_dir/"matches"
        match_files=list(matches_dir.glob("*.json"))

        all_examples=[]

        for match_file in tqdm(match_files, desc="Processing matches"):
            with open(match_file, 'r') as f:
                match_data=json.load(f)
            examples=self._create_examples_from_match(match_data)
            all_examples.extend(examples)
        
        #Shuffle and split
        import random
        random.shuffle(all_examples)

        n_test=int(len(all_examples)*test_split)
        n_val=int(len(all_examples)*val_split)

        test_examples=all_examples[:n_test]
        val_examples=all_examples[n_test:n_test+n_val]
        train_examples=all_examples[n_test+n_val:]

        #Save splits
        with open(self.processed_dir/"train.pkl", "wb") as f:
            pickle.dump(train_examples, f)
        with open(self.processed_dir/"val.pkl", "wb") as f:
            pickle.dump(val_examples, f)
        with open(self.processed_dir/"test.pkl", "wb") as f:
            pickle.dump(test_examples, f)
        self.logger.info(f"Saved {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test examples")

    def _create_examples_from_match(self, match_data: Dict) -> List[Dict]:
        """Create training examples from a single match."""
        picks_bans = match_data.get('picks_bans', [])
        if len(picks_bans) < 20:
            return []
        
        # First pass: collect all hero IDs and filter valid ones
        valid_hero_ids = set()
        for action in picks_bans:
            hero_id = action['hero_id']
            if 1 <= hero_id <= 150:  # Only accept valid heroes
                valid_hero_ids.add(hero_id)
        
        # If too few valid heroes, skip this match
        if len([a for a in picks_bans if 1 <= a['hero_id'] <= 150]) < 20:
            return []
        
        radiant_win = match_data.get('radiant_win', False)
        examples = []
        
        # Create hero_id -> role mapping
        # Roles: 0:Unknown, 1:Pos1, 2:Pos2, 3:Pos3, 4:Pos4, 5:Pos5
        hero_role_map = {}
        hero_gold_map={}

        players = match_data.get('players', [])
        for p in players:
            h_id = p.get('hero_id')
            lane = p.get('lane') # 1:Safe, 2:Mid, 3:Off
            role = p.get('lane_role') # 1:Core, 2:Supp, 3:Roam

            #Get gold at 10 min
            gold_t=p.get('gold_t')
            gold_10=0
            if gold_t and len(gold_t)>10:
                gold_10=gold_t[10]
            else:
                #Fallback: Estimate based on GPM
                gold_10=p.get('gold_per_min', 300)
            
            if h_id:
                hero_gold_map[h_id]=gold_10
     
            # Heuristic mapping to standard 1-5 positions
            pos = 0
            if lane == 2: # Mid
                pos = 2
            elif lane == 1: # Safe
                if role == 1: pos = 1 # Safe Core
                else: pos = 5 # Safe Supp
            elif lane == 3: # Off
                if role == 1: pos = 3 # Off Core
                else: pos = 4 # Off Supp
            
            if h_id:
                hero_role_map[h_id] = pos
        
        #Calculate lane outcomes (radiant perspective)
        rad_pos={}
        dire_pos={}

        #Fill the pos maps based on picks
        for action in picks_bans:
            if action['is_pick']:
                h_id=action['hero_id']
                team=action['team'] #0=radiant, 1=dire
                pos=hero_role_map.get(h_id, 0)
                gold=hero_gold_map.get(h_id, 0)

                if pos>0:
                    if team==0:
                        rad_pos[pos]=gold
                    else:
                        dire_pos[pos]=gold

        #Calcualte diffs (Normalize by 1000)
        #Safe Lane: Rad Safe (1+5) vs Dire Off (3+4)
        safe_diff=((rad_pos.get(1,0)+rad_pos.get(5,0))-(dire_pos.get(3,0)+dire_pos.get(4,0)))/1000.0

        #Mid Lane
        mid_diff=(rad_pos.get(2,0)-dire_pos.get(2,0))/1000.0

        #Off Lane
        off_diff=((rad_pos.get(3,0)+rad_pos.get(4,0))-(dire_pos.get(1,0)+dire_pos.get(5,0)))/1000.0

        lane_outcome=[safe_diff, mid_diff, off_diff]


        for step in range(len(picks_bans)):
            current_action = picks_bans[step]
            hero_id = current_action['hero_id']
            
            # CRITICAL: Skip if hero_id is invalid
            if hero_id < 1 or hero_id > 150:
                continue
            
            # Build history (only valid heroes)
            history = [a for a in picks_bans[:step] if 1 <= a['hero_id'] <= 150]
            
            # Build hero sequence (pad to 24)
            hero_sequence = [0] * 24
            for i, action in enumerate(history[:24]):  # Cap at 24
                hero_sequence[i] = action['hero_id']
            
            #Build type sequence
            type_sequence=[0]*24
            for i, action in enumerate(history[:24]):
                type_sequence[i]=int(action['is_pick'])
            
            #Build team sequence
            team_sequence=[0]*24
            for i, action in enumerate(history[:24]):
                team_sequence[i]=action['team']

            #Build role sequence (New Feature)
            role_sequence=[0]*24
            for i, action in enumerate(history[:24]):
                if action['is_pick']:
                    # If it's a pick, use the mapped role
                    role_sequence[i] = hero_role_map.get(action['hero_id'], 0)
                else:
                    # If it's a ban, role is 0 (unknown/irrelevant)
                    role_sequence[i] = 0
            
            # Track available heroes
            picked_banned = set(action['hero_id'] for action in history)
            valid_actions = [hero_id not in picked_banned for hero_id in range(1, 151)]
            
            # Determine outcome based on team
            team = current_action['team']
            outcome = 1.0 if (team == 0 and radiant_win) or (team == 1 and not radiant_win) else 0.0
            
            examples.append({
                'hero_sequence': hero_sequence,
                'type_sequence': type_sequence,
                'team_sequence': team_sequence,
                'role_sequence': role_sequence,
                'valid_actions': valid_actions,
                'target_actions': hero_id - 1,  # Now guaranteed to be 0-123
                'outcome': outcome,
                'lane_outcome':lane_outcome,
                'team': team,
                'is_pick': int(current_action['is_pick'])
            })
        
        return examples
        
            

