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

    def get_pro_matches(self, less_than_match_id: Optional[int]=None)->List[Dict]:
        url = f"{self.BASE_URL}/proMatches"
        params={'less_than_match_id':less_than_match_id} if less_than_match_id else 0
        time.sleep(self.rate_limit_delay)
        response=requests.get(url, params=params)
        return response.json()
    
    def get_match(self, match_id: int)->Dict:
        url=f"{self.BASE_URL}/matches/{match_id}"
        time.sleep(self.rate_limit_delay)
        response=requests.get(url)
        if response.status_code==200:
            return response.json()
        return None

    def get_heroes(self)->Dict:
        url=f"{self.BASE_URL}/heroes"
        response=requests.get(url)
        return response.json()

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
            examples=self._create_example_from_matches(match_data)
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

    def _create_examples_from_match(self, match_data: Dict)->List[Dict]:
        """Create training examples from a single match"""
        picks_bans=match_data.get('picks_bans', [])
        if len(picks_bans)<20:
            return []
        radiant_win=match_data.get('radiant_win', False)
        examples=[]

        for step in range(len(picks_bans)):
            history=picks_bans[:step]
            current_action=picks_bans[step]

            #Build hero sequence, pad to 24
            hero_sequence=[0] * 24
            for i, action in enumerate(history):
                hero_sequence[i]=action['hero_id']
            
            #Track available heroes
            picked_banned=set(action['hero_id'] for action in history)
            valid_actions=[hero_id not in picked_banned for hero_id in range(1, 125)]

            #Determine action based on team
            team=current_action['team']
            outcome=1.0 if (team==0 and radiant_win) or (team==1 and not radiant_win) else 0.0

            examples.append({
                'hero_sequence':hero_sequence,
                'valid_actions':valid_actions,
                'target_actions': current_action['hero_id'] - 1, #0-indexed
                'outcome':outcome,
                'team':team,
                'is_pick':int(current_action['is_pick'])
            })
            
            return examples
    
        

