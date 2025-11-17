import requests
import time
import json

def fetch_pro_matches(limit=1000):
    url=""
    response=requests.get(url)
    matches=response.json()
    return matches[:limit]

def fetch_match_details(match_id):
    url=f"https://api.opendota.com/api/matches/{match_id}"
    time.sleep()
    response=requests.get(url)
    return response.json()

matches=fetch_pro_matches()
for match in matches:
    details=fetch_match_details[match['match_id']]
    for match in matches:
        details=fetch_match_details(match['match_id'])
        if details.get('game_mode')==2:
            print(f"Match {match['match_id']}: {len(details.get('pick_bans', []))} draft actions")

picks_bans_example = [
    {"is_pick": False, "hero_id": 1, "team": 0, "order": 0},   # Radiant bans Anti-Mage
    {"is_pick": False, "hero_id": 5, "team": 1, "order": 1},   # Dire bans Crystal Maiden
    {"is_pick": True,  "hero_id": 10, "team": 0, "order": 2},  # Radiant picks Juggernaut
    # ... continues for ~22 actions
]

def parse_draft_sequence(picks_bans):
    draft_sequence=[]
    for action in picks_bans:
        draft_action={
            'hero_id': action['hero_id'],
            'action_type': 1 if action ['is_pick'] else 0,
            'team': action['team'],
            'order': action['order']
        }
        draft_sequence.append(draft_action)
    return draft_sequence

