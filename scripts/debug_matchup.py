import json
import os
from collections import defaultdict

# Configuration
MATCHES_DIR = "data/raw/matches"
HEROES_FILE = "data/raw/heroes.json"

def load_heroes():
    with open(HEROES_FILE, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(h['id']): h for h in data}
    return data

# Quick test with first 100 matches
matchups = defaultdict(lambda: [0, 0])
heroes = load_heroes()

# Find AM and Slardar IDs
am_id = None
slardar_id = None
for hid, hdata in heroes.items():
    if hdata.get('localized_name') == 'Anti-Mage':
        am_id = int(hid)
    if hdata.get('localized_name') == 'Slardar':
        slardar_id = int(hid)

print(f"AM ID: {am_id}, Slardar ID: {slardar_id}")

files = [f for f in os.listdir(MATCHES_DIR) if f.endswith('.json')]
print(f"Scanning {len(files)} total matches...")
am_vs_slardar_games = []

for fname in files:
    try:
        with open(os.path.join(MATCHES_DIR, fname), 'r') as f:
            data = json.load(f)
            
        if 'radiant_win' not in data or 'draft_timings' not in data:
            continue
            
        radiant_win = data['radiant_win']
        radiant_team = []
        dire_team = []
        
        for pick in data['draft_timings']:
            if pick['pick']:
                if pick['active_team'] == 2:
                    radiant_team.append(pick['hero_id'])
                else:
                    dire_team.append(pick['hero_id'])
        
        # Check if AM vs Slardar matchup
        am_on_radiant = am_id in radiant_team
        slardar_on_radiant = slardar_id in radiant_team
        am_on_dire = am_id in dire_team
        slardar_on_dire = slardar_id in dire_team
        
        if (am_on_radiant and slardar_on_dire) or (am_on_dire and slardar_on_radiant):
            am_team = 'Radiant' if am_on_radiant else 'Dire'
            slardar_team = 'Radiant' if slardar_on_radiant else 'Dire'
            winner = 'Radiant' if radiant_win else 'Dire'
            am_won = (am_team == winner)
            
            am_vs_slardar_games.append({
                'am_team': am_team,
                'slardar_team': slardar_team,
                'winner': winner,
                'am_won': am_won
            })
            
            # Update matchup stats
            for r_hero in radiant_team:
                for d_hero in dire_team:
                    matchups[(r_hero, d_hero)][1] += 1
                    if radiant_win:
                        matchups[(r_hero, d_hero)][0] += 1
                    
                    matchups[(d_hero, r_hero)][1] += 1
                    if not radiant_win:
                        matchups[(d_hero, r_hero)][0] += 1
                        
    except Exception as e:
        continue

print(f"\nFound {len(am_vs_slardar_games)} AM vs Slardar games in first 500 matches")
print(f"AM wins: {sum(1 for g in am_vs_slardar_games if g['am_won'])}")
print(f"Slardar wins: {sum(1 for g in am_vs_slardar_games if not g['am_won'])}")
if am_vs_slardar_games:
    am_wr = sum(1 for g in am_vs_slardar_games if g['am_won']) / len(am_vs_slardar_games)
    print(f"AM Winrate: {am_wr:.1%}")

print(f"\nFrom matchups dict:")
print(f"matchups[(AM, Slardar)] = {matchups[(am_id, slardar_id)]}")
wins, total = matchups[(am_id, slardar_id)]
if total > 0:
    print(f"AM WR from dict: {wins/total:.1%}")
