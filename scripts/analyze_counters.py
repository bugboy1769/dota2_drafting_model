import json
import os
from collections import defaultdict
import numpy as np

# Configuration
MATCHES_DIR = "data/raw/matches"
HEROES_FILE = "data/raw/heroes.json"

def load_heroes():
    with open(HEROES_FILE, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(h['id']): h for h in data}
    return data

def analyze_matchups():
    print(f"Scanning matches in {MATCHES_DIR}...")
    
    # Store results: matchups[(hero_a, hero_b)] = [wins, total_games]
    # hero_a is the hero we are checking winrate for against hero_b
    matchups = defaultdict(lambda: [0, 0])
    
    files = [f for f in os.listdir(MATCHES_DIR) if f.endswith('.json')]
    print(f"Found {len(files)} matches.")
    
    for fname in files:
        try:
            with open(os.path.join(MATCHES_DIR, fname), 'r') as f:
                data = json.load(f)
                
            # Check if valid match
            if 'radiant_win' not in data or 'draft_timings' not in data:
                continue
                
            radiant_win = data['radiant_win']
            
            # Extract teams
            radiant_team = []
            dire_team = []
            
            for pick in data['draft_timings']:
                if pick['pick']: # If it's a pick (not a ban)
                    if pick['active_team'] == 2: # Radiant
                        radiant_team.append(pick['hero_id'])
                    else: # Dire
                        dire_team.append(pick['hero_id'])
            
            # Update stats for every pair
            # Radiant Heroes vs Dire Heroes
            for r_hero in radiant_team:
                for d_hero in dire_team:
                    # R vs D
                    matchups[(r_hero, d_hero)][1] += 1
                    if radiant_win:
                        matchups[(r_hero, d_hero)][0] += 1
                    
                    # D vs R (inverse perspective)
                    matchups[(d_hero, r_hero)][1] += 1
                    if not radiant_win:
                        matchups[(d_hero, r_hero)][0] += 1
                        
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

    return matchups

def print_report(matchups, heroes, focus_hero_name="Anti-Mage"):
    # Find ID for focus hero
    focus_id = None
    for hid, hdata in heroes.items():
        if hdata.get('localized_name') == focus_hero_name:
            focus_id = int(hid)
            break
            
    if focus_id is None:
        print(f"Hero {focus_hero_name} not found.")
        return

    print(f"\n--- Analysis for {focus_hero_name} (ID: {focus_id}) ---")
    print(f"{'Opponent':<20} | {'Win Rate':<10} | {'Games':<10}")
    print("-" * 45)
    
    results = []
    for (h_a, h_b), stats in matchups.items():
        if h_a == focus_id:
            wins, total = stats
            if total < 5: continue # Skip low sample size
            wr = wins / total
            results.append((h_b, wr, total))
            
    # Sort by lowest winrate (best counters against focus hero)
    results.sort(key=lambda x: x[1])
    
    for opp_id, wr, total in results[:10]: # Top 10 counters
        opp_name = heroes.get(str(opp_id), {}).get('localized_name', f"ID {opp_id}")
        print(f"{opp_name:<20} | {wr:.1%}     | {total:<10}")

if __name__ == "__main__":
    heroes = load_heroes()
    data = analyze_matchups()
    print_report(data, heroes, "Anti-Mage")
    print_report(data, heroes, "Meepo")
