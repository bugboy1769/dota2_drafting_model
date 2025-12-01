import pickle
from collections import Counter
import torch
from pathlib import Path
from tqdm import tqdm

def check_distribution():
    data_path = Path("data/processed/train.pkl")
    if not data_path.exists():
        print("Training data not found!")
        return

    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        examples = pickle.load(f)

    print(f"Analyzing {len(examples)} examples...")
    
    hero_counts = Counter()
    total_picks = 0
    
    for ex in tqdm(examples):
        # Count target action (the hero being picked in this example)
        if ex['is_pick']:
            # target_actions is 0-indexed (hero_id - 1)
            hero_id = ex['target_actions'] + 1
            hero_counts[hero_id] += 1
            total_picks += 1

    print("\nTop 20 Most Frequent Heroes:")
    print(f"{'Rank':<5} {'Hero ID':<10} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    
    sorted_heroes = hero_counts.most_common(20)
    for rank, (h_id, count) in enumerate(sorted_heroes, 1):
        pct = (count / total_picks) * 100
        print(f"{rank:<5} {h_id:<10} {count:<10} {pct:.2f}%")

    # Check Anti-Mage specifically (ID 1)
    am_count = hero_counts.get(1, 0)
    am_pct = (am_count / total_picks) * 100
    print(f"\nAnti-Mage (ID 1): {am_count} picks ({am_pct:.2f}%)")

if __name__ == "__main__":
    check_distribution()
