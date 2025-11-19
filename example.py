# inspect_examples2.py
import pickle
from pathlib import Path
from collections import Counter

p = Path("data/processed/train.pkl")   # adjust path if needed
if not p.exists():
    print("File not found:", p)
    raise SystemExit(1)

with p.open("rb") as f:
    examples = pickle.load(f)

print("Total examples:", len(examples))
if len(examples) == 0:
    raise SystemExit("No examples found")

# show keys of first 5 examples
for i in range(min(5, len(examples))):
    ex = examples[i]
    print(f"\n--- example {i} keys ---")
    keys = list(ex.keys())
    print("keys:", keys)
    # show sample values for likely keys
    for k in keys:
        v = ex[k]
        preview = None
        try:
            if isinstance(v, (list, tuple)):
                preview = v[:12]
            else:
                preview = repr(v)[:200]
        except Exception:
            preview = "<could not preview>"
        print(f"  {k}: {type(v).__name__} -> {preview}")

# try to find typical target key names and show distribution
possible_target_keys = ['target_action','target','action','target_hero','target_id','hero_target','label','y']
found_counts = {}
for ex in examples:
    for k in possible_target_keys:
        if k in ex:
            found_counts[k] = found_counts.get(k, 0) + 1

print("\nDetected counts for common target key names:", found_counts)

# if nothing found among common keys, search for numeric scalar keys (int/float)
scalar_key_candidates = Counter()
for ex in examples[:200]:
    for k, v in ex.items():
        if isinstance(v, (int, float)):
            scalar_key_candidates[k] += 1

print("\nTop scalar keys seen (int/float) among first 200 examples (key -> count):")
print(scalar_key_candidates.most_common(20))

