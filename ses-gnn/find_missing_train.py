# find_missing_train.py
import os

# Из dMaSIF training.txt
with open('training.txt') as f:
    their_train = set(ln.strip() for ln in f if ln.strip())

# Из нашего overlap_train_filtered.txt
with open('overlap_train_filtered.txt') as f:
    ours = set(ln.strip() for ln in f if ln.strip())

# Наши ключи = PDB_CHAIN1 из наших графов
our_keys = set()
for name in ours:
    parts = name.split('_')
    if len(parts) >= 2:
        our_keys.add(f"{parts[0]}_{parts[1]}")

# Что осталось
missing = their_train - our_keys
print(f"Missing: {len(missing)}")

with open('training_missing.txt', 'w') as f:
    for x in sorted(missing):
        f.write(x + '\n')