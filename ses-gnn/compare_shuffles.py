import glob
import random
import os

# 1. Что сейчас в файлах (от 8L new, shuffle на 5461)
with open('train_files.txt') as f:
    train_current = [l.strip() for l in f if l.strip()]
with open('val_files.txt') as f:
    val_current = [l.strip() for l in f if l.strip()]

print(f"current train: {len(train_current)}")
print(f"current val:   {len(val_current)}")

# 2. Реконструкция правильного shuffle (на 5462 файлах)
old_paths = sorted(glob.glob('./processed/graphs_multiscale/*.pt'))
print(f"\nold total: {len(old_paths)}")

old_names = [os.path.basename(p) for p in old_paths]
random.Random(42).shuffle(old_names)

train_expected = old_names[:5000]
val_expected   = old_names[5000:5050]
test_expected  = old_names[5050:]

print(f"expected train: {len(train_expected)}")
print(f"expected val:   {len(val_expected)}")
print(f"expected test:  {len(test_expected)}")

# 3. Сравнение
train_set = set(train_current)
expected_train_set = set(train_expected)

val_set = set(val_current)
expected_val_set = set(val_expected)

print(f"\n=== TRAIN ===")
print(f"совпадает полностью: {train_set == expected_train_set}")
print(f"пересечение: {len(train_set & expected_train_set)} / {len(expected_train_set)}")

print(f"\n=== VAL ===")
# Текущий val = 461 файл, ожидаемый = 50 + 411 из test
val_overlap_with_val = val_set & expected_val_set
val_overlap_with_test = val_set & set(test_expected)
print(f"текущий val ∩ ожидаемый val (50):  {len(val_overlap_with_val)} / 50")
print(f"текущий val ∩ ожидаемый test (412): {len(val_overlap_with_test)} / 412")

print(f"\n=== TEST (текущий val минус ожидаемый val) ===")
# Что сейчас в val, но должно было быть test
test_leaked_into_val = val_overlap_with_test
print(f"файлов из test попали в val: {len(test_leaked_into_val)}")

# 4. Сколько test-файлов попало в текущий train?
test_set_expected = set(test_expected)
leaked_into_train = test_set_expected & train_set
print(f"\ntest-файлов утекло в текущий train: {len(leaked_into_train)} / {len(test_set_expected)}")

# 5. Примеры
print(f"\nпервые 5 текущий train: {train_current[:5]}")
print(f"первые 5 ожидаемый train: {train_expected[:5]}")
print(f"первые 5 текущий val: {val_current[:5]}")
print(f"первые 5 ожидаемый val: {val_expected[:5]}")