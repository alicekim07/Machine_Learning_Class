## 이 파일은 Raw한 데이터에서 test_list.txt, train_list.txt, val_list.txt를 자동으로 만드는 파일입니다.
import os
import random

LABELS = ["yes", "no", "stop", "go"]
ALL_CLASSES = os.listdir("Data/archive")

# print(ALL_CLASSES)
other_classes = [c for c in ALL_CLASSES if c not in LABELS and c != "_background_noise_"]
# print(other_classes)

train_list = []
val_list = []
test_list = []

for label in LABELS:
    files = os.listdir(f"Data/archive/{label}")
    random.shuffle(files)
    n = len(files)
    train, val, test = files[:int(0.8*n)], files[int(0.8*n):int(0.9*n)], files[int(0.9*n):]
    for f in train:
        train_list.append((f"{label}/{f}", label))
    for f in val:
        val_list.append((f"{label}/{f}", label))
    for f in test:
        test_list.append((f"{label}/{f}", label))

# Unknown (other)
for label in random.sample(other_classes, 8):
    files = os.listdir(f"Data/archive/{label}")
    files = random.sample(files, 300)
    for f in files[:200]:
        train_list.append((f"{label}/{f}", "other"))
    for f in files[200:250]:
        val_list.append((f"{label}/{f}", "other"))
    for f in files[250:]:
        test_list.append((f"{label}/{f}", "other"))

# Shuffle and Save
random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

def save_list(lst, filename):
    with open(filename, "w") as f:
        for path, label in lst:
            f.write(f"{path},{label}\n")

save_list(train_list, "train_list.txt")
save_list(val_list, "val_list.txt")
save_list(test_list, "test_list.txt")