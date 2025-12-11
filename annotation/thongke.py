from collections import Counter

label_map = {
    "1": "Neutrality",
    "2": "Enjoyment",
    "3": "Confusion",
    "4": "Fatigue",
    "5": "Distraction"
}

def count_labels(file_path):
    counter = Counter()
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                label = line.split()[-1]   # token cuối
                counter[label] += 1
    return counter


def print_stats(name, counter):
    print(f"\n=== {name} LABEL COUNTS ===")
    for lbl, cnt in sorted(counter.items(), key=lambda x: int(x[0])):
        print(f"Label {lbl} ({label_map[lbl]}): {cnt}")


train_counts = count_labels("train.txt")
test_counts = count_labels("test.txt")

print_stats("TRAIN", train_counts)
print_stats("TEST", test_counts)
