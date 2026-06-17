import os
import csv

train1_dir = "data/audiocaps/train/train1"
train_csv = "data/audiocaps/metadata/train.csv"
output_csv = "data/audiocaps/metadata/train1.csv"

# 从 train1 目录获取所有 audiocap_id
train1_ids = {os.path.splitext(f)[0] for f in os.listdir(train1_dir) if f.endswith(".wav")}

# 筛选并写入
with open(train_csv, "r") as fin, open(output_csv, "w", newline="") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    header = next(reader)
    writer.writerow(header)
    count = 0
    for row in reader:
        if row[0] in train1_ids:
            writer.writerow(row)
            count += 1

print(f"train1.csv: {count} rows (from {len(train1_ids)} wav files)")
