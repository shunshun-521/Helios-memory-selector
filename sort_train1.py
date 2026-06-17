import pandas as pd

df = pd.read_csv('data/audiocaps/metadata/train1.csv')
df = df.sort_values('audiocap_id').reset_index(drop=True)
df.to_csv('data/audiocaps/metadata/train1_1.csv', index=False)
print(f"Saved train1_1.csv with {len(df)} rows, sorted by audiocap_id.")
