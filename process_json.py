import json
import csv
import os

csv_file = 'data/audiocaps/metadata/train1.csv'
json_file = 'data/audiocaps/helios_train.json'
output_dir = 'data/audiocaps/t2a_output'
output_file = os.path.join(output_dir, 'new_train.json')

os.makedirs(output_dir, exist_ok=True)

# Read JSON into a dictionary indexed by caption
with open(json_file, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

json_dict = {}
for item in json_data:
    if 'cap' in item and len(item['cap']) > 0:
        # Assuming the first caption is what we match against
        cap = item['cap'][0]
        json_dict[cap] = item

new_json_data = []

# Read CSV and build the new JSON list in order
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        caption = row['caption']
        if caption in json_dict:
            # Create a new item based on the existing one
            new_item = json_dict[caption].copy()
            
            # Apply modifications
            new_item['sample_rate'] = 48000
            new_item['duration'] = 10
            new_item['cut'] = [0.0, 10.0]
            
            new_json_data.append(new_item)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_json_data, f, indent=4, ensure_ascii=False)

print(f"Successfully created {output_file} with {len(new_json_data)} entries.")
