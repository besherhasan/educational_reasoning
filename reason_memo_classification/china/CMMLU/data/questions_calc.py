import os
import pandas as pd

root_path = '/home/besher.hassan/educational/educational_reasoning/blooms_classifications/china/CMMLU/data'

splits = ['dev', 'test']

total_questions = 0
split_counts = {}

for split in splits:
    split_path = os.path.join(root_path, split)
    if not os.path.exists(split_path):
        print(f"Warning: Directory {split_path} does not exist.")
        continue

    count = 0
    for filename in os.listdir(split_path):
        if filename.endswith('.csv'):
            full_path = os.path.join(split_path, filename)
            try:
                df = pd.read_csv(full_path)
                count += len(df)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    split_counts[split] = count
    total_questions += count

print(f"Split counts: {split_counts}")
print(f"Total questions: {total_questions}")
