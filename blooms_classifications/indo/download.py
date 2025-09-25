from datasets import load_dataset

# Load the IndoMMLU dataset (main 'test' split)
dataset = load_dataset('indolem/IndoMMLU')

# Count the number of questions
num_questions = len(dataset['test'])
print(f"Total questions in test split: {num_questions}")
