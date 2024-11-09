from model_generate import generate_with_kv_cache
from datasets import load_dataset
from tqdm import tqdm

# Load TriviaQA dataset
ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="test")
print(ds[1])

# # Run inference on examples
results = {}
for question in tqdm(ds):
    try:
        key = question["question_id"] + "--" + question["entity_pages"]["filename"][0]

        prompt = f"Question: {question['question']}\nContext: {question['entity_pages']['wiki_context'][0]}\nAnswer:"
        prediction = "Alright"
        results[key] = prediction
    except:
        print(question["search_results"])
        break

# # Save results
import json

with open("triviaqa_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
