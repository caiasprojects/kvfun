import json
import random
import os
from tqdm import tqdm

from model_generate import KV_hybrid_model

n = 400
model_str = "Llama8B_aux_8B_base_hyprid"

recalculate_args = {
    "recalculate": False,
    "interval_size": 80,
    "num_intervals": 1,
}

model = KV_hybrid_model(baseline_base=True, baseline_aux=False)

# Open and read the JSON file
with open("squad/squad_data.json", "r") as file:
    data = json.load(file)["data"]


def create_squad_questions(data):

    questions_total = []
    for article in data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if len(questions_total) >= n:
                    print("reached n")
                    return questions_total

                question = qa["question"]
                id = qa["id"]

                final_question = f"Answer in as few words as possible. Context: {context}\nQuestion: {question}. Answer in as few words as possible"
                if len(final_question) > 1500:
                    questions_total.append({"id": id, "prompt": final_question})

    return questions_total


questions = create_squad_questions(data)
print("max prompt length: ", max(len(q["prompt"]) for q in questions))
print("min prompt length: ", min(len(q["prompt"]) for q in questions))

answers = {}
total_ttft = 0

prompt_lens = []
for question in tqdm(questions):
    id = question["id"]
    prompt = question["prompt"]

    response, ttft, cache, prompt_len = model.call_with_prompt(
        prompt, max_new_tokens=100, recalculate_args=recalculate_args
    )

    total_ttft += ttft
    answers[id] = {"response": response, "ttft": ttft, "prompt_len": prompt_len}
    prompt_lens.append(prompt_len)
print("Average ttft: ", total_ttft / n)

print("min prompt length: ", min(prompt_lens))
print("max prompt length: ", max(prompt_lens))

save_dir = "squad/answers/"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, f"{model_str}@{n}.json")

with open(output_path, "w") as f:
    json.dump(answers, f)
