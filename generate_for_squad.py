import json
import random
import os
from tqdm import tqdm

from model_generate import KV_hybrid_model

# Open and read the JSON file
with open("squad/squad_data.json", "r") as file:
    data = json.load(file)["data"]


def create_squad_questions(data):

    questions = []
    for article in data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id = qa["id"]

                final_question = f"Context: {context}\nQuestion: {question}. Answer in as few words as possible"
                questions.append({"id": id, "prompt": final_question})

    return questions


random.seed(42)

questions = create_squad_questions(data)


n = 400
model_str = "Llama1B_baseline_hyprid"

# select n questions
random_questions = random.sample(questions, n)

model = KV_hybrid_model(baseline=False)


answers = {}
total_ttft = 0

for question in tqdm(random_questions):
    id = question["id"]
    prompt = question["prompt"]

    response, ttft, cache, prompt_len = model.call_with_prompt(
        prompt, max_new_tokens=100, recalculate=False
    )

    total_ttft += ttft
    answers[id] = {"response": response, "ttft": ttft, "prompt_len": prompt_len}

print("Average ttft: ", total_ttft / n)

save_dir = "squad/answers/"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, f"{model_str}@{n}.json")

with open(output_path, "w") as f:
    json.dump(answers, f)
