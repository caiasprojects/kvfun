import json
import random
import os
from tqdm import tqdm

from model_generate import Baseline_model, KV_prediction_model

n = 200
model_str = "Llama_8B_base_1B_aux"
model_name = f"meta-llama/{model_str}"

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

# select n questions
random_questions = random.sample(questions, n)

model = Baseline_model(model_name)
# model = KV_prediction_model()

answers = {}
total_ttft = 0

for question in tqdm(random_questions):
    id = question["id"]
    prompt = question["prompt"]
    # print(prompt)

    response, ttft = model.call_with_prompt(prompt)
    # print("response: ", response)
    total_ttft += ttft
    answers[id] = response

print("Average ttft: ", total_ttft / n)

save_dir = "squad/answers/"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, f"{model_str}.json")

with open(output_path, "w") as f:
    json.dump(answers, f)
