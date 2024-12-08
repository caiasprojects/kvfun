import json
import random
import os
from tqdm import tqdm

from model_generate import KV_hybrid_model
from transformers import AutoTokenizer

n = 400
model_str = "1B_aux_8B_base_no_recalc"

recalculate_args = {
    "recalculate": False,
    "interval_size": 400,
    "num_intervals": 1,
}

model = KV_hybrid_model(baseline_base=False, baseline_aux=False)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Open and read the JSON file
with open("squad/squad_data.json", "r") as file:
    data = json.load(file)["data"]


def create_squad_questions(data):
    questions_total = []

    for article in data:
        if len(questions_total) >= n:
            print("reached n")
            return questions_total

        # Collect all paragraphs from the article
        article_paragraphs = []
        for paragraph in article["paragraphs"]:
            article_paragraphs.append(paragraph["context"])
        # print(len(article_paragraphs))

        for i, paragraph in enumerate(article["paragraphs"][:-4]):
            num_paragraphs = 4  # of paragraphs to add to extend prompt length
            combined_context = " ".join(article_paragraphs[i : i + num_paragraphs])
            # print(len(combined_context))

            for qa in paragraph["qas"]:
                if len(questions_total) >= n:
                    return questions_total

                question = qa["question"]
                id = qa["id"]

                final_question = f"Answer in as few words as possible. Context: {combined_context}\nQuestion: {question}. Answer in as few words as possible"

                message = [{"role": "user", "content": final_question}]

                tokens = tokenizer.apply_chat_template(
                    message, tokenize=True, add_generation_prompt=True
                )

                # make sure its long enough to be useful
                if 1000 <= len(tokens):
                    questions_total.append({"id": id, "prompt": final_question})

    return questions_total


questions = create_squad_questions(data)

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

print(model_str)
