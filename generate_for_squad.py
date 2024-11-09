import json
import random

from model_generate import call_with_prompt

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

                final_question = f"Context: {context}\nQuestion: {question}\nAnswer:"
                questions.append({"id": id, "prompt": final_question})

    return questions


random.seed(42)

questions = create_squad_questions(data)
random_questions = random.sample(questions, 400)

for question in random_questions:
    id = question["id"]
    prompt = question["prompt"]
    print(prompt)

    response = call_with_prompt(prompt)
    print("REPSONSE: \n")
    print(response)

    break
