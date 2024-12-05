import json


def calculate_correct_percentage(json_file):

    # Load JSON data from the file
    with open(json_file, "r") as file:
        data = json.load(file)

    # Initialize counters
    total_questions = 0
    correct_answers = 0

    # Loop through each entry in the JSON
    for entry in data.values():
        total_questions += 1

        generated_answer = entry.get("generated_answer", "").strip().lower()
        rl_answer = entry.get("rl_ans", "").strip().lower()
        iddd = entry.get("unid", "")

        # Check if all words in the generated_answer are in rl_ans
        if all(word in generated_answer.split() for word in rl_answer.split()) or all(
            word in rl_answer.split() for word in generated_answer.split()
        ):
            correct_answers += 1
            print("corrr")
            print(iddd)

    # Calculate percentage
    if total_questions == 0:
        print("No questions found in the dataset.")
        return

    percentage_correct = (correct_answers / total_questions) * 100
    print(f"Percentage of Correct Answers: {percentage_correct:.2f}%")


# Replace 'data.json' with the path to your JSON file
json_file_path = "loogle_results/shortdep_qa_results.json"
calculate_correct_percentage(json_file_path)
