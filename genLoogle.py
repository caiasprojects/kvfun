import os
import json
import argparse
from datasets import load_dataset
from model_generate import KV_hybrid_model

n = 400

# to test baseline, recalculate=False.
# Then the corresponding baseline_model flag set to true

# otherwise, recalculate=True and both baseline flags set to false. for hybrid

model_str = "Llama1B_aux_8B_base_recomputed_1n_80x_hyprid"

# try to keep num_intervals to the lowest possible and interval size to a higher number.
# num_intervals is # of forward passes. They are expensive
recalculate_args = {
    "recalculate": True,
    "interval_size": 80,
    "num_intervals": 1,
}

model = KV_hybrid_model(baseline_base=False, baseline_aux=False)

save_dir = "loogle/answers/"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, f"{model_str}@{n}.json")


def generate_responses(messages, tokenizer, sampling_params, llm):
    """Generate responses from the model based on input messages."""
    prompt_token_ids = [
        tokenizer.apply_chat_template(message, add_generation_prompt=True)
        for message in messages
    ]
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
    )
    return [output.outputs[0].text for output in outputs]


def generate_responses_and_save(
    input_prompts,
    ids,
    texts,
    questions,
    answers,
    output_file,
    iss,
):
    """Generate responses using the LLM and save them to output files."""
    # Split messages into batches based on the number of output files
    results = {}
    for i, prompt in enumerate(input_prompts):
        # Generate responses for the current batch
        response, ttft, cache, prompt_len = model.call_with_prompt(
            prompt, max_new_tokens=100, recalculate_args=recalculate_args
        )

        processed_response = response.strip()

        results[ids[i]] = {
            "unid": iss[i],
            "text": texts[i][:10],
            "question": questions[i],
            "generated_answer": processed_response,
            "rl_ans": answers[i],
            "ttft": ttft,
            "prompt_len": prompt_len,
        }

    # Save results to the output file
    with open(output_file, "w") as out_file:
        json.dump(results, out_file)
    print(f"Q&A pairs saved to {output_file}")


def evaluate_loogle(testset, output_file):
    """Evaluate the model on a specific LooGLE dataset."""
    # Load dataset split
    data = load_dataset(
        "bigainlco/LooGLE", testset, split="test", trust_remote_code=True
    )

    ids = []
    iss = []
    texts = []
    questions = []
    answers = []
    input_prompts = []
    mj = 0

    # Prepare prompts for generating answers
    for i, entry in enumerate(data):
        input_text = entry["input"]
        qa_pairs_raw = entry["qa_pairs"]

        # Parse qa_pairs field
        try:
            qa_pairs = (
                eval(qa_pairs_raw) if isinstance(qa_pairs_raw, str) else qa_pairs_raw
            )
        except (ValueError, SyntaxError):
            print(f"Error parsing qa_pairs for entry {i}")
            continue

        for j, qa in enumerate(qa_pairs):
            if not isinstance(qa, dict) or "Q" not in qa or "A" not in qa:
                print(f"Invalid QA pair format in entry {i}: {qa}")
                continue

            question = qa["Q"]
            ground_truth_answer = qa["A"]

            # Store data for results
            ids.append(i)
            texts.append(input_text)
            questions.append(question)
            answers.append(ground_truth_answer)
            iss.append(mj)
            mj = mj + 1

            # Create the prompt message
            prompt = f"""
                    You are a highly knowledgeable assistant. Based on the provided text, answer the question in an extremely concise and accurate manner.

                    # TEXT:
                    {input_text.strip()}

                    # QUESTION:
                    {question.strip()}

                    Requirements:
                    1. Provide a clear and precise answer.
                    2. Respond in a few words, up to 1 sentence

                    # ANSWER:
                    """
            if len(prompt) > 60000:
                continue

            input_prompts.append(prompt)
            print(len(prompt))
            if len(input_prompts) > n:
                print("Reached n")
                break

    # Generate and save responses
    generate_responses_and_save(
        input_prompts,
        ids,
        texts,
        questions,
        answers,
        output_file,
        iss,
    )


def main():
    """Main function to evaluate the model on all LooGLE datasets."""
    # os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

    # Datasets in LooGLE
    datasets = ["shortdep_qa"]

    # Evaluate on each dataset
    for testset in datasets:
        save_dir = "loogle/answers/"
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{model_str}@{n}.json")
        evaluate_loogle(testset, output_path)


if __name__ == "__main__":
    main()
