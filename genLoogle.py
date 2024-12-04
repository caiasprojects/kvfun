import os
import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset

def create_dir(directory):
    """Create a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def generate_responses(messages, tokenizer, sampling_params, llm):
    """Generate responses from the model based on input messages."""
    prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]

def process_response(response):
    """Clean up the model's response."""
    return response.strip()

def generate_responses_and_save(messages, ids, texts, questions, answers, output_files, tokenizer, sampling_params, llm, iss):
    """Generate responses using the LLM and save them to output files."""
    # Split messages into batches based on the number of output files
    batch_size = len(messages) // len(output_files)
    batches = [messages[i:i + batch_size] for i in range(0, len(messages), batch_size)]

    for idx, output_file in enumerate(output_files):
        # Generate responses for the current batch
        batch = batches[idx]
        batch_responses = generate_responses(batch, tokenizer, sampling_params, llm)

        results = []
        for i in range(len(batch_responses)):
            response = batch_responses[i]
            processed_response = process_response(response)

            result = {
                "id": ids[i],
                "unid" : iss[i],
                "text": texts[i][:10],
                "question": questions[i],
                "generated_answer": processed_response,
                "rl_ans": answers[i]
            }
            results.append(result)

        # Save results to the output file
        with open(output_file, 'w') as out_file:
            json.dump(results, out_file, indent=4)
        print(f"Q&A pairs saved to {output_file}")

def evaluate_loogle(testset, output_files, tokenizer, sampling_params, llm):
    """Evaluate the model on a specific LooGLE dataset."""
    # Load dataset split
    data = load_dataset("bigainlco/LooGLE", testset, split="test")

    ids = []
    iss = []
    texts = []
    questions = []
    answers = []
    messages = []
    mj = 0

    # Prepare prompts for generating answers
    for i, entry in enumerate(data):
        input_text = entry["input"]
        qa_pairs_raw = entry["qa_pairs"]

        # Parse qa_pairs field
        try:
            qa_pairs = eval(qa_pairs_raw) if isinstance(qa_pairs_raw, str) else qa_pairs_raw
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
            prompt = [
                {
                    "role": "user",
                    "content": f"""
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
                }
            ]
            messages.append(prompt)
            if len(messages) > 200:
                break

    # Generate and save responses
    generate_responses_and_save(
        messages,
        ids,
        texts,
        questions,
        answers,
        output_files,
        tokenizer,
        sampling_params,
        llm,
        iss
    )

def main(model_name, max_model_len, outputdir, temperature):
    """Main function to evaluate the model on all LooGLE datasets."""
    # os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1500, stop_token_ids=[tokenizer.eos_token_id])

    # Datasets in LooGLE
    datasets = ["shortdep_qa"]

    # Create output directory
    create_dir(outputdir)

    # Evaluate on each dataset
    for testset in datasets:
        output_file = os.path.join(outputdir, f"{testset}_results.json")
        evaluate_loogle(testset, [output_file], tokenizer, sampling_params, llm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LooGLE Q&A Evaluation Script")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The name of the model to be used.")
    parser.add_argument('--max_model_len', type=int, default=36051, help="Maximum model length.")
    parser.add_argument('--outputdir', type=str, default='loogle_results', help="Output directory to store results.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    args = parser.parse_args()

    main(args.model_name, args.max_model_len, args.outputdir, args.temperature)
