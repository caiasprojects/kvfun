import os
import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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

def generate_qa_pairs(input_file, output_files, tokenizer, sampling_params, llm):
    """Generate and save Q&A solutions based on the input JSON."""
    # Load text and questions from the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    print(data[0])
    question_list = data
    messages = []
    ids = []
    texts = []
    questions = []
    answers = []


    # Prepare prompts for generating answers
    for entry in question_list:
        q_id = entry["id"]
        text = entry["text"]
        question = entry["question"]
        answer = entry["answer"]

        ids.append(q_id)
        texts.append(text)
        questions.append(question)
        answers.append(answer)

        prompt = [
            {
                "role": "user",
                "content": f"""
                You are a highly knowledgeable assistant. Based on the provided text, answer the question in a extremely concise and accurate manner.

                # TEXT:
                {text.strip()}

                # QUESTION:
                {question.strip()}

                Requirements:
                1. Provide a clear and precise answer.
                2. Respond in 1 to 10 words. 

                # ANSWER:
                """
            }
        ]
        messages.append(prompt)

    # Generate responses using the LLM
    for idx, output_file in enumerate(output_files):
        responses = generate_responses(messages, tokenizer, sampling_params, llm)

        results = []
        for i in range(len(responses)):
            response = responses[i]
            processed_response = process_response(response)

            result = {
                "id": ids[i],
                "text": texts[i],
                "question": questions[i],
                "generated_answer": processed_response,
                "rl_ans" : answers[i]
            }
            results.append(result)

        # Save results to the output file
        with open(output_file, 'w') as out_file:
            json.dump(results, out_file, indent=4)
        print(f"Q&A pairs saved to {output_file}")

def main(model_name, max_model_len, outputdir, num_samples, temperature):
    """Main function to initialize components and run the Q&A generation."""
    # Set environment variables for the LLM
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=1, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1500, stop_token_ids=[tokenizer.eos_token_id])

    # Input file with text and questions
    input_file = 'haystack.json'

    # Output file paths
    output_files = [f'{outputdir}/qa_output_{i}.json' for i in range(num_samples)]

    generate_qa_pairs(input_file, output_files, tokenizer, sampling_params, llm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A Pair Generation Script")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The name of the model to be used.")
    parser.add_argument('--max_model_len', type=int, default=1500, help="Maximum model length.")
    parser.add_argument('--outputdir', type=str, default='ans', help="Output directory to store results.")
    parser.add_argument('--numsamples', type=int, default=10, help="Number of samples to process.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    args = parser.parse_args()

    main(args.model_name, args.max_model_len, args.outputdir, args.numsamples, args.temperature)
