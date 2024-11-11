from transformers import AutoModelForCausalLM, DynamicCache, AutoTokenizer
import torch
import time
import argparse

BASE_DEFAULT = "meta-llama/Llama-3.1-8B"
AUX_DEFAULT = "meta-llama/Llama-3.2-1B"


def format_messages_for_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"System: {message['content']}\n\n"
        elif message["role"] == "user":
            prompt += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            prompt += f"Assistant: {message['content']}\n"
    prompt += "Assistant: "
    return prompt


class KV_prediction_model:
    def __init__(self, base_model=BASE_DEFAULT, aux_model=AUX_DEFAULT):
        self.base_model_str = base_model
        self.aux_model_str = aux_model

        self.initialize()

    def initialize(self):
        aux_tokenizer = AutoTokenizer.from_pretrained(self.aux_model_str)
        aux_model = AutoModelForCausalLM.from_pretrained(
            self.aux_model_str, torch_dtype=torch.bfloat16, device_map="auto"
        )

        base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_str)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_str, torch_dtype=torch.bfloat16, device_map="auto"
        )


class Baseline_model:
    def __init__(self, model_name=BASE_DEFAULT):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def run(self, messages, max_new_tokens=100):

        prompt = format_messages_for_prompt(messages)

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model.to(device)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        start_time = time.time()
        past_key_values = DynamicCache()
        # Run a forward pass and store the past key values (KV cache)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        ttft = time.time() - start_time

        # Display the KV cache for debugging
        # print("KV Cache:")
        # for i, layer_cache in enumerate(past_key_values):
        #     print(f"Layer {i+1}:")
        #     print("Keys:", layer_cache[0].shape)
        #     print("Values:", layer_cache[1].shape)

        # print(f"TTFT: {ttft:.4f} seconds")

        generated_text = ""

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=device,
                        ),
                    ],
                    dim=-1,
                )

                new_token = self.tokenizer.decode(next_token, skip_special_tokens=True)
                generated_text += new_token

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Check for repeating
                if "User:" in generated_text:
                    user_idx = generated_text.find("User:")
                    generated_text = generated_text[:user_idx]
                    break

        generated_text = generated_text.rstrip()
        print(f"Generated text: \n{generated_text}'")
        return generated_text, ttft

    def call_with_prompt(self, prompt, max_new_tokens=2048):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft = self.run(messages, max_new_tokens=max_new_tokens)

        return response, ttft


if __name__ == "__main__":

    prompt = "How do you make a cake?"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=BASE_DEFAULT, help="Name of the model to load"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate after the first token",
    )

    args = parser.parse_args()
    baseline_model = Baseline_model()
    response, ttft = baseline_model.call_with_prompt(prompt)
    print(response, ttft)
