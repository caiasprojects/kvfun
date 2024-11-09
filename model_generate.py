import transformers
import torch
import time
import argparse

BASE_DEFAULT = "meta-llama/Llama-3.2-1B"
AUX_DEFAULT = "meta-llama/Llama-3.2-1B"


class KV_prediction_model:
    def __init__(self, base_model=BASE_DEFAULT, aux_model=AUX_DEFAULT):
        self.base_model_str = base_model
        self.aux_model_str = aux_model

        self.initialize()

    def initialize(self):
        aux_tokenizer = transformers.AutoTokenizer.from_pretrained(self.aux_model_str)
        aux_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.aux_model_str, torch_dtype=torch.bfloat16, device_map="auto"
        )

        base_tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_model_str)
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model_str, torch_dtype=torch.bfloat16, device_map="auto"
        )


class Baseline_model:
    def __init__(self, model_name=BASE_DEFAULT):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def run(self, messages, max_new_tokens=2048):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        prompt = self.tokenizer.apply_chat_template(messages)

        # no training being done
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model.to(device)

        tokenized_input = self.tokenizer.apply_chat_template(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )

        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device)

        start_time = time.time()

        # Run a forward pass and store the past key values (KV cache)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            logits = outputs.logits
            kv_cache = outputs.past_key_values

        ttft = time.time() - start_time

        # Display the KV cache for debugging
        print("KV Cache:")
        for i, layer_cache in enumerate(kv_cache):
            print(f"Layer {i+1}:")
            print("Keys:", layer_cache[0].shape)
            print("Values:", layer_cache[1].shape)

        # Generate text based on the prompt

        first_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        # print(f"{first_token_id=}")

        first_token = self.tokenizer.decode(first_token_id)
        print(f"{first_token=}")

        # Display TTFT and first token
        print(f"TTFT: {ttft:.4f} seconds")
        print(f"First token generated: '{first_token}'")

        new_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1), dtype=torch.long, device=device
                ),
            ],
            dim=1,
        )

        new_input_ids = torch.cat([input_ids, first_token_id.unsqueeze(0)], dim=-1)

        with torch.no_grad():
            generated_ids = self.model.generate(
                new_input_ids,
                attention_mask=new_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                past_key_values=kv_cache,  # Pass the cached KV states
            )

        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        print(f"Generated text: '{generated_text}'")
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
