import transformers
import torch
import time
import argparse

# Specify the model ID
# model_id = "meta-llama/Llama-3.1-8B"  # default if no arg was passed
model_id = "meta-llama/Llama-3.2-1B"


# Define a function to generate text and print the KV cache
def generate_with_kv_cache(prompt, model_name, max_new_tokens):

    # Load the model and tokenizer
    print("loading")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print("Done loading")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # no training being done
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    tokenized_input = tokenizer(
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
        outputs = model(
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

    first_token = tokenizer.decode(first_token_id)
    print(f"{first_token=}")

    # Display TTFT and first token
    print(f"TTFT: {ttft:.4f} seconds")
    print(f"First token generated: '{first_token}'")

    new_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device),
        ],
        dim=1,
    )

    new_input_ids = torch.cat([input_ids, first_token_id.unsqueeze(0)], dim=-1)

    with torch.no_grad():
        generated_ids = model.generate(
            new_input_ids,
            attention_mask=new_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            past_key_values=kv_cache,  # Pass the cached KV states
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text: '{generated_text}'")
    return generated_text


if __name__ == "__main__":

    prompt = "Hello how do I bake a cake?"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=model_id, help="Name of the model to load"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate after the first token",
    )

    args = parser.parse_args()

    response = generate_with_kv_cache(
        prompt, args.model_name, max_new_tokens=args.max_new_tokens
    )
