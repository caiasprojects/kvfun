import transformers
import torch
import time
import argparse

# Specify the model ID
model_id = "meta-llama/Llama-3.1-8B"  # default if no arg was passed


# Define a function to generate text and print the KV cache
def generate_with_kv_cache(prompt, model_name, max_new_tokens):

    # Load the model and tokenizer
    print("loading")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_auth_token=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", use_auth_token=True
    )
    print("Done loading")

    # no training being done
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_time = time.time()

    # Run a forward pass and store the past key values (KV cache)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        print(outputs)
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
    print(f"{first_token_id=}")

    first_token = tokenizer.decode(first_token_id)
    print(f"{first_token=}")

    # Display TTFT and first token
    print(f"TTFT: {ttft:.4f} seconds")
    print(f"First token generated: '{first_token}'")

    input_ids = torch.cat([input_ids, first_token_id.unsqueeze(0)], dim=-1)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            past_key_values=kv_cache,  # Pass the cached KV states
        )


if __name__ == "__main__":

    prompt = """Please generate some more tokens of this: Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. 
    Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, 
    pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate 
    eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis pretium. Integer 
    tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, c
    onsequat vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla u
    t metus varius laoreet. Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. 
    Curabitur ullamcorper ultricies nisi. Nam eget dui. Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, 
    sem quam semper libero, sit amet adipiscing sem neque sed ipsum. Nam quam nunc, blandit vel, luctus pulvinar, hendrerit 
    id, lorem. Maecenas nec odio et ante tincidunt tempus. Donec vitae sapien ut libero venenatis faucibus. Nullam quis ante. 
    Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. Donec sodales sagittis magna. 
    Sed consequat, leo eget bibendum sodales, augue velit cursus nunc
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=model_id, help="Name of the model to load"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Number of new tokens to generate after the first token",
    )

    args = parser.parse_args()

    response = generate_with_kv_cache(
        prompt, args.model_name, max_new_tokens=args.max_new_tokens
    )

    print(f"Generated Text: \n {response}")
