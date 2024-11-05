import transformers
import torch

# Specify the model ID
model_id = "meta-llama/Llama-3.1-8B"

# Load the model and tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Define a function to generate text and print the KV cache
def generate_with_kv_cache(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Run a forward pass and store the past key values (KV cache)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        
    # Access the past key-value cache
    kv_cache = outputs.past_key_values

    # Display the KV cache for debugging
    print("KV Cache:")
    for i, layer_cache in enumerate(kv_cache):
        print(f"Layer {i+1}:")
        print("Keys:", layer_cache[0].shape)
        print("Values:", layer_cache[1].shape)
    
    # Generate text based on the prompt
    generated_ids = model.generate(inputs["input_ids"], max_length=100)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
if __name__ == "__main__":
    prompt = "Hey, how are you doing today?"
    response = generate_with_kv_cache(prompt)
    print("\nGenerated Text:")
    print(response)
