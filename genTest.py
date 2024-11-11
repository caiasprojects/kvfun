import torch
from functools import partial
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

import torchvision
torchvision.disable_beta_transforms_warning()

# Set up device and model
device = 'cuda:0'
torch.cuda.set_device(device)
torch.set_default_dtype(torch.bfloat16)

# Initialize tokenizer and auxiliary model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
aux_model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/CaiaMaiCode/KVFun/sft_output/checkpoint-3000").to(device)

# Load projection tensors for KV cache
import safetensors
tensors = {}
with safetensors.safe_open("/home/ubuntu/CaiaMaiCode/KVFun/projections.step3200.safetensors", framework="pt", device=device) as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Use specified prompt
sample_text = """Alice's Adventures in Wonderland is an 1865 novel by Lewis Carroll. It tells the story of a girl named Alice who falls through a rabbit hole into a fantasy world filled with strange and anthropomorphic creatures. It is an example of literary nonsense and has been widely influential in both literature and popular culture."""

# Tokenize sample text
input_ids = tokenizer(sample_text, return_tensors="pt").to(device)

# Initialize KV cache for prompt
prompt_cache = StaticCache(config=aux_model.config, batch_size=1, max_cache_len=3000, device=device, dtype=torch.bfloat16)

# KV hook for recording keys and values during generation
def kv_hook(module, in_x, output, index, kvs):
    if kvs.get(index) is None:
        kvs[index] = output

# Register hooks to capture KV cache
aux_ks, aux_vs = {}, {}
for i in range(16):
    aux_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index=i, kvs=aux_vs))
    aux_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index=i, kvs=aux_ks))

# Generate prompt cache with KV hook active
with torch.no_grad():
    prompt_cache = aux_model(**input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1, do_sample=True, top_k=50).past_key_values

# Prepare a prompt for testing continuation generation
new_prompt = " In this curious tale,"

# Generate with KV cache
new_inputs = tokenizer(sample_text + new_prompt, return_tensors="pt").to(device)
outputs_cached = aux_model.generate(**new_inputs, past_key_values=prompt_cache, max_new_tokens=500, temperature=0.5, do_sample=True, top_k=50, pad_token_id=tokenizer.eos_token_id)
response_cached = tokenizer.batch_decode(outputs_cached, skip_special_tokens=True)[0]
print("Generated output with KV cache:", response_cached)

# Generate without KV cache
outputs_no_cache = aux_model.generate(**new_inputs, max_new_tokens=500, temperature=0.5, do_sample=True, top_k=50, pad_token_id=tokenizer.eos_token_id)
response_no_cache = tokenizer.batch_decode(outputs_no_cache, skip_special_tokens=True)[0]
print("Generated output without KV cache:", response_no_cache)
