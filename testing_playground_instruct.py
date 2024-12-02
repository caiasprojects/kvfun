import os
import torch
from functools import partial
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import copy
import matplotlib.pyplot as plt
import numpy as np

# torchvision.disable_beta_transforms_warning()
import safetensors
from transformers import StaticCache, DynamicCache
from safetensors.torch import save_file

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda:0"

torch.cuda.set_device(device)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(device)

# load models
aux_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    aux_model.config.pad_token_id = aux_model.config.eos_token_id

# load tensors
repo_id = "caiacost/matrix-fun"
filename = "Instruct-8b-1b1projections.3968.New.safetensors"
file_path = hf_hub_download(repo_id=repo_id, filename=filename)
tensors = load_file(file_path)

# sample prompt
sample = "My name is John. What is my name?"
messages = [
    {"role": "user", "content": sample},
]

# fill input ids
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    device="cuda:0",
)

len_prompt = input_ids.shape[1]

# predicted aux model cache
prompt_cache_aux = DynamicCache()

# Constants
aux_xs = {}
n_layers_base = 32
n_layers_aux = 16
aux_dim = 2048
base_dim = 4096
aux_kv_dim = 512
base_kv_dim = 1024
cache_len = 3000
kvheads = 8
head_dim_base = 128


# hooks to get aux activations
def x_hook(module, in_x, output, index, xs):
    if xs.get(index) == None:
        xs[index] = in_x[0]


for i in range(n_layers_aux):
    aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(
        partial(x_hook, index=i, xs=aux_xs)
    )

#### Fill prompt_cache_aux by calling small model
with torch.no_grad():
    prompt_cache_aux = aux_model(
        input_ids,
        use_cache=True,
        past_key_values=prompt_cache_aux,
    ).past_key_values

print(
    "prompt_cache_aux",
    len(prompt_cache_aux.key_cache),
    prompt_cache_aux.key_cache[0].shape,
)


# helper function to generate text autoregressively with a given cache
def generate_text(model, input_ids, cache, tokenizer, max_new_tokens=100):
    generated_text = ""

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=input_ids[:, -1:],
                past_key_values=cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            new_token = tokenizer.decode(next_token, skip_special_tokens=True)
            generated_text += new_token
    return generated_text.rstrip(), past_key_values


past_key_values_aux = copy.deepcopy(prompt_cache_aux)

generated_text, past_key_values_aux = generate_text(
    aux_model, input_ids, past_key_values_aux, tokenizer
)

print("output small model :", generated_text)

# predicted base model cache
base_prompt_cache = DynamicCache()

# real base model cache
base_prompt_cache_real = DynamicCache()

# fill input ids (not needed, but just in case)
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    device=device,
)
print("input_ids", len(input_ids[0]))


# fill real base model cache
with torch.no_grad():
    base_prompt_cache_real = base_model(
        input_ids,
        use_cache=True,
        past_key_values=base_prompt_cache_real,
    ).past_key_values

# fill predicted base model cache
for i in range(n_layers_base):
    if i == 0:
        print("shapes baseL")
        print("aux_xs[i//2].shape", aux_xs[i // 2].shape)
        print("tensors['Lk.' +str(i)].shape", tensors["Lk." + str(i)].shape)

    v_len = aux_xs[i // 2].shape[1]

    key_state = (
        (aux_xs[i // 2] @ tensors["Lk." + str(i)].to(torch.bfloat16))
        .view(1, v_len, kvheads, head_dim_base)
        .transpose(1, 2)
    )
    value_state = (
        (aux_xs[i // 2] @ tensors["Lv." + str(i)].to(torch.bfloat16))
        .view(1, v_len, kvheads, head_dim_base)
        .transpose(1, 2)
    )

    # deprecated since we use dynamic cache
    # base_prompt_cache.key_cache[i][:, :, :v_len, :] = key_state
    # base_prompt_cache.value_cache[i][:, :, :v_len, :] = value_state

    base_prompt_cache.update(
        key_states=key_state, value_states=value_state, layer_idx=i
    )

    # copy random stuff
    # copy_last = 2510
    # base_prompt_cache.key_cache[i][:, :, copy_last:, :] = (
    #     base_prompt_cache_real.key_cache[i][:, :, copy_last:, :]
    # )
    # base_prompt_cache.value_cache[i][:, :, copy_last:, :] = (
    #     base_prompt_cache_real.value_cache[i][:, :, copy_last:, :]
    # )

    # copy_first = 15

    # base_prompt_cache.key_cache[i][:, :, :copy_first, :] = (
    #     base_prompt_cache_real.key_cache[i][:, :, :copy_first, :]
    # )
    # base_prompt_cache.value_cache[i][:, :, :copy_first, :] = (
    #     base_prompt_cache_real.value_cache[i][:, :, :copy_first, :]
    # )

    # copy_start = 40
    # copy_end = 44

    # base_prompt_cache.key_cache[i][:, :, copy_start:copy_end, :] = (
    #     base_prompt_cache_real.key_cache[i][:, :, copy_start:copy_end, :]
    # )
    # base_prompt_cache.value_cache[i][:, :, copy_start:copy_end, :] = (
    #     base_prompt_cache_real.value_cache[i][:, :, copy_start:copy_end, :]
    # )

    # copy_last = 50
    # base_prompt_cache.key_cache[i][:, :, -copy_last:, :] = (
    #     base_prompt_cache_real.key_cache[i][:, :, -copy_last:, :]
    # )
    # base_prompt_cache.value_cache[i][:, :, -copy_last:, :] = (
    #     base_prompt_cache_real.value_cache[i][:, :, -copy_last:, :]
    # )


og_input_ids = input_ids


# See what tokens are per the input
for i in range(og_input_ids.shape[1]):
    token_id = input_ids[0, i].item()
    decoded = tokenizer.decode(token_id)
    print(f"Token {i}: {token_id} -> {decoded}")

# generate text with predicted cache
past_key_values = copy.deepcopy(base_prompt_cache)
generated_text, past_key_values = generate_text(
    base_model, input_ids, past_key_values, tokenizer
)
print("output big model :", generated_text)


def plot_kv_differences(
    base_prompt_cache,
    base_prompt_cache_real,
    seq_start=0,
    seq_end=len_prompt,
    save_path="kv_differences/kv_differences.png",
):
    # Calculate differences for keys and values
    key_differences = []
    value_differences = []

    for layer in range(n_layers_base):
        # L1 error for keys - convert to float32 before numpy
        key_diff = (
            torch.abs(
                base_prompt_cache.key_cache[layer][:, :, seq_start:seq_end, :]
                - base_prompt_cache_real.key_cache[layer][:, :, seq_start:seq_end, :]
            )
            .mean(dim=(0, 1, 3))
            .float()  # Convert to float32
            .cpu()
            .numpy()
        )

        # L1 error for values - convert to float32 before numpy
        value_diff = (
            torch.abs(
                base_prompt_cache.value_cache[layer][:, :, seq_start:seq_end, :]
                - base_prompt_cache_real.value_cache[layer][:, :, seq_start:seq_end, :]
            )
            .mean(dim=(0, 1, 3))
            .float()  # Convert to float32
            .cpu()
            .numpy()
        )

        key_differences.append(key_diff)
        value_differences.append(value_diff)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot key differences
    im1 = ax1.imshow(key_differences, aspect="auto", cmap="magma", origin="lower")
    ax1.set_title("Predicted Key L1 Error")
    ax1.set_xlabel("Sequence Idx")
    ax1.set_ylabel("Layer Idx")
    plt.colorbar(im1, ax=ax1)

    # Plot value differences
    im2 = ax2.imshow(value_differences, aspect="auto", cmap="magma", origin="lower")
    ax2.set_title("Predicted Value L1 Error")
    ax2.set_xlabel("Sequence Idx")
    ax2.set_ylabel("Layer Idx")
    plt.colorbar(im2, ax=ax2)

    # Set proper sequence indices
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(0, seq_end - seq_start, 4))
        ax.set_xticklabels(range(seq_start, seq_end, 4))

    plt.tight_layout()
    print("saving to", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Call the function with a save path
plot_kv_differences(base_prompt_cache, base_prompt_cache_real)
