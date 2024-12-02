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
sample = "Respond very briefly: My name is John. What is my name?"
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

    # copy_first = 5

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

    # copy_last = 3
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


# TODO: Right now, earlier tokens will have higher importance scores. since
# they appear more. We need to normalize for this. But if we do it by position,
# then the later tokens will have higher scores. Etc. Actually a hard problem lol
def calculate_token_importance(attention_matrices, percent_recalculate):
    # attention_matrices is a tuple of tensors, one per layer
    # Each tensor has shape [batch_size, num_heads, seq_len, seq_len]

    # Stack all layers' attention matrices

    print("attention_matrices", len(attention_matrices))
    print("attention_matrices[0].shape", attention_matrices[0].shape)

    stacked_attention = torch.stack(
        attention_matrices
    )  # [num_layers, batch_size, num_heads, seq_len, seq_len]

    # Average across layers, batch, and heads
    avg_attention = stacked_attention.mean(dim=(0, 1, 2))  # [seq_len, seq_len]
    print("avg_attention.shape", avg_attention.shape)

    token_importance = avg_attention.sum(dim=0)
    print("token_importance.shape", token_importance.shape)
    print("token_importance", token_importance)

    k = int(percent_recalculate * len_prompt)
    print("k", k)
    top_k_values, top_k_indices = torch.topk(token_importance, k=k)

    # always add last five tokens if not already in list
    last_five_tokens = torch.tensor(
        [len_prompt - 1, len_prompt - 2, len_prompt - 3, len_prompt - 4, len_prompt - 5]
    )

    for token in last_five_tokens:
        if token not in top_k_indices:
            top_k_indices = torch.cat([top_k_indices, token.unsqueeze(0)])

    sorted_top_k_indices = torch.sort(top_k_indices)[0]
    print("sorted_top_k_indices", sorted_top_k_indices)

    return sorted_top_k_indices


# TODO:
# Here are my notes for me. Very imporant okay.

# curreently my impleentation is slower. Too many forward passes.
# as well, the mechanism for recomputation is not correct.
# it does it layer by layer, but it should be doing it token by token.
# you should see black lines in the kv plots.
# Lets say you have a list of 10 tokens you want to recompute.
# There are two ways to do this. You do a forward pass with these 10 tokens.
# and then fit the kv cache of these 10 tokens back into the hybrid cache.
# I dont think this is a good approach. It'll work functionally, but it can be
# wrong.
# another way is to do the forward pass up to the first token with the aux model.
# then do one more token (auto regressively, so using the cahce you jsut built) for the next token.
# then again aux model to the next token.

# the other way is a full restructing of the model. Maybe look at the homework implementation of kv cache.
# basically you would be told to compute the cahce of the prompt, but you already have a predicted one.
# you would check if you have to do

# This also works. But is random
# def calculate_attention_scores():
#     # Create random importance scores for each token in the sequence
#     sequence_length = len_prompt  # Using the global len_prompt variable
#     token_importance = torch.rand(sequence_length, device=device)
#     return token_importance


# so this attention is wrong because it does key matmul with key.
# it works, but it's not the right way to do it.
# def calculate_attention_scores(cache, layer_idx, seq_len):
#     """Calculate attention scores for the sequence"""
#     # Get key and value states for the layer
#     key_states = cache.key_cache[layer_idx]  # [1, num_heads, seq_len, head_dim]
#     value_states = cache.value_cache[layer_idx]

#     # Calculate attention scores
#     attention_scores = torch.matmul(key_states, key_states.transpose(-1, -2))
#     attention_scores = attention_scores / math.sqrt(head_dim_base)
#     attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)

#     # Average across heads
#     importance_scores = attention_scores.mean(dim=1).squeeze(0)  # [seq_len, seq_len]

#     # Get maximum attention received by each token
#     token_importance = importance_scores.max(dim=0)[0]  # [seq_len]
#     return token_importance


def create_hybrid_cache(aux_model, base_model, input_ids, percent_recalculate=0.5):
    # Get auxiliary model cache first
    aux_cache = DynamicCache()
    with torch.no_grad():
        aux_outputs = aux_model(
            input_ids,
            use_cache=True,
            past_key_values=aux_cache,
            output_attentions=True,
        )
        aux_cache = aux_outputs.past_key_values
        attention_scores = aux_outputs.attentions

    # Calculate importance scores using attention patterns
    important_positions = calculate_token_importance(
        attention_scores, percent_recalculate
    )

    print(f"Number of important tokens: {len(important_positions)}")

    # Initialize hybrid cache with projected aux states
    hybrid_cache = DynamicCache()
    for layer_idx in range(n_layers_base):
        v_len = aux_xs[layer_idx // 2].shape[1]
        key_state = (
            (aux_xs[layer_idx // 2] @ tensors[f"Lk.{layer_idx}"].to(torch.bfloat16))
            .view(1, v_len, kvheads, head_dim_base)
            .transpose(1, 2)
        )
        value_state = (
            (aux_xs[layer_idx // 2] @ tensors[f"Lv.{layer_idx}"].to(torch.bfloat16))
            .view(1, v_len, kvheads, head_dim_base)
            .transpose(1, 2)
        )
        hybrid_cache.update(
            key_states=key_state, value_states=value_state, layer_idx=layer_idx
        )

    # Process important tokens one by one
    current_cache = hybrid_cache
    print("important_positions", important_positions)

    for pos in important_positions:
        print("pos", pos)
        # Create input for single token
        token_input = input_ids[:, : pos + 1]  # Include all tokens up to this position
        print("token_input.shape", token_input.shape)

        # Forward pass with current cache
        with torch.no_grad():
            outputs = base_model(
                token_input[:, -1:],  # Only process the last token
                use_cache=True,
                past_key_values=current_cache,
            )

            # Update cache with the new KV values for this token
            new_cache = outputs.past_key_values

            for layer_idx in range(n_layers_base):

                new_key = new_cache.key_cache[layer_idx][
                    :, :, -1:, :
                ]  # Take only the last position

                new_value = new_cache.value_cache[layer_idx][:, :, -1:, :]

                hybrid_cache.key_cache[layer_idx][:, :, pos : pos + 1, :] = new_key
                hybrid_cache.value_cache[layer_idx][:, :, pos : pos + 1, :] = new_value

            current_cache = hybrid_cache

    return hybrid_cache


final_cache = create_hybrid_cache(aux_model, base_model, input_ids)

generated_text, final_cache = generate_text(
    base_model, input_ids, final_cache, tokenizer
)
print("output hybrid model :", generated_text)

plot_kv_differences(
    base_prompt_cache_real,
    final_cache,
    save_path="kv_differences/hybrid_kv_differences.png",
)
