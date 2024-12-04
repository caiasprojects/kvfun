import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.benchmark as benchmark
from transformers import AutoModelForCausalLM, DynamicCache, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from functools import partial
import copy

BASE_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
AUX_DEFAULT = "meta-llama/Llama-3.2-1B-Instruct"

device = "cuda:0"
torch.cuda.set_device(device)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(device)

n_layers_base = 32
n_layers_aux = 16
# aux_dim = 2048
# base_dim = 4096
# aux_kv_dim = 512
# base_kv_dim = 1024
# cache_len = 3000
kvheads = 8
head_dim_base = 128

# load tensors
repo_id = "caiacost/matrix-fun"
filename = "Instruct-8b-1b1projections.3968.New.safetensors"
file_path = hf_hub_download(repo_id=repo_id, filename=filename)
tensors = load_file(file_path)


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


def plot_kv_differences(
    base_prompt_cache,
    base_prompt_cache_real,
    prompt_len,
    save_path="kv_differences/kv_differences.png",
):
    # Calculate differences for keys and values
    key_differences = []
    value_differences = []

    seq_start = 0
    seq_end = prompt_len

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


# positions expected to be sorted
def positions_to_intervals(positions):
    intervals = []
    if not positions:
        return intervals

    start = positions[0]
    prev = start

    for pos in positions[1:]:
        if pos != prev + 1:
            # End of current interval, start new one
            intervals.append((start, prev))
            start = pos
        prev = pos

    # Add the last interval
    intervals.append((start, prev))

    return intervals


def select_important_intervals(
    token_importance, interval_size=10, num_intervals=3, offset=0
):
    """
    Creates intervals around the most important tokens while avoiding overlap.

    Args:
        token_importance: List of importance scores for each token
        interval_size: Number of tokens to include on each side of the important token
        num_intervals: Number of intervals to create

    Returns:
        List of tuples containing (start, end) positions for each interval
    """
    # Convert to list if not already
    scores = list(token_importance)
    intervals = []
    seq_len = len(scores)
    interval_size = interval_size // 2

    for _ in range(num_intervals):
        # Find the highest remaining score
        max_idx = scores.index(max(scores))

        # Calculate interval boundaries
        start = max(0, max_idx - interval_size)
        end = min(seq_len, max_idx + interval_size)

        # Add the interval
        intervals.append((start, end))

        # Mask out the tokens in this interval to avoid overlap
        for i in range(start, end):
            scores[i] = float("-inf")

    # Sort intervals by start position
    intervals = [(start + offset, end + offset) for start, end in intervals]
    return sorted(intervals)


class KV_prediction_model_with_copy:
    def __init__(self, base_model=BASE_DEFAULT, aux_model=AUX_DEFAULT):
        self.base_model_str = base_model
        self.aux_model_str = aux_model

        # base aux configs
        self.aux_xs = {}

        self.initialize()

    def initialize(self):

        # ------ initialie aux -------
        self.aux_model = AutoModelForCausalLM.from_pretrained(
            self.aux_model_str, torch_dtype=torch.bfloat16, device_map=device
        )

        self.aux_model.eval()
        self.aux_model.to(device)

        # set hooks to get activations for projections
        def x_hook(module, in_x, output, index, xs):
            if xs.get(index) == None:
                xs[index] = in_x[0]

        for i in range(n_layers_aux):
            self.aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(
                partial(x_hook, index=i, xs=self.aux_xs)
            )

        # ------- initialize base ------
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_str)
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_str)

        self.base_model.eval()
        self.base_model.to(device)

        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id

    def build_cache(self, input_ids):

        # process prompt with aux
        aux_prompt_cache = DynamicCache()
        with torch.no_grad():
            outputs = self.aux_model(
                input_ids,
                use_cache=True,
                past_key_values=aux_prompt_cache,
            )
            aux_prompt_cache = outputs.past_key_values

        base_prompt_cache_real = DynamicCache()
        with torch.no_grad():
            base_prompt_cache_real = self.base_model(
                input_ids,
                use_cache=True,
                past_key_values=base_prompt_cache_real,
            ).past_key_values

        # predicted cache
        predicted_cache_base = DynamicCache()

        # go through each layer of the base
        for i in range(n_layers_base):

            v_len = self.aux_xs[i // 2].shape[1]

            # get keys for layer
            key_state = (
                (self.aux_xs[i // 2] @ tensors[f"Lk.{i}"].to(torch.bfloat16))
                .view(1, v_len, kvheads, head_dim_base)
                .transpose(1, 2)
            )

            # get values
            value_state = (
                (self.aux_xs[i // 2] @ tensors["Lv." + str(i)].to(torch.bfloat16))
                .view(1, v_len, kvheads, head_dim_base)
                .transpose(1, 2)
            )

            predicted_cache_base.update(
                key_states=key_state, value_states=value_state, layer_idx=i
            )

            copy_first = 5

            predicted_cache_base.key_cache[i][:, :, :copy_first, :] = (
                base_prompt_cache_real.key_cache[i][:, :, :copy_first, :]
            )
            predicted_cache_base.value_cache[i][:, :, :copy_first, :] = (
                base_prompt_cache_real.value_cache[i][:, :, :copy_first, :]
            )

            # copy_start = 40
            # copy_end = 44

            # predicted_cache_base.key_cache[i][:, :, copy_start:copy_end, :] = (
            #     base_prompt_cache_real.key_cache[i][:, :, copy_start:copy_end, :]
            # )
            # predicted_cache_base.value_cache[i][:, :, copy_start:copy_end, :] = (
            #     base_prompt_cache_real.value_cache[i][:, :, copy_start:copy_end, :]
            # )

            copy_last = 5
            predicted_cache_base.key_cache[i][:, :, -copy_last:, :] = (
                base_prompt_cache_real.key_cache[i][:, :, -copy_last:, :]
            )
            predicted_cache_base.value_cache[i][:, :, -copy_last:, :] = (
                base_prompt_cache_real.value_cache[i][:, :, -copy_last:, :]
            )

        past_key_values = copy.deepcopy(predicted_cache_base)
        return past_key_values

    def run(self, messages, max_new_tokens=100, ttft_k=1):

        # tokenize prompt
        input_ids = self.base_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)

        past_key_values = self.build_cache(input_ids)

        stmt = """
                past_key_values = self.build_cache(input_ids)   
            """

        t = benchmark.Timer(
            stmt=stmt,
            globals={
                "self": self,
                "input_ids": input_ids,
            },
        )

        ttft = t.timeit(ttft_k)
        ttft = ttft.mean

        generated_text, past_key_values = generate_text(
            self.base_model,
            input_ids,
            past_key_values,
            self.base_tokenizer,
            max_new_tokens,
        )

        return generated_text, ttft, past_key_values, input_ids.shape[1]

    def call_with_prompt(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft, kv_cache, prompt_len = self.run(
            messages, max_new_tokens=max_new_tokens
        )

        return response, ttft, kv_cache, prompt_len


class KV_prediction_model:
    def __init__(self, base_model=BASE_DEFAULT, aux_model=AUX_DEFAULT):
        self.base_model_str = base_model
        self.aux_model_str = aux_model

        # base aux configs
        self.aux_xs = {}

        self.initialize()

    def initialize(self):

        # ------ initialie aux -------
        self.aux_model = AutoModelForCausalLM.from_pretrained(self.aux_model_str)

        self.aux_model.eval()
        self.aux_model.to(device)

        # set hooks to get activations for projections
        def x_hook(module, in_x, output, index, xs):
            if xs.get(index) == None:
                xs[index] = in_x[0]

        for i in range(n_layers_aux):
            self.aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(
                partial(x_hook, index=i, xs=self.aux_xs)
            )

        # ------- initialize base ------
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_str)
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_str)

        self.base_model.eval()
        self.base_model.to(device)

        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id

    def build_cache(self, input_ids):

        # process prompt with aux
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.aux_model(
                input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        # predicted cache
        predicted_cache_base = DynamicCache()

        # go through each layer of the base
        for i in range(n_layers_base):

            v_len = self.aux_xs[i // 2].shape[1]

            key_state = (
                (self.aux_xs[i // 2] @ tensors["Lk." + str(i)].to(torch.bfloat16))
                .view(1, v_len, kvheads, head_dim_base)
                .transpose(1, 2)
            )

            value_state = (
                (self.aux_xs[i // 2] @ tensors["Lv." + str(i)].to(torch.bfloat16))
                .view(1, v_len, kvheads, head_dim_base)
                .transpose(1, 2)
            )

            predicted_cache_base.update(
                key_states=key_state, value_states=value_state, layer_idx=i
            )

        past_key_values = copy.deepcopy(predicted_cache_base)
        return past_key_values

    def run(self, messages, max_new_tokens=100, ttft_k=1):

        # tokenize prompt
        input_ids = self.base_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)

        past_key_values = self.build_cache(input_ids)

        stmt = """
                past_key_values = self.build_cache(input_ids)   
            """

        t = benchmark.Timer(
            stmt=stmt,
            globals={
                "self": self,
                "input_ids": input_ids,
            },
        )

        ttft = t.timeit(ttft_k)
        ttft = ttft.mean

        generated_text, past_key_values = generate_text(
            self.base_model,
            input_ids,
            past_key_values,
            self.base_tokenizer,
            max_new_tokens,
        )
        return generated_text, ttft, past_key_values, input_ids.shape[1]

    def call_with_prompt(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft, kv_cache, prompt_len = self.run(
            messages, max_new_tokens=max_new_tokens
        )

        return response, ttft, kv_cache, prompt_len


class Baseline_model:
    def __init__(self, model_name=BASE_DEFAULT):

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def build_cache(self, input_ids):
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, use_cache=True, past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
        return past_key_values

    def run(self, messages, max_new_tokens=1024, ttft_k=1):

        self.model.eval()

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)

        for i in range(input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            decoded = self.tokenizer.decode(token_id)
            print(f"Token {i}: {token_id} -> {decoded}")

        past_key_values = self.build_cache(input_ids)

        stmt = """
                past_key_values = self.build_cache(input_ids)   
            """

        t = benchmark.Timer(
            stmt=stmt,
            globals={
                "self": self,
                "input_ids": input_ids,
            },
        )

        ttft = t.timeit(ttft_k)
        ttft = ttft.mean
        # print("TTFT ", ttft)

        generated_text, _ = generate_text(
            self.model, input_ids, past_key_values, self.tokenizer
        )

        return generated_text, ttft, past_key_values, input_ids.shape[1]

    def call_with_prompt(self, prompt, max_new_tokens=100):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft, kv_cache, prompt_len = self.run(
            messages, max_new_tokens=max_new_tokens
        )

        return response, ttft, kv_cache, prompt_len
