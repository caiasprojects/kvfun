from transformers import AutoModelForCausalLM, DynamicCache, AutoTokenizer
import torch
import time
import argparse
import torch.utils.benchmark as benchmark
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from functools import partial
import copy

import matplotlib.pyplot as plt
import numpy as np

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


class KV_hybrid_model:
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

    def calculate_token_importance(
        self, input_ids, attention_matrices, percent_recalculate
    ):

        len_prompt = input_ids.shape[1]

        # print("attention_matrices", len(attention_matrices))
        # print("attention_matrices[0].shape", attention_matrices[0].shape)

        stacked_attention = torch.stack(
            attention_matrices
        )  # [num_layers, batch_size, num_heads, seq_len, seq_len]

        # Average across layers, batch, and heads
        avg_attention = stacked_attention.mean(dim=(0, 1, 2))  # [seq_len, seq_len]
        # print("avg_attention.shape", avg_attention.shape)

        token_importance = avg_attention.sum(dim=0)
        # print("token_importance.shape", token_importance.shape)
        # print("token_importance", token_importance)

        # k = int(percent_recalculate * len_prompt)
        k = 0
        # print("k", k)
        top_k_values, top_k_indices = torch.topk(token_importance, k=k)

        # always add last five tokens if not already in list
        tokens = list(range(0, 10))
        tokens.extend(list(range(len_prompt - 10, len_prompt)))
        extra_indices = []

        for token in tokens:
            if token not in top_k_indices:
                extra_indices.append(token)

        top_k_indices = torch.cat([top_k_indices, torch.tensor(extra_indices)])

        sorted_top_k_indices = torch.sort(top_k_indices)[0]
        # print("sorted_top_k_indices", sorted_top_k_indices)

        return sorted_top_k_indices

    def build_cache(self, input_ids, percent_recalculate=0.1):
        # Get auxiliary model cache first
        aux_cache = DynamicCache()
        with torch.no_grad():
            aux_outputs = self.aux_model(
                input_ids,
                use_cache=True,
                past_key_values=aux_cache,
                output_attentions=True,
            )
            aux_cache = aux_outputs.past_key_values
            attention_scores = aux_outputs.attentions

        # Calculate importance scores using attention patterns
        important_positions = self.calculate_token_importance(
            input_ids, attention_scores, percent_recalculate
        )

        token_groups = []
        current_group = []
        for i, pos in enumerate(important_positions):
            if not current_group or pos == current_group[-1] + 1:
                current_group.append(pos.item())
            else:
                token_groups.append(current_group)
                current_group = [pos.item()]
        if current_group:
            token_groups.append(current_group)

        print(token_groups)

        # print(f"Number of important tokens: {len(important_positions)}")

        # Initialize hybrid cache with projected aux states
        hybrid_cache = DynamicCache()
        for layer_idx in range(n_layers_base):
            v_len = self.aux_xs[layer_idx // 2].shape[1]
            key_state = (
                (
                    self.aux_xs[layer_idx // 2]
                    @ tensors[f"Lk.{layer_idx}"].to(torch.bfloat16)
                )
                .view(1, v_len, kvheads, head_dim_base)
                .transpose(1, 2)
            )

            value_state = (
                (
                    self.aux_xs[layer_idx // 2]
                    @ tensors[f"Lv.{layer_idx}"].to(torch.bfloat16)
                )
                .view(1, v_len, kvheads, head_dim_base)
                .transpose(1, 2)
            )
            hybrid_cache.update(
                key_states=key_state, value_states=value_state, layer_idx=layer_idx
            )

        # Process token groups
        current_cache = hybrid_cache
        # print("important_positions", important_positions)

        for group in token_groups:
            start_pos = group[0]
            end_pos = group[-1] + 1  # Add 1 to include the last token

            # Create input for token group
            token_input = input_ids[
                :, :end_pos
            ]  # Include all tokens up to end of group
            # print("token_input", token_input)
            # print("token_input.shape", token_input.shape)
            # print("input_ids.shape", input_ids.shape)
            # print("current_cache.key_cache[0].shape", current_cache.key_cache[0].shape)

            truncated_cache = DynamicCache()
            for layer_idx in range(n_layers_base):
                truncated_cache.update(
                    key_states=current_cache.key_cache[layer_idx][:, :, :start_pos, :],
                    value_states=current_cache.value_cache[layer_idx][
                        :, :, :start_pos, :
                    ],
                    layer_idx=layer_idx,
                )

            print(
                "truncated_cache.key_cache[0].shape", truncated_cache.key_cache[0].shape
            )

            # Forward pass with current cache
            with torch.no_grad():
                outputs = self.base_model(
                    token_input[
                        :, -len(group) :
                    ],  # Process only the tokens in the group
                    use_cache=True,
                    past_key_values=truncated_cache,
                )

                # Update cache with the new KV values for this group
                new_cache = outputs.past_key_values
                print("new_cache.key_cache[0].shape", new_cache.key_cache[0].shape)

                for layer_idx in range(n_layers_base):
                    new_keys = new_cache.key_cache[layer_idx][:, :, -len(group) :, :]
                    new_values = new_cache.value_cache[layer_idx][
                        :, :, -len(group) :, :
                    ]

                    hybrid_cache.key_cache[layer_idx][
                        :, :, start_pos:end_pos, :
                    ] = new_keys
                    hybrid_cache.value_cache[layer_idx][
                        :, :, start_pos:end_pos, :
                    ] = new_values

                current_cache = hybrid_cache

        return hybrid_cache

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


if __name__ == "__main__":
    # prompt = "France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible. France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible. France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible. France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible. France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible. France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible.France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible.France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible.France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible.France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible.France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars. Question:  When did French Republicans back building the English empire?. Answer in as few words as possible."

    prompt = "Briefly answer this. My name is John, what is my name?"
    baseline_model = Baseline_model(BASE_DEFAULT)
    response, ttft, base_cache, prompt_len = baseline_model.call_with_prompt(
        prompt, max_new_tokens=100
    )
    print("prompt_len", prompt_len)
    print("Baseline model", response, ttft)
    del baseline_model

    # kv_model = KV_prediction_model()
    # kv_response, kv_ttft, kv_cache, _ = kv_model.call_with_prompt(
    #     prompt, max_new_tokens=100
    # )
    # print("KV model", kv_response, kv_ttft)
    # plot_kv_differences(
    #     kv_cache,
    #     base_cache,
    #     prompt_len,
    #     save_path="kv_differences/kv_base.png",
    # )
    # del kv_model

    # kv_model_with_copy = KV_prediction_model_with_copy()
    # kv_with_copy_response, kv_with_copy_ttft, kv_copy_cache, _ = (
    #     kv_model_with_copy.call_with_prompt(prompt, max_new_tokens=100)
    # )
    # print("KV with copy: ", kv_with_copy_response, kv_with_copy_ttft)
    # plot_kv_differences(
    #     kv_copy_cache,
    #     base_cache,
    #     prompt_len,
    #     save_path="kv_differences/copy_base.png",
    # )
    # del kv_model_with_copy

    kv_hybrid_model = KV_hybrid_model()
    kv_hybrid_response, kv_hybrid_ttft, kv_hybrid_cache, prompt_len = (
        kv_hybrid_model.call_with_prompt(prompt, max_new_tokens=100)
    )
    print("KV hybrid: ", kv_hybrid_response, kv_hybrid_ttft)
    plot_kv_differences(
        kv_hybrid_cache,
        base_cache,
        prompt_len,
        save_path="kv_differences/hybrid_base.png",
    )
    del kv_hybrid_model
