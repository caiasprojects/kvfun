from transformers import AutoModelForCausalLM, DynamicCache, AutoTokenizer
import torch
import time
import argparse
import torch.utils.benchmark as benchmark
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

        k = int(percent_recalculate * len_prompt)
        # k = 10
        # print("k", k)
        top_k_values, top_k_indices = torch.topk(token_importance, k=k)

        # always add last five tokens if not already in list
        tokens = list(range(0, 5))
        tokens.extend(list(range(len_prompt - 6, len_prompt)))
        extra_indices = []

        for token in tokens:
            if token not in top_k_indices:
                extra_indices.append(token)

        top_k_indices = torch.cat([top_k_indices, torch.tensor(extra_indices)])

        sorted_top_k_indices = torch.sort(top_k_indices)[0]
        # print("sorted_top_k_indices", sorted_top_k_indices)

        return sorted_top_k_indices

    def build_cache(self, input_ids, percent_recalculate=0.5):
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

        # Process important tokens one by one
        current_cache = hybrid_cache
        # print("important_positions", important_positions)

        for pos in important_positions:
            # print("pos", pos)
            # Create input for single token
            token_input = input_ids[
                :, : pos + 1
            ]  # Include all tokens up to this position
            # print("token_input.shape", token_input.shape)

            # Forward pass with current cache
            with torch.no_grad():
                outputs = self.base_model(
                    token_input[:, -1:],  # Only process the last token
                    use_cache=True,
                    past_key_values=current_cache,
                )

                # Update cache with the new KV values for this token
                new_cache = outputs.past_key_values

                for layer_idx in range(n_layers_base):
                    # print(
                    #     "new_cache.key_cache[layer_idx].shape",
                    #     new_cache.key_cache[layer_idx].shape,
                    # )

                    new_key = new_cache.key_cache[layer_idx][
                        :, :, -1:, :
                    ]  # Take only the last position

                    # print("new_key.shape", new_key.shape)
                    new_value = new_cache.value_cache[layer_idx][:, :, -1:, :]

                    hybrid_cache.key_cache[layer_idx][:, :, pos : pos + 1, :] = new_key
                    hybrid_cache.value_cache[layer_idx][
                        :, :, pos : pos + 1, :
                    ] = new_value

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

        # input_ids_base = self.base_tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        # ).to(device)

        generated_text, past_key_values = generate_text(
            self.base_model,
            input_ids,
            past_key_values,
            self.base_tokenizer,
            max_new_tokens,
        )
        return generated_text, ttft

    def call_with_prompt(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft = self.run(messages, max_new_tokens=max_new_tokens)

        return response, ttft


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
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.aux_model(
                input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

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

        return generated_text, ttft

    def call_with_prompt(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft = self.run(messages, max_new_tokens=max_new_tokens)

        return response, ttft


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
        return generated_text, ttft

    def call_with_prompt(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft = self.run(messages, max_new_tokens=max_new_tokens)

        return response, ttft


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

        return generated_text, ttft

    def call_with_prompt(self, prompt, max_new_tokens=100):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft = self.run(messages, max_new_tokens=max_new_tokens)

        return response, ttft


if __name__ == "__main__":
    prompt = "My name is John. What is my name?"

    baseline_model = Baseline_model(BASE_DEFAULT)
    response, ttft = baseline_model.call_with_prompt(prompt, max_new_tokens=100)
    print("Baseline model", response, ttft)
    del baseline_model

    # kv_model = KV_prediction_model()
    # kv_response, kv_ttft = kv_model.call_with_prompt(prompt, max_new_tokens=100)
    # print("KV model", kv_response, kv_ttft)
    # del kv_model

    # kv_model_with_copy = KV_prediction_model_with_copy()
    # kv_with_copy_response, kv_with_copy_ttft = kv_model_with_copy.call_with_prompt(
    #     prompt, max_new_tokens=100
    # )
    # print("KV with copy: ", kv_with_copy_response, kv_with_copy_ttft)
    # del kv_model_with_copy

    kv_hybrid_model = KV_hybrid_model()
    kv_hybrid_response, kv_hybrid_ttft = kv_hybrid_model.call_with_prompt(
        prompt, max_new_tokens=100
    )
    print("KV hybrid: ", kv_hybrid_response, kv_hybrid_ttft)
    del kv_hybrid_model
