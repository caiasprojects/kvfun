from transformers import AutoModelForCausalLM, DynamicCache, AutoTokenizer
import torch
import time
import argparse
import torch.utils.benchmark as benchmark
from functools import partial
import copy

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    plot_kv_differences,
    positions_to_intervals,
    generate_text,
    select_important_intervals,
    tensors,
    BASE_DEFAULT,
    AUX_DEFAULT,
    device,
    Baseline_model,
    KV_prediction_model_with_copy,
    KV_prediction_model,
)

BASELINE = False
n_layers_base = 32
n_layers_aux = 16
# aux_dim = 2048
# base_dim = 4096
# aux_kv_dim = 512
# base_kv_dim = 1024
# cache_len = 3000
kvheads = 8
head_dim_base = 128


class KV_hybrid_model:
    def __init__(
        self,
        base_model=BASE_DEFAULT,
        aux_model=AUX_DEFAULT,
        baseline_base=False,
        baseline_aux=False,
    ):
        self.base_model_str = base_model
        self.aux_model_str = aux_model
        self.baseline_base = baseline_base
        self.baseline_aux = baseline_aux

        # base aux configs

        self.initialize()

    def initialize(self):

        # ------ initialie aux -------
        self.aux_model = AutoModelForCausalLM.from_pretrained(
            self.aux_model_str,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )

        self.aux_model.eval()
        self.aux_model.to(device)

        # ------- initialize base ------
        if self.baseline_aux:
            print("Will use aux model on inference (generate_text)")

        if self.baseline_base:
            print("Will use base on aux creation")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_str,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )

        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_str)

        self.base_model.eval()
        self.base_model.to(device)

        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id

    # set hooks to get activations for projections
    def x_hook(self, module, in_x, output, index, xs):
        if xs.get(index) == None:
            xs[index] = in_x[0]

    def calculate_token_importance(
        self, input_ids, attention_matrices, interval_size=20, num_intervals=2
    ):

        len_prompt = input_ids.shape[1]

        stacked_attention = torch.stack(attention_matrices)

        # Average across layers, batch, and heads
        avg_attention = stacked_attention.mean(dim=(0, 1, 2))  # [seq_len, seq_len]

        token_importance = avg_attention.sum(dim=0)

        # Add the first 10 and last 10 percent of tokens for recalculation
        ten_percent = len_prompt // 10
        token_importance = token_importance[ten_percent:-ten_percent]

        intervals = select_important_intervals(
            token_importance.tolist(),
            interval_size=interval_size,
            num_intervals=num_intervals,
            offset=ten_percent,
        )

        full_recalculate = (
            [(0, ten_percent)] + intervals + [(len_prompt - ten_percent, len_prompt)]
        )

        merged_recompute_intervals = []
        current_start, current_end = full_recalculate[0]

        # Merge intervals if they overlap
        for start, end in full_recalculate[1:]:
            if start <= current_end + 1:
                # Extend current interval
                current_end = max(current_end, end)
            else:
                # Add current interval to result and start new interval
                merged_recompute_intervals.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last interval
        merged_recompute_intervals.append((current_start, current_end))

        # print("merged_recompute_intervals", merged_recompute_intervals)

        return full_recalculate

    def up_project_cache(self):

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
        return hybrid_cache

    def build_cache(self, input_ids, recalculate_args={}):

        recalculate = recalculate_args.get("recalculate", True)
        interval_size = recalculate_args.get("interval_size", 20)
        num_intervals = recalculate_args.get("num_intervals", 2)

        aux_cache = DynamicCache()

        # for baseline
        if self.baseline_base:
            with torch.no_grad():
                aux_outputs = self.base_model(
                    input_ids,
                    use_cache=True,
                    past_key_values=aux_cache,
                    output_attentions=True,
                )
                attention_scores = aux_outputs.attentions
                aux_cache = aux_outputs.past_key_values
            return aux_cache

        with torch.no_grad():
            aux_outputs = self.aux_model(
                input_ids,
                use_cache=True,
                past_key_values=aux_cache,
                output_attentions=True,
            )
            aux_cache = aux_outputs.past_key_values
            attention_scores = aux_outputs.attentions

        if self.baseline_aux:
            return aux_cache

        hybrid_cache = self.up_project_cache()

        if not recalculate:
            return hybrid_cache

        # Calculate intervals to recompute
        intervals = self.calculate_token_importance(
            input_ids, attention_scores, interval_size, num_intervals
        )

        current_cache = hybrid_cache
        # print("important_positions", important_positions)
        # print("num of intervals", len(intervals))

        for interval in intervals:
            start_pos = interval[0]
            end_pos = interval[-1] + 1  # Add 1 to include the last token

            # Create input for token group
            token_input = input_ids[:, start_pos:end_pos]

            # cache up to start of interval
            truncated_cache = DynamicCache()
            for layer_idx in range(n_layers_base):
                truncated_cache.update(
                    key_states=current_cache.key_cache[layer_idx][:, :, :start_pos, :],
                    value_states=current_cache.value_cache[layer_idx][
                        :, :, :start_pos, :
                    ],
                    layer_idx=layer_idx,
                )

            # Forward pass with truncated cache
            with torch.no_grad():
                outputs = self.base_model(
                    token_input,  # Process only the tokens in the group
                    use_cache=True,
                    past_key_values=truncated_cache,
                )

                # Update cache with the new KV values for this group
                new_cache = outputs.past_key_values
                # print("new_cache.key_cache[0].shape", new_cache.key_cache[0].shape)

                for layer_idx in range(n_layers_base):
                    new_keys = new_cache.key_cache[layer_idx][
                        :, :, start_pos:end_pos, :
                    ]
                    new_values = new_cache.value_cache[layer_idx][
                        :, :, start_pos:end_pos, :
                    ]

                    hybrid_cache.key_cache[layer_idx][
                        :, :, start_pos:end_pos, :
                    ] = new_keys
                    hybrid_cache.value_cache[layer_idx][
                        :, :, start_pos:end_pos, :
                    ] = new_values

                current_cache = hybrid_cache

        return hybrid_cache

    def run(self, messages, max_new_tokens=100, ttft_k=2, recalculate_args={}):

        self.aux_xs = {}
        handles = []

        for i in range(n_layers_aux):
            handle = self.aux_model.model.layers[
                i
            ].self_attn.q_proj.register_forward_hook(
                partial(self.x_hook, index=i, xs=self.aux_xs)
            )
            handles.append(handle)

        # tokenize prompt
        input_ids = self.base_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)

        # Time to build cache
        stmt = """
                past_key_values = self.build_cache(input_ids, recalculate_args=recalculate_args)   
            """

        t = benchmark.Timer(
            stmt=stmt,
            globals={
                "self": self,
                "input_ids": input_ids,
                "recalculate_args": recalculate_args,
            },
        )

        ttft = t.timeit(ttft_k)
        ttft = ttft.mean

        past_key_values = self.build_cache(input_ids, recalculate_args=recalculate_args)

        if self.baseline_aux:
            generated_text, past_key_values = generate_text(
                self.aux_model,
                input_ids,
                past_key_values,
                self.base_tokenizer,
                max_new_tokens,
            )
        else:
            generated_text, past_key_values = generate_text(
                self.base_model,
                input_ids,
                past_key_values,
                self.base_tokenizer,
                max_new_tokens,
            )

        for handle in handles:
            handle.remove()

        return generated_text, ttft, past_key_values, input_ids.shape[1]

    def call_with_prompt(self, prompt, max_new_tokens=1024, recalculate_args={}):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft, kv_cache, prompt_len = self.run(
            messages, max_new_tokens=max_new_tokens, recalculate_args=recalculate_args
        )

        return response, ttft, kv_cache, prompt_len


if __name__ == "__main__":

    prompt = """Context: France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, only became supportive when Germany started to build her own colonial empire. As it developed, the new empire took on roles of trade with France, supplying raw materials and purchasing manufactured items, as well as lending prestige to the motherland and spreading French civilization and language as well as Catholicism. It also provided crucial manpower in both World Wars.
        Question:  When did French Republicans back building the English empire?. Answer in as few words as possible
        """

    def clear_gpu_cache():
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # real base cache
    clear_gpu_cache()
    baseline_model = KV_hybrid_model(baseline_base=True)
    response, ttft, base_cache, prompt_len = baseline_model.call_with_prompt(
        prompt, max_new_tokens=100
    )
    print("prompt_len", prompt_len)
    print("Baseline model", response, ttft)
    del baseline_model  # free GPU

    clear_gpu_cache()

    # call kv hybrid
    recalculate_args = {
        "recalculate": True,
        "interval_size": 40,
        "num_intervals": 3,
    }

    kv_hybrid_model = KV_hybrid_model(baseline_base=False, baseline_aux=False)
    kv_hybrid_response, kv_hybrid_ttft, kv_hybrid_cache, prompt_len = (
        kv_hybrid_model.call_with_prompt(
            prompt, max_new_tokens=100, recalculate_args=recalculate_args
        )
    )
    print("KV hybrid: ", kv_hybrid_response, kv_hybrid_ttft)
    plot_kv_differences(
        kv_hybrid_cache,
        base_cache,
        prompt_len,
        save_path="kv_differences/hybrid_base.png",
    )

    del kv_hybrid_model
