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


class KV_prediction_model:
    def __init__(self, base_model=BASE_DEFAULT, aux_model=AUX_DEFAULT):
        self.base_model_str = base_model
        self.aux_model_str = aux_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)

        # base aux configs
        self.aux_xs = {}

        self.n_layers_base = 32
        self.n_layers_aux = 16
        self.aux_dim = 2048
        self.base_dim = 4096
        self.aux_kv_dim = 512
        self.base_kv_dim = 1024
        self.cache_len = 3000
        self.kvheads = 8
        self.head_dim_base = 128

        self.initialize()

    def initialize(self):

        # get projection matrices from huggingface
        repo_id = "caiacost/matrix-fun"
        filename = "projections.safetensors"
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.tensors = load_file(file_path)

        # ------ initialie aux -------
        self.aux_model = AutoModelForCausalLM.from_pretrained(
            self.aux_model_str, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.aux_tokenizer = AutoTokenizer.from_pretrained(self.aux_model_str)

        self.aux_model.eval()
        self.aux_model.to(self.device)

        if self.aux_tokenizer.pad_token is None:
            self.aux_tokenizer.pad_token = self.aux_tokenizer.eos_token
            self.aux_model.config.pad_token_id = self.aux_model.config.eos_token_id

        # ------- initialize base ------
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_str, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_str)

        self.base_model.eval()
        self.base_model.to(self.device)

        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id

    def run(self, messages, max_new_tokens=1024):

        # tokenize prompt
        input_ids_aux = self.aux_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.device)

        # set hooks to get activations from previous
        def x_hook(module, in_x, output, index, xs):
            if xs.get(index) == None:
                xs[index] = in_x[0].to(self.device)

        for i in range(self.n_layers_aux):
            self.aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(
                partial(x_hook, index=i, xs=self.aux_xs)
            )

        # process prompt with aux
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.aux_model(
                input_ids_aux,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        print(
            "prompt_cache_aux: ",
            len(past_key_values.key_cache),
            past_key_values.key_cache[0].shape,
        )

        # get ttft
        stmt = """
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.aux_model(
                input_ids_aux,
                use_cache=True,
                past_key_values=past_key_values,
            )
        """

        t = benchmark.Timer(
            stmt=stmt,
            globals={
                "self": self,
                "input_ids_aux": input_ids_aux,
                "DynamicCache": DynamicCache,
            },
        )

        ttft = t.timeit(1)
        ttft = ttft.mean

        # predicted cache
        predicted_cache_base = DynamicCache(num_hidden_layers=self.n_layers_base)

        # base_prompt_cache = StaticCache(
        #     config=base_model.config,
        #     batch_size=1,
        #     max_cache_len=3000,
        #     device="cuda:0",
        #     dtype=torch.bfloat16,
        # )

        # go through each layer of the base
        for i in range(self.n_layers_base):

            # for first layer
            if i == 0:
                print("shapes baseL")
                print("aux_xs[i//2].shape", self.aux_xs[i // 2].shape)
                print(
                    "tensors['Lk.' +str(i)].shape", self.tensors["Lk." + str(i)].shape
                )

            #
            v_len = self.aux_xs[i // 2].shape[1]

            # get keys for layer
            key_state = (
                (self.aux_xs[i // 2] @ self.tensors["Lk." + str(i)].to(self.device))
                .view(1, v_len, self.kvheads, self.head_dim_base)
                .transpose(1, 2)
            ).to(self.device)

            # get values
            value_state = (
                (self.aux_xs[i // 2] @ self.tensors["Lv." + str(i)].to(self.device))
                .view(1, v_len, self.kvheads, self.head_dim_base)
                .transpose(1, 2)
            ).to(self.device)

            predicted_cache_base.update(
                key_states=key_state, value_states=value_state, layer_idx=i
            )

            # base_prompt_cache.key_cache[i][:, :, :v_len, :] = key_state
            # base_prompt_cache.value_cache[i][:, :, :v_len, :] = value_state

            # copy random stuff
            # copy_last = 2510
            # base_prompt_cache.key_cache[i][:, :, copy_last:, :] = (
            #     base_prompt_cache_real.key_cache[i][:, :, copy_last:, :]
            # )
            # base_prompt_cache.value_cache[i][:, :, copy_last:, :] = (
            #     base_prompt_cache_real.value_cache[i][:, :, copy_last:, :]
            # )

            # copy_start = 100
            # base_prompt_cache.key_cache[i][:, :, :copy_start, :] = (
            #     base_prompt_cache_real.key_cache[i][:, :, :copy_start, :]
            # )
            # base_prompt_cache.value_cache[i][:, :, :copy_start, :] = (
            #     base_prompt_cache_real.value_cache[i][:, :, :copy_start, :]
            # )

        #
        # new_inputs = tokenizer(sample + prompt, return_tensors="pt").to("cuda")
        past_key_values = copy.deepcopy(predicted_cache_base)

        input_ids_base = self.base_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.device)

        generated_text = ""

        with torch.no_grad():
            for _ in range(max_new_tokens):

                outputs = self.base_model(
                    input_ids=input_ids_base[:, -1:],
                    past_key_values=past_key_values,
                    pad_token_id=self.base_tokenizer.pad_token_id,
                    eos_token_id=self.base_tokenizer.eos_token_id,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                input_ids_base = torch.cat(
                    [input_ids_base, next_token.unsqueeze(-1)], dim=-1
                )

                # check for eos
                if next_token.item() == self.base_tokenizer.eos_token_id:
                    break

                new_token = self.base_tokenizer.decode(
                    next_token, skip_special_tokens=True
                )
                # print(new_token)

                generated_text += new_token

        generated_text = generated_text.rstrip()
        # print(f"Generated text: \n{generated_text}'")
        return generated_text, ttft

    def call_with_prompt(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft = self.run(messages, max_new_tokens=max_new_tokens)

        return response, ttft


class Baseline_model:
    def __init__(self, model_name):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def run(self, messages, max_new_tokens=1024):

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        self.model.to(device)

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(device)

        past_key_values = DynamicCache()
        # start_time = time.time()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        stmt = """
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
        """

        t = benchmark.Timer(
            stmt=stmt,
            globals={
                "self": self,
                "input_ids": input_ids,
                "DynamicCache": DynamicCache,
            },
        )

        ttft = t.timeit(1)
        ttft = ttft.mean
        # print("TTFT ", ttft)

        # Display the KV cache for debugging
        # print("KV Cache:")
        # for i, layer_cache in enumerate(past_key_values):
        #     print(f"Layer {i+1}:")
        #     print("Keys:", layer_cache[0].shape)
        #     print("Values:", layer_cache[1].shape)

        generated_text = ""

        with torch.no_grad():
            for _ in range(max_new_tokens):

                outputs = self.model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

                # check for eos
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                new_token = self.tokenizer.decode(next_token, skip_special_tokens=True)
                # print(new_token)

                generated_text += new_token

        generated_text = generated_text.rstrip()
        # print(f"Generated text: \n{generated_text}'")
        return generated_text, ttft

    def call_with_prompt(self, prompt, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response, ttft = self.run(messages, max_new_tokens=max_new_tokens)

        return response, ttft


if __name__ == "__main__":

    prompt = "How do you make a cake?"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=AUX_DEFAULT, help="Name of the model to load"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate after the first token",
    )

    args = parser.parse_args()
    # baseline_model = Baseline_model(args.model_name)
    # response, ttft = baseline_model.call_with_prompt(prompt)

    kv_model = KV_prediction_model()
    response, ttft = kv_model.call_with_prompt(prompt, max_new_tokens=100)

    print(response, ttft)
