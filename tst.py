import os
import torch
from functools import partial
import math 


import torchvision
torchvision.disable_beta_transforms_warning()

from safetensors.torch import save_file

from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer =  AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

device = 'cuda:0'

torch.cuda.set_device(device)

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(device) 

apple = False

aux_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
   
import safetensors
tensors = {}

with safetensors.safe_open("Instruct-8b-1b1projections.3968.New.safetensors", framework="pt", device=device) as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

sample = "Put answer in brackets like [Answer], Respond very briefly just the answer: My name is John. What is my name?"

input_ids = tokenizer(sample, return_tensors="pt")

print("input_ids", input_ids)


len_prompt = input_ids['input_ids'].shape[1]
print("len prompt", len_prompt )

from transformers import StaticCache


prompt_cache = StaticCache(config=aux_model.config, batch_size=1, max_cache_len=3000, device="cuda:0", dtype=torch.bfloat16)

aux_ks = {}
aux_vs = {}
aux_qs = {}

aux_xs = {}

base_ks = {}
base_vs = {}
base_qs = {}


def kv_hook(module, in_x, output, index, kvs):
   if  kvs.get(index) == None:
      kvs[index] = output


def x_hook(module, in_x, output, index, xs):
   if  xs.get(index) == None:
      xs[index] = in_x[0]


      
for i in range(16):
   aux_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_vs))
   aux_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_ks))
   aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_qs))
   aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(x_hook, index = i, xs = aux_xs))


with torch.no_grad():
     prompt_cache = aux_model(**input_ids,
                              pad_token_id=tokenizer.eos_token_id,
                              max_new_tokens=1, temperature=0.3, do_sample=True,  top_k=50,
                              past_key_values = prompt_cache).past_key_values

print("prompt_cache", len( prompt_cache.key_cache),  prompt_cache.key_cache[0].shape) #,  "\n",  prompt_cache.value_cache[0])
#exit()

prompt = " ["

import copy

new_inputs = tokenizer(sample + prompt, return_tensors="pt").to("cuda")
past_key_values = copy.deepcopy(prompt_cache)
outputs = aux_model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=10,  temperature=0.5, do_sample=True,  top_k=50,
                         pad_token_id=tokenizer.eos_token_id
                         )
response = tokenizer.batch_decode(outputs , skip_special_tokens=True )[0]
print("output:", response)


# Now.  load the base model

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


base_prompt_cache = StaticCache(config=base_model.config, batch_size=1, max_cache_len=3000, device="cuda:0", dtype=torch.bfloat16)

base_prompt_cache1 = StaticCache(config=base_model.config, batch_size=1, max_cache_len=3000, device="cuda:0", dtype=torch.bfloat16)


with torch.no_grad():
   base_prompt_cache1 = base_model(**input_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             max_new_tokens=1, temperature=0.5, do_sample=True,  top_k=50,
                             past_key_values = base_prompt_cache1).past_key_values
   

n_layers_base = 32
n_layers_aux = 16
aux_dim = 2048
base_dim = 4096

aux_kv_dim = 512
base_kv_dim = 1024

cache_len = 3000

for i in range(n_layers_base):
   v_len = aux_xs[i//2].shape[1]
   
   key_state = (aux_xs[i//2] @ tensors["Lk." +str(i)]).view(1,v_len,8,128 ).transpose(1,2)
   value_state = (aux_xs[i//2] @ tensors["Lv." +str(i)]).view(1, v_len,8,128).transpose(1,2)
   
   base_prompt_cache.key_cache[i][:,:,:v_len,:] = key_state
   base_prompt_cache.value_cache[i][:,:,:v_len,:] = value_state

   if True:
      
#       copy_start = 2500
#       base_prompt_cache.key_cache[i][:,:,copy_start:,:] = base_prompt_cache1.key_cache[i][:,:,copy_start:,:]
#       base_prompt_cache.value_cache[i][:,:,copy_start:,:] = base_prompt_cache1.value_cache[i][:,:,copy_start:,:]
      
      
      copy_start = 20
      base_prompt_cache.key_cache[i][:,:,:copy_start,:] = base_prompt_cache1.key_cache[i][:,:,:copy_start,:]
      base_prompt_cache.value_cache[i][:,:,:copy_start,:] = base_prompt_cache1.value_cache[i][:,:,:copy_start,:]
      


new_inputs = tokenizer(sample + prompt, return_tensors="pt").to("cuda")
past_key_values = copy.deepcopy(base_prompt_cache)
outputs = base_model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=2,  temperature=0.3, do_sample=True,  top_k=50,
	                 pad_token_id=tokenizer.eos_token_id
                         )
response = tokenizer.batch_decode(outputs , skip_special_tokens=True )[0]
print("output:", response)

