import os
import math
import torch
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb


# --- Constants and Hyperparameters ---
doc_len = 1024
batch = 4
batch_size = 4
gradient_accumulation_steps = 8
warmup_iters = 50
lr_decay_iters = 1 * 1024 * 4
min_lr = 4e-4
learning_rate = 4e-3
beta1, beta2 = 0.9, 0.95
eval_interval = 20
eval_iters = 10
save_interval = 128
x_not_kv = True
decay_lr = True # whether to decay the learning rate
start_idx = 0


# --- Tokenizer Initialization ---
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Dataset Preprocessing ---
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=1025, padding="max_length", truncation=True)

c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")
tokenized_datasets = c4_subset.map(tokenize_function, batched=True, num_proc=32)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(160 * 10 * 1000))
small_test_dataset = tokenized_datasets["train"].shuffle(seed=142).select(range(10 * 1000))

# --- Distributed Training Setup ---
# ddp = int(os.environ.get('RANK', -1)) != -1
# ddp_rank = int(os.environ.get('RANK', 0)) if ddp else 0
# ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0)) if ddp else 0
# ddp_world_size = int(os.environ.get('WORLD_SIZE', 1)) if ddp else 1
# device = f"cuda:{ddp_local_rank}" if ddp else "cuda:0"
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
ddp_rank = 0
ddp_world_size  = 1
device = 'cuda:0'
backend='nccl'

gradient_accumulation_steps = 8
seed_offset = 0
if ddp:
    

    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    gradient_accumulation_steps  = gradient_accumulation_steps//ddp_world_size 
    seed_offset = ddp_rank # each process gets a different seed
    
    init_process_group(backend=backend)

torch.manual_seed(1337 + seed_offset)

torch.cuda.set_device(device)

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(device) 



# if ddp:
#     init_process_group(backend="nccl")

# --- Model Initialization ---
# Change base for for 8B -> 1B configuration aka Llama-3.1-8B-Instruct
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=access_token)
aux_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=access_token)

if ddp:
    aux_model = DDP(aux_model, device_ids=[ddp_local_rank])

print("Models loaded successfully.")

# --- Layer Definitions and Dimensions ---
m8b_1b = False  # Enable for 8B -> 1B configuration
m3b_1b = True   # Enable for 3B -> 1B configuration

if m8b_1b:
    # 8B -> 1B Model Configuration
    n_layers_base = 32         # Number of layers in the base model
    n_layers_aux = 16          # Number of layers in the auxiliary model
    base_dim = 4096            # Hidden dimension for the base model
    aux_dim = 2048             # Hidden dimension for the auxiliary model
    base_kv_dim = 1024         # Key/Value projection dimension for the base model
    aux_kv_dim = 512           # Key/Value projection dimension for the auxiliary model

    # Map base model layers to auxiliary model layers
    base_aux_map = {}
    for i in range(n_layers_base):
        base_aux_map[i] = i // 2

if m3b_1b:
    # 3B -> 1B Model Configuration
    n_layers_base = 28         # Number of layers in the base model
    n_layers_aux = 16          # Number of layers in the auxiliary model
    base_dim = 3072            # Hidden dimension for the base model
    aux_dim = 2048             # Hidden dimension for the auxiliary model
    base_kv_dim = 1024         # Key/Value projection dimension for the base model
    aux_kv_dim = 512           # Key/Value projection dimension for the auxiliary model

    # Map base model layers to auxiliary model layers with a custom mapping
    base_aux_map = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 3,
        5: 4, 6: 5, 7: 6, 8: 6, 9: 6,
        10: 7, 11: 7, 12: 8, 13: 8, 14: 9,
        15: 9, 16: 10, 17: 10, 18: 11, 19: 11,
        20: 12, 21: 12, 22: 13, 23: 13, 24: 14,
        25: 14, 26: 15, 27: 15,
    }

    # Comment: Custom mapping ensures certain base layers are merged or reused in the auxiliary model.

# --- Hook Storage ---
base_ks, base_vs, base_qs = {}, {}, {}
aux_ks, aux_vs, aux_qs, aux_xs = {}, {}, {}, {}
Lv, Lk = {}, {}

# --- Hook Functions ---
def kv_hook(module, in_x, output, index, kvs, name):
    kvs[index] = output

def x_hook(module, in_x, output, index, xs):
    xs[index] = in_x[0]

def kv_m_hook(module, in_x, output, index, kvs, akvs, L):
    kvs[index] = output

# --- Register Hooks ---
for i in range(n_layers_base):
    bvhook = base_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_m_hook, index = i, kvs = base_vs, akvs= aux_xs if x_not_kv else aux_vs , L = Lv  ))
    bkhook  = base_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_m_hook, index = i, kvs = base_ks,  akvs= aux_xs  if x_not_kv else aux_ks , L = Lk ))
    bqhook = base_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(kv_hook, index = i, kvs = base_qs, name="base q"))
        
      
for i in range(n_layers_aux):
    avhook = aux_model.module.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_vs, name="v"))
    akhook = aux_model.module.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_ks, name="k"))
    aqhook = aux_model.module.model.layers[i].self_attn.q_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_qs, name="q"))
    axhook = aux_model.module.model.layers[i].self_attn.q_proj.register_forward_hook(partial(x_hook, index = i, xs = aux_xs))

        

print("Hooks registered.")

# --- Initialize Learnable Projections ---
with torch.no_grad():
    Lv = {}
    Lk = {}
    params = []
    for j in range(n_layers_base):  # the number of layers in aux_model
        Lv[j] = torch.randn( (aux_dim, base_kv_dim) )/25.0
        Lk[j] = torch.randn( (aux_dim, base_kv_dim) )/25.0

        params.append(Lv[j])
        params.append(Lk[j])

        if ddp:
            torch.distributed.broadcast( Lv[j], 0)
            torch.distributed.broadcast( Lk[j], 0)
    
for j in range(n_layers_base):    
    Lv[j].requires_grad = True
    Lk[j].requires_grad = True


for param in base_model.parameters():
    param.requires_grad = False


for param in aux_model.parameters():
    param.requires_grad = False




# --- Optimizer Setup ---
optim_groups = [{'params': params, 'weight_decay': 0.0}]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

# --- Optimizer Setup ---
if ddp_rank == 0:
    # wandb.login()
    wandb.init(
        project="KV-training",
        name="run-kv-training",
        config={
            "batch_size": batch_size,
            "doc_len": doc_len,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "eval_interval": eval_interval,
            "save_interval": save_interval,
            "n_layers_base": n_layers_base,
            "n_layers_aux": n_layers_aux,
        }
    )


# --- Helper Functions ---
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(split,idx):
    global small_train_dataset
    global batch
    global start_idx
    global doc_len
    xlen = doc_len + 1
    idx = start_idx
    toks = [ [] for _ in range(batch) ]
    for b in range(batch):
        tok_len  = 0
        while tok_len < doc_len:
            i = ddp_rank * batch  + (  start_idx * ddp_world_size * batch )
            start_idx += 1
            next_docs = small_train_dataset[i]['input_ids']
            tok_len += len(next_docs) + 1
            toks[b] = toks[b] + [128001] + next_docs
    for b in range(batch):
        toks[b] = toks[b][0:xlen+1]
        
    X = torch.tensor(toks).to(device)
    Y = X[:,1:xlen].to(device).contiguous()
    X = X[:,0:xlen-1].to(device)
    return  X, Y

def get_eval_batch(split,idx):
    global small_test_dataset
    global batch
    xlen = doc_len + 1
    i = ddp_rank * batch  + (  idx * ddp_world_size * batch )
    X = torch.tensor(small_train_dataset[i: i+batch]['input_ids']).to(device)
    Y = X[:,1:xlen].to(device).contiguous()
    X = X[:,0:xlen-1].to(device)
    return  X, Y

# --- Training Loop ---
# I know what youre thinking why write a training loop? why not ? 
for i in range(lr_decay_iters):
    # Evaluate the model periodically
    if i % eval_interval == 0:
        if ddp_rank == 0:
            print("Evaluating the model...")
        
        with torch.no_grad():
            # Initialize evaluation metrics
            eval_loss = torch.zeros(1, dtype=torch.float)
            avg_vloss, avg_kloss, avg_qkloss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            
            # Evaluation loop
            for ii in range(eval_iters):
                for k in range(gradient_accumulation_steps):
                    # Fetch evaluation batch
                    X, Y = get_eval_batch("train", ii * 32 + k)
                    
                    # Forward pass
                    aux_logits = aux_model(X).logits
                    re_base_logits = base_model(X).logits
                    targets = Y.flatten()

                    aux_logitsr = aux_logits.reshape(-1, aux_logits.shape[-1])
                    re_base_logitsr = re_base_logits.reshape(-1, re_base_logits.shape[-1])
                    
                    eloss = float(0.0)
                    
                    # Compute per-layer losses
                    for layer in range(n_layers_base):
                        aux_layer = base_aux_map[layer]
                        
                        # Compute Value Loss (VLoss) and Key Loss (KLoss)
                        vloss =  torch.mean( torch.abs( ((aux_xs[aux_layer] @ Lv[layer ]) - base_vs[layer]).to(torch.float) ))/( n_layers_base )
                        kloss =  torch.mean( torch.abs( ((aux_xs[aux_layer] @ Lk[layer ]) - base_ks[layer]).to(torch.float) ))/( n_layers_base )

                        eloss += vloss 
                        eloss += kloss
                        avg_vloss += vloss
                        avg_kloss += kloss
                        
                        # Compute Query-Key Alignment Loss (QKLoss)
                        bs, slen, n_kv_heads, head_dim = batch_size, doc_len, aux_kv_dim // 64, 128
                        n_rep = base_dim // base_kv_dim

                        kr = (aux_xs[aux_layer] @ Lk[layer ]).view(bs, slen, n_kv_heads,  head_dim)
                        kr = kr[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
                        q = base_qs[layer]
                        bk =  base_ks[layer].view(bs, slen, n_kv_heads,  head_dim )
                        bkr = bk[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim )
                        q = q.view(bs, slen, n_kv_heads * n_rep,  head_dim )
                        aqk = torch.matmul(q.transpose(1,2), kr.transpose(1,2).transpose(2, 3))
                        bqk = torch.matmul(q.transpose(1,2), bkr.transpose(1,2).transpose(2, 3)) 

                        qkloss =  torch.mean( torch.abs(  (aqk - bqk).to(torch.float) ))/(n_layers_base * n_layers_base )   
                        
                        eloss += qkloss
                        avg_qkloss += qkloss

                    # Accumulate evaluation loss
                    eval_loss += eloss / eval_iters

                # Reduce metrics across processes
                torch.distributed.all_reduce(eval_loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(avg_vloss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(avg_kloss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(avg_qkloss, op=torch.distributed.ReduceOp.AVG)
            
            # Log evaluation metrics
            #TODO REPLACE W wandb
            if ddp_rank == 0:
                print(
                    f"Step {i} - Eval Loss: {eval_loss.item() / gradient_accumulation_steps:.4f}, "
                    f"VLoss: {avg_vloss.item() / (eval_iters * gradient_accumulation_steps):.4f}, "
                    f"KLoss: {avg_kloss.item() / (eval_iters * gradient_accumulation_steps):.4f}, "
                    f"QKLoss: {avg_qkloss.item() / (eval_iters * gradient_accumulation_steps):.4f}"
                )
                wandb.log({
                    "eval_loss": eval_loss.item() / gradient_accumulation_steps,
                    "eval_vloss": avg_vloss.item() / (eval_iters * gradient_accumulation_steps),
                    "eval_kloss": avg_kloss.item() / (eval_iters * gradient_accumulation_steps),
                    "eval_qkloss": avg_qkloss.item() / (eval_iters * gradient_accumulation_steps)
                })

    # Adjust learning rate
    lr = get_lr(i) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Training Step
    Tavg_loss = torch.zeros(1, dtype=torch.float)  # Reset average loss
    Tavg_loss[0] = 1e-8

    Tavg_qkloss = Tavg_loss.clone()
    Tavg_kloss = Tavg_loss.clone()
    Tavg_vloss = Tavg_loss.clone()

    for k in range(gradient_accumulation_steps):
        if ddp:
            aux_model.require_backward_grad_sync = (k == gradient_accumulation_steps - 1)
            
        # Fetch training batch
        X, Y = get_batch("train", i * 32 + k)
        
        # Forward pass
        aux_logits = aux_model(X).logits

        
        re_base_logits = base_model(X).logits
        targets = Y.flatten()
        
        loss = float(0.0)  # Initialize total loss train this time 
        
        # Compute per-layer losses
        for layer in range(n_layers_base):
            aux_layer = base_aux_map[layer]
            
            # Compute VLoss and KLoss
            vloss =  torch.mean( torch.abs( ((aux_xs[aux_layer] @ Lv[layer ]) - base_vs[layer]).to(torch.float)))/( n_layers_base)
            kloss = torch.mean( torch.abs( ((aux_xs[aux_layer] @ Lk[layer ]) - base_ks[layer]).to(torch.float) ))/(  n_layers_base)

            loss += vloss 
            Tavg_vloss += vloss
            loss += kloss
            Tavg_kloss +=kloss
            
            # Compute QKLoss
            bs, slen, n_kv_heads, head_dim = batch_size, doc_len, aux_kv_dim // 64, 128
            n_rep = base_dim // base_kv_dim

            kr = (aux_xs[aux_layer] @ Lk[layer ]).view(bs, slen, n_kv_heads,  head_dim)
            kr = kr[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
            q = base_qs[layer]
            bk =  base_ks[layer].view(bs, slen, n_kv_heads,  head_dim )
            bkr = bk[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim )
            q = q.view(bs, slen, n_kv_heads * n_rep,  head_dim )
            aqk = torch.matmul(q.transpose(1,2), kr.transpose(1,2).transpose(2, 3))
            bqk = torch.matmul(q.transpose(1,2), bkr.transpose(1,2).transpose(2, 3)) 

            loss += qkloss

            Tavg_qkloss += qkloss


        # Backward pass
        loss.backward()
        Tavg_loss += loss

    # Gradient synchronization for DDP
    with torch.no_grad():
        if ddp:
            for layer in range(n_layers_base):
                if Lv[layer].grad != None:
                    torch.distributed.all_reduce(Lv[layer].grad, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(Lk[layer].grad , op=torch.distributed.ReduceOp.AVG)
                else:
                    if ddp_rank == 0:
                        print("Lv[layer].grad == None", layer)
        
        #TODO REPLACE W wandb
        if ddp_rank == 0:
           print( i, "\tloss", Tavg_loss.item()/(gradient_accumulation_steps ),  "\tvloss", Tavg_vloss.item()/(gradient_accumulation_steps ), "\tqk loss", Tavg_qkloss.item()/(gradient_accumulation_steps ),  "\tkloss", Tavg_kloss.item()/(gradient_accumulation_steps ))
           wandb.log({
                "train_loss": Tavg_loss.item() / gradient_accumulation_steps,
                "train_vloss": Tavg_vloss.item() / gradient_accumulation_steps,
                "train_kloss": Tavg_kloss.item() / gradient_accumulation_steps,
                "train_qkloss": Tavg_qkloss.item() / gradient_accumulation_steps,
                "lr": lr
            })

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Save checkpoints periodically
    if i % save_interval == 0 and ddp_rank == 0:
        save_file(
            {f"Lv.{layer}": Lv[layer] for layer in range(n_layers_base)} | 
            {f"Lk.{layer}": Lk[layer] for layer in range(n_layers_base)},
            f"projections_step_{i}.safetensors"
        )
        print(f"Checkpoint saved at step {i}")
