import os
import math
import torch
from functools import partial
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, DatasetDict, load_from_disk
from safetensors.torch import save_file
from trl import SFTTrainer
import torch
import wandb
from functools import partial
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, ModelConfig, get_peft_config
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import TrainerCallback
import torch
# General Python utilities
import os
import math
from functools import partial

# PyTorch
import torch
from torch.optim import AdamW  # Optimizer
from transformers import get_scheduler, TrainingArguments


# Initialize WandB for experiment tracking
wandb.init(project="sft-training", name="sft-run-kv-training-instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Constants



n_layers_base, n_layers_aux = 32, 16
aux_dim, base_dim, base_kv_dim, aux_kv_dim = 2048, 4096, 1024, 512
x_not_kv = True
batch = 4  # or any valid batch size
doc_len = 1024  # or any valid sequence length

gradient_accumulation_steps = 2
batch_size = 4
# eval_batch_size = 2

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

IGNORE_INDEX = 128001


# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=1025, padding="max_length", truncation=True)

# Load or cache the tokenized dataset
try:
    tokenized_datasets = load_from_disk("cached_tokenized_datasets")
except FileNotFoundError:
    dataset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz", split="train")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=32)
    tokenized_datasets.save_to_disk("cached_tokenized_datasets")

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(160 * 10 * 1000))
eval_dataset = tokenized_datasets["train"].shuffle(seed=142).select(range(1000))

half_eval_size = len(eval_dataset) // 10

eval_dataset = eval_dataset.select(range(half_eval_size))


# Model setup
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
aux_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


# Move models to the correct device
base_model = base_model.to(device)
aux_model = aux_model.to(device)


Lv, Lk, params = {}, {}, []

with torch.no_grad():
    for j in range(n_layers_base):  # The number of layers in base_model
        Lv[j] = torch.randn((aux_dim, base_kv_dim), device=device) / 25.0
        Lk[j] = torch.randn((aux_dim, base_kv_dim), device=device) / 25.0

        # Add tensors to trainable parameters list
        params.append(Lv[j])
        params.append(Lk[j])
    
        
for j in range(n_layers_base):    
    Lv[j].requires_grad = True
    Lk[j].requires_grad = True

# Hook functions
# could use 4 apple
def kv_m_hook(module, in_x, output, index, kvs, akvs, L):
    """Forward hook to capture key/value projections."""
    # print(f"Hook triggered for layer {index}")
    kvs[index] = output


def kv_hook(module, in_x, output, index, kvs, name):
    """Simple hook to capture outputs."""
    # print(f"Hook triggered for {name} at layer {index}")
    kvs[index] = output


def x_hook(module, in_x, output, index, xs):
    """Capture inputs to the model."""
    # print(f"Input hook triggered for layer {index}")
    xs[index] = in_x[0]


# Register hooks for K/V projections on base_model
base_vs, base_ks, base_qs = {}, {}, {}
aux_xs, aux_vs, aux_ks, aux_qs = {}, {}, {}, {}

for i in range(n_layers_base):
    # mhooks
    bvhook = base_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_m_hook, index = i, kvs = base_vs, akvs= aux_xs, L = Lv  ))
    bkhook = base_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_m_hook, index = i, kvs = base_ks,  akvs= aux_xs , L = Lk ))

    # kvhook
    bqhook = base_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(kv_hook, index = i, kvs = base_qs, name="base q"))
        
for i in range(n_layers_aux):
    # kv hooks
    avhook = aux_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_vs , name="v" ))
    akhook = aux_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_ks, name="k"))
    aqhook = aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(kv_hook, index = i, kvs = aux_qs, name="q"))

    # inp hook
    axhook = aux_model.model.layers[i].self_attn.q_proj.register_forward_hook(partial(x_hook, index = i, xs = aux_xs))


for param in base_model.parameters():
    param.requires_grad = False
    
for param in aux_model.parameters():
    param.requires_grad = False


def compute_combined_loss(model, inputs, ev=False):
    """
    Compute the combined loss for the training step with consistent structure.

    Args:
        model: The auxiliary model.
        inputs: The input batch containing "input_ids" and "labels".
        ev (bool): If True, returns detailed loss components for evaluation.

    Returns:
        loss: The total computed loss (and optionally detailed losses if `ev` is True).
    """
    device = next(model.parameters()).device
    targets = inputs["labels"].flatten()

    # Forward pass to compute logits
    aux_logits = model(**inputs).logits
    base_logits = base_model(**inputs).logits

    aux_logitsr = aux_logits.reshape(-1, aux_logits.shape[-1])

    loss = 0.0
    avg_value_loss, avg_key_loss, avg_qk_loss = 0.0, 0.0, 0.0

    for layer in range(n_layers_base):
        aux_layer = layer // 2

        # Move tensors to the correct device
        aux_xs_layer = aux_xs[aux_layer].to(device)
        Lv_layer = Lv[layer].to(device)
        Lk_layer = Lk[layer].to(device)
        base_vs_layer = base_vs[layer].to(device)
        base_ks_layer = base_ks[layer].to(device)
        base_qs_layer = base_qs[layer].to(device)

        # Get number of elements in the base_vs_layer tensor
        num_el = base_vs_layer.numel()

        # Value loss
        value_loss = torch.mean(
            torch.linalg.matrix_norm((aux_xs_layer @ Lv_layer - base_vs_layer).float(), ord="fro")
        ) / (num_el * n_layers_base)
        loss += value_loss
        avg_value_loss += value_loss

        # Key loss
        key_loss = torch.mean(
            torch.linalg.matrix_norm((aux_xs_layer @ Lk_layer - base_ks_layer).float(), ord="fro")
        ) / (num_el * n_layers_base)
        loss += key_loss
        avg_key_loss += key_loss

        # QK similarity loss (optional)
        bs, slen = aux_xs_layer.size(0), aux_xs_layer.size(1)
        n_kv_heads = 8
        head_dim = 128
        total_elements = base_qs_layer.numel()
        n_rep = total_elements // (bs * slen * n_kv_heads * head_dim)

        kr = (aux_xs_layer @ Lk_layer).view(bs, slen, n_kv_heads, head_dim)
        kr = kr.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        q = base_qs_layer.view(bs, slen, n_kv_heads * n_rep, head_dim)
        bkr = base_ks_layer.view(bs, slen, n_kv_heads, head_dim)
        bkr = bkr.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

        qk_loss = torch.mean(torch.abs(torch.matmul(q.transpose(1, 2), kr.transpose(1, 2).transpose(2, 3)) -
                                       torch.matmul(q.transpose(1, 2), bkr.transpose(1, 2).transpose(2, 3)).float())) / (num_el * n_layers_base)
        loss += qk_loss
        avg_qk_loss += qk_loss

    # Cross-entropy loss
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=128001)
    ce_loss = loss_func(aux_logitsr.float(), targets)
    loss += ce_loss

    if ev:
        return loss, avg_value_loss.item(), avg_key_loss.item(), avg_qk_loss.item()

    wandb.log({
            "Value Loss": avg_value_loss.item(),
            "Key Loss": avg_key_loss.item(),
            "QK Loss": avg_qk_loss.item(),
            "Tot Loss": loss.item()
        })
    return loss

# Define the combined loss function OLD 
# def compute_combined_loss1(model, inputs, ev = False):
#     """
#     Compute the combined loss for the training step with dynamic shape handling.

#     Args:
#         model: The auxiliary model.
#         inputs: The input batch containing "input_ids" and "labels".
#         base_vs: The base model's value projections.
#         base_ks: The base model's key projections.
#         Lv: Learned value projection matrices for auxiliary model.
#         Lk: Learned key projection matrices for auxiliary model.
#         n_layers_base: Number of layers in the base model.
#         aux_xs: Auxiliary model's inputs for projection.
#         base_qs: The base model's query projections.
#     Returns:
#         loss: The total computed loss.
#     """
#     # Move inputs to the correct device
#     device = next(model.parameters()).device
#     # inputs = {key: value.to(device) for key, value in inputs.items()}

#     # Forward pass to compute logits
#     aux_logits = model(**inputs).logits
#     base_logits = base_model(**inputs).logits

#     # Flatten and reshape
#     targets = inputs["labels"].flatten()
#     aux_logitsr = aux_logits.reshape(-1, aux_logits.shape[-1])
#     base_logitsr = base_logits.reshape(-1, base_logits.shape[-1])

#     # Initialize loss components
#     loss, avg_loss1, avg_loss3, avg_loss2, avg_loss4 = 0.0, 0.0, 0.0, 0.0, 0.0

#     for layer in range(n_layers_base):
#         aux_layer = layer // 2
#         # Debug: Check for missing projections
#         if aux_layer not in aux_xs or layer not in base_vs or layer not in base_ks:
#             raise KeyError(f"Layer {layer} or auxiliary layer {aux_layer} not found in projections!")

#         # Extract and move tensors to the correct device
#         aux_xs_layer = aux_xs[aux_layer].to(device)
#         Lv_layer = Lv[layer].to(device)
#         Lk_layer = Lk[layer].to(device)

#         base_vs_layer = base_vs[layer].to(device)
#         base_ks_layer = base_ks[layer].to(device)
#         base_qs_layer = base_qs[layer].to(device)

#         # Value loss
#         # here you would add x_not_kv of apple
#         vloss = torch.mean(torch.abs((aux_xs_layer @ Lv_layer) - base_vs_layer).float()) / n_layers_base
    
#         loss += vloss
#         avg_loss1 += vloss

#         # Key loss
#         kloss = torch.mean(torch.abs((aux_xs_layer @ Lk_layer) - base_ks_layer).float()) / n_layers_base
#         loss += kloss
#         avg_loss3 += kloss

#         # QK similarity loss
#         bs, slen = aux_xs_layer.size(0), aux_xs_layer.size(1)
#         total_elements = base_qs_layer.numel()

#         # Dynamically calculate n_kv_heads, head_dim, and n_rep
#         n_kv_heads = 8  # Assuming this is fixed
#         head_dim = 128  # Assuming this is fixed
#         n_rep = total_elements // (bs * slen * n_kv_heads * head_dim)

#         # Validate the computed dimensions
#         expected_elements = bs * slen * n_kv_heads * n_rep * head_dim
#         if expected_elements != total_elements:
#             raise ValueError(
#                 f"Mismatch in sizes: expected {expected_elements} elements but got {total_elements}. "
#                 f"bs={bs}, slen={slen}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, n_rep={n_rep}"
#             )

#         # Reshape tensors
#         kr = (aux_xs_layer @ Lk_layer).view(bs, slen, n_kv_heads, head_dim)
#         kr = kr.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

#         q = base_qs_layer.view(bs, slen, n_kv_heads * n_rep, head_dim)

#         bk = base_ks_layer.view(bs, slen, n_kv_heads, head_dim)
#         bkr = bk.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

#         aqk = torch.matmul(q.transpose(1, 2), kr.transpose(1, 2).transpose(2, 3))
#         bqk = torch.matmul(q.transpose(1, 2), bkr.transpose(1, 2).transpose(2, 3))

#         qkloss = torch.mean(torch.abs((aqk - bqk).float())) / (n_layers_base * n_layers_base)
#         loss += qkloss
#         avg_loss2 += qkloss
#         #CE loss 
#         # loss_func=  torch.nn.CrossEntropyLoss( ignore_index=IGNORE_INDEX)
#         # base_loss = torch.mean(loss_func(base_logitsr.float(), targets))
#         # avg_loss5 += base_loss
#     print(f"Losses:\n Value Loss: {avg_loss1.item()}, Key Loss: {avg_loss3.item()}, QK Loss: {avg_loss2.item()}")
#     if not ev:
#         wandb.log({
#             "Value Loss": avg_loss1.item(),
#             "Key Loss": avg_loss3.item(),
#             "QK Loss": avg_loss2.item()
#         })
        
#         return loss/n_layers_base
#     return loss/n_layers_base, avg_loss1.item(),  avg_loss3.item(), avg_loss2.item()


# Define save function for tensors
def save_tensors_and_model(iteration, Lv, Lk, aux_model, output_dir="output", prefix="checkpoint"):
    tensors = {f"Lv.{layer}": Lv[layer] for layer in range(n_layers_base)}
    tensors.update({f"Lk.{layer}": Lk[layer] for layer in range(n_layers_base)})
    os.makedirs(output_dir, exist_ok=True)

    tensor_filename = os.path.join(output_dir, f"{prefix}_Lv_Lk_{iteration}.safetensors")
    save_file(tensors, tensor_filename)

    model_dir = os.path.join(output_dir, f"{prefix}_model_{iteration}")
    aux_model.save_pretrained(model_dir)
    print(f"Saved tensors to {tensor_filename} and model to {model_dir}")

    
def get_eval_batch(dataset, idx, batch_size, doc_len, device):
    """
    Fetches a batch of evaluation data.

    Args:
        dataset: Dictionary of lists (e.g., {"input_ids": [...]}).
        idx: Current batch index.
        batch_size: Number of samples in a batch.
        doc_len: Maximum document length.
        device: Device to use (e.g., CPU or GPU).

    Returns:
        X, Y: Input and target tensors.
    """
    xlen = doc_len + 1
    i =  batch_size  + (  idx  * batch_size )
    X = torch.tensor(dataset[i: i+batch_size]['input_ids']).to(device)
    Y = X[:,1:xlen].to(device).contiguous()
    X = X[:,0:xlen-1].to(device)
    return  X, Y



def evaluate_model(eval_dataset, aux_model ):
    """
    Evaluate the auxiliary model with projections and compute evaluation loss.

    Args:
        eval_dataset: The evaluation dataset.
        aux_model: Auxiliary model.
        base_vs: Base value matrices.
        base_ks: Base key matrices.
        Lv: Learned value projection matrices.
        Lk: Learned key projection matrices.
        n_layers_base: Number of layers in the base model.
        aux_xs: Auxiliary projections.
        base_qs: Base query matrices.
        doc_len: Document length.
        batch_size: Number of samples in a batch.
        gradient_accumulation_steps: Steps for gradient accumulation.

    Returns:
        None
    """
    total_loss = 0.0
    totv = 0.0
    totk = 0.0
    totqk = 0.0
    eval_iters = len(eval_dataset) // (batch_size * gradient_accumulation_steps)

    for ii in range(eval_iters - 1 ):
        for k in range(gradient_accumulation_steps):
            batch_idx = ii * gradient_accumulation_steps + k
            print(f"Evaluating batch {batch_idx + 1}/{eval_iters * gradient_accumulation_steps}")

            # Get evaluation batch
            X, Y = get_eval_batch(eval_dataset, batch_idx, batch_size, doc_len, device)

            # Prepare input for compute_combined_loss
            inputs = {
                "input_ids": X,
                "labels": Y
            }

            # Compute combined loss
            eval_loss, val, key, qk = compute_combined_loss(
                aux_model, inputs, True
            )

            # Accumulate loss
            total_loss += eval_loss.item()

            totv += val
            totk += key
            totqk += qk

            

    # Average evaluation loss
    avg_loss = total_loss / (eval_iters * gradient_accumulation_steps)
    avg_v = totv /  (eval_iters * gradient_accumulation_steps)
    avg_k = totk /  (eval_iters * gradient_accumulation_steps)
    avg_qk = totqk /  (eval_iters * gradient_accumulation_steps)

    # Log metrics
    wandb.log({
        "Average Evaluation Loss": avg_loss,
        "Average v loss": avg_v,
        "Average v loss": avg_k,
        "Average v loss": avg_qk
    })

    print(f"Final Evaluation Loss: {avg_loss}")
    print(f"Final v Loss: {avg_v}")
    print(f"Final q Loss: {avg_k}")
    print(f"Final qk Loss: {avg_qk}")




# Callback to save projections periodically
class SaveProjectionCallback(TrainerCallback):
    def __init__(self, save_interval):
        """
        Initialize the callback.

        Args:
            save_interval: Interval to save projections and evaluate the model.
            eval_dataset: Dataset for evaluation.
            aux_model: Auxiliary model being trained.
            base_model: Base model used for comparisons.
            Lv, Lk: Learned value and key projection matrices.
            n_layers_base: Number of layers in the base model.
        """

        self.save_interval = save_interval
        self.Lv = Lv
        self.Lk = Lk 
        self.n_layers_base = n_layers_base

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training has started. Save interval is set to:", self.save_interval)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_interval == 0:
            print(f"Saving projections at step {state.global_step}")
            save_projections(state.global_step, self.save_interval, self.Lv, self.Lk, self.n_layers_base)

            print(f"Evaluating model at step {state.global_step}...")

            
            eval_loss = evaluate_model( eval_dataset = eval_dataset, aux_model = aux_model)

            print(f"Evaluation Loss at step {state.global_step}: {eval_loss}")
            wandb.log({"Evaluation Loss": eval_loss}, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        print("Training has ended. Saving final projections.")
        save_projections(state.global_step, self.save_interval, self.Lv, self.Lk, self.n_layers_base)



# Custom trainer with a combined loss function
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = compute_combined_loss(
            model=model,
            inputs=inputs
        )
        if return_outputs:
            outputs = model(**inputs)
            return loss, outputs
        return loss

# Save projections
def save_projections(step, save_interval, Lv, Lk, n_layers_base):
    tensors = {f"Lv.{layer}": Lv[layer] for layer in range(n_layers_base)}
    tensors.update({f"Lk.{layer}": Lk[layer] for layer in range(n_layers_base)})
    os.makedirs("./saved_projections", exist_ok=True)
    save_path = f"./saved_projections/projections_step_{step}.safetensors"
    save_file(tensors, save_path)
    print(f"Projections saved at step {step} to {save_path}")



# Optimizer setup
optim_groups = [
    {'params': params, 'weight_decay': 0.01}
]

optimizer = torch.optim.AdamW(optim_groups, lr=4e-5, betas= (0.9, 0.95))

# Scheduler setup
total_training_steps = 1000  # Replace with your actual training steps
num_warmup_steps = 25  # Adjust warmup to 10% of total steps

# Create the scheduler with "cosine_with_min_lr"
lr_scheduler = get_scheduler(
    name="cosine",  # Use "cosine" for the standard cosine schedule
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,  # Define warmup steps
    num_training_steps=total_training_steps  # Total training steps
)


# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps, 
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-4,
    max_steps=1000,
    logging_steps=10,
    output_dir="./sft_output",
    eval_strategy="steps",
    eval_steps=10,
    logging_dir="./logs",
    report_to="wandb",
    save_total_limit=2,
    load_best_model_at_end=True
)


# Create the custom trainer instance
trainer = CustomSFTTrainer(
    model=aux_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=1025,
    tokenizer=tokenizer,
    optimizers=(optimizer, lr_scheduler),  # Pass optimizer and scheduler explicitly
    callbacks=[
        SaveProjectionCallback(
            save_interval=100
        )
   ]
)

# Start training
trainer.train()

# Save the final model
trainer.save_model(training_args.output_dir)

# Finalize WandB
wandb.finish()