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


# Initialize WandB for experiment tracking
wandb.init(
    project="sft-training",
    name="sft-run-kv-training"
)

IGNORE_INDEX = -100
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=1025, padding="max_length", truncation=True)


try:
    tokenized_datasets = load_from_disk("cached_tokenized_datasets")
except FileNotFoundError:
    # Load dataset in streaming mode
    dataset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz", streaming=True)
    
    # Tokenize with streaming and parallel processing (note that streaming data does not support `.select` or `.shuffle`)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=8)
    
    # You may need to convert the dataset back to non-streaming before saving, as streaming datasets cannot be saved
    tokenized_datasets = DatasetDict({"train": tokenized_datasets})
    tokenized_datasets.save_to_disk("cached_tokenized_datasets")


# Select the train and evaluation subsets (shuffling cached data)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(160 * 10 * 1000))
eval_dataset = tokenized_datasets["train"].shuffle(seed=142).select(range(1000))

# Load models
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
aux_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the appropriate device
base_model.to(device)
aux_model.to(device)

# Freeze base model parameters
for param in base_model.parameters():
    param.requires_grad = False

# Set up auxiliary model for training
for param in aux_model.parameters():
    param.requires_grad = True

# Initialize hooks for K/V extraction
base_ks, base_vs = {}, {}
aux_ks, aux_vs = {}, {}
n_layers_base = 32
n_layers_aux = 16
aux_kv_dim = 512
base_kv_dim = 1024

# Initialize Lv and Lk projection matrices
Lv, Lk = {}, {}
for j in range(n_layers_base):
    Lv[j] = torch.randn((aux_kv_dim, base_kv_dim), requires_grad=True).to(device)
    Lk[j] = torch.randn((aux_kv_dim, base_kv_dim), requires_grad=True).to(device)

# Hook function to extract K/V matrices
def kv_hook(module, in_x, output, index, kvs):
    kvs[index] = output

# Register hooks for K/V extraction
for i in range(n_layers_base):
    base_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index=i, kvs=base_vs))
    base_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index=i, kvs=base_ks))

for i in range(n_layers_aux):
    aux_model.model.layers[i].self_attn.v_proj.register_forward_hook(partial(kv_hook, index=i, kvs=aux_vs))
    aux_model.model.layers[i].self_attn.k_proj.register_forward_hook(partial(kv_hook, index=i, kvs=aux_ks))

# Combined loss function

# Determine the device (cuda if available, otherwise cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the appropriate device
base_model.to(device)
aux_model.to(device)

def compute_combined_loss(model, inputs, base_model, base_vs, base_ks, aux_vs, aux_ks, Lv, Lk, n_layers_base):
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    aux_logits = model(**inputs).logits
    targets = inputs["labels"]
    base_logits = base_model(inputs["input_ids"]).logits

    loss = 0
    aux_logitsr = aux_logits.reshape(-1, aux_logits.shape[-1])
    targets_flat = targets.flatten()

    # Custom K/V loss
    for layer in range(n_layers_base):
        aux_layer = layer // 2
        num_el = base_vs[layer].numel() if base_vs[layer].numel() > 0 else 1
        loss += torch.mean(torch.linalg.matrix_norm(((aux_vs[aux_layer] @ Lv[layer]) - base_vs[layer]).float(), ord='fro')) / (num_el * n_layers_base)
        loss += torch.mean(torch.linalg.matrix_norm(((aux_ks[aux_layer] @ Lk[layer]) - base_ks[layer]).float(), ord='fro')) / (num_el * n_layers_base)

    # Cross-entropy loss
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    ce_loss = loss_func(aux_logitsr, targets_flat)
    total_loss = loss + ce_loss

    return total_loss
    

def save_projections(step, save_interval, Lv, Lk, n_layers_base):
    if step % save_interval == 0:
        tensors = {**{f"Lv.{layer}": Lv[layer] for layer in range(n_layers_base)},
                   **{f"Lk.{layer}": Lk[layer] for layer in range(n_layers_base)}}
        save_file(tensors, f"out/projections.step{step}.safetensors")
        print(f"Saved projections at step {step}")

# Configure training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=6e-4,
    max_steps=8000,  # Set maximum number of training steps
    num_train_epochs=1,
    logging_steps=25,
    eval_steps=100,
    evaluation_strategy="steps",
    output_dir="./sft_output",
    logging_dir="./logs",
    report_to="wandb",
    save_total_limit=2,
    load_best_model_at_end=True,
)

class SaveProjectionCallback(TrainerCallback):
    def __init__(self, save_interval):
        self.save_interval = save_interval

    # Called at the beginning of training
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training has started. Save interval is set to:", self.save_interval)

    # Save projections periodically at the end of each step
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_interval == 0:
            save_projections(state.global_step, self.save_interval, Lv, Lk, n_layers_base)

    # Called at the end of training to save final projections
    def on_train_end(self, args, state, control, **kwargs):
        print("Training has ended. Saving final projections.")
        save_projections(state.global_step, self.save_interval, Lv, Lk, n_layers_base)

class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Calculate the loss using your custom loss function
        loss = compute_combined_loss(model, inputs, base_model, base_vs, base_ks, aux_vs, aux_ks, Lv, Lk, n_layers_base)

        if return_outputs:
            # Return both loss and outputs if return_outputs is True
            outputs = model(**inputs)
            return loss, outputs
        else:
            # Otherwise, return only the loss
            return loss

# Create the SFTTrainer instance
# Create the custom trainer instance
trainer = CustomSFTTrainer(
    model=aux_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[SaveProjectionCallback(save_interval=800)]
)

# Start training and automatically evaluate every 100 steps
trainer.train()

# Save the final model
trainer.save_model(training_args.output_dir)

# Finish the WandB run
wandb.finish()
