# K/V Projection Training with Auxiliary Model and Custom Loss

This project demonstrates a supervised fine-tuning (SFT) setup for a large language model. We use a base model (`Llama-3.1-8B`) and an auxiliary model (`Llama-3.2-1B`) to train the auxiliary model with a custom loss function. The loss incorporates both standard cross-entropy loss on the output and a custom loss based on the alignment of the key/value (K/V) matrices between the base and auxiliary models.

Additionally, the training process periodically saves custom projection matrices (`Lv` and `Lk`) in the `safetensors` format for efficient storage and reusability.

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- `safetensors` library
- `wandb` for experiment tracking

Install required libraries with:
```bash
pip install torch transformers datasets safetensors wandb
```

