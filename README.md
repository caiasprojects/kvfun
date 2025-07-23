# Cache Me If You Can: KV Projections and Recomputation

A research implementation for accelerating Large Language Model (LLM) inference through **KV cache projection** and **selective recomputation**. This project explores novel approaches to reduce Time-to-First-Token (TTFT) while maintaining generation quality by using smaller auxiliary models to predict KV caches for larger base models.

## 🎯 Project Overview

**"Cache Me If You Can: KV Projections and Recomputation"** ([📄 Paper Link](https://drive.google.com/file/d/11ArntGE9wBYyUaTdvZx3iSV5pvgG96DO/view?usp=sharing)). It proposes a new approach to optimize TTFT in large language models. Unlike previous methods that directly project KV caches, we:

1. **Project from auxiliary activations** rather than KV caches directly
2. **Use a combined loss function** (key alignment, value alignment, query-key similarity)
3. **Implement selective KV recomputation** during inference to maintain quality

### Key Results
- **4% speedup** in TTFT over 8B model
- **11% performance improvement** over 1B model 
- Evaluated on SQuAD v2 dataset with prompts ranging from 1000-2400 tokens

## 🏗️ Architecture

### Core Components

1. **Auxiliary Model (Llama 3.2 1B)**: Processes prompts and provides activations for projection
2. **Base Model (Llama 3.1 8B)**: Target model for generation with projected KV cache
3. **Projection Matrices**: Learned matrices (`Lk`, `Lv`) that map auxiliary activations to base model's key/value spaces
4. **Selective Recomputation**: Dynamic recalculation of high-attention KV pairs during inference

### Model Classes

- **`KV_hybrid_model`**: Main inference class with selective recomputation
- **`KV_prediction_model`**: Basic KV projection without recomputation  
- **`KV_prediction_model_with_copy`**: KV projection with static copying of first/last tokens
- **`Baseline_model`**: Standard model for comparison

## 🔬 Technical Approach

### 1. Activation-Based Projection

Instead of projecting KV caches directly, we leverage auxiliary model activations:

```python
K_proj = X_aux @ L_K  # Project activations to keys
V_proj = X_aux @ L_V  # Project activations to values
```

This approach is more effective because activations retain richer intermediate representations than the final KV outputs.

### 2. Multi-Component Loss Function

Our training optimizes three loss components:

- **Value Alignment Loss**: `L_V = ||X_aux @ L_V - V_base||_1`
- **Key Alignment Loss**: `L_K = ||X_aux @ L_K - K_base||_1`  
- **Query-Key Similarity Loss**: `L_QK = ||Q_base^T K_aux - Q_base^T K_base||_F^2`

### 3. Selective KV Recomputation

During inference, we:
1. Calculate token importance using auxiliary model attention scores
2. Select high-importance intervals for recomputation
3. Dynamically recompute KV pairs for these intervals using the base model
4. Merge intervals to minimize the number of forward passes

## 🚀 Installation

### Requirements

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Training Projection Matrices

Train new projection matrices from scratch:

```bash
python train_caia.py
```

**Key hyperparameters:**
- Document length: 1024 tokens
- Batch size: 4 (effective: 32 with gradient accumulation)
- Learning rate: 4e-3 → 4e-4 (cosine decay)
- Training steps: 4,096
- Dataset: C4 (English split)

### Inference

#### Basic Inference

```python
from model_generate import KV_hybrid_model

# Initialize model
model = KV_hybrid_model(baseline_base=False, baseline_aux=False)

# Configure recomputation
recalculate_args = {
    "recalculate": True,
    "interval_size": 80,      # Tokens per recomputation interval
    "num_intervals": 2        # Number of intervals to recompute
}

# Generate response
response, ttft, cache, prompt_len = model.call_with_prompt(
    "Your prompt here", 
    max_new_tokens=100,
    recalculate_args=recalculate_args
)
```

#### Baseline Comparisons

```python
# Pure base model (8B)
baseline_8b = KV_hybrid_model(baseline_base=True)

# Pure auxiliary model (1B) 
baseline_1b = KV_hybrid_model(baseline_aux=True)

# KV projection without recomputation
kv_only = KV_hybrid_model()
```

### Evaluation

#### SQuAD v2 Evaluation

Generate predictions:
```bash
python generate_for_squad.py
```

Evaluate results:
```bash
cd squad/
python evaluate_squad.py squad_data.json answers/
```

## 📊 Results & Analysis

### Key Findings

1. **TTFT Optimization**: 4% speedup over 8B baseline while maintaining significantly better quality than 1B model
2. **Quality-Speed Tradeoff**: Achieves 11% performance improvement over 1B model with minimal speed penalty
3. **Scalability**: Successfully demonstrates scaling to larger model pairs (1B→8B)

### Attention Visualization

The project includes visualization tools for analyzing KV differences:

```python
from utils import plot_kv_differences
plot_kv_differences(predicted_cache, true_cache, prompt_len, "output.png")
```

## 📁 Project Structure

```
kvfun/
├── train_caia.py              # Main training script for projection matrices
├── model_generate.py          # Inference models with selective recomputation  
├── utils.py                   # Utility functions and model classes
├── generate_for_squad.py      # SQuAD dataset generation script
├── get_info.py                # Model analysis utilities
├── squad/                     # SQuAD evaluation
│   ├── evaluate_squad.py      # Official SQuAD evaluation script
│   ├── squad_data.json        # SQuAD v2 dataset
│   └── answers/               # Generated predictions
├── kv_differences/            # KV cache analysis plots
├── triviaqa/                  # TriviaQA evaluation (optional)
└── requirements.txt           # Dependencies
```

