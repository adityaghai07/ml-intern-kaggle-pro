# NVIDIA Nemotron Model Reasoning Challenge — Baseline Solution

## Competition Summary
- **Slug**: `nvidia-nemotron-model-reasoning-challenge`
- **Prize**: $106,388
- **Deadline**: June 15, 2026
- **Metric**: NVIDIA Nemotron Metric
- **Top LB Score**: 0.87 | Competitive threshold: ~0.86

## Key Findings from Research

### Competition Structure
- **NOT a CSV submission** — this is a model adapter submission
- You submit a **`submission.zip`** containing a trained LoRA adapter
- The evaluator loads the adapter on top of `metric/nemotron-3-nano-30b-a3b-bf16`
  and runs it against a hidden test benchmark
- No raw data files are listed in competition data — training data is accessed via
  `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv` inside notebooks

### Model
- **Nemotron-3 Nano 30B A3B** (Mamba-based MoE, ~3B active params)
- Available via: `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")`
- Runs on single H100 with LoRA

### Submission Format
```
submission.zip
├── adapter_config.json       ← REQUIRED
└── adapter_model.safetensors ← REQUIRED
```

### What Top Notebooks Do
| Notebook | Votes | Key Insight |
|----------|-------|-------------|
| NVIDIA Nemotron Submission Demo | 1831 | Official demo: LoRA r=32, show format |
| SFTTrainer training | 745 | SFTTrainer + SFTConfig, `all-linear` targets |
| Training (CoT + Labels) | 629 | CoT reasoning chains, 2048 seq len, 2 epochs → **best quality** |
| Nemotron SFT LoRA with CoT | 427 | Similar CoT approach |

### Winning Recipe (from top community notebooks)
```python
LORA_RANK    = 32
LORA_ALPHA   = 32
target_modules = "all-linear"
MAX_SEQ_LEN  = 2048       # Supports full reasoning chains
NUM_EPOCHS   = 2
LR           = 5e-5
SUBSAMPLE    = 1200        # rows from train.csv
```

**Critical patches required on Kaggle:**
1. Pure PyTorch `rmsnorm_fn` replacement (avoids Triton kernel crash)
2. `ptxas-blackwell` binary copy + chmod (Kaggle H100 Blackwell GPU)
3. `is_fast_path_available = False` on Mamba modules
4. `TRITON_PTXAS_BLACKWELL_PATH` env var

## How to Use

### Run on Kaggle (recommended)
1. Upload `train_adapter.py` as a Kaggle notebook
2. Add dataset: `nvidia-nemotron-3-reasoning-challenge`
3. Add model: `metric/nemotron-3-nano-30b-a3b-bf16`
4. Enable H100 GPU (required)
5. Run — output `submission.zip` appears in `/kaggle/working/`
6. Submit via Kaggle UI → "Submit to Competition"

### Files
- `train_adapter.py` — Main training script (baseline, ~0.84-0.86 target)

## Improvement Ideas (to reach 0.87+)
1. **Use CoT augmented dataset** (`kienngx/nemotron-30b-competition-trainingdata-cot-labels`)
   instead of raw `train.csv` — adds chain-of-thought reasoning traces
2. **More data** — increase `SUBSAMPLE_SIZE` to full dataset
3. **More epochs** — try 3 epochs with cosine schedule
4. **Higher rank** — r=32 is the max allowed by competition rules
5. **Better prompting** — include step-by-step reasoning instructions
