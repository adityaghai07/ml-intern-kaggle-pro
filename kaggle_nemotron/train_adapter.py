"""
NVIDIA Nemotron Model Reasoning Challenge - Baseline Training Script
Adapted from top community notebooks:
  - ryanholbrook/nvidia-nemotron-submission-demo (official demo, 1831 votes)
  - dennisfong/nvidia-nemotron-sfttrainer-training (SFTTrainer approach, 745 votes)
  - kienngx/nvidia-nemotron-training-cot-labels (CoT + Labels, 629 votes)

Strategy:
  - Fine-tune Nemotron-3-Nano-30B-A3B via LoRA (rank=32)
  - SFT on train.csv with chat-template formatting: user=prompt+boxed instruction, assistant=answer
  - Package adapter into submission.zip (adapter_config.json + adapter_model.safetensors)

Key findings from notebooks:
  - Model: metric/nemotron-3-nano-30b-a3b-bf16 (Mamba-based MoE, ~3B active params)
  - Submission format: ZIP with PEFT adapter files
  - Triton/RMSNorm patches required for Kaggle H100 runtime
  - Best approach: CoT labels + longer sequences (2048) + 2 epochs → ~0.86 score
  - LoRA targets: "all-linear", rank=32, alpha=32
"""

import os
import sys
import subprocess
import stat
import shutil
import gc
import zipfile

# Install packages — try multiple strategies
# Strategy 1: check if packages already available in docker image
_NEED_INSTALL = False
try:
    import trl, peft, datasets
    print("Packages already available in docker image.")
except ImportError:
    _NEED_INSTALL = True

if _NEED_INSTALL:
    # Strategy 2: offline dataset (no internet needed)
    # Dataset sources mount at /kaggle/input/datasets/<owner>/<slug>/
    _OFFLINE_CANDIDATES = [
        "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages",
        "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages",
        "/kaggle/input/nvidia-nemotron-offline-packages/offline_packages",
        "/kaggle/input/nvidia-nemotron-offline-packages",
    ]
    _offline_dir = next((p for p in _OFFLINE_CANDIDATES if os.path.isdir(p)), "")
    if not _offline_dir:
        # Brute-force search for .whl files under /kaggle/input
        import glob as _glob
        _whl_files = _glob.glob("/kaggle/input/**/*.whl", recursive=True)
        if _whl_files:
            _offline_dir = os.path.dirname(_whl_files[0])
            print(f"Found wheel files at: {_offline_dir}")
    if _offline_dir:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "--no-index", f"--find-links={_offline_dir}",
            "trl", "peft", "datasets", "--ignore-installed",
        ])
        print(f"Installed packages from offline dataset: {_offline_dir}")
    else:
        # Debug: show what's under /kaggle/input for diagnosis
        for _root, _dirs, _files in os.walk("/kaggle/input"):
            _depth = _root.replace("/kaggle/input", "").count(os.sep)
            if _depth < 5:
                print(f"{'  ' * _depth}{os.path.basename(_root)}/  files={_files[:5]}")
        # Strategy 3: PyPI (requires internet — will fail in no-internet kernels)
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "trl", "peft", "datasets",
            ])
            print("Installed packages from PyPI.")
        except subprocess.CalledProcessError:
            print("WARNING: pip install failed (no internet?). Checking if packages exist...")
            import trl, peft, datasets  # will raise if truly missing

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add cutlass from utility script (required by mamba_ssm for Nemotron model)
import site
_CUTLASS_PATH = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/"
if os.path.isdir(_CUTLASS_PATH):
    site.addsitedir(_CUTLASS_PATH)
    print(f"Added cutlass from: {_CUTLASS_PATH}")

import torch
import torch.nn.functional as F
try:
    import kagglehub
except ImportError:
    kagglehub = None
try:
    import polars as pl
except ImportError:
    pl = None
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── 1. Triton / RMSNorm patches (required on Kaggle H100 runtime) ──────────
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn

# Kernel source is mounted at /kaggle/usr/lib/notebooks/<owner>/<slug_underscored>/
# Try multiple possible paths
_PTXAS_CANDIDATES = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell",
]
src = next((p for p in _PTXAS_CANDIDATES if os.path.exists(p)), "")
dst = "/tmp/ptxas-blackwell"
if os.path.exists(src):
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    import triton.backends.nvidia as nv_backend
    src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
    dst_bin = "/tmp/triton_nvidia_bin"
    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
    for f in os.listdir(dst_bin):
        fp = os.path.join(dst_bin, f)
        if os.path.isfile(fp):
            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
    os.environ["TRITON_PTXAS_PATH"] = dst
    print("✓ Triton ptxas fix applied.")
else:
    print("⚠ Triton ptxas fix skipped (not on Kaggle runtime).")

# ── 2. Hyperparameters ──────────────────────────────────────────────────────
SUBSAMPLE_SIZE = 1200       # From CoT notebook; more data → better reasoning
LORA_RANK      = 32         # Max allowed by competition
LORA_ALPHA     = 32         # Equal alpha for stable training
MAX_SEQ_LEN    = 2048       # Supports full reasoning chains (CoT approach)
NUM_EPOCHS     = 2          # 2 epochs for better convergence (CoT notebook)
BATCH_SIZE     = 1
GRAD_ACCUM     = 4          # Effective batch = 4
LR             = 5e-5       # From CoT notebook (better than 2e-4 for reasoning)
OUTPUT_DIR     = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 3. Load data ────────────────────────────────────────────────────────────
# Model is mounted via model_sources at /kaggle/input/models/...
_MODEL_MOUNTED = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
if os.path.isdir(_MODEL_MOUNTED):
    MODEL_PATH = _MODEL_MOUNTED
    print(f"Model path (mounted): {MODEL_PATH}")
elif kagglehub is not None:
    MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    print(f"Model path (downloaded): {MODEL_PATH}")
else:
    raise RuntimeError("Model not found at mounted path and kagglehub not available.")

# Competition data — search all possible mount locations recursively
import glob
TRAIN_CSV = None
_SEARCH_ROOTS = ["/kaggle/input"]
for root in _SEARCH_ROOTS:
    if not os.path.isdir(root):
        continue
    # Walk up to 3 levels deep looking for train.csv
    for depth_pattern in ["*/train.csv", "*/*/train.csv", "*/*/*/train.csv"]:
        matches = glob.glob(os.path.join(root, depth_pattern))
        if matches:
            TRAIN_CSV = matches[0]
            break
    if TRAIN_CSV:
        break

if TRAIN_CSV is None:
    # Debug: show full directory tree
    for root_dir, dirs, files in os.walk("/kaggle/input"):
        depth = root_dir.replace("/kaggle/input", "").count(os.sep)
        if depth < 4:
            print(f"{'  ' * depth}{os.path.basename(root_dir)}/  files={files[:5]}")
    raise FileNotFoundError("train.csv not found anywhere under /kaggle/input/")

print(f"Found train.csv at: {TRAIN_CSV}")

if pl is not None:
    train_df = pl.read_csv(TRAIN_CSV)
    print(f"Total training samples: {len(train_df)}")
    train_df = train_df.sample(n=min(SUBSAMPLE_SIZE, len(train_df)), seed=42)
    print(f"Subsampled to: {len(train_df)}")
    hf_dataset = Dataset.from_pandas(train_df.to_pandas())
else:
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Total training samples: {len(train_df)}")
    train_df = train_df.sample(n=min(SUBSAMPLE_SIZE, len(train_df)), random_state=42)
    print(f"Subsampled to: {len(train_df)}")
    hf_dataset = Dataset.from_pandas(train_df)
print(f"Dataset columns: {hf_dataset.column_names}")

# ── 4. Tokenizer & prompt formatting ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_training_text(example):
    """
    Format: user asks problem + boxed instruction; assistant gives answer.
    If CoT column exists, prepend the chain-of-thought before the boxed answer.
    This mirrors the CoT approach from kienngx which targets ~0.86 score.
    """
    prompt = example["prompt"]
    answer = example["answer"]

    user_msg = prompt + "\nPut your final answer inside \\boxed{}."

    # Use CoT if available (from augmented datasets); otherwise use direct answer
    cot = example.get("generated_cot", "")
    if cot:
        assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"
    else:
        assistant_msg = f"\\boxed{{{answer}}}"

    try:
        messages = [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    return {"text": text}

hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names)
print(f"Example formatted text (first 400 chars):\n{hf_dataset[0]['text'][:400]}")

# ── 5. Model loading ─────────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()

# Disable fast-path Mamba CUDA kernels (stability fix for Kaggle)
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"  Patched {name}: is_fast_path_available = False")

# ── 6. LoRA setup ─────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",   # All linear layers → best reasoning perf
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 7. Triton compiler env fix ───────────────────────────────────────────────
try:
    import triton.backends.nvidia.compiler as nv_compiler
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
    nv_compiler.get_ptxas_version = lambda arch: "12.0"
    print("✓ Triton compiler patched.")
except Exception as e:
    print(f"⚠ Triton compiler patch skipped: {e}")

# ── 8. Training ──────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=30,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

print("Starting training...")
trainer.train()

# ── 9. Save adapter ──────────────────────────────────────────────────────────
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nAdapter saved to {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f}  ({size/1024:.1f} KB)")

# ── 10. Package submission.zip ───────────────────────────────────────────────
zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        zf.write(fpath, fname)

print(f"\nCreated {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")

with zipfile.ZipFile(zip_path, 'r') as zf:
    names = zf.namelist()
    print(f"Contents: {names}")
    assert "adapter_config.json" in names,      "MISSING: adapter_config.json"
    assert "adapter_model.safetensors" in names, "MISSING: adapter_model.safetensors"

print("\n✅ submission.zip is ready to submit to the competition!")
