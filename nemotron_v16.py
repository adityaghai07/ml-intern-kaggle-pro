# nemotron_v16.py — Golden reference script
# Base: train_cot_v3.py (v9) which RAN SUCCESSFULLY on Kaggle and submitted.
# Changes from v9:
#   - Full 2907 CoT samples, 1 epoch (instead of 1200 samples, 2 epochs)
#   - LR=1e-4 (kienngx experiments show better than 5e-5)
#   - lora_alpha=64 (2x rank for better scaling)
#   - enable_thinking: CoT includes <think> tags for reasoning
# RULE: Do NOT touch sections 1-2 (env setup + imports) unless they break.
#       They are proven to work on Kaggle's BYOD runtime.

import subprocess, sys, os, stat, shutil, gc, zipfile, types, glob
from pathlib import Path

# =============================================================================
# 1.  ENVIRONMENT SETUP  — PROVEN WORKING (from v9/train_cot_v3.py)
#     DO NOT MODIFY unless you have proof it breaks on current Kaggle runtime.
# =============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Cutlass (required by mamba_ssm) ---
# Try both hyphen and underscore variants — Kaggle runtime may use either.
cutlass_candidates = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/",
]
import site as _site
for _cp in cutlass_candidates:
    if os.path.isdir(_cp):
        _site.addsitedir(_cp)
        print(f"Cutlass: {_cp}")
        break
else:
    print("WARNING: cutlass path not found")

# --- Triton ptxas-blackwell fix ---
ptxas_candidates = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell",
]
ptxas_dst = "/tmp/ptxas-blackwell"
ptxas_src = None
for _p in ptxas_candidates:
    if os.path.exists(_p):
        ptxas_src = _p
        break

if ptxas_src and not os.path.exists(ptxas_dst):
    shutil.copy2(ptxas_src, ptxas_dst)
    os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    import triton.backends.nvidia as _nv
    src_bin = os.path.dirname(ptxas_src)
    dst_bin = "/tmp/triton_nvidia_bin"
    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
    for f in os.listdir(dst_bin):
        fp = os.path.join(dst_bin, f)
        if os.path.isfile(fp):
            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _nv.__file__ = os.path.join(dst_bin, "..", "__init__.py")
    os.environ["TRITON_PTXAS_PATH"] = ptxas_dst
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst
    print(f"Triton ptxas fix: {ptxas_src}")
elif os.path.exists(ptxas_dst):
    os.environ["TRITON_PTXAS_PATH"] = ptxas_dst
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst
    print("Triton ptxas: already present")
else:
    print("WARNING: ptxas-blackwell not found")

# --- RMSNorm pure-PyTorch bypass ---
import importlib, torch, torch.nn.functional as F

def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast: x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None: out = out + bias.float()
    if z is not None: out = out * F.silu(z.float())
    return out.to(dtype)

for _name, _mod in list(sys.modules.items()):
    if hasattr(_mod, "rmsnorm_fn"):
        _mod.rmsnorm_fn = _pure_rmsnorm_fn
print("RMSNorm: patched to pure PyTorch")

# --- Offline package install (v9 pattern + v12 robustness) ---
# Strategy: try system imports first; only install from offline if missing.
# Use --find-links (not --no-deps alone) for proper dependency resolution.
# If that fails, fall back to --no-deps per .whl (v12 pattern).
offline_candidates = [
    "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages",
    "/kaggle/input/nvidia-nemotron-offline-packages/offline_packages",
    "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages",
]
offline_dir = next((p for p in offline_candidates if os.path.isdir(p)), "")

# Also check for utility script site dir
utility_candidates = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/",
]
for _u in utility_candidates:
    if os.path.isdir(_u):
        _site.addsitedir(_u)
        print(f"Utility script: {_u}")
        break

target_dir = "/kaggle/working/packages"
os.makedirs(target_dir, exist_ok=True)

def _try_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None

# Check what's already available
_trl = _try_import("trl")
_ds  = _try_import("datasets")
print(f"System trl: {'v' + _trl.__version__ if _trl else 'NOT FOUND'}")
print(f"System datasets: {'v' + _ds.__version__ if _ds else 'NOT FOUND'}")

if not _trl or not _ds:
    installed = False
    # Method 1: v9 pattern — --find-links with full dependency resolution
    if offline_dir:
        print(f"Installing from offline: {offline_dir}")
        pkgs = []
        if not _ds:  pkgs.append("datasets")
        if not _trl: pkgs.append("trl")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "--no-index", "--find-links", offline_dir,
                 "--target", target_dir] + pkgs,
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                installed = True
                print(f"Installed {pkgs} (--find-links)")
            else:
                print(f"--find-links failed: {result.stderr[-300:]}")
        except Exception as e:
            print(f"--find-links exception: {e}")

    # Method 2: v12 fallback — --no-deps per .whl file
    if not installed:
        print("Fallback: --no-deps per .whl")
        for pkg, pattern in [("datasets", "datasets-*.whl"), ("trl", "trl-*.whl")]:
            if _try_import(pkg):
                continue
            wheels = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
            if wheels:
                w = wheels[-1]  # latest version
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "--no-index", "--no-deps", "--target", target_dir, w],
                    capture_output=True, text=True,
                )
                print(f"Installed {pkg} from {os.path.basename(w)}")
        # Also install lightweight deps that --no-deps skips
        for pkg, pattern in [("xxhash", "xxhash-*.whl"), ("dill", "dill-*.whl"),
                             ("multiprocess", "multiprocess-*.whl")]:
            if not _try_import(pkg):
                wheels = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
                if wheels:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q",
                         "--no-index", "--no-deps", "--target", target_dir, wheels[-1]],
                        capture_output=True, text=True,
                    )

    # Add target to path — use addsitedir for .pth file support (v9 pattern)
    sys.path.insert(0, target_dir)
    _site.addsitedir(target_dir)

    # CRITICAL: if offline install put transformers in target_dir, delete it
    # to avoid CUDA version conflicts (v10/v11 root cause)
    local_tf = os.path.join(target_dir, "transformers")
    if os.path.isdir(local_tf):
        shutil.rmtree(local_tf, ignore_errors=True)
        for f in glob.glob(os.path.join(target_dir, "transformers*")):
            if os.path.isdir(f):
                shutil.rmtree(f, ignore_errors=True)
            elif os.path.isfile(f):
                os.remove(f)
        print("Cleaned offline transformers (avoid CUDA conflict)")

# Stub problematic mamba submodules (from v9)
for _mod_name in [
    "mamba_ssm.modules.mamba3",
    "mamba_ssm.ops.cute",
    "mamba_ssm.ops.cute.mamba3",
    "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
]:
    sys.modules[_mod_name] = types.ModuleType(_mod_name)
sys.modules["mamba_ssm.modules.mamba3"].Mamba3 = None

# =============================================================================
# 2.  IMPORTS  — Verify everything loads before doing expensive model download
# =============================================================================
import polars as pl
import kagglehub
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}, mem: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")
print(f"transformers: {__import__('transformers').__version__} from {__import__('transformers').__file__}")
print(f"peft: {__import__('peft').__version__}")
print(f"trl: {__import__('trl').__version__}")

# =============================================================================
# 3.  HYPERPARAMETERS
#     v16 changes from v9: LR 5e-5→1e-4, alpha 32→64, samples 1200→2907, epochs 2→1
# =============================================================================
LORA_RANK      = 32
LORA_ALPHA     = 64        # 2x rank (v9 used 32)
MAX_SEQ_LEN    = 2048
NUM_EPOCHS     = 1          # 1 epoch with full dataset (v9 used 2 with 1200)
BATCH_SIZE     = 1
GRAD_ACCUM     = 4
LR             = 1e-4       # kienngx showed better than 5e-5
SUBSAMPLE_SIZE = 2907       # full CoT dataset, 1 epoch fits ~5h
OUTPUT_DIR     = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nConfig: rank={LORA_RANK}, alpha={LORA_ALPHA}, lr={LR}, "
      f"epochs={NUM_EPOCHS}, seq={MAX_SEQ_LEN}, samples={SUBSAMPLE_SIZE}")

# =============================================================================
# 4.  DATA
# =============================================================================
# Search for CoT dataset (kienngx verified-correct CoT reasoning chains)
cot_candidates = [
    "/kaggle/input/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels/final_Nemotron_training_data.csv",
    "/kaggle/input/nemotron-30b-competition-trainingdata-cot-labels/final_Nemotron_training_data.csv",
]
cot_path = None
for c in cot_candidates:
    if os.path.exists(c):
        cot_path = c
        break
if not cot_path:
    for f in glob.glob("/kaggle/input/**/final_Nemotron_training_data.csv", recursive=True):
        cot_path = f
        break
if not cot_path:
    for f in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        cot_path = f
        break

print(f"Dataset: {cot_path}")
train_df = pl.read_csv(cot_path)
USE_COT = "generated_cot" in train_df.columns
print(f"Rows: {len(train_df)}, CoT: {USE_COT}, Cols: {train_df.columns}")

if len(train_df) > SUBSAMPLE_SIZE:
    train_df = train_df.sample(n=SUBSAMPLE_SIZE, seed=42)
    print(f"Subsampled to {SUBSAMPLE_SIZE}")

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Model: {MODEL_PATH}")

# =============================================================================
# 5.  PROMPT FORMATTING
# =============================================================================
def build_training_text(example):
    prompt = example["prompt"]
    answer = str(example["answer"])
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."

    if USE_COT and example.get("generated_cot"):
        cot = str(example["generated_cot"])
        assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"
    else:
        assistant_msg = f"<think>\nLet me work through this step by step.\n</think>\n\\boxed{{{answer}}}"

    try:
        messages = [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    return {"text": text}

hf_dataset = Dataset.from_pandas(train_df.to_pandas())
hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names)
print(f"Formatted: {len(hf_dataset)} examples")
print(f"Sample: {hf_dataset[0]['text'][:300]}")

# =============================================================================
# 6.  MODEL LOADING
# =============================================================================
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()

for _name, _mod in sys.modules.items():
    if "modeling_nemotron_h" in _name:
        _mod.is_fast_path_available = False
print("Model loaded, fast_path disabled")

# =============================================================================
# 7.  LoRA
# =============================================================================
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =============================================================================
# 8.  TRITON COMPILER FIX
# =============================================================================
try:
    import triton.backends.nvidia.compiler as _nv_compiler
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst
    _nv_compiler.get_ptxas_version = lambda arch: "12.0"
    print("Triton compiler patched")
except Exception as e:
    print(f"Triton compiler patch skipped: {e}")

# =============================================================================
# 9.  TRAINING
# =============================================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=30,
    disable_tqdm=True,
    logging_strategy="steps",
    logging_first_step=True,
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

print(f"\nTraining: {len(hf_dataset)} samples x {NUM_EPOCHS} epochs, LR={LR}, batch={BATCH_SIZE}x{GRAD_ACCUM}")
trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train()
print("Training complete!")

# =============================================================================
# 10.  SAVE + ZIP
# =============================================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nAdapter files in {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    print(f"  {f} ({os.path.getsize(os.path.join(OUTPUT_DIR, f))/1024:.1f} KB)")

zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        zf.write(os.path.join(OUTPUT_DIR, fname), fname)

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
with zipfile.ZipFile(zip_path, "r") as zf:
    contents = zf.namelist()
    print(f"Contents: {contents}")
    assert "adapter_config.json" in contents, "MISSING: adapter_config.json"
    assert "adapter_model.safetensors" in contents, "MISSING: adapter_model.safetensors"

print("\nDone! submission.zip ready.")
