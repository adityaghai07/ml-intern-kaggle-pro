<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# ML Intern Kaggle Pro

An autonomous ML agent that researches, trains, and competes on Kaggle — with full support for pushing notebooks to Kaggle GPUs, polling results, error recovery, and cross-session learning. Forked from [huggingface/ml-intern](https://github.com/huggingface/ml-intern) and extended with a complete Kaggle competition integration.

## What's New (vs upstream ml-intern)

### Kaggle Competition Integration
A full-loop autonomous Kaggle workflow built on top of the existing HuggingFace agent:

- **17 Kaggle operations** — browse competitions, read notebooks/discussions, push scripts to Kaggle GPUs, poll execution, download output, submit predictions, and track scores
- **Autonomous polling** — after pushing a notebook, the agent polls status at increasing intervals (5min → 10min → 15min) and auto-recovers from errors
- **Error recovery** — on notebook failure, downloads logs, reads working notebooks for solutions, fixes the script, and re-pushes
- **Cross-session learning** — run history persists at `~/.kaggle/agent_runs/`, so the agent remembers every error, fix, and score from previous sessions and never repeats the same mistake
- **Daily submission cap** — enforces 3 submissions/day to conserve quota; excess time goes to research
- **Runtime pitfall avoidance** — hard-won lessons about Kaggle mount paths, GPU accelerators, BYOD docker images, offline packages, and cutlass dependencies are baked into the system prompt

### New Files

| File | Purpose |
|------|---------|
| `agent/tools/kaggle_tool.py` | Core Kaggle tool — 17 operations (list, read, push, poll, submit, save_run, run_history) with httpx REST + score/run log persistence |
| `agent/tools/kaggle_notebooks.py` | Notebook generation utility — creates competition notebooks with nbformat |
| `tests/unit/test_kaggle_tool.py` | Unit tests for auth, score persistence, dispatch, approval gates |
| `kaggle_nemotron/` | Working example: NVIDIA Nemotron Reasoning Challenge baseline |

### Modified Files

| File | Change |
|------|--------|
| `agent/core/tools.py` | Registered kaggle tool in `create_builtin_tools()` |
| `agent/core/agent_loop.py` | Added approval gates for `submit` and `push_notebook` |
| `agent/tools/research_tool.py` | Added kaggle to research sub-agent tools + hints |
| `agent/prompts/system_prompt_v3.yaml` | Added autonomous Kaggle workflow (5 phases) + 8 runtime pitfalls |

## Kaggle Operations

| Operation | Description | Approval |
|-----------|-------------|----------|
| `list_competitions` | Browse active competitions | No |
| `competition_details` | Get eval metric, deadline, rules | No |
| `list_data_files` | List competition data files | No |
| `list_notebooks` | Find top notebooks (by votes/score/date) | No |
| `read_notebook` | Read full notebook source as markdown | No |
| `notebook_metadata` | Get exact sources, accelerator, docker image from working notebooks | No |
| `list_discussions` | Browse competition forum | No |
| `read_discussion` | Read discussion with replies | No |
| `leaderboard` | View top leaderboard entries | No |
| `my_submissions` | List your submissions + scores | No |
| `submit` | Submit predictions file | **Yes** |
| `push_notebook` | Push script to Kaggle GPU and run it | **Yes** |
| `notebook_status` | Poll execution status | No |
| `notebook_output` | Download output files/logs | No |
| `score_history` | Local score tracking with trend analysis | No |
| `save_run` | Log a run event (push, error, fix, submission) | No |
| `run_history` | View full run log — errors, fixes, scores | No |

## Autonomous Kaggle Workflow

The agent follows a 5-phase loop for each competition:

```
Phase 0: Session Startup
  └─ Read run_history (avoid past mistakes) + check daily submission count

Phase 1: Competition Analysis (first time)
  └─ Details, data files, leaderboard, notebook_metadata from official demo

Phase 2: Research (every session)
  └─ Check for new top notebooks, read discussions, deep paper research

Phase 3: Implement & Push
  └─ Write script with pre-flight checklist → push_notebook with correct
     accelerator, docker_image, sources from notebook_metadata

Phase 4: Poll & Recover
  └─ 5min → 10min → 15min polling
  └─ On error: download logs → analyze → fix → re-push → log via save_run
  └─ On success: download output → submit with hypothesis

Phase 5: Iterate
  └─ Check score → analyze → research new ideas → implement next version
  └─ Max 3 submissions/day enforced
```

## Quick Start

### Installation

```bash
git clone <your-repo-url>
cd ml-intern-kaggle-pro
pip install -e ".[dev]"
```

### Environment Setup

Create a `.env` file:

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key>
HF_TOKEN=<your-hugging-face-token>
KAGGLE_USERNAME=<your-kaggle-username>
KAGGLE_KEY=<your-kaggle-api-key>
```

Get Kaggle credentials from [kaggle.com/settings → API](https://www.kaggle.com/settings) or place `kaggle.json` at `~/.kaggle/kaggle.json`.

### Usage

```bash
# Interactive mode
ml-intern

# Headless mode — run a competition autonomously
ml-intern "compete on nvidia-nemotron-model-reasoning-challenge"

# With specific model
ml-intern --model anthropic/claude-sonnet-4-6 "compete on titanic"
```

## Example: NVIDIA Nemotron Reasoning Challenge

See `kaggle_nemotron/` for a complete working example. The agent:

1. Analyzed the competition (adapter submission, Nemotron-3-Nano-30B model, LoRA rank 32 max)
2. Read 3+ top notebooks to extract the winning recipe (CoT labels, 2048 seq len, 2 epochs)
3. Wrote `train_adapter.py` with all Kaggle runtime patches (cutlass, Triton, RMSNorm)
4. Pushed to Kaggle with correct GPU (`NvidiaRtxPro6000`) and competition docker image
5. Recovered from 4 runtime errors autonomously:
   - Missing cutlass → `site.addsitedir()` from utility script
   - OOM on P100 → switched to RTX Pro 6000 via `notebook_metadata`
   - No internet in BYOD image → offline package installation
   - Wrong offline packages path → recursive search at correct mount point

## Architecture

Inherits the upstream ml-intern architecture (async queue-based agent loop with LiteLLM) and adds:

```
ToolRouter
  ├─ HF docs & research
  ├─ HF repos, datasets, jobs, papers
  ├─ GitHub code search
  ├─ Sandbox & local tools
  ├─ Planning
  ├─ Kaggle tool (NEW)          ← 17 operations
  │   ├─ httpx REST client       ← read operations
  │   ├─ Kaggle kernels push     ← notebook execution
  │   ├─ Score persistence       ← ~/.kaggle/agent_scores/
  │   └─ Run log persistence     ← ~/.kaggle/agent_runs/
  └─ MCP server tools
```

## Development

### Running Tests

```bash
pytest tests/unit/test_kaggle_tool.py -v
```

### Adding Kaggle Operations

Edit `agent/tools/kaggle_tool.py`:
1. Add an async handler function: `async def _my_op(args, limit) -> ToolResult`
2. Register in `_OPERATIONS` dict
3. Update `KAGGLE_TOOL_SPEC` description and parameters

## Credits

- Forked from [huggingface/ml-intern](https://github.com/huggingface/ml-intern)
- Kaggle integration by [@adityaghai01](https://github.com/adityaghai01)
