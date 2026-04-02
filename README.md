<h1 align="center">
SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization
</h1>
<font size=4><div align='center'>
[[📖 Paper](https://arxiv.org/abs/2509.11543)] [[🤗 Daily Paper](https://huggingface.co/papers/2509.11543)]
</div></font>

## 🔥 Overview

We introduce **SKILL0**, an in-context reinforcement learning framework designed for *skill internalization*.

<div align="center">
  <img src="docs/skillzero/motivation.png" alt="Logo" style="width:80%;">
</div>
<div align="center">
  <img src="docs/skillzero/method.png" alt="Logo" style="width:80%;">
</div>
Ours <b>UI-S1-7B</b> achieves SOTA performance on both semi-online metric (SOP) and online metric (AndroidWorld) among open-source 7B models.

<div align="center">
  <img src="assets/skillzero/metric.png" alt="Logo" style="width:80%;">
</div>

## Detailed results

<div align="center">
  <img src="assets/result.png" alt="Logo" style="width:80%;">
</div>
---

## 🛠️ Installation


### Python environment

```bash
conda create -n skillzero python=3.12 -y
conda activate skillzero

pip install vllm==0.10.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

Log in to Weights & Biases if you use WandB logging (scripts pass `trainer.logger=['console','wandb']` in many cases):

```bash
export WANDB_API_KEY=your_key_here
```

### ALFWorld

```bash
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
```

Game and PDDL assets are cached under `~/.cache/alfworld/` by default. Override with:

```bash
export ALFWORLD_DATA=/path/to/your/alfworld_data
```

### Search environment

```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

---

## Data and Search pipeline

### Layout

| Purpose | Default path |
|--------|----------------|
| Search-R1 processed parquet (train/test) | `~/data/searchR1_processed_direct/` |
| ALFWorld / verl-agent parquet (after `prepare`) | `~/data/verl-agent/` |
| FAISS index + wiki corpus for retriever | `~/data/searchR1/` (override with `SEARCH_R1_INDEX_DIR`) |

Prepare Search training data under `~/data/searchR1_processed_direct` using your upstream preprocessing (e.g. from Search-R1 or your own pipeline). File names expected by the training scripts include `train.parquet`, `test.parquet`.

### Validation parquet for SkillZero Search (required before SkillZero search training)

Run from the **repository root**:

```bash
python -m examples.data_preprocess.generate_search_r1_val
```

Defaults read `~/data/searchR1_processed_direct/test.parquet` and write `~/data/searchR1_processed_direct/val_<max_sample>.parquet` (default `--max_sample 1000` → `val_1000.parquet`). The SkillZero search scripts use `val_1000.parquet` by default; if you change `--max_sample`, set `data.val_files` accordingly or regenerate with matching options.

### Retriever server

1. Download index and corpus (example: use `examples/search/searchr1_download.py` with `local_dir` such as `~/data/searchR1`), build `e5_Flat.index` and decompress `wiki-18.jsonl` as needed.
2. Start the API (from repo root):

```bash
# Optional: index directory and encoder model
export SEARCH_R1_INDEX_DIR=$HOME/data/searchR1
export E5_RETRIEVER_MODEL=intfloat/e5-base-v2

bash examples/search/retriever/retrieval_launch.sh
```

Training scripts use `http://127.0.0.1:8000/retrieve` by default. Override with:

```bash
export SEARCH_URL=http://your-host:8000/retrieve
```

---

## Training

All scripts live under `scripts/` and assume the repo root as working directory (they `cd` there automatically). You can run either:

```bash
bash scripts/train_alfworld_text.sh
# or, from repo root:
bash train_alfworld_text.sh
```

### SkillZero (method)

| Script | Description |
|--------|-------------|
| `scripts/train_alfworld_skillzero_3b.sh` | ALFWorld, Qwen2.5-VL-3B, curriculum + skills |
| `scripts/train_alfworld_skillzero_7b.sh` | ALFWorld, Qwen2.5-VL-7B |
| `scripts/train_search_skillzero_3b.sh` | Search, Qwen2.5-VL-3B |
| `scripts/train_search_skillzero_7b.sh` | Search, Qwen2.5-VL-7B |

GPUs: `trainer.n_gpus_per_node=4` in these entrypoints.

### Baselines

| Script | Description |
|--------|-------------|
| `scripts/train_alfworld_text.sh` | ALFWorld text-only |
| `scripts/train_alfworld_agentocr.sh` | ALFWorld AgentOCR-style visual |
| `scripts/train_alfworld_text_skill.sh` | ALFWorld text + skill file |
| `scripts/train_search_text.sh` | Search text |
| `scripts/train_search_agentocr.sh` | Search visual (AgentOCR-style) |
| `scripts/train_search_text_skill.sh` | Search text + skill file |

### Merge checkpoints

See `scripts/model_merger.py` for FSDP/Megatron merge examples using paths under `./checkpoints/...`.

---

## Acknowledgement

This project builds on [AgentOCR](https://github.com/langfengQ/AgentOCR), [verl-agent](https://github.com/langfengQ/verl-agent), and [veRL](https://github.com/volcengine/verl), [ALFWorld](https://github.com/alfworld/alfworld), [SkillRL](https://github.com/aiming-lab/SkillRL), and [Search-R1](https://github.com/PeterGriffinJin/Search-R1). We thank the authors of those projects.
