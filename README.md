<h1 align="center">
SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization
</h1>
<div align='center' style="font-size:18px;">
<p>
    <a href="https://arxiv.org/abs/2604.02268">
      <img src="https://img.shields.io/badge/Paper-arxiv%3A2604.02268-blue" alt="Paper"/>
    </a>
    <a href="https://huggingface.co/papers/2604.02268">
      <img src="https://img.shields.io/badge/Daily%20Paper-huggingface-yellow" alt="HF Paper"/>
    </a>
  </p>
</div>


## 🔥 Overview

We introduce **SKILL0**, an in-context reinforcement learning framework designed for *skill internalization*.
<div align="center" style="display:flex; justify-content:center; gap:20px; align-items:flex-start;">
  <img src="docs/skillzero/motivation.png" alt="motivation" style="width:40%;">
  <img src="docs/skillzero/method.png" alt="method" style="width:58%;">
</div>




SKILL0 achieves substantial improvements over the standard RL baseline on ALFWorld and Search-QA.
<div align="center">
  <img src="docs/skillzero/metric.png" alt="Logo" style="width:80%;">
</div>

## 🗞️ News
- **`2026-5-15`**: 🔥🔥 Our new work was released: [SDAR](https://github.com/ZJU-REAL/SDAR), which introduces Self-Distilled Agentic Reinforcement Learning.
- **`2026-5-07`**: 🔥 Our new work was released: [SKILL1](https://github.com/AlphaLab-USTC/Skill1), which evloves skill-augmented agents in **one** unified policy.
- **`2026-4-03`**: We release our paper and code.

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

### Install Supported Environments

#### 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

#### 2. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Build Retriever environments:
```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4 
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```

Validation parquet for SkillZero Search
```bash
python -m examples.data_preprocess.generate_search_r1_val
```


### Training

All scripts live under `scripts/` and assume the repo root as working directory (they `cd` there automatically). You can run either:

```bash
bash scripts/train_alfworld_skillzero_3b.sh
bash scripts/train_search_skillzero_3b

### Merge checkpoints

See `scripts/model_merger.py` for FSDP/Megatron merge examples using paths under `./checkpoints/...`.
```

## ⭐️ Citation

If you find this project useful, welcome to cite us.

```bit
@misc{lu2026skill0,
      title={SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization}, 
      author={Zhengxi Lu and Zhiyuan Yao and Jinyang Wu and Chengcheng Han and Qi Gu and Xunliang Cai and Weiming Lu and Jun Xiao and Yueting Zhuang and Yongliang Shen},
      year={2026},
      eprint={2604.02268},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.02268}, 
}
@misc{shi2026skill1,
      title={Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning}, 
      author={Yaorui Shi and Yuxin Chen and Zhengxi Lu and Yuchun Miao and Shugui Liu and Qi GU and Xunliang Cai and Xiang Wang and An Zhang},
      year={2026},
      eprint={2605.06130},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.06130}, 
}
@misc{lu2026sdar,
      title={Self-Distilled Agentic Reinforcement Learning}, 
      author={Zhengxi Lu and Zhiyuan Yao and Zhuowen Han and Zi-Han Wang and Jinyang Wu and Qi Gu and Xunliang Cai and Weiming Lu and Jun Xiao and Yueting Zhuang and Yongliang Shen},
      year={2026},
      eprint={2605.15155},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.15155}, 
}
```

## 🤝 Acknowledgement

This project builds on [AgentOCR](https://github.com/langfengQ/AgentOCR), [verl-agent](https://github.com/langfengQ/verl-agent), [veRL](https://github.com/volcengine/verl), [ALFWorld](https://github.com/alfworld/alfworld), [SkillRL](https://github.com/aiming-lab/SkillRL), and [Search-R1](https://github.com/PeterGriffinJin/Search-R1). We thank the authors of those projects.
