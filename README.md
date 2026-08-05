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
- **`2026-7-29`**: 🔥🔥 We released [SkillRise](https://arxiv.org/abs/2607.26784) and its [code](https://github.com/Within-yao/SkillRise), introducing **cross-task skill evolution** via agentic RL.
- **`2026-7-17`**: 🔥🔥 We released [SEED](https://arxiv.org/abs/2607.14777) and its [code](https://github.com/jinyangwu/SEED), introducing **self-evolving** opd beyond skill internalization.
- **`2026-6-25`**: 🔥 We released [OPID](https://arxiv.org/abs/2606.26790) and its [code](https://github.com/jinyangwu/OPID), introducing **skill evolving** beyond skill internalization.
- **`2026-5-15`**: 🔥 Our new work was released: [SDAR](https://github.com/ZJU-REAL/SDAR), which introduces Self-Distilled Agentic Reinforcement Learning.
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
@article{lu2026skill0,
  title={Skill0: In-context agentic reinforcement learning for skill internalization},
  author={Lu, Zhengxi and Yao, Zhiyuan and Wu, Jinyang and Han, Chengcheng and Gu, Qi and Cai, Xunliang and Lu, Weiming and Xiao, Jun and Zhuang, Yueting and Shen, Yongliang},
  journal={arXiv preprint arXiv:2604.02268},
  year={2026}
}
@article{lu2026sdar,
  title={Self-distilled agentic reinforcement learning},
  author={Lu, Zhengxi and Yao, Zhiyuan and Han, Zhuowen and Wang, Zi-Han and Wu, Jinyang and Gu, Qi and Cai, Xunliang and Lu, Weiming and Xiao, Jun and Zhuang, Yueting and others},
  journal={arXiv preprint arXiv:2605.15155},
  year={2026}
}
@article{shi2026skill1,
  title={Skill1: Unified evolution of skill-augmented agents via reinforcement learning},
  author={Shi, Yaorui and Chen, Yuxin and Lu, Zhengxi and Miao, Yuchun and Liu, Shugui and Gu, Qi and Cai, Xunliang and Wang, Xiang and Zhang, An},
  journal={arXiv preprint arXiv:2605.06130},
  year={2026}
}
@article{wu2026seed,
  title={SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning},
  author={Wu, Jinyang and Yang, Shuo and Lu, Zhengxi and Zhang, Fan and Shen, Yuhao and Feng, Lang and Luo, Haoran and Lian, Zheng and Zhang, Shuai and Wen, Zhengqi and others},
  journal={arXiv preprint arXiv:2607.14777},
  year={2026}
}
@article{yang2026opid,
  title={Opid: On-policy skill distillation for agentic reinforcement learning},
  author={Yang, Shuo and Wu, Jinyang and Lu, Zhengxi and Shen, Yuhao and Zhang, Fan and Feng, Lang and Zhang, Shuai and Luo, Haoran and Lian, Zheng and Wen, Zhengqi and others},
  journal={arXiv preprint arXiv:2606.26790},
  year={2026}
}
@article{yao2026skillrise,
  title={SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution},
  author={Yao, Zhiyuan and Chen, Yuxin and Lu, Zhengxi and Xu, Zishan and Sun, Yueqing and Guo, Yifu and Lu, Yuquan and Cai, Zhengzhou and Zhang, Kangning and Han, Zhuowen and others},
  journal={arXiv preprint arXiv:2607.26784},
  year={2026}
}
```

## 🤝 Acknowledgement

This project builds on [AgentOCR](https://github.com/langfengQ/AgentOCR), [verl-agent](https://github.com/langfengQ/verl-agent), [veRL](https://github.com/volcengine/verl), [ALFWorld](https://github.com/alfworld/alfworld), [SkillRL](https://github.com/aiming-lab/SkillRL), and [Search-R1](https://github.com/PeterGriffinJin/Search-R1). We thank the authors of those projects.
