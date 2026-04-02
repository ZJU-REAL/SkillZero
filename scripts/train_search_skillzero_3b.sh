#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

set -x

# Retriever HTTP API (override before running if your server is not local)
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"

###############
# Search task: highlight <search> / <information> (same as train_search_text.sh)
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
###############

EXP_LOG_NAME="skillzero_search_vl_3b"
export LOG_PATH="$REPO_ROOT/log/$EXP_LOG_NAME.log"
mkdir -p "$REPO_ROOT/log/"

export USE_SKILL=True
export SKILL_DIR="$REPO_ROOT/skills/search"

num_cpus_per_env_worker=0.1

train_data_size=128
val_data_size=512
group_size=8

# val_1000.parquet from: python -m examples.data_preprocess.generate_search_r1_val (default --max_sample 1000)
python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=8 \
    algorithm.adv_estimator=grpo \
    data.train_files="$HOME/data/searchR1_processed_direct/train.parquet" \
    data.val_files="$HOME/data/searchR1_processed_direct/val_1000.parquet" \
    data.test_files="$HOME/data/searchR1_processed_direct/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.use_skill=$USE_SKILL \
    env.seed=0 \
    env.max_steps=4 \
    env.history_length=4 \
    env.rollout.n=$group_size \
    env.search.search_url=$SEARCH_URL \
    env.curriculum_learning.enable=True \
    env.curriculum_learning.max_set_schedule=[5,3,0] \
    env.curriculum_learning.skill_mapping_file=$SKILL_DIR/skill_mapping.json \
    ocr.use_ocr=True \
    ocr.max_workers=64 \
    ocr.font_size=12 \
    ocr.max_width=560 \
    ocr.agent_select_compression.enable=True \
    ocr.agent_select_compression.compression_reward_coef=0.01 \
    ocr.agent_select_compression.compression_reward_every_n_steps=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='SkillZero_search' \
    trainer.experiment_name=$EXP_LOG_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.test_after_train=True \
    trainer.total_training_steps=180 \
    trainer.val_before_train=True \
    2>&1 | tee "$LOG_PATH"
