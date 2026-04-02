set -x
###############
# Highlight configs: use environment variable to avoid Hydra parsing issues with < > characters
# Format: "context1:r,g,b;context2:r,g,b"
# Observation  are highlighted in blue (0,0,255)
# Action are highlighted in red (255,0,0)
export HIGHLIGHT_CONFIGS='[Observation]:0,0,255;[Action]:255,0,0'
################

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

# OCR settings
use_ocr=True
ocr_use_parallel=True
ocr_max_workers=64
ocr_font_size=10
ocr_max_width=392

# Self-compression settings
agent_select_compression_enable=True
compression_reward_coef=0.01  # base coefficient for compression reward
compression_reward_every_n_steps=5  # apply compression reward every n steps

# Data settings
train_data_size=16
val_data_size=128
group_size=8

# Set mode based on use_ocr: visual if use_ocr=True, text otherwise
if [ "$use_ocr" = "True" ]; then
    mode="visual"
    model=Qwen/Qwen2.5-VL-3B-Instruct
    max_prompt_length=2048
    experiment_name="agentocr_selfcompress${agent_select_compression_enable}_coef${compression_reward_coef}_everyn${compression_reward_every_n_steps}_fs${ocr_font_size}_maxwidth${ocr_max_width}_maxprompt${max_prompt_length}_qwen25_vl_3b"
else
    mode="text"
    model=Qwen/Qwen2.5-3B-Instruct
    max_prompt_length=5120
    experiment_name="text_maxprompt${max_prompt_length}_qwen25_3b"
fi

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode $mode \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/$mode/train.parquet \
    data.val_files=$HOME/data/verl-agent/$mode/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.history_length=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    ocr.use_ocr=$use_ocr \
    ocr.use_parallel=$ocr_use_parallel \
    ocr.max_workers=$ocr_max_workers \
    ocr.font_size=$ocr_font_size \
    ocr.max_width=$ocr_max_width \
    ocr.agent_select_compression.enable=$agent_select_compression_enable \
    ocr.agent_select_compression.compression_reward_coef=$compression_reward_coef \
    ocr.agent_select_compression.compression_reward_every_n_steps=$compression_reward_every_n_steps \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='AgentOCR_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@
