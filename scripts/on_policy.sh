export CUDA_VISIBLE_DEVICES=0
SEED=44413
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
LR=1e-6

REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_44413
POLICY_MODEL_PATH=models/$MODEL/on_policy_dpo_beta_0.1_$SEED

# vary the following parameters to fit your GPU memory
gradient_accumulation_steps=8 # bigger fits better on GPU
local_micro_batch_size=16 # smaller fits better on GPU
local_eval_batch_size=2 # smaller fits better on GPU


poetry run accelerate launch --config_file deepspeed.yaml \
    --main_process_port=2950 \
    summarize_from_feedback_details/on_policy.py \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --local_rollout_forward_batch_size=64\
    --warm_up_steps=50 \
    --local_micro_batch_size=$local_micro_batch_size \
    --response_length=128 \
    --temperature=0.7 \
    --base_model=$MODEL \
    --num_updates=1500 \
    --optimizer=adamw \
    --nminibatches=2 \
    --num_train_epochs=2\
    --sft_model_path=$SFT_MODEL_PATH \
    --save_steps=15360 \
    --reward_model_path=$REWARD_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$POLICY_MODEL_PATH \
    --seed=$SEED