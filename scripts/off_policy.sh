export CUDA_VISIBLE_DEVICES=0,2
SEED=44413
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
LR=1e-6

REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED

# vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=2 # smaller fits better on GPU
gradient_accumulation_steps=8 # bigger fits better on GPU
local_micro_batch_size=4 # smaller fits better on GPU
local_eval_batch_size=2 # smaller fits better on GPU

OFF_POLICY_MODEL_PATH=models/$MODEL/length_IS_policy_beta_0.1_lr_1e-6_epochs_2_$SEED

betas=("0.01" "0.05" "0.1")
# losses=("clipped" "sigmoid")

# for beta in "${betas[@]}"; do
# for loss_name in "${losses[@]}"; do
        # --reward_model_path=$REWARD_MODEL_PATH \
    poetry run accelerate launch --config_file deepspeed.yaml \
        summarize_from_feedback_details/off_policy.py \
        --base_model=$MODEL \
        --warm_up_steps=300\
        --num_train_epochs=3\
        --sft_model_path=$SFT_MODEL_PATH \
        --output_dir=$OFF_POLICY_MODEL_PATH \
        --lr=$LR\
        --save_steps=10800 \
        --beta=0.1 \
        --local_micro_batch_size=$local_micro_batch_size\
        --gradient_accumulation_steps=$gradient_accumulation_steps\
        --deepspeed \
        --run_eval \
        --track \
        --push_to_hub \
        --local_eval_batch_size=1 \
        --seed=$SEED
# done
# done

# for beta in "${betas[@]}"; do
# DPO_POLICY_MODEL_PATH=models/$MODEL/dpo_policy_shift_beta_${beta}_$SEED
# poetry run accelerate launch --config_file deepspeed.yaml \
#     summarize_from_feedback_details/dpo.py 
#     --base_model=$MODEL \
#     --sft_model_path=$SFT_MODEL_PATH \
#     --output_dir=$DPO_POLICY_MODEL_PATH \
#     --lr=$LR\
#     --save_steps=1800 \
#     --beta=${beta} \
#     --local_micro_batch_size=$local_micro_batch_size\
#     --gradient_accumulation_steps=$gradient_accumulation_steps\
#     --deepspeed \
#     --run_eval \
#     --track \
#     --push_to_hub \
#     --local_eval_batch_size=1 \
#     --seed=$SEED
# done