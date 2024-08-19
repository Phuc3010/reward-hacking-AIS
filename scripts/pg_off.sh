export CUDA_VISIBLE_DEVICES=3,4,5,6
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
POLICY_MODEL_PATH=models/$MODEL/policy_model_dpo_beta_0,1_lr_1e-6_$SEED
OFF_POLICY_MODEL_PATH=models/$MODEL/off_policy_model_tau_2.0_beta_0.1_$SEED

# vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=2 # smaller fits better on GPU
gradient_accumulation_steps=4 # bigger fits better on GPU
local_micro_batch_size=4 # smaller fits better on GPU
local_eval_batch_size=2 # smaller fits better on GPU

OFF_POLICY_MODEL_PATH=models/$MODEL/off_policy_beta_0.1_lr_1e-6_$SEED

betas=("0.01" "0.02" "0.05" "0.1" "0.5")
losses=("clipped" "sigmoid")

# for beta in "${betas[@]}"; do
# for loss_name in "${losses[@]}"; do
#     OFF_POLICY_MODEL_PATH=models/$MODEL/off_policy_beta_${beta}_loss_${loss_name}_ratio_0.7_$SEED
#     poetry run accelerate launch --config_file deepspeed.yaml \
#         summarize_from_feedback_details/ppo_off.py \
#         --base_model=$MODEL \
#         --sft_model_path=$SFT_MODEL_PATH \
#         --output_dir=$OFF_POLICY_MODEL_PATH \
#         --loss_name=$loss_name \
#         --tau=0.5 \
#         --clipping_ratio=0.7 \
#         --lr=$LR\
#         --save_steps=1800 \
#         --beta=$beta \
#         --local_micro_batch_size=$local_micro_batch_size\
#         --gradient_accumulation_steps=$gradient_accumulation_steps\
#         --deepspeed \
#         --run_eval \
#         --track \
#         --push_to_hub \
#         --local_eval_batch_size=1 \
#         --seed=$SEED
# done
# done

for beta in "${betas[@]}"; do
DPO_POLICY_MODEL_PATH=models/$MODEL/dpo_policy_shift_beta_${beta}_$SEED
poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/dpo.py \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --output_dir=$DPO_POLICY_MODEL_PATH \
    --lr=$LR\
    --save_steps=1800 \
    --beta=${beta} \
    --local_micro_batch_size=$local_micro_batch_size\
    --gradient_accumulation_steps=$gradient_accumulation_steps\
    --deepspeed \
    --run_eval \
    --track \
    --push_to_hub \
    --local_eval_batch_size=1 \
    --seed=$SEED
done