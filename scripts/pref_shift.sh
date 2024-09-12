export CUDA_VISIBLE_DEVICES=5,6
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
gradient_accumulation_steps=2 # bigger fits better on GPU
local_micro_batch_size=4 # smaller fits better on GPU
local_eval_batch_size=2 # smaller fits better on GPU
gradient_steps=('46800' "79200")

betas=("0.01" "0.03" "0.05" "0.1" "0.3" "1.0" "3" "10") 

for steps in "${gradient_steps[@]}"; do
for beta in "${betas[@]}"; do
    DPO_POLICY_MODEL_PATH=models/$MODEL/dpo_policy_shift_beta_${beta}_steps_${steps}_$SEED
    poetry run accelerate launch --config_file deepspeed.yaml \
        summarize_from_feedback_details/pref_shift.py \
        --exp_name=dpo_shift_beta_${beta}_steps_${steps} \
        --base_model=$MODEL \
        --gradient_steps=$steps\
        --warm_up_steps=150\
        --num_train_epochs=1\
        --sft_model_path=$SFT_MODEL_PATH \
        --output_dir=$DPO_POLICY_MODEL_PATH \
        --lr=$LR\
        --save_steps=2500 \
        --beta=$beta \
        --local_micro_batch_size=$local_micro_batch_size\
        --gradient_accumulation_steps=$gradient_accumulation_steps\
        --deepspeed \
        --run_eval \
        --track \
        --push_to_hub \
        --local_eval_batch_size=1 \
        --seed=$SEED
done
done