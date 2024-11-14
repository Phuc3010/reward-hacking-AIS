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
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED

# vary the following parameters to fit your GPU memory
gradient_accumulation_steps=4 # bigger fits better on GPU
local_micro_batch_size=16 # smaller fits better on GPU


betas=("0.01" "0.05" "0.1")
losses=("length_IS" "ipo")
epochs=("1")

for epoch in "${epochs[@]}"; do
for beta in "${betas[@]}"; do
for loss_name in "${losses[@]}"; do
        # --reward_model_path=$REWARD_MODEL_PATH \
    OFF_POLICY_MODEL_PATH=models/$MODEL/${loss_name}_policy_beta_${beta}_$SEED
    poetry run accelerate launch --config_file deepspeed.yaml \
        summarize_from_feedback_details/ipo.py \
        --base_model=$MODEL \
        --exp_name=${loss_name}_ipo_pythia-1b_beta-${beta}\
        --loss_name=$loss_name\
        --optimizer=adamw\
        --num_train_epochs=$epoch\
        --warm_up_steps=150\
        --sft_model_path=$SFT_MODEL_PATH \
        --output_dir=$OFF_POLICY_MODEL_PATH \
        --lr=$LR\
        --save_steps=7200\
        --beta=$beta\
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
done

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