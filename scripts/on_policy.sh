
export CUDA_VISIBLE_DEVICES=5,6
SEED=0
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
gradient_accumulation_steps=4 # bigger fits better on GPU
local_micro_batch_size=4 # smaller fits better on GPU
local_eval_batch_size=2 # smaller fits better on GPU

# poetry run accelerate launch --config_file deepspeed.yaml \
#     summarize_from_feedback_details/reward.py \
#     --base_model=$MODEL \
#     --sft_model_path=$SFT_MODEL_PATH \
#     --warm_up_steps=150 \
#     --exp_name=reward_pythia-1b \
#     --local_micro_batch_size=4\
#     --gradient_accumulation_steps=8\
#     --lr=$LR \
#     --deepspeed \
#     --run_eval \
#     --track \
#     --output_dir=$REWARD_MODEL_PATH \
#     --local_eval_batch_size=$local_eval_batch_size \
#     --seed=$SEED

poetry run accelerate launch --config_file deepspeed.yaml \
    --main_process_port=2950 \
    summarize_from_feedback_details/on_policy.py \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --warm_up_steps=600 \
    --local_micro_batch_size=$local_micro_batch_size \
    --response_length=64 \
    --base_model=$MODEL \
    --num_updates=2900 \
    --optimizer=rmsprop \
    --temperature=1.0 \
    --num_train_epochs=2\
    --sft_model_path=$SFT_MODEL_PATH \
    --save_steps=18304 \
    --reward_model_path=$REWARD_MODEL_PATH \
    --lr=5e-7 \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$POLICY_MODEL_PATH \
    --seed=$SEED