export CUDA_VISIBLE_DEVICES=3,4
SEED=0

if [ -z "$MODEL" ]; then
    MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
    # MODEL=EleutherAI/pythia-160m-deduped
    # MODEL=openai-community/gpt2-large
    # MODEL=meta-llama/Llama-2-7b-hf
fi
LR=2e-5

REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED

# vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=2 # smaller fits better on GPU
gradient_accumulation_steps=2 # bigger fits better on GPU
local_micro_batch_size=16 # smaller fits better on GPU
local_eval_batch_size=2 # smaller fits better on GPU


poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/sft.py \
    --base_model=$MODEL \
    --warm_up_steps=150\
    --local_micro_batch_size=16\
    --gradient_accumulation_steps=4\
    --num_train_epochs=2\
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$SFT_MODEL_PATH\
    --local_eval_batch_size=$local_eval_batch_size \
    --seed=$SEED

poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/reward.py \
    --base_model=$MODEL \
    --warm_up_steps=150\
    --local_micro_batch_size=16\
    --gradient_accumulation_steps=4\
    --num_train_epochs=2\
    --lr=1e-6 \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$SFT_MODEL_PATH\
    --local_eval_batch_size=$local_eval_batch_size \
    --seed=$SEED