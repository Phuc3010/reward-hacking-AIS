export CUDA_VISIBLE_DEVICES=2

# MODEL_NAME_OR_PATH='DatPySci/EleutherAI_pythia-1b-deduped__dpo__tldr'
betas=("0.005" "0.01" "0.02" "0.05" "0.1")

steps=("75600" "79200" "82800" "86400" "90000")

for step in "${steps[@]}"; do
    MODEL_NAME_OR_PATH=models/EleutherAI/pythia-1b-deduped/off_policy_beta_0.1_loss_clipped_ratio_0.7_44413/step_${step}
    KEY=ppo_clipped_beta_0.1_eps_0.7_step_$step
    python visualize_tokens.py --model_name_or_path=$MODEL_NAME_OR_PATH \
                            --key=$KEY \
                            --seed=0\
                            --n_samples=512 

#     MODEL_NAME_OR_PATH=models/EleutherAI/pythia-1b-deduped/dpo_policy_beta_${beta}_44413
#     KEY=dpo_beta_${beta}_step_92820
#     python visualize_tokens.py --model_name_or_path=$MODEL_NAME_OR_PATH \
#                             --key=$KEY \
#                             --seed=0\
#                             --n_samples=512 

#     MODEL_NAME_OR_PATH=models/EleutherAI/pythia-1b-deduped/off_policy_beta_${beta}_loss_clipped_44413
#     KEY=ppo_clipped_beta_${beta}_step_92820
#     python visualize_tokens.py --model_name_or_path=$MODEL_NAME_OR_PATH \
#                             --key=$KEY \
#                             --seed=0\
#                             --n_samples=512 
done