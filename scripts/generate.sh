export CUDA_VISIBLE_DEVICES=6

# steps=("50400")
steps=("140400" "270000")

for step in "${steps[@]}"; do
output_path=data/synthetic/step_${step}
python on_policy_data_gen/generate.py --model=models/EleutherAI/pythia-1b-deduped/on_policy_dpo_beta_0.1_44413/step_${step}\
                    --output_dir=$output_path
# python on_policy_data_gen/generate.py --model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr --output_dir=$output_path
done

                


