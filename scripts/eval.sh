export CUDA_VISIBLE_DEVICES=0

SEED=44413
# betas=("0.01" "0.03" "0.05" "0.1" "1.0" "10") 
# betas=('0.01' 
# betas=("10")
# betas=('0.1' '0.05')
betas=('0.05')
# betas=('0.01')
loss_names=("on")
# loss_names=('length_IS')
# loss_names=('dpo')

# steps=("10800" "21760" "32640" "43520" "54400" "65280" "76160" "87040" "97920" "108800" "119680" "130560" "141440" "152320" "163200" "174080" "184960" "195840" "206720" "217600" "228480" "239360" "250240" "261120" "272000" "282880" "293760" "304640" "315520" "326400" "337280" "348160")
# steps=("5440" "10880" "16320" "21760" "27200" "32640" "38080" "43520" "48960")
# steps=("4992" "9984" "14976" "19968" "24960" "29952" "34944" "39936" "44928"  "49920")
steps=('15360' '30720' '46080' '61440' '76800' '92160' '107520' '122880' '138240' '153600' '168960' '184320' '199680' '215040' '230400')

# steps=('14400' '28800' '43200' '57600' '72000' '86400')

for loss_name in "${loss_names[@]}"; do
for beta in "${betas[@]}"; do
for step in "${steps[@]}"; do
    KEY=${loss_name}_pythia-1b_beta_${beta}_step_${step}
    MODEL_NAME_OR_PATH=models/EleutherAI/pythia-1b-deduped/on_policy_dpo_beta_${beta}_${SEED}/step_${step}
    python visualize_tokens.py --model_name_or_path=$MODEL_NAME_OR_PATH \
                            --key=$KEY \
                            --seed=0\
                            --n_samples=512
done
done
done
done