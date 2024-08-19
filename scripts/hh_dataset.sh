poetry run python summarize_from_feedback_details/hh_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --hh_params.max_sft_response_length=128\
    --hh_params.max_sft_query_response_length=640 \
    --hh_params.max_rm_response_length=128 \
    --hh_params.max_rm_query_response_length=640 \
    --hh_params.padding="pad_token" \
    --push_to_hub # you can optionally push to hub