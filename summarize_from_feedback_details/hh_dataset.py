import copy
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from datasets import load_dataset
import re
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
from transformers import AutoTokenizer

api = HfApi()


"""
poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --hh_params.max_sft_response_length=53 \
    --hh_params.max_sft_query_response_length=562 \
    --hh_params.max_rm_response_length=169 \
    --hh_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --push_to_hub \

poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --hh_params.max_sft_response_length=53 \
    --hh_params.max_sft_query_response_length=562 \
    --hh_params.max_rm_response_length=169 \
    --hh_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --push_to_hub \
    --hh_params.padding="empty_space" \
    --cnndm_params.padding="empty_space" \
"""


@dataclass
class TaskQueryHParams:
    length: Optional[int] = None
    format_str: Optional[str] = None
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[Literal["empty_space", "pad_token"]] = None
    pad_token: Optional[str] = None
    pad_side: Optional[str] = None
    max_sft_response_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    max_rm_response_length: Optional[int] = None
    max_rm_query_response_length: Optional[int] = None

@dataclass
class Args:
    base_model: str = "EleutherAI/pythia-1b-deduped"  #  "gpt2"
    hf_entity: str = None
    push_to_hub: bool = False
    check_length_correctness: bool = True
    debug: bool = False
    hh_params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=512,
            truncate_field="chosen",
            truncate_text="\n\nHuman: ",
            padding="pad_token",
            pad_side="left",
            max_sft_response_length=128,
            max_sft_query_response_length=640,
            max_rm_response_length=128,
            max_rm_query_response_length=640,
        )
    )

def _ensure_length(toks, l, pad_sequence=None, pad_side=None, truncate_side=None):
    assert pad_side in (None, "left", "right")
    assert truncate_side in (None, "left", "right")
    if len(toks) < l:
        assert pad_sequence is not None
        pad_amt = l - len(toks)
        assert len(pad_sequence) >= pad_amt, f"{len(pad_sequence)} < {pad_amt}"
        if pad_side is None:
            assert len(toks) == l, f"Needed to pad! {len(toks)} < {l}"
            return toks
        elif pad_side == "left":
            return pad_sequence[-pad_amt:] + toks
        else:
            assert pad_side == "right"
            return toks + pad_sequence[:pad_amt]
    if truncate_side is None:
        assert len(toks) == l, f"Needed to truncate! {len(toks)} > {l}"
        return toks
    elif truncate_side == "left":
        return toks[-l:]
    else:
        assert truncate_side == "right"
        return toks[:l]

def _get_query_padding_for_task(encoder, hparams: TaskQueryHParams):
    return hparams.pad_token * hparams.length

def process_query(query, *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)

    chunks = []
    for chunk in re.split(r'\s*(Human:|Assistant:)\s+', query): 
        if chunk.startswith('Human'):
            chunk = "\n\nHuman: "
        elif chunk.startswith('Assistant'):
            chunk = "\n\nAssistant: "
        else:
            pass
        if chunk != '':
            chunks.append(chunk)
        
    query = ''.join(chunks)

    indices = [idx for idx, ele in enumerate(chunks) if ele =='\n\nHuman: ' or ele == '\n\nAssistant: ']
    query_tokens = encoder.encode(query)
    i = 0
    while len(query_tokens) > hparams.length:
        i += 1
        try:
            query = "".join(chunks[indices[i]:])
        except:
            truncate_indx = query.rindex("\n")
            prev_len = len(query)
            query = query[:truncate_indx]
            if len(query) == prev_len:
                query = query[:-1]

        query_tokens = encoder.encode(query)

    query_token = _ensure_length(query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
    query = encoder.decode(query_token, skip_special_tokens=True).lstrip()
    return dict(
        query_token=query_token,
        query=query,
    )

def ceil_div(a, b):
    return (a - 1) // b + 1

def extract_anthropic_prompt(prompt_and_response):
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def split_prompt_and_responses(ex):
    prompt = extract_anthropic_prompt(ex['chosen'])
    chosen_response = ex['chosen'][len(prompt):]
    rejected_response = ex['rejected'][len(prompt):]
    return prompt, chosen_response, rejected_response

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
        assert isinstance(args.hf_entity, str)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # post init
    args.hh_params.pad_token = [tokenizer.pad_token_id]
    pprint(args)

    timestamp = int(time.time())
    sft_ds = load_dataset('Anthropic/hh-rlhf')

    def process_query_data(x):
        # the `x['summary']` in `vwxyzjn/summarize_from_feedback_tldr_3_filtered`
        # DOES NOT HAVE a leading space so we are adding the leading space and
        # `<|endoftext|>` token
        prompt, chosen, rejected = split_prompt_and_responses(x)
        reference_response = f" {chosen.strip()}<|endoftext|>"
        y = {
            **process_query(prompt, encoder=tokenizer, hparams=args.hh_params),
            'reference_response': reference_response,
            "reference_response_token": tokenizer.encode(
                reference_response,
                padding="max_length",
                max_length=args.hh_params.max_sft_response_length,
                truncation=True,
            ),
            "reference_response_token_len": len(tokenizer.encode(reference_response)),
        }
        y["query_reference_response"] = y["query"].strip() + y["reference_response"]

        y["query_reference_response_token"] = tokenizer.encode(
            y["query_reference_response"],
            padding="max_length",
            max_length=args.hh_params.max_sft_query_response_length,
            truncation=True,
        )
        y["query_reference_response_token_response_label"] = copy.deepcopy(y["query_reference_response_token"])
        unpadded_query_token = [token for token in y["query_token"] if token != tokenizer.pad_token_id]
        y["query_reference_response_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
        y["query_reference_response_token_len"] = len(tokenizer.encode(y["query_reference_response"]))
        return y

    sft_ds = sft_ds.map(process_query_data, load_from_cache_file=False, num_proc=1 if args.debug else 12)
    if args.push_to_hub:
        sft_dataset_hf_path = f"{args.hf_entity}/anthropic_hh_rlhf_filtered_oai_preprocessing"
        sft_ds.push_to_hub(sft_dataset_hf_path)
        sft_card = RepoCard.load(sft_dataset_hf_path, repo_type="dataset")
        sft_card.push_to_hub(sft_dataset_hf_path, repo_type="dataset")

    label_ds = load_dataset('Anthropic/hh-rlhf')

    def process_response_data(x):
        prompt, chosen, rejected = split_prompt_and_responses(x)
        chosen = f" {chosen.strip()}<|endoftext|>"
        rejected = f" {rejected.strip()}<|endoftext|>"

        chosen_policy = x["summaries"][choice]["policy"]
        rejected_policy = x["summaries"][1 - choice]["policy"]
        policies = "--".join(sorted([chosen_policy, rejected_policy]))
        format_params = args.cnndm_params if x["batch"] in cnndm_batches else args.hh_params
        max_rm_response_length = (
            args.cnndm_params.max_rm_response_length
            if x["batch"] in cnndm_batches
            else args.hh_params.max_rm_response_length
        )
        max_rm_query_response_length = (
            args.cnndm_params.max_rm_query_response_length
            if x["batch"] in cnndm_batches
            else args.hh_params.max_rm_query_response_length
        )
        y = {
            **process_query(x["info"], encoder=tokenizer, hparams=format_params),
            "chosen": chosen,
            "chosen_token": tokenizer.encode(
                chosen, padding="max_length", max_length=max_rm_response_length, truncation=True
            ),
            "chosen_token_len": len(tokenizer.encode(chosen)),
            "rejected": rejected,
            "rejected_token": tokenizer.encode(
                rejected, padding="max_length", max_length=max_rm_response_length, truncation=True
            ),
            "rejected_token_len": len(tokenizer.encode(rejected)),
            "chosen_policy": chosen_policy,
            "rejected_policy": rejected_policy,
            "policies": policies,
        }
        y["query_chosen"] = y["query"].strip() + y["chosen"]
        # if padding is space, then we can just concatenate the tokens
        if args.hh_params.padding == "empty_space":
            y["query_chosen_token"] = y["query_token"] + y["chosen_token"]
        else:
            y["query_chosen_token"] = tokenizer.encode(
                y["query_chosen"], padding="max_length", max_length=max_rm_query_response_length, truncation=True
            )
        y["query_chosen_token_len"] = len(tokenizer.encode(y["query_chosen"]))
        y["query_rejected"] = y["query"].strip() + y["rejected"]
        # if padding is space, then we can just concatenate the tokens
        if args.hh_params.padding == "empty_space":
            y["query_rejected_token"] = y["query_token"] + y["rejected_token"]
        else:
            y["query_rejected_token"] = tokenizer.encode(
                y["query_rejected"], padding="max_length", max_length=max_rm_query_response_length, truncation=True
            )
        y["query_rejected_token_len"] = len(tokenizer.encode(y["query_rejected"]))
        y["query_token_len"] = len(tokenizer.encode(y["query"]))
        unpadded_query_token = [token for token in y["query_token"] if token != tokenizer.pad_token_id]
        y["query_chosen_token_response_label"] = copy.deepcopy(y["query_chosen_token"])
        y["query_chosen_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
        y["query_rejected_token_response_label"] = copy.deepcopy(y["query_rejected_token"])
        y["query_rejected_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
        return y

    label_ds = label_ds.map(process_response_data, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
    if args.push_to_hub:
        rm_dataset_hf_path = f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{timestamp}"
        label_ds.push_to_hub(f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{timestamp}")

    ####################################
    # visualize token length distribution
    ####################################
    calculated_hh_params = TaskQueryHParams(
        max_sft_query_response_length=0,
        max_sft_response_length=0,
        max_rm_response_length=0,
        max_rm_query_response_length=0,
    )
    calculated_cnndm_params = TaskQueryHParams(
        max_rm_query_response_length=0,
        max_rm_response_length=0,
    )

    os.makedirs("dataset_visuals", exist_ok=True)
    num_sft_visuals = 2
    num_label_visuals = 5
    num_subplots = len(sft_ds) * num_sft_visuals + len(label_ds) * num_label_visuals
    num_cols = 3
    print(f"{num_subplots=}")
    fig, axs = plt.subplots(ceil_div(num_subplots, num_cols), num_cols, figsize=(16, 16))
    axs = axs.flatten()
    j = 0
    for _, key in enumerate(sft_ds.keys()):
        df = sft_ds[key].to_pandas()
        axs[j].hist(df["reference_response_token_len"], bins=100)
        axs[j].set_title(f"{key} split: reference response token length\nmax_length={max(df['reference_response_token_len'])}")
        axs[j + 1].hist(df["query_reference_response_token_len"], bins=100)
        axs[j + 1].set_title(
            f"{key} split: query.strip() + reference response token length\nmax_length={max(df['query_reference_response_token_len'])}"
        )
        calculated_hh_params.max_sft_response_length = max(
            calculated_hh_params.max_sft_response_length, max(df["reference_response_token_len"])
        )
        calculated_hh_params.max_sft_query_response_length = max(
            calculated_hh_params.max_sft_query_response_length, max(df["query_reference_response_token_len"])
        )
        j += num_sft_visuals
    offset = len(sft_ds)
    for _, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        axs[j].hist(df["chosen_token_len"], bins=100)
        axs[j].set_title(f"{split} split: chosen token length\nmax_length={max(df['chosen_token_len'])}")
        axs[j + 1].hist(df["rejected_token_len"], bins=100)
        axs[j + 1].set_title(f"{split} split: rejected token length\nmax_length={max(df['rejected_token_len'])}")
        axs[j + 2].hist(df["query_chosen_token_len"], bins=100)
        axs[j + 2].set_title(
            f"{split} split: query.strip() + chosen token length\nmax_length={max(df['query_chosen_token_len'])}"
        )
        axs[j + 3].hist(df["query_rejected_token_len"], bins=100)
        axs[j + 3].set_title(
            f"{split} split: query.strip() + rejected token length\nmax_length={max(df['query_rejected_token_len'])}"
        )
        axs[j + 4].hist(df["query_token_len"], bins=100)
        axs[j + 4].set_title(f"{split} split: query token length\nmax_length={max(df['query_token_len'])}")
        if split in ["train", "validation"]:
            calculated_hh_params.max_rm_response_length = max(
                calculated_hh_params.max_rm_response_length, max(df["chosen_token_len"]), max(df["rejected_token_len"])
            )
            calculated_hh_params.max_rm_query_response_length = max(
                calculated_hh_params.max_rm_query_response_length,
                max(df["query_chosen_token_len"]),
                max(df["query_rejected_token_len"]),
            )
        elif split == "validation_cnndm":
            calculated_cnndm_params.max_rm_response_length = max(
                calculated_cnndm_params.max_rm_response_length, max(df["chosen_token_len"]), max(df["rejected_token_len"])
            )
            calculated_cnndm_params.max_rm_query_response_length = max(
                calculated_cnndm_params.max_rm_query_response_length,
                max(df["query_chosen_token_len"]),
                max(df["query_rejected_token_len"]),
            )
        else:
            raise ValueError(f"Unknown dataset split: {split}")
        j += num_label_visuals
    fig.suptitle(f"{args.base_model} Tokenizer: Token length distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/token_len.png")

    pprint({"calculated_hh_params": calculated_hh_params})
    pprint({"calculated_cnndm_params": calculated_cnndm_params})
    if args.check_length_correctness:
        assert calculated_hh_params.max_sft_response_length == args.hh_params.max_sft_response_length
        assert calculated_hh_params.max_sft_query_response_length == args.hh_params.max_sft_query_response_length
        assert calculated_hh_params.max_rm_response_length == args.hh_params.max_rm_response_length
        assert calculated_hh_params.max_rm_query_response_length == args.hh_params.max_rm_query_response_length
        assert calculated_cnndm_params.max_rm_response_length == args.cnndm_params.max_rm_response_length
        assert calculated_cnndm_params.max_rm_query_response_length == args.cnndm_params.max_rm_query_response_length
        print("✨ calculated lenghts are ok!")

    # visualize confidence distribution
    fig, axs = plt.subplots(len(label_ds), 1, figsize=(8, 8))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        axs[i].hist(df["extra.confidence"])
        axs[i].set_title(f"{split} split: confidence distribution")
    fig.suptitle("Confidence distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/confidence.png")

    # visualize policies used
    fig, axs = plt.subplots(1, len(label_ds), figsize=(8, 12))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        cat = pd.concat([df["chosen_policy"], df["rejected_policy"]], axis=0)
        cat.hist(ax=axs[i], xrot=90, orientation="horizontal")
        axs[i].set_title(f"{split} split: policy distribution")
    fig.suptitle("Policy distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/policies.png")

    # visualize compairson distribution
    fig, axs = plt.subplots(1, len(label_ds), figsize=(24, 30))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        df["policies"].hist(ax=axs[i], xrot=90, orientation="horizontal")
        axs[i].set_title(f"{split} split: policy comparison distribution")
    fig.suptitle("Policy comparison distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/policy_comparisons.png")

    if args.push_to_hub:
        # upload the `dataset_visuals`
        api.upload_folder(
            folder_path="dataset_visuals",
            path_in_repo="dataset_visuals",
            repo_id=f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{timestamp}",
            repo_type="dataset",
        )
        # upload current file
        print(f"{__file__=}")
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="create_dataset.py",
            repo_id=f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{timestamp}",
            repo_type="dataset",
        )
        print(f"✨ Pushed to hub: https://huggingface.co/datasets/{sft_dataset_hf_path}")
        print(f"✨ Pushed to hub: https://huggingface.co/datasets/{rm_dataset_hf_path}")

