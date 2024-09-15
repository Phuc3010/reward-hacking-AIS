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
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
from transformers import AutoTokenizer

api = HfApi()


"""
poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --tldr_params.max_sft_response_length=53 \
    --tldr_params.max_sft_query_response_length=562 \
    --tldr_params.max_rm_response_length=169 \
    --tldr_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --push_to_hub \

poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --tldr_params.max_sft_response_length=53 \
    --tldr_params.max_sft_query_response_length=562 \
    --tldr_params.max_rm_response_length=169 \
    --tldr_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --push_to_hub \
    --tldr_params.padding="empty_space" \
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
    sft: bool = False
    step: int = 0
    push_to_hub: bool = False
    check_length_correctness: bool = True
    debug: bool = False
    tldr_params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=550,
            format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
            truncate_field="post",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
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


def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        # copy to avoid mutating input
        query_info = dict(**query_info)

    format_str = hparams.format_str or "{query}"
    query_tokens = encoder.encode(format_str.format(**query_info))
    truncate_field = hparams.truncate_field or "query"

    if truncate_field not in query_info:
        raise ValueError(f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!")
    while len(query_tokens) > hparams.length:
        if not len(query_info[truncate_field]):
            raise ValueError("Could not truncate enough!")

        i = -1  # default to just remove one character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_tokens = encoder.encode(format_str.format(**query_info))

    query_token = _ensure_length(query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
    query = encoder.decode(query_token, skip_special_tokens=True).lstrip()
    return dict(
        query_token=query_token,
        query=query,
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
        assert isinstance(args.hf_entity, str)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # post init
    args.tldr_params.pad_token = [tokenizer.pad_token_id]
    pprint(args)
    timestamp = int(time.time())
    if args.sft:
        label_ds = load_dataset("json", data_files=f'data/synthetic_tldr/output_sft_50k_samples.json')
    else:
        label_ds = load_dataset("json", data_files=f'data/synthetic_tldr/output_{args.step}_50k_samples.json')
    print(label_ds)
    
    def process_response_data(x):
        response1 = f" {x['response_A'].replace('<|endoftext|>', '').strip()}<|endoftext|>"
        response2 = f" {x['response_B'].replace('<|endoftext|>', '').strip()}<|endoftext|>"
        max_rm_response_length = 128
        max_rm_query_response_length = 640
        y = {
            'query': x['prompt'],
            'query_token': _ensure_length(tokenizer.encode(x['prompt']), args.tldr_params.length, pad_side=args.tldr_params.pad_side, pad_sequence=_get_query_padding_for_task(tokenizer, args.tldr_params)),
            'response1': response1,
            'response2': response2,
            "response1_token": tokenizer.encode(
                response1, padding="max_length", max_length=max_rm_response_length, truncation=True
            ),
            "response1_token_len": len(tokenizer.encode(response1)),
            "response2_token": tokenizer.encode(
                response2, padding="max_length", max_length=max_rm_response_length, truncation=True
            ),
            "response2_token_len": len(tokenizer.encode(response2)),
        }
        y["query_response1"] = y["query"].strip() + y["response1"]
        y["query_response2"] = y["query"].strip() + y["response2"]
        # if padding is space, then we can just concatenate the tokens
        y["query_response1_token"] = tokenizer.encode(
            y["query_response1"], padding="max_length", max_length=max_rm_query_response_length, truncation=True
        )

        y["query_response1_token_len"] = len(tokenizer.encode(y["query_response1"]))
        y["query_response2"] = y["query"].strip() + y["response2"]
        # if padding is space, then we can just concatenate the tokens
        y["query_response2_token"] = tokenizer.encode(
            y["query_response2"], padding="max_length", max_length=max_rm_query_response_length, truncation=True
        )
        y["query_response2_token_len"] = len(tokenizer.encode(y["query_response2"]))
        y["query_token_len"] = len(tokenizer.encode(y["query"]))
        unpadded_query_token = [token for token in y["query_token"] if token != tokenizer.pad_token_id]
        y["query_response1_token_response_label"] = copy.deepcopy(y["query_response1_token"])
        y["query_response1_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
        y["query_response2_token_response_label"] = copy.deepcopy(y["query_response2_token"])
        y["query_response2_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
        return y

    label_ds = label_ds.map(process_response_data, load_from_cache_file=False, num_proc=8)
    if args.sft:
        label_ds.save_to_disk(f"data/synthetic/sft")
    else:
        label_ds.save_to_disk(f"data/synthetic/step_{args.step}")
    
    if args.push_to_hub:
        rm_dataset_hf_path = f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{timestamp}"
        label_ds.push_to_hub(f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{timestamp}")

    ####################################
    # visualize token length distribution
    ####################################