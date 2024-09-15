from collections import defaultdict
from dataclasses import dataclass
import tyro
import json
import os
import numpy as np
from transformers import AutoModelForSequenceClassification
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
torch.autograd.grad_mode.set_grad_enabled(False)

def get_reward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).logits

    sequence_lengths = (torch.eq(query_responses, tokenizer.pad_token_id).long().argmax(-1) - 1).to(query_responses.device)
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths]

def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)

@dataclass
class Args:
    step: int = 0
    sft: bool = False

if __name__=="__main__":
    ######
    # Start
    ######
    args = tyro.cli(Args)
    table = defaultdict(list)
    split = 'train'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    if args.sft:
        pref_dataset = load_from_disk(f"data/synthetic/sft")
    else:
        pref_dataset = load_from_disk(f"data/synthetic/steo_{args.step}")
    
    pref_dataset = pref_dataset.remove_columns("prompt")

    rm = AutoModelForSequenceClassification.from_pretrained("cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr", torch_dtype=torch.bfloat16, device_map='auto')
    print(rm)
    rm.config.pad_token_id = tokenizer.pad_token_id
    
    table = defaultdict(list)
    split = 'train'
    if args.sft:
        filename = f'data/tldr_pref/sft.json'
    else:
        filename = f'data/tldr_pref/{args.step}.json'
    import json
    running_rewards = 0.0
    iter_tqdm = tqdm(pref_dataset)
    flips = 0

    iter_tqdm = tqdm(range(0, len(pref_dataset[split])))

    for i in iter_tqdm:
        query = torch.LongTensor(pref_dataset[split][i : i + 1]["query_token"]).to(device)
        context_length = query.shape[1]
        query_response1 = torch.cat((query, torch.LongTensor(tokenizer.encode(pref_dataset[split][i]["response_A"])).to(device).unsqueeze(0)), dim=1)

        query_response2 = torch.cat((query, torch.LongTensor(tokenizer.encode(pref_dataset[split][i]["response_B"])).to(device).unsqueeze(0)), dim=1)
        response1_rewards = get_reward(rm, query_response1, tokenizer).squeeze().item()
        response2_rewards = get_reward(rm, query_response2, tokenizer).squeeze().item()

        d = pref_dataset[split][i]
        if i == 0:
            print(tokenizer.batch_decode(query_response1, skip_special_tokens=True)[0])
            print(d.keys())
            print(response1_rewards)

        d["response1_score"] = response1_rewards
        d["response2_score"] = response2_rewards

        running_rewards += (response1_rewards + response2_rewards) / 2
        average_reward = running_rewards / (i + 1)
        iter_tqdm.set_postfix(average_reward=average_reward)

        with open(filename, 'a') as f:
            json.dump(d, f)
            f.write('\n')