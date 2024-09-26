from collections import defaultdict
from dataclasses import dataclass
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from tqdm.contrib import tzip
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
torch.autograd.grad_mode.set_grad_enabled(False)

@dataclass
class RunRecord:
    wandb_url: str
    hf_repo_url: str
    hf_repo_id: str
    revision: str

def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer

def forward(model, query_responses, labels, tokenizer, gather: bool = True):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    labels = labels[:, 1:].clone()
    logits = output.logits[:, :-1, :]
    loss_mask = (labels != tokenizer.pad_token_id)
    if gather:
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        all_logps = (per_token_logps * loss_mask).sum(-1)
    else:
        all_logps = (logits.log_softmax(-1) * loss_mask.unsqueeze(2)).squeeze(2)
    return all_logps

def get_reward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = (torch.eq(query_responses, tokenizer.pad_token_id).long().argmax(-1) - 1).to(query_responses.device)
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths], reward_logits

def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)

class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


if __name__=="__main__":
    console = Console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_step", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=256)

    args = parser.parse_args()
    print(args.key)

    if not os.path.exists("release_runs.csv"):
        import wandb
        keys = {
            "sft": "refactor-chosen-rejected3",
            "reward": "refactor-chosen-rejected3",
            "ppo_left_padding_new_nowhiten_reward": "refactor-chosen-rejected3",
            "dpo": "refactor-chosen-rejected2",
        }
        runs = []
        for exp_name, tag in keys.items():
            runs.extend(list(wandb.Api().runs(
                path=f"costa-huang/tldr_summarize",
                filters={
                    "$and": [
                        {f"config.exp_name.value": exp_name},
                        {"tags": {"$in": [tag]}},
                    ]
                }
            )))
        table = defaultdict(list)
        for i in range(len(runs)):
            table["base_model"].append(runs[i].config["base_model"])
            table["exp"].append(runs[i].config["exp_name"])
            table["seed"].append(runs[i].config["seed"])
            table["wandb_url"].append(runs[i].url)
            table["hf_repo_url"].append(runs[i].config["hf_repo_url"])
            table["hf_repo_id"].append(runs[i].config["hf_repo_id"])
            table["revision"].append(runs[i].config["run_name"])
        df = pd.DataFrame(table)
        df.to_csv("release_runs.csv", index=False)
    else:
        df = pd.read_csv("release_runs.csv")


    df = df.groupby(["base_model", "exp", "seed"]).agg(lambda x: x.tolist()[0])
    model_seed = 44413
    sft_record = RunRecord(**df.loc[("EleutherAI/pythia-1b-deduped", "sft", model_seed)])

    ######
    # Start
    ######
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    response_length = 128
    validation_generation_config = GenerationConfig(
        max_new_tokens=response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sft_dataset = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144")

    console.print("loading", sft_record)
    sft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        # sft_record.hf_repo_id,
        f'models/EleutherAI/pythia-1b-deduped/dpo_policy_beta_0.05_44413/step_{args.model_step}',
        # revision=sft_record.revision,
        use_cache=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        trust_remote_code=True,
    ).to(device)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        use_cache=True,
        trust_remote_code=True,
    ).to(device)

    table = defaultdict(list)
    import random
    random.seed(args.seed)
    sft_dataset = sft_dataset['validation']
    sft_dataset = sft_dataset.shuffle(args.seed)
    sft_dataset = sft_dataset.select(range(args.n_samples))
    sft_dataset = sft_dataset.with_format(
    "torch",
    columns=[
        "query_token",
        'query_reference_response_token',
        'query_reference_response',
        'reference_response_token',
    ]
    )

    sft_loader = torch.utils.data.DataLoader(
        sft_dataset, 
        batch_size=8
    )

    all_model_query_responses = []
    all_reference_query_responses = []
    context_lengths = []
    i = 0 

    for batch in tqdm(sft_loader):
        with torch.no_grad():
            query = batch['query_token'].to(device)
            query_reference_response = batch['query_reference_response_token']
            tokenizer.padding_side = 'left'
            query_reference_response = tokenizer(batch['query_reference_response'], padding='max_length', max_length=640, truncation=True, return_tensors='pt')['input_ids'].to(device)
            tokenizer.padding_side = 'right'
            context_length = query.shape[1]
            table['Prompt'].extend(tokenizer.batch_decode(query, skip_special_tokens=True))
            model_query_response = generate(model, query, tokenizer, validation_generation_config)

            model_query_response_lst = model_query_response.cpu().split(1, dim=0)
            reference_query_response_lst = query_reference_response.cpu().split(1, dim=0)
            
            all_model_query_responses.extend(model_query_response_lst)
            all_reference_query_responses.extend(reference_query_response_lst)
            context_lengths.extend([context_length] * 8)

            model_response = model_query_response[:, context_length:]

            labels = model_query_response.clone()
            labels[:, :context_length] = tokenizer.pad_token_id
            
            model_logprobs = forward(model, model_query_response, labels, tokenizer, gather=False)
            reference_model_logprobs = forward(sft_model, model_query_response, labels, tokenizer, gather=False)
            kl_div = torch.nn.functional.kl_div(input=reference_model_logprobs, target=model_logprobs, log_target=True, reduction='none').sum(dim=(1,2))

            table[f"{args.key} Model Response"].extend(tokenizer.batch_decode(model_response, skip_special_tokens=True))
            table["Reference Model Response"].extend(tokenizer.batch_decode(batch['reference_response_token'], skip_special_tokens=True))
            table['KL Divergence'].extend(kl_div.cpu().tolist())

    print(f"KL divergence: {sum(table['KL Divergence']) / len(table['KL Divergence'])}")

    model.to('cpu')
    sft_model.to('cpu')
    del model, sft_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    model_seed = 44413
    # rm_record = RunRecord(**df.loc[("EleutherAI/pythia-6.9b-deduped", "reward", model_seed)]) # larger (in some sense gold) RM
    hf_repo_id = 'vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr'
    revision = 'reward__44413__1708628552'

    scalar_model_config = ScalarModelConfig.from_pretrained(
        hf_repo_id,
        revision=revision,
        trust_remote_code=True,
    )

    original_model = "/".join(scalar_model_config.base_config["_name_or_path"].split("/")[1:3])
    scalar_model_config.base_config["_name_or_path"] = original_model
    scalar_model_config.base_model = original_model

    rm: PreTrainedModel = ScalarModel.from_pretrained(
        hf_repo_id,
        revision=revision,
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True,
        config=scalar_model_config,
    )

    for context_length, model_query_response, reference_query_response in tzip(context_lengths, all_model_query_responses, all_reference_query_responses):
        with torch.no_grad():
            model_query_response = model_query_response.to(device)
            model_reward, model_reward_logits = get_reward(rm, model_query_response, tokenizer)
            model_reward = model_reward.squeeze().item()
            table["Model Score (RM)"].append(model_reward)
            table["Model Length"].append(model_query_response.shape[1] - context_length)

            reference_query_response = reference_query_response.to(device)
            reference_reward, reference_reward_logits = get_reward(rm, reference_query_response, tokenizer)
            reference_reward = reference_reward.squeeze().item()
            table["Reference Score (RM)"].append(reference_reward)
            table["Reference Length"].append(reference_query_response.shape[1] - context_length)

            if reference_reward < model_reward:
                table['Policy Wins'].append('wins')
            elif reference_reward > model_reward:
                table['Policy Wins'].append('lose')
            else:
                table['Policy Wins'].append('tie')
    
    df = pd.DataFrame(table)
    df.to_csv("results/table_results.csv", index=False)
    print_rich_table("Results", df.head(15), console)

    kl_div = table['KL Divergence']
    win_rate = table['Policy Wins']
    policy_wins = sum([1 for ele in win_rate if ele == 'wins'])
    reference_wins = sum([1 for ele in win_rate if ele == 'lose'])

    policy_lengths = table['Model Length']
    reference_lengths = table['Reference Length']

    policy_scores = table["Model Score (RM)"]
    reference_scores = table['Reference Score (RM)']
    results = {
        'total': args.n_samples,
        'seed': args.seed,
        'judge' : 'EleutherAI/pythia-6.9b-golld-rm',
        'candidate': {
            'name': args.key,
            'wins': policy_wins,
            'lengths': sum(policy_lengths) / len(policy_lengths),
            'scores': sum(policy_scores) / len(policy_scores),
            'KL divergence': sum(kl_div) / len(kl_div)
        },
        'baseline': {
            'name': 'reference',
            'wins': reference_wins,
            'lengths': sum(reference_lengths) / len(reference_lengths),
            'scores': sum(reference_scores) / len(reference_scores),
        },
    }
    
    import json
    with open('results/results.json', 'a+') as f:
        json.dump(results, f)
        f.write('\n')



    
