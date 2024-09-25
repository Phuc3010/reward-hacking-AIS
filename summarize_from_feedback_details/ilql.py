import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    Adafactor,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    get_scheduler,
)
from huggingface_hub import HfApi
from models import ScalarModel, ScalarModelConfig

api = HfApi()


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw", 'adafactor'] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 92832
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 1
    """per rank eval batch size"""
    loss_name: str ='expectile'

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token  id"""
    temperature: float = 0.01
    """the sampling temperature"""
    label_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    ipo: bool = False
    """Whether to use IPO loss https://arxiv.org/abs/2310.12036"""
    label_smoothing: float = 0.0
    """Label smoothing for DPO (Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf))"""
    beta: float = 0.05
    """The beta value for DPO"""

    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    critic_model_path: str = ''
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/dpo_model"
    """Where to save the model"""
    save_steps: int = 3600
    mixture_coef: float = 0.125


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"{args.exp_name}__{args.seed}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def expectile_losses(diff, expectile=0.5):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def quantile_losses(diff, quantile=0.6):
    weight = torch.where(diff > 0, quantile, (1 - quantile))
    return weight * torch.abs(diff)

def get_value(model, query, tokenizer):
    attention_mask = query != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query, ~attention_mask, 0)
    value_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    return value_logits[:, -1]

def forward(model, query_responses, labels, tokenizer, average_log_prob: bool = False):
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
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        all_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        all_logps = (per_token_logps * loss_mask).sum(-1)
    return all_logps

def model_forward(model, query_responses, query, labels, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

    query_attention_mask = query != tokenizer.pad_token_id
    query_input_ids = torch.masked_fill(query, ~query_attention_mask, 0)

    output, vpred = model(
        query_responses_input_ids=input_ids,
        query_responses_attention_mask=attention_mask,
        query_input_ids=query_input_ids,
        query_attention_mask=query_attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
    labels = labels[:, 1:].clone()
    logits = output.logits[:, :-1, :]
    loss_mask = (labels != tokenizer.pad_token_id)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    all_logps = (per_token_logps * loss_mask).sum(-1)
    values = vpred[:, -1].squeeze(-1)
    return all_logps, values

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

def geometric_mixture_forward(model, ref_model, query_responses, labels, tokenizer, mixture_coef: float, average_log_prob: bool = False):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    model_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    ref_model_output = ref_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    labels = labels[:, 1:].clone()
    loss_mask = (labels != tokenizer.pad_token_id)

    model_logits = model_output.logits[:, :-1, :]
    ref_model_logits = ref_model_output.logits[:, :-1, :]

    per_token_logps = torch.gather(model_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    all_logps = (per_token_logps * loss_mask).sum(-1)

    ref_per_token_logps = torch.gather(ref_model_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    ref_all_logps = (ref_per_token_logps * loss_mask).sum(-1)

    mixture_per_token_logps = torch.gather(F.log_softmax(mixture_coef * ref_model_logits + (1 - mixture_coef) * model_logits, dim=-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    mixture_all_logps = (mixture_per_token_logps * loss_mask).sum(-1)
    if average_log_prob:
        mixture_all_logps /= loss_mask.sum(1)
        all_logps /= loss_mask.sum(1)
        ref_all_logps /= loss_mask.sum(1)
    
    return all_logps, ref_all_logps, mixture_all_logps 

def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses

class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, query_responses_input_ids, query_responses_attention_mask, query_input_ids, query_attention_mask, **kwargs):
        return self.policy(input_ids=query_responses_input_ids, attention_mask=query_responses_attention_mask, **kwargs), self.critic(input_ids=query_input_ids, attention_mask=query_attention_mask, **kwargs)
    
    def gradient_checkpointing_enable(self):
        self.critic.gradient_checkpointing_enable()
        self.policy.gradient_checkpointing_enable()

def evaluate_rm(args: Args, accelerator, tokenizer, model, ref_model, dataloader):
    model.eval()
    with torch.no_grad():
        items = defaultdict(list)
        for data in tqdm(dataloader):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            labels = torch.cat((data["query_chosen_token_response_label"], data["query_rejected_token_response_label"]), dim=0)
            ref_chosen_logps, ref_rejected_logps = forward(ref_model, query_responses, labels, tokenizer)
            chosen_logps, rejected_logps = forward(model, query_responses, labels, tokenizer)
            reward_preferred = args.beta * (chosen_logps - ref_chosen_logps)
            reward_rejected = args.beta * (rejected_logps - ref_rejected_logps)
            accuracy = reward_preferred > reward_rejected
            accuracy = accelerator.gather(accuracy)
            reward_preferred = accelerator.gather(reward_preferred)
            reward_rejected = accelerator.gather(reward_rejected)
            for k in data:
                data[k] = gather_object(data[k])
            for i in range(len(accuracy)):
                items["query"].append(tokenizer.decode(data["query_token"][i], skip_special_tokens=True))
                items["chosen"].append(tokenizer.decode(data["chosen_token"][i]))
                items["rejected"].append(tokenizer.decode(data["rejected_token"][i]))
                items["batch"].append(data["batch"][i])
                items["split"].append(data["split"][i])
                items["confidence"].append(data["extra.confidence"][i].item())
                items["choice"].append(data["choice"][i].item())
                items["policies"].append(data["policies"][i])
                items["chosen_policy"].append(data["chosen_policy"][i])
                items["rejected_policy"].append(data["rejected_policy"][i])
                items["accuracy"].append(accuracy[i].item())
                items["reward_preferred"].append(reward_preferred[i].item())
                items["reward_rejected"].append(reward_rejected[i].item())
    model.train()
    return pd.DataFrame(items)



@dataclass
class EvalStorage:
    query_token: List[str] = field(default_factory=list)
    postprocessed_response_token: List[str] = field(default_factory=list)
    reference_response_token: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    reference_score: List[float] = field(default_factory=list)

    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    reference_response: List[str] = field(default_factory=list)


def evaluate_policy(args: Args, model, tokenizer, dataloader, generation_config, sampling=True):
    eval_storage = EvalStorage()
    with torch.no_grad():
        for data in tqdm(dataloader):
            queries = data["query_token"]
            reference_response_token = data["reference_response_token"]
            context_length = queries.shape[1]
            query_responses = generate(
                model,
                queries,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            eval_storage.query_token.extend(queries)
            eval_storage.reference_response_token.extend(reference_response_token)
            eval_storage.postprocessed_response_token.extend(postprocessed_responses)
            if sampling:
                break

    eval_storage.query = tokenizer.batch_decode(eval_storage.query_token, skip_special_tokens=True)
    eval_storage.reference_response = tokenizer.batch_decode(eval_storage.reference_response_token)
    eval_storage.postprocessed_response = tokenizer.batch_decode(
        eval_storage.postprocessed_response_token, skip_special_tokens=True
    )
    # eval_score = torch.cat(eval_storage.score).float().cpu().numpy().tolist()
    # eval_reference_score = torch.cat(eval_storage.reference_score).float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "reference_responses": gather_object(eval_storage.reference_response),
            # "scores": gather_object(eval_score),
            # "reference_scores": gather_object(eval_reference_score),
        }
    )
    return eval_storage, eval_df

# def train(args: Args):
if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    # load dataset
    dataset = load_dataset('DatPySci/summarize_from_feedback_oai_preprocessing_pythia-6.9b-gold', split="train")
    dataset = dataset.shuffle(seed=local_seed)

    def add_query_eos_token(example):
        example['query_eos_token'] = example['query_token'] + [tokenizer.eos_token_id]
        return example
    
    dataset = dataset.map(add_query_eos_token, num_proc=8, batched=False)
    dataset = dataset.select(range(args.total_episodes))
    dataset = dataset.with_format(
        "torch",
        columns=[
            # 'query_eos_token',
            'query_token',
            "query_chosen_token",
            "query_chosen_token_response_label",
            "query_rejected_token",
            "query_rejected_token_response_label",
        ],
    )
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)
    eval_datasets = []
    eval_dataloaders = {}
    for split in ["validation", "validation_cnndm"]:
        validation_dataset = load_dataset(args.label_dataset, split=split).flatten()
        validation_dataset = validation_dataset.with_format(
            "torch",
            columns=[
                "choice",
                "chosen_token",
                "query_chosen_token",
                "query_chosen_token_response_label",
                "rejected_token",
                "query_rejected_token",
                "query_rejected_token_response_label",
                "batch",
                "split",
                "extra.confidence",
                "chosen_policy",
                "rejected_policy",
                "policies",
            ],
        )
        eval_datasets.append(validation_dataset)
        eval_dataloaders[split] = DataLoader(validation_dataset, batch_size=args.local_eval_batch_size)
    sft_validation_dataset = load_dataset(args.query_dataset, split="validation")
    sft_validation_dataset = sft_validation_dataset.with_format("torch", columns=["query_token", "reference_response_token", "query_reference_response_token_response_label"])
    sft_validation_dataloader = DataLoader(sft_validation_dataset, batch_size=args.local_eval_batch_size)
    args.num_updates = args.total_episodes // args.batch_size


    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    model_config = AutoConfig.from_pretrained('vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr', use_cache=False)
    critic_config = AutoConfig.from_pretrained(args.critic_model_path, use_cache=False)
    
    scalar_model_config = ScalarModelConfig(
        # base_model='vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr',
        # base_model='EleutherAI/pythia-410m-deduped',
        base_model=args.critic_model_path,
        base_config=critic_config,
        hidden_size=critic_config.hidden_size,
    )

    critic: PreTrainedModel = ScalarModel(scalar_model_config)
    # reward_model: PreTrainedModel = ScalarModel.from_pretrained(
    #     args.critic_model_path,
    #     trust_remote_code=True,
    # )
    
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        'vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr',
        config=model_config,
        revision='sft__44413__1708611267',
        trust_remote_code=True,
    )
    
    ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        'vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr',
        config=model_config,
        revision='sft__44413__1708611267',
        trust_remote_code=True,
    )
    for module in [policy, ref_model, critic]:
        disable_dropout(module)

    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / paddin

    model = PolicyAndValueWrapper(policy, critic)
    model.gradient_checkpointing_enable()
    ref_model.gradient_checkpointing_enable()

    AcceleratorState().deepspeed_plugin.deepspeed_config["gradient_checkpointing"] = True
    AcceleratorState().deepspeed_plugin.deepspeed_config["gradient_clipping"] = 10.0
    
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == 'adafactor':
        optimizer = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False)

    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )
    
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    torch.manual_seed(local_seed)  # reset the local seed again

    if args.deepspeed:
        import deepspeed
        deepspeed_states = AcceleratorState().deepspeed_plugin
        deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size

        eval_ds_config = {
            "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        accelerator.print(f"{eval_ds_config=}")
        ref_model, *_ = deepspeed.initialize(model=ref_model, config=eval_ds_config)
        ref_model.eval()
    else:
        ref_model = ref_model.to(device)
        ref_model.eval()
    
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=128,
        min_new_tokens=128,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training model===")
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    policy_losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    value_losses = torch.zeros((args.gradient_accumulation_steps,), device=device)

    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    model.train()

    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in dataloader:
            if global_step % args.save_steps == 0 and global_step > 0:
                unwrapped: PreTrainedModel = accelerator.unwrap_model(model).policy
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped.save_pretrained(
                        args.output_dir+f"/step_{global_step}",
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(model.policy),
                        safe_serialization=False,
                    )
            
            update += 1
            global_step += args.micro_batch_size
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            query_tokens = data['query_token'].repeat(2, 1)
            labels = torch.cat((data["query_chosen_token_response_label"], data["query_rejected_token_response_label"]), dim=0)
            
            bsz = query_responses.shape[0]
            with torch.no_grad():
                scores = torch.ones(size=(bsz,), device=query_responses.device)
                scores[query_responses.shape[0] // 2:] = -1
                scores /= 4
                policy_logps, reference_logps, mixture_logps = geometric_mixture_forward(accelerator.unwrap_model(model).policy, ref_model, query_responses, labels, tokenizer, args.mixture_coef, average_log_prob=True)
                values = get_value(accelerator.unwrap_model(model).critic, query_tokens, tokenizer)
                KL = policy_logps - reference_logps
                rlhf_reward = scores - args.beta * KL
            
            with accelerator.accumulate(model):
                new_policy_logps = forward(model.policy, query_responses, labels, tokenizer, average_log_prob=True)
                new_values = get_value(model.critic, query_tokens, tokenizer)
                
                if args.loss_name == 'expectile':
                    value_loss = expectile_losses(rlhf_reward - new_values)
                elif args.loss_name == 'quantile':
                    value_loss = quantile_losses(rlhf_reward - new_values)

                policy_loss = 0.5 * (scores - values - args.beta * (new_policy_logps - reference_logps)).pow(2) / args.beta
                value_loss = value_loss.mean()
                policy_loss = policy_loss.mean()
                loss = value_loss + policy_loss
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            with torch.no_grad():
                losses[gradient_accumulation_idx] = loss
                value_losses[gradient_accumulation_idx] = value_loss
                policy_losses[gradient_accumulation_idx] = policy_loss

            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                train_loss = accelerator.gather(losses).mean().item()
                writer.add_scalar("train/rm/loss", accelerator.gather(losses).mean().item(), global_step)
                writer.add_scalar("train/rm/value", accelerator.gather(values).mean().item(), global_step)
                writer.add_scalar("train/rm/rlhf_reward", accelerator.gather(rlhf_reward).mean().item(), global_step)
                writer.add_scalar("train/rm/policy_loss", accelerator.gather(policy_losses).mean().item(), global_step)
                writer.add_scalar("train/rm/value_loss", accelerator.gather(value_losses).mean().item(), global_step)
                writer.add_scalar("train/rm/forward_kl", accelerator.gather(reference_logps - policy_logps.detach()).mean().item(), global_step)
                writer.add_scalar("train/rm/lr", scheduler.get_last_lr()[0], global_step)
                accelerator.print(
                    f"{train_loss=}, {scheduler.get_last_lr()=}, {optimizer.param_groups[0]['lr']=}, {update=}"
                )
                del KL, policy_logps
                torch.cuda.empty_cache()

    # if args.run_eval:
    #     _, evaluate_df = evaluate_policy(args, model.policy, tokenizer, sft_validation_dataloader, validation_generation_config, sampling=False)
    #     if accelerator.is_main_process:
    #         evaluate_df.to_csv(f"runs/{args.run_name}/table.csv")
    #         if args.track:
    #             wandb.log({"eval/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
    #     del evaluate_df
    #     torch.cuda.empty_cache()
    #     for eval_split in eval_dataloaders:
    #         evaluate_df = evaluate_rm(args, accelerator, tokenizer, model, ref_model, eval_dataloaders[eval_split])
    #         for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
    #             writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split}", row["accuracy"], global_step)
    #             accelerator.print(f"eval/rm/{eval_split}/accuracy/split/{split}: {row['accuracy']}")
    #         for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
    #             writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], global_step)
    #             accelerator.print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
    #         for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
    #             writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], global_step)
    #             accelerator.print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
    #         writer.add_scalar(f"eval/rm/{eval_split}/accuracy", evaluate_df["accuracy"].mean(), global_step)
    #         accelerator.print(f"eval/rm/{eval_split}/accuracy: {evaluate_df['accuracy'].mean()}")
    #         if accelerator.is_main_process:
    #             os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
    #             evaluate_df.to_csv(f"eval_tables/{args.run_name}/eval_{eval_split}_{update}.csv")
    #             if args.track:
    #                 wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
    #         del evaluate_df
    #         torch.cuda.empty_cache()

    # save model
    if args.output_dir and args.num_train_epochs > 0:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
        unwrapped: PreTrainedModel = accelerator.unwrap_model(model.policy)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model.policy),
                safe_serialization=False,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
                accelerator.print(f"🔥 pushed to {args.hf_repo_url}")