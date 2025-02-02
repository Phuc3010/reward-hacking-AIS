import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Literal, Optional, Dict
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast, gather_object
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    get_scheduler,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)

torch.set_printoptions(precision=4, sci_mode=False)
api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class PpoHParams:
    nminibatches: int = 1
    noptepochs: int = 1
    cliprange: float = 0.2
    kl_coef: float = 0.05

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
    """Whether to use deepspeed to train the policy"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 150
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 64
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 1000000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    nminibatches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 2
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 32
    """per rank no grad forward pass in the rollout phase"""
    missing_eos_penalty: float = 1.0
    save_steps: int = 3600

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained policy to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    response_length: int = 64
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    rloo_k: int = 2
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = 1.0
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = True
    """whether to penalize responses that do not contain `truncate_token_id`"""
    offload: bool = False
    """Whether to offload ref policy and reward policy to CPU"""
    reward_model_path: str = ""
    """the path to the reward policy"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft policy"""

    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved policy to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the policy repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved policy in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved policy in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved policy in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/ppo_model"
    """Where to save the policy"""
    ppo: PpoHParams = field(default_factory=PpoHParams)


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps * args.nminibatches
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(args.batch_size, args.nminibatches)
    args.local_mini_batch_size = exact_div(args.local_batch_size, args.nminibatches)
    # `per_rank_rollout_batch_size` is our `args.local_batch_size`
    # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
    args.num_updates = args.total_episodes // args.batch_size
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
def disable_dropout(policy: torch.nn.Module):
    """Disable dropout in a policy."""
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def formatted_dict(d: Dict) -> Dict:
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}

@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    tokenizer,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            tokenizer,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)
    return torch.cat(query_responses, 0), torch.cat(logitss, 0)

def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


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


def get_reward(policy, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )

def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


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
        output_scores=True
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


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


def forward(policy, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def evaluate(reward_model, policy, tokenizer, dataloader, generation_config, sampling=True):
    eval_storage = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(dataloader):
            queries = data["query_token"]
            reference_response_token = data["reference_response_token"]
            context_length = queries.shape[1]
            query_reference_responses = torch.cat((data["query_token"], data["reference_response_token"]), dim=1)
            _, reference_score, _ = get_reward(reward_model, query_reference_responses, tokenizer, queries.shape[1])

            query_responses, _ = generate(
                policy,
                queries,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            _, score, _ = get_reward(reward_model, postprocessed_query_responses, tokenizer, queries.shape[1])

            eval_storage["query_token"].extend(queries)
            eval_storage["reference_response_token"].extend(reference_response_token)
            eval_storage["reference_score"].append(reference_score)
            eval_storage["postprocessed_response_token"].extend(postprocessed_responses)
            eval_storage["score"].append(score)
            if sampling:
                break

    eval_storage["query"] = tokenizer.batch_decode(eval_storage["query_token"], skip_special_tokens=True)
    eval_storage["reference_response"] = tokenizer.batch_decode(eval_storage["reference_response_token"])
    eval_storage["postprocessed_response"] = tokenizer.batch_decode(
        eval_storage["postprocessed_response_token"], skip_special_tokens=True
    )
    eval_score = torch.cat(eval_storage["score"]).float().cpu().numpy().tolist()
    eval_reference_score = torch.cat(eval_storage["reference_score"]).float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage["query"]),
            "postprocessed_response": gather_object(eval_storage["postprocessed_response"]),
            "reference_responses": gather_object(eval_storage["reference_response"]),
            "scores": gather_object(eval_score),
            "reference_scores": gather_object(eval_reference_score),
        }
    )
    return eval_storage, eval_df


if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    # load dataset
    dataset = load_dataset(args.query_dataset, split="train")
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
    args.total_episodes = 2 * len(dataset)
    args.num_updates = args.total_episodes // args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
    eval_dataloaders = {}
    for split in ["validation", "test"]:
        eval_dataset = load_dataset(args.query_dataset, split=split)
        eval_dataset = eval_dataset.with_format("torch", columns=["query_token", "reference_response_token"])
        eval_dataloaders[split] = DataLoader(eval_dataset, batch_size=args.local_eval_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the policy
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
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
    model_config = AutoConfig.from_pretrained(args.base_model)
    scalar_model_config = ScalarModelConfig(
        base_model=args.base_model,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if not args.reward_model_path:
        reward_model: PreTrainedModel = ScalarModel(scalar_model_config)
    else:
        reward_model: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    
    model_config = AutoConfig.from_pretrained('vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr')
    
    policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        'vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr',
        config=model_config,
        revision='sft__44413__1708611267',
        trust_remote_code=True,
    )
    
    ref_policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        'vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr',
        config=model_config,
        revision='sft__44413__1708611267',
        trust_remote_code=True,
    )
    
    for module in [policy, ref_policy, reward_model]:
        disable_dropout(module)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

    if args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(policy.parameters(), lr=args.lr, eps=args.eps)

    # scheduler = get_scheduler(
    #     args.scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.warm_up_steps,
    #     num_training_steps=args.num_updates,
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (args.warm_up_steps + 1)))

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    torch.manual_seed(local_seed)  # reset the local seed again

    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
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
        if args.offload or args.base_model == "EleutherAI/pythia-6.9b-deduped":
            deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
            eval_ds_config["zero_optimization"] = {
                "stage": 3,
                "stage3_param_persistence_threshold": 1e4,
                "offload_param": {"device": "cpu"},
            }
        accelerator.print(f"{eval_ds_config=}")
        reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        reward_model.eval()
        ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
        ref_policy.eval()
    else:
        ref_policy = ref_policy.to(device)
        reward_model = reward_model.to(device)

    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        temperature=(args.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training policy===")
    global_step = 0
    start_time = time.time()
    eval_split = list(eval_dataloaders.keys())[0]
    stats_shape = (args.ppo.noptepochs, args.nminibatches, args.gradient_accumulation_steps)
    approxkl_stats = torch.zeros(stats_shape, device=device)
    pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
    pg_loss_stats = torch.zeros(stats_shape, device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)
    policy.train()

    for update in range(1, args.num_updates + 1):
        if global_step % args.save_steps == 0 and global_step > 0:
            unwrapped: PreTrainedModel = accelerator.unwrap_model(policy)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped.save_pretrained(
                    args.output_dir+f"/step_{global_step}",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(policy),
                    safe_serialization=False,
                )
        global_step += args.batch_size
        data = next(iter_dataloader)
        with torch.no_grad():
            queries = data["query_token"].to(device)
            queries = queries.repeat(args.rloo_k, 1)
            context_length = queries.shape[1]
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []

            query_responses, logitss = batch_generation(
                accelerator.unwrap_model(policy),
                queries,
                args.local_rollout_forward_batch_size,
                tokenizer,
                generation_config,
            )

            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i : i + args.local_rollout_forward_batch_size]
                query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                response = query_response[:, context_length:]
                logits = logitss[i : i + args.local_rollout_forward_batch_size]
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del logits, all_logprob
                torch.cuda.empty_cache()

                ref_output = forward(ref_policy, query_response, tokenizer)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logits /= args.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                postprocessed_response = truncate_response(args, tokenizer, response)

                # Response Processing 2. run reward policy on the truncated responses
                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                _, score, _ = get_reward(reward_model, postprocessed_query_response, tokenizer, context_length)

                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                sequence_lengths.append(sequence_length)
                scores.append(score)

            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)
            del (logprob, ref_logprob, score)
            torch.cuda.empty_cache()
            gc.collect()

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
            if args.non_eos_penalty:
                scores[~contain_eos_token] -= args.missing_eos_penalty
            
            # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            kl = logprobs - ref_logprobs
            non_score_reward = (-args.ppo.kl_coef * kl).sum(1)
            rlhf_reward = scores + non_score_reward

            # vectorized RLOO advantages implementation
            rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
            baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
            advantages = rlhf_reward - baseline
            advantages = advantages.flatten()
            torch.cuda.empty_cache()
        
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            b_inds = np.random.permutation(args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, args.local_micro_batch_size):
                    with accelerator.accumulate(policy):
                        micro_batch_end = micro_batch_start + args.local_micro_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]

                        output = forward(policy, mb_query_responses, tokenizer)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        new_ratio = (new_logprobs - mb_logprobs).exp()
                        new_logprobs = new_logprobs.sum(1)
                        mb_logprobs = mb_logprobs.sum(1)
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_loss = pg_losses.mean()
                        loss = pg_loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        with torch.no_grad():
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            # pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                # del everything and empty cache
                # fmt: off
                del (
                    output, logits, new_all_logprobs, new_logprobs, logprobs_diff, ratio, pg_losses,
                    pg_loss, loss,  prob_dist, entropy, approxkl, 
                    mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                )
                # fmt: on
                torch.cuda.empty_cache()
            if accelerator.is_main_process:
                console.print(
                    "ppo_epoch_idx",
                    ppo_epoch_idx,
                    "approxkl",
                    approxkl_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_loss",
                    pg_loss_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_clipfrac",
                    pg_clipfrac_stats[: ppo_epoch_idx + 1].mean().item(),
                    "ratio",
                    ratio_stats[: ppo_epoch_idx + 1].mean().item(),
                )
        with torch.no_grad():
            scheduler.step()
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.mean()
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar(
                "objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update
            )
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkl_stats).mean().item(), update)
            # writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(pg_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_loss_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropy_stats).mean().item(), update)
            writer.add_scalar("ppo/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("ppo/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("ppo/lr", scheduler.get_last_lr()[0], update)
            writer.add_scalar("ppo/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("ppo/eps", eps, update)
            metrics = {
                "loss": accelerator.gather(pg_loss_stats).mean().item(),
                "kl": accelerator.gather(mean_kl).mean().item(),
                "entropy": accelerator.gather(mean_entropy).mean().item(),
                "scores": accelerator.gather(scores).mean().item(),
                "non_score_reward": accelerator.gather(mean_non_score_reward).mean().item(),
                "rlhf_reward": accelerator.gather(rlhf_reward).mean().item(),
                "num_eos_tokens": (responses == tokenizer.eos_token_id).sum().item(),
                'lr': scheduler.get_last_lr()[0]
            }
            
            metrics = formatted_dict(metrics)
            accelerator.print(f'train stats after {global_step} examples: {metrics}')
        del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
        torch.cuda.empty_cache()

    if args.run_eval:
        for eval_split in eval_dataloaders:
            eval_storage, eval_df = evaluate(
                reward_model,
                accelerator.unwrap_model(policy),
                tokenizer,
                eval_dataloaders[eval_split],
                validation_generation_config,
                sampling=False,
            )
            if accelerator.is_main_process:
                eval_ds = Dataset.from_pandas(eval_df)
                eval_ds.save_to_disk(f"runs/{args.run_name}/{eval_split}_dataset")
                if args.track:
                    wandb.log({f"eval/{eval_split}_query_responses": wandb.Table(dataframe=eval_df)}, step=update)

    # save policy
    if args.output_dir and args.num_train_epochs > 0:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
        unwrapped: PreTrainedModel = accelerator.unwrap_model(policy)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(unwrapped),
                safe_serialization=False,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
                accelerator.print(f"🔥 pushed to https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}")