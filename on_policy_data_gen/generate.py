from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GenerationConfig
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, PreTrainedModel, AutoModelForCausalLM
import os
import argparse
import json
from rich import print as rich_print
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.9,
                    help='Temperature for sampling')
parser.add_argument('--sft', type=bool, default=False,
                    help='is loading SFT model')
parser.add_argument('--top_p', type=float, default=0.9, help='Top-p probability for sampling')
parser.add_argument('--step', type=int, default=0,
                    help='model steps')
parser.add_argument('--max_tokens', type=int, default=128,
                    help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="data/synthetic_tldr",
                    help='output_dir')
args = parser.parse_args()

print(args)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
train_dataset= load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144', split='train')
train_dataset = train_dataset.shuffle(seed=args.seed)
train_dataset = train_dataset.select(range(50_000))

model_config = AutoConfig.from_pretrained('vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr')

if args.sft:
    rich_print("Loading SFT weights")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        'vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr',
        config=model_config,
        use_flash_attention_2=True,
        revision='sft__44413__1708611267',
        device_map='auto',
        trust_remote_code=True,
    )
else:
    rich_print("Loading pre-trained weights")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        use_flash_attention_2=True,
        device_map='auto',
        trust_remote_code=True,
    )

train_dataset = train_dataset.with_format(
"torch",
columns=[
    "query_token",
]
)

batch_size = 16
dataloader= DataLoader(train_dataset, batch_size=batch_size)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
generation_config = GenerationConfig(
    do_sample=True,
    top_p=0.95, 
    temperature=0.9,
    max_new_tokens=128,
    num_return_sequences=2,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    return_scores=True
)
rich_print(model)
device = 'cuda'
output_data = []

import tqdm
for batch in tqdm.tqdm(dataloader):
    with torch.no_grad():
        prompt_mask = (batch['query_token'] != tokenizer.pad_token_id).long()
        with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
            context_length = batch['query_token'].size(1)
            sequences = model.generate(
                input_ids=batch['query_token'].to(device),
                attention_mask=prompt_mask.to(device),
                generation_config=generation_config
            ).sequences
        sequences = sequences.view(batch_size, 2, -1)[..., context_length:]
        sequences1, sequences2 = torch.split(sequences, dim=1, split_size_or_sections=1)
        decoded_sequences1 = tokenizer.batch_decode(sequences1.squeeze(1), skip_special_tokens=True)
        decoded_sequences2 = tokenizer.batch_decode(sequences2.squeeze(1), skip_special_tokens=True)
        decoded_prompts = tokenizer.batch_decode(batch['query_token'], skip_special_tokens=True)
        output_data.extend([{"prompt": prompt, "response_A": seq1, "response_B": seq2} for prompt, seq1, seq2 in zip(decoded_prompts, decoded_sequences1, decoded_sequences2)])

if args.sft:
    output_file = f'output_sft_50k_samples.json'
else:
    output_file = f'output_{args.step}_50k_samples.json'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")