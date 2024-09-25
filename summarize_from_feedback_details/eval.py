from collections import defaultdict
from configs import Config
import torch
from tqdm import tqdm
from utils import get_reward, forward, generate, truncate_response
from collections import defaultdict
from accelerate.utils import gather_object, broadcast
import collections
import pandas as pd
import evaluate as hf_evaluate
rouge = hf_evaluate.load("rouge")

def evaluate_sft(args: Config, accelerator, tokenizer, model, dataloader, generation_config):
    model.eval()
    rouge_scores = collections.defaultdict(list)
    all_decode_queries = []
    all_decode_responses = []
    all_decode_reference_responses = []
    all_losses = []
    unwrapped = accelerator.unwrap_model(model)
    for _, data in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            queries = data["query_token"]
            reference_responses = data["reference_response_token"]
            context_length = queries.shape[1]
            query_reference_responses = torch.cat((queries, reference_responses), dim=1)
            output = forward(model, query_reference_responses, tokenizer)
            labels = query_reference_responses.masked_fill(query_reference_responses == tokenizer.pad_token_id, -1)
            lm_logits = output.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-1,
            )
            loss = accelerator.gather(loss)
            all_losses.append(loss)

            generated_responses = generate(
                unwrapped,
                queries,
                tokenizer,
                generation_config,
            )
            responses = generated_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            decode_queries = tokenizer.batch_decode(queries)
            decode_reference_responses = tokenizer.batch_decode(
                reference_responses,
                skip_special_tokens=True,
            )
            decode_responses = tokenizer.batch_decode(
                postprocessed_responses,
                skip_special_tokens=True,
            )
            rouge_score = rouge.compute(predictions=decode_responses, references=decode_reference_responses)
            decode_queries = gather_object(decode_queries)
            decode_responses = gather_object(decode_responses)
            decode_reference_responses = gather_object(decode_reference_responses)
            rouge_scores["rouge1"].append(np.mean(gather_object([rouge_score["rouge1"]])))
            rouge_scores["rouge2"].append(np.mean(gather_object([rouge_score["rouge2"]])))
            rouge_scores["rougeL"].append(np.mean(gather_object([rouge_score["rougeL"]])))
            all_decode_queries.extend(decode_queries)
            all_decode_responses.extend(decode_responses)
            all_decode_reference_responses.extend(decode_reference_responses)
    return (
        pd.DataFrame(
            {
                "query": all_decode_queries,
                "response": all_decode_responses,
                "reference": all_decode_reference_responses,
            }
        ),
        rouge_scores,
        all_losses,
    )
def evaluate_rm(args: Config, accelerator, tokenizer, model, dataloader):
    model.eval()
    with torch.no_grad():
        items = defaultdict(list)
        for data in tqdm(dataloader):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            with accelerator.accumulate(model):
                predicted_reward = get_reward(model, query_responses, tokenizer)
                chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
                rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
                accuracy = (chosen_rewards > rejected_rewards).float()
                accuracy = accelerator.gather(accuracy)
                chosen_rewards = accelerator.gather(chosen_rewards)
                rejected_rewards = accelerator.gather(rejected_rewards)
            for k in data:
                data[k] = gather_object(data[k])
            for i in range(len(accuracy)):
                items["query"].append(tokenizer.decode(data["query_token"][i], skip_special_tokens=True))
                items["chosen"].append(tokenizer.decode(data["chosen_token"][i]))
                items["rejected"].append(tokenizer.decode(data["rejected_token"][i]))
                items["batch"].append(data["batch"][i])
                items["split"].append(data["split"][i])
                items["choice"].append(data["choice"][i].item())
                items["policies"].append(data["policies"][i])
                items["chosen_policy"].append(data["chosen_policy"][i])
                items["rejected_policy"].append(data["rejected_policy"][i])
                items["accuracy"].append(accuracy[i].item())
                items["chosen_rewards"].append(chosen_rewards[i].item())
                items["rejected_rewards"].append(rejected_rewards[i].item())
    model.train()
    return pd.DataFrame(items)
