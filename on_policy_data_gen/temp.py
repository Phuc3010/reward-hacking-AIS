from datasets import load_from_disk
import torch

if __name__=="__main__":
    steps = "46800"
    dataset = load_from_disk(f"data/tldr_pref/step_{steps}")['train']
    dataset = dataset.with_format(
        "torch",
        columns=[
            "query_token",
            "chosen_token",
            "query_chosen_token",
            "query_chosen_token_response_label",
            "rejected_token",
            "query_rejected_token",
            "query_rejected_token_response_label",
        ],
    )
    sample = dataset[0]
    print(sample['chosen_token'].size())
    print(sample['query_chosen_token'].size())
    print(sample['query_rejected_token'].size())
    print(sample['rejected_token'].size())
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    # for batch in dataloader:
    #     print(batch)
    #     break