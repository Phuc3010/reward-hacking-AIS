from datasets import load_dataset

if __name__ == "__main__":
    step = "50400"

    ds = load_dataset("json", data_files=f'data/synthetic/step_{step}/labeled_train.json')

    def reformat(ex):
        ex = {k.replace("reponse1", "response1"): v for k, v in ex.items()}
        if ex['response1_score'] > ex['response2_score']:
            ex = {k.replace("response1", "chosen"): v for k, v in ex.items()}
            ex = {k.replace("response2", "rejected"): v for k, v in ex.items()}
        else:
            ex = {k.replace("response2", "chosen"): v for k, v in ex.items()}
            ex = {k.replace("response1", "rejected"): v for k, v in ex.items()}        
        return ex

    ds = ds.map(reformat, load_from_cache_file=False, num_proc=8)
    ds = ds.remove_columns([col for col in ds.column_names if "response1" in col or "response2" in col])
    ds.save_to_disk(f"data/tldr_pref/step_{step}")