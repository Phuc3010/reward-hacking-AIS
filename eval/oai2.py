# you can download the CSV from https://wandb.ai/costa-huang/tldr_summarize/runs/gb2dian5

import asyncio
import random
from dataclasses import dataclass, field
import time
from typing import Optional
import openai

import pandas as pd
import tyro
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


@dataclass
class LLMJudgeConfig:
    n: int = 64
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: Optional[int] = None

    def __post_init__(self):
        if "gpt-3.5" in self.model:
            # gpt-3.5 generates so fast that it will exceeds the
            # token limit per minute
            self.max_parallel_requests = 6
        elif "gpt-4" in self.model:
            self.max_parallel_requests = 13

@dataclass
class Args:
    csv: str = "trained_response.csv"
    output_path: Optional[str] = None
    num_trails: int = 1

@dataclass
class ArgsWrapper:
    args: tyro.conf.OmitArgPrefixes[Args] = field(default_factory=Args)
    ljc: tyro.conf.OmitArgPrefixes[LLMJudgeConfig] = field(default_factory=LLMJudgeConfig)

    def __post_init__(self):
        if self.args.output_path is None:
            self.args.output_path = self.args.csv.split(".csv")[0] + f"{self.ljc.model}.csv"

TEMPLATE = r"""
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post:
{{post}}

### Summary A:
{{response0}}

### Summary B:
{{response1}}

### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""

def llm_judge(ljc: LLMJudgeConfig, df: pd.DataFrame):
    limiter = asyncio.Semaphore(ljc.max_parallel_requests)
    async_client = AsyncOpenAI()

    async def process_text(post: str, response0: str, response1: str, i: int):
        text = TEMPLATE.replace("{{post}}", post)
        text = text.replace("{{response0}}", response0)
        text = text.replace("{{response1}}", response1)  # Ensure this split logic is correct for your data

        async with limiter:
            response = None
            while response is None:
                try:
                    response = await async_client.chat.completions.create(
                        model=ljc.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": text},
                        ],
                    )
                    r = response.choices[0].message.content
                except Exception as e:
                    print(f"error in {i}: {e}")
                    time.sleep(30) # deal with rate limit
                    continue
            
            try:
                comparison = r.split("Comparison:")[1].split("Preferred:")[0].strip()
                preferred = r.split("Preferred:")[1].strip()
                return comparison, preferred, i, text + r
            except:
                print(f"error in {i}")
                return "", random.choice(["A", "B"]), i, text + r


    async def main(ljc: LLMJudgeConfig, df: pd.DataFrame):
        """`df` should have columns: `prompt`, `response0`, `response1`"""
        tasks = []
        df["explanation"] = [None for _ in range(len(df))]
        df["preferred"] = [None for _ in range(len(df))]
        df["shuffled_index"] = [None for _ in range(len(df))]
        df["entire_conversation"] = [None for _ in range(len(df))]
        r = range(min(ljc.n, len(df)))
        if ljc.n == -1:
            r = range(len(df))
        for i in r:
            post = df["prompt"].iloc[i].strip()
            # shuffled the index to avoid GPT4's preference bias in the content's order
            shuffled_index = random.randint(0, 1)
            df.at[i, "shuffled_index"] = shuffled_index
            responses = [
                df["response0"].iloc[i].strip(),
                df["response1"].iloc[i].strip(),
            ]
            response0 = responses[shuffled_index]
            response1 = responses[1 - shuffled_index]
            task = asyncio.create_task(process_text(post, response0, response1, i))
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks)

        for _, (comparison, preferred, i, entire_conversation) in enumerate(results):
            df.at[i, "explanation"] = comparison
            df.at[i, "entire_conversation"] = entire_conversation
            preferred_label = (
                "response0"
                if (df.at[i, "shuffled_index"] == 0 and preferred == "A")
                or (df.at[i, "shuffled_index"] == 1 and preferred == "B")
                else "response1"
            )
            df.at[i, "preferred"] = preferred_label
        print(df["preferred"].value_counts())
        return df
    return asyncio.run(main(ljc, df))

if __name__ == "__main__":
    all_args = tyro.cli(ArgsWrapper)
    args = all_args.args
    ljc = all_args.ljc
    df = pd.read_csv(args.csv)
    df["reference_responses"] = df["reference_responses"].map(lambda x: x.split("<|endoftext|>")[0].strip())
    df["prompt"] = df["query"].map(lambda x: x.strip())
    df["response0"] = df["postprocessed_response"].map(lambda x: x.strip())
    df["response1"] = df["reference_responses"].map(lambda x: x.strip())
    judge_df = llm_judge(ljc, df)
    judge_df.to_csv(args.output_path)