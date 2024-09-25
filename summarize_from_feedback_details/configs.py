from dataclasses import dataclass
import os
from typing import Optional, Dict, List, Literal

@dataclass
class Config:
    warm_up_steps: int = 150
    local_micro_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True
    deepspeed: bool = False
    lr: float = 3e-6

    run_name: Optional[str] = None
    eps: float = 1e-5
    optimizer: Literal['adam', 'adamw', 'rmsprop'] = 'adamw'
    world_size: int = 1
    num_train_epochs: int = 1
    base_model: str = 'EleutherAI/pythia-1b-deduped'
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    max_new_tokens: int = 64
    temperature: float = 0.7
    track: bool = True
    truncate_token: str = 'eos'
    project_name: str = "tldr_summarize"
    entity: Optional[str] = None
    output_dir: str = 'models/sft_model'
    scheduler: str = 'cosine'
    num_updates: int = 1

@dataclass 
class DPOConfig(Config):
    beta: float = 0.1
    reference_model_path: str = ''

@dataclass
class RLOOConfig(Config):
    beta: float = 0.1
    rloo_k: int = 2