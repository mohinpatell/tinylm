"""Model and training hyperparameters."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 65       # shakespeare char vocab
    block_size: int = 128      # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_steps: int = 5000
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_steps: int = 50
    checkpoint_dir: str = 'checkpoints'
