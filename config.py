"""Model and training hyperparameters."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 65       # shakespeare char vocab
    block_size: int = 256      # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_steps: int = 5000
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_steps: int = 50
    checkpoint_dir: str = 'checkpoints'


# smaller config for quick experiments
SMALL = GPTConfig(
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=128,
)
