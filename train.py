"""Training loop."""

import os
import time
import math
import torch
from torch.utils.data import DataLoader

from config import GPTConfig, TrainConfig
from model import GPT
from dataset import get_datasets


def get_lr(step, config):
    """Linear warmup then cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    # cosine decay to 10% of max lr
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * (config.learning_rate * 0.9)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, n_steps, device):
    """Average loss over n_steps batches."""
    model.eval()
    losses = {}
    for name, loader in [('train', train_loader), ('val', val_loader)]:
        total_loss = 0
        count = 0
        for i, (x, y) in enumerate(loader):
            if i >= n_steps:
                break
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y)
            total_loss += loss.item()
            count += 1
        losses[name] = total_loss / max(count, 1)
    model.train()
    return losses


def train():
    gpt_config = GPTConfig()
    train_config = TrainConfig()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'Using device: {device}')

    # data
    train_ds, val_ds, tok = get_datasets(block_size=gpt_config.block_size)
    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=False, drop_last=True)

    gpt_config.vocab_size = tok.vocab_size

    # model
    model = GPT(gpt_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} parameters')

    # only apply weight decay to 2d+ params (skip biases, layernorms)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': train_config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=train_config.learning_rate, betas=(0.9, 0.95))

    print(f'Decay params: {sum(p.numel() for p in decay_params):,}')
    print(f'No-decay params: {sum(p.numel() for p in nodecay_params):,}')
    print()

    # training loop
    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()

    for step in range(train_config.max_steps):
        # get batch (cycle through dataset)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # logging
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f'step {step:5d} | loss: {loss.item():.4f} | lr: {lr:.2e} | {elapsed:.1f}s')

        # periodic save
        if step > 0 and step % train_config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader,
                                   train_config.eval_steps, device)
            print(f'  estimate | train: {losses["train"]:.4f} | val: {losses["val"]:.4f}')

    # save final model
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(train_config.checkpoint_dir, 'model.pt')
    torch.save({
        'model': model.state_dict(),
        'config': gpt_config,
        'tokenizer_text': tok.idx_to_char,
    }, ckpt_path)
    print(f'\nSaved checkpoint to {ckpt_path}')

    # quick sample to see how it's doing
    model.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    kv_caches = [None] * gpt_config.n_layer
    with torch.no_grad():
        logits, _, kv_caches = model(prompt, kv_caches=None)
        for _ in range(200):
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat([prompt, next_tok], dim=1)
            logits, _, kv_caches = model(next_tok, kv_caches=kv_caches)
    generated = tok.decode(prompt[0].tolist())
    print(f'\nGenerated sample:\n{generated}')


if __name__ == '__main__':
    train()
