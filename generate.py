"""Text generation."""

import argparse
import torch

from model import GPT
from tokenizer import CharTokenizer


def load_model(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = GPT(config).to(device)
    model.load_state_dict(ckpt['model'])

    # rebuild tokenizer from saved char map
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.idx_to_char = ckpt['tokenizer_text']
    tok.char_to_idx = {c: i for i, c in tok.idx_to_char.items()}
    tok.vocab_size = len(tok.idx_to_char)

    return model, tok, config


@torch.no_grad()
def generate(model, tok, config, prompt='', max_new_tokens=500,
             temperature=0.8, top_k=None, top_p=None, device='cpu'):
    """Generate text token by token with optional top-k/top-p filtering."""
    if prompt:
        tokens = tok.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    all_tokens = idx
    kv_caches = None

    for _ in range(max_new_tokens):
        if kv_caches is not None:
            input_ids = all_tokens[:, -1:]
        else:
            input_ids = all_tokens[:, -config.block_size:]

        logits, _, kv_caches = model(input_ids, kv_caches=kv_caches)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # drop cache if we've hit the context limit
        if kv_caches[0][0].shape[2] >= config.block_size:
            kv_caches = None

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        all_tokens = torch.cat([all_tokens, next_tok], dim=1)

    return tok.decode(all_tokens[0].tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.pt')
    parser.add_argument('--prompt', type=str, default='ROMEO:\n')
    parser.add_argument('--tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    args = parser.parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

    model, tok, config = load_model(args.checkpoint, device)
    # set to inference mode (disables dropout)
    model.eval()
    print(f'Loaded model ({sum(p.numel() for p in model.parameters()):,} params)')
    print(f'Generating with temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}')
    print('---')

    text = generate(model, tok, config, prompt=args.prompt,
                    max_new_tokens=args.tokens, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p, device=device)
    print(text)
