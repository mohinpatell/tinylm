# tinylm

Small GPT-style language model, built from scratch. Every transformer component (multi-head attention, positional embeddings, causal masking, feed-forward blocks) is implemented using raw PyTorch tensor operations. No `nn.TransformerEncoder` or pre-built attention modules.

This is the follow-up to [nanograd](https://github.com/mohinpatell/nanograd), where I built an autograd engine from scratch. Now I wanted to understand transformers at the same level of depth.

Trained on Shakespeare. It writes decent fake Shakespeare.

## Sample output

```
ROMEO:
What is the beauty flatter of my hand:
Shall was good so live the best thing,
Which this weigh to me stand this discalled,
Farewell for what what his sound as spite...
```

```
JULIET:
I am cause.

KING EDWARD IV:
A marcious man and most me the word.

VILLIA:
God will I, am born to my lord,
That you day toogh thee lesser, but of thy beast...
```

Not perfect, but the model picks up character names, dialogue structure, and verse-like phrasing from ~1MB of training data. ~818K parameters, trained for 5K steps on Apple Silicon.

## Architecture

Decoder-only transformer (GPT-2 style):

```
Token Embedding + Positional Embedding
  |
  v
[TransformerBlock x 4]
  - LayerNorm -> Multi-Head Causal Self-Attention -> Residual
  - LayerNorm -> Feed-Forward (GELU) -> Residual
  |
  v
LayerNorm -> Linear Head (weight-tied with embedding)
```

Key details:
- Pre-norm (LayerNorm before attention/FFN, not after)
- Weight tying between token embedding and output head
- Residual projections scaled by 1/sqrt(2*n_layers) at init
- GELU activation in FFN (not ReLU)
- Learned positional embeddings

## Generation

Supports multiple sampling strategies:

```bash
# temperature sampling
python generate.py --prompt "ROMEO:" --temperature 0.8

# top-k sampling (more coherent)
python generate.py --prompt "HAMLET:" --temperature 0.7 --top-k 40

# nucleus (top-p) sampling
python generate.py --prompt "KING:" --temperature 0.8 --top-p 0.9
```

## Training

```bash
pip install torch numpy matplotlib

# train (takes ~5 min on Apple Silicon MPS, ~15 min CPU)
python train.py

# generate text from checkpoint
python generate.py --prompt "ROMEO:" --tokens 500
```

## Config

Default model (818K params):
- `n_embd=128`, `n_head=4`, `n_layer=4`, `block_size=128`
- Character-level tokenizer (65 chars)
- AdamW with cosine LR decay + warmup
