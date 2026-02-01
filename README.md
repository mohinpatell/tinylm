# tinylm

Small GPT style language model, built from scratch. Every transformer component (multi-head attention, positional embeddings, causal masking, feed forward blocks) is implemented using raw PyTorch tensor operations. No `nn.TransformerEncoder` or prebuilt attention modules.

Trained on Shakespeare. It writes decent fake Shakespeare.

## Sample output

```
ROMEO:
Hath pleased liver himself to shride so the king.

ROMEO:
He hath sent to live him to speak.

BENVOLIO:
My lord, I will give him that my know
And think there, being in his love and long
The ambitious trades of war; I must to make
His power comfort and be this heart...
```

```
KING HENRY:
Now will to all his self--

QUEEN ELIZABETH:
So do not that be gone, which is no poison,
Or 'tis more than the king and with man.

KING RICHARD III:
Six With it were he patience and Lord
Should desire the common of good lords...
```

The model picks up character names, dialogue turns, verse like phrasing, and stage directions from ~1MB of training data. ~2.7M parameters, trained for 5K steps on Apple Silicon.

## Architecture

Decoder only transformer (GPT-2 style):

```
Token Embedding + Positional Embedding
  |
  v
[TransformerBlock x 6]
  - LayerNorm -> Multi-Head Causal Self-Attention -> Residual
  - LayerNorm -> Feed-Forward (GELU) -> Residual
  |
  v
LayerNorm -> Linear Head (weight-tied with embedding)
```

Key details:
- Pre norm (LayerNorm before attention/FFN, not after)
- Weight tying between token embedding and output head
- Residual projections scaled by 1/sqrt(2*n_layers) at init
- GELU activation in FFN (not ReLU)
- Learned positional embeddings
- KV cache for O(n) generation instead of O(n^2)

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
pip install torch numpy

# train (takes ~30 min on Apple Silicon MPS, ~90 min CPU)
python train.py

# generate text from checkpoint
python generate.py --prompt "ROMEO:" --tokens 500
```

## Config

Default model (2.7M params):
- `n_embd=192`, `n_head=6`, `n_layer=6`, `block_size=256`
- Character level tokenizer (65 chars)
- AdamW with cosine LR decay + linear warmup

A smaller config is available for quick experiments:

```python
from config import SMALL  # 4 layers, 128d, 128 context
```
