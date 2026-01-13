# tinylm

Building a GPT-style language model from scratch. After building [nanograd](https://github.com/mohinpatell/nanograd) to understand autograd, I wanted to get the same depth of understanding for transformers.

Using PyTorch for autograd (already proved I understand that part in nanograd), but implementing every transformer component myself: multi-head attention, positional embeddings, causal masking, layer norm placement, all of it. No `nn.TransformerEncoder` or pre-built attention modules.

Training on Shakespeare to start. Goal is to generate passable fake Shakespeare.

## Plan
- Character-level tokenizer (simple, no external deps)
- Decoder-only transformer (GPT-2 style, pre-norm)
- Train on ~1MB of Shakespeare
- Multiple sampling strategies for generation (temperature, top-k, top-p)
