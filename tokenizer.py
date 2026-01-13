"""Character-level tokenizer.

Simple but effective for a small model on Shakespeare.
Each unique character gets an integer ID.
"""


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices):
        return ''.join(self.idx_to_char[i] for i in indices)


def get_shakespeare():
    """Download Shakespeare text if not already cached."""
    import os
    path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
    if not os.path.exists(path):
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f'Downloading shakespeare to {path}...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with open(path, 'r') as f:
        return f.read()


if __name__ == '__main__':
    text = get_shakespeare()
    print(f'Total characters: {len(text):,}')
    print(f'First 200 chars:\n{text[:200]}')

    tok = CharTokenizer(text)
    print(f'\nVocab size: {tok.vocab_size}')
    print(f'Characters: {"".join(tok.idx_to_char[i] for i in range(tok.vocab_size))}')

    # roundtrip test
    encoded = tok.encode('Hello World')
    decoded = tok.decode(encoded)
    print(f'\n"Hello World" -> {encoded} -> "{decoded}"')
