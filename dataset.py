"""Dataset for next-token prediction.

Takes a long string of text, tokenizes it, and serves up
(input, target) pairs where target is just the input shifted by 1.
"""

import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer, get_shakespeare


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_datasets(block_size=128, train_split=0.9):
    """Load Shakespeare and split into train/val datasets."""
    text = get_shakespeare()
    tok = CharTokenizer(text)
    data = tok.encode(text)

    n = int(len(data) * train_split)
    train_data = data[:n]
    val_data = data[n:]

    train_ds = TextDataset(train_data, block_size)
    val_ds = TextDataset(val_data, block_size)

    return train_ds, val_ds, tok


if __name__ == '__main__':
    train_ds, val_ds, tok = get_datasets(block_size=32)
    print(f'Train samples: {len(train_ds):,}')
    print(f'Val samples: {len(val_ds):,}')

    x, y = train_ds[0]
    print(f'\nSample input:  "{tok.decode(x.tolist())}"')
    print(f'Sample target: "{tok.decode(y.tolist())}"')
    print(f'Shapes: x={x.shape}, y={y.shape}')
