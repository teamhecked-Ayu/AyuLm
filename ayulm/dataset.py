import torch
from torch.utils.data import Dataset


class AyuLMDataset(Dataset):
    """
    Creates next-token prediction examples
    from encoded text.
    """

    def __init__(self, text, tokenizer, context_length=128):if not isinstance(context_length, int) or isinstance(context_length, bool):
    raise ValueError("context_length must be a positive integer")

if context_length <= 0:
    raise ValueError("context_length must be positive")

self.tokens = tokenizer.encode(text)
self.context_length = context_length

        if len(self.tokens) <= context_length:
            raise ValueError(
                "Text must contain more tokens than context_length"
            )

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, index):
        start = index
        end = start + self.context_length

        x = torch.tensor(
            self.tokens[start:end],
            dtype=torch.long
        )

        y = torch.tensor(
            self.tokens[start + 1:end + 1],
            dtype=torch.long
        )

        return x, y
