
import pytest

from ayulm.dataset import AyuLMDataset
from ayulm.tokenizer import AyuLMTokenizer


def test_dataset_creates_next_token_pairs():
    tokenizer = AyuLMTokenizer()

    text = "AyuLM is learning language modeling. " * 30

    dataset = AyuLMDataset(
        text,
        tokenizer,
        context_length=8,
    )

    x, y = dataset[0]

    assert len(x) == 8
    assert len(y) == 8
    assert x.tolist()[1:] == y.tolist()[:-1]


def test_dataset_rejects_short_text():
    tokenizer = AyuLMTokenizer()

    with pytest.raises(ValueError):
        AyuLMDataset(
            "short",
            tokenizer,
            context_length=128,
        )


def test_dataset_rejects_invalid_context_length():
    tokenizer = AyuLMTokenizer()

    with pytest.raises(ValueError):
        AyuLMDataset(
            "AyuLM " * 30,
            tokenizer,
            context_length=0,
      )
      
