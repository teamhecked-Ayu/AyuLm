
from pathlib import Path

import pytest

from ayulm.bpe_tokenizer import AyuLMBPETokenizer
from ayulm.dataset import AyuLMDataset


@pytest.fixture
def tokenizer():
    model_path = Path("tokenizers/ayulm_bpe.model")

    if not model_path.exists():
        pytest.skip("BPE tokenizer model not available")

    return AyuLMBPETokenizer(model_path)


def test_bpe_dataset_creates_next_token_pairs(tokenizer):
    text = (
        "AyuLM is learning language modeling. "
        "નમસ્તે ગુજરાતી. "
        "नमस्ते हिंदी. "
        "Ciao mondo. "
    ) * 30

    dataset = AyuLMDataset(
        text,
        tokenizer,
        context_length=8,
    )

    x, y = dataset[0]

    assert len(x) == 8
    assert len(y) == 8

    assert x.tolist()[1:] == y.tolist()[:-1]

    assert all(
        0 <= token_id < tokenizer.vocab_size
        for token_id in x.tolist() + y.tolist()
    )


def test_bpe_dataset_length(tokenizer):
    text = "AyuLM multilingual research. " * 30

    dataset = AyuLMDataset(
        text,
        tokenizer,
        context_length=8,
    )

    assert len(dataset) > 0
  
