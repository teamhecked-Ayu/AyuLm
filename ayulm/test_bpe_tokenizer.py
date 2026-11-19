
from pathlib import Path

import pytest

from ayulm.bpe_tokenizer import AyuLMBPETokenizer


@pytest.fixture
def tokenizer():
    model_path = Path("ayulm_tokenizer_v3.model")

    if not model_path.exists():
        pytest.skip("Experimental tokenizer model not available")

    return AyuLMBPETokenizer(model_path)


@pytest.mark.parametrize(
    "text",
    [
        "AyuLM is learning.",
        "નમસ્તે, ગુજરાતી ભાષા.",
        "नमस्ते, हिंदी भाषा।",
        "Ciao, intelligenza artificiale.",
        "Unknown text: विज्ञान and technology!",
        "123456789 ! ? @ # $ %",
    ],
)
def test_encode_decode_round_trip(tokenizer, text):
    token_ids = tokenizer.encode(text)

    assert all(
        0 <= token_id < tokenizer.vocab_size
        for token_id in token_ids
    )

    assert tokenizer.decode(token_ids) == text


def test_special_token_ids(tokenizer):
    assert tokenizer.unk_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.pad_id == 3


def test_invalid_input(tokenizer):
    with pytest.raises(TypeError):
        tokenizer.encode(123)

    with pytest.raises(ValueError):
        tokenizer.decode([tokenizer.vocab_size])


def test_missing_model():
    with pytest.raises(FileNotFoundError):
        AyuLMBPETokenizer("missing_tokenizer.model")
  
