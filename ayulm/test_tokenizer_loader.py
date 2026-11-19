
from pathlib import Path

import pytest

from ayulm.tokenizer import AyuLMTokenizer
from ayulm.bpe_tokenizer import AyuLMBPETokenizer
from ayulm.tokenizer_loader import load_tokenizer


def test_load_byte_tokenizer():
    tokenizer = load_tokenizer({"type": "byte"})

    assert isinstance(tokenizer, AyuLMTokenizer)
    assert tokenizer.vocab_size == 256


def test_load_bpe_tokenizer():
    model_path = Path("tokenizers/ayulm_bpe.model")

    if not model_path.exists():
        pytest.skip("BPE model unavailable")

    tokenizer = load_tokenizer({
        "type": "bpe",
        "model_path": str(model_path),
    })

    assert isinstance(tokenizer, AyuLMBPETokenizer)
    assert tokenizer.vocab_size == 512


def test_default_tokenizer_is_byte():
    tokenizer = load_tokenizer({})

    assert isinstance(tokenizer, AyuLMTokenizer)


def test_unsupported_tokenizer():
    with pytest.raises(ValueError):
        load_tokenizer({"type": "unknown"})


def test_invalid_settings():
    with pytest.raises(TypeError):
        load_tokenizer("byte")

def test_empty_bpe_model_path():
    with pytest.raises(ValueError):
        load_tokenizer({
            "type": "bpe",
            "model_path": "",
        })


def test_non_string_tokenizer_type():
    with pytest.raises(ValueError):
        load_tokenizer({
            "type": 123,
        })


def test_none_tokenizer_type():
    with pytest.raises(ValueError):
        load_tokenizer({
            "type": None,
        })
