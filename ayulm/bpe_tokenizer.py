
from pathlib import Path

import sentencepiece as spm


class AyuLMBPETokenizer:
    """SentencePiece BPE tokenizer for AyuLM."""

    def __init__(self, model_path):
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}"
            )

        self.processor = spm.SentencePieceProcessor(
            model_file=str(model_path)
        )

        self.vocab_size = self.processor.get_piece_size()

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        return self.processor.encode(
            text,
            out_type=int
        )

    def decode(self, token_ids):
        if not all(
            isinstance(token, int)
            for token in token_ids
        ):
            raise TypeError("Token IDs must be integers")

        if any(
            token < 0 or token >= self.vocab_size
            for token in token_ids
        ):
            raise ValueError("Invalid token ID")

        return self.processor.decode(token_ids)

    @property
    def unk_id(self):
        return self.processor.unk_id()

    @property
    def bos_id(self):
        return self.processor.bos_id()

    @property
    def eos_id(self):
        return self.processor.eos_id()

    @property
    def pad_id(self):
        return self.processor.pad_id()
          
