class AyuLMTokenizer:
    """
    Initial byte-level tokenizer for AyuLM.
    Maps UTF-8 bytes to token IDs from 0 to 255.
    """

    vocab_size = 256

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        if any(
            not isinstance(token, int) or token < 0 or token > 255
            for token in token_ids
        ):
            raise ValueError("Invalid token ID")

        return bytes(token_ids).decode(
            "utf-8",
            errors="replace"
        )
