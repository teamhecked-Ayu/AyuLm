
from ayulm.tokenizer import AyuLMTokenizer
from ayulm.bpe_tokenizer import AyuLMBPETokenizer


def load_tokenizer(settings):
    if not isinstance(settings, dict):
        raise TypeError("Tokenizer settings must be a dictionary")

    tokenizer_type = settings.get("type", "byte")

    if tokenizer_type == "byte":
        return AyuLMTokenizer()

    if tokenizer_type == "bpe":
        model_path = settings.get(
            "model_path",
            "tokenizers/ayulm_bpe.model",
        )

        return AyuLMBPETokenizer(model_path)

    raise ValueError(
        f"Unsupported tokenizer type: {tokenizer_type}"
      )
  
