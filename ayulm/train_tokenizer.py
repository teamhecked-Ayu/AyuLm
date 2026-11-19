
import argparse
from pathlib import Path

import sentencepiece as spm


def train_tokenizer(
    input_path,
    output_prefix,
    vocab_size=512,
):
    input_path = Path(input_path)
    output_prefix = Path(output_prefix)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training corpus not found: {input_path}"
        )

    if vocab_size < 300:
        raise ValueError(
            "Vocabulary size must be at least 300 "
            "for byte fallback."
        )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        hard_vocab_limit=False,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    print("Tokenizer training completed.")
    print("Model:", f"{output_prefix}.model")
    print("Vocabulary:", f"{output_prefix}.vocab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="tokenizers/ayulm_bpe",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    train_tokenizer(
        args.input,
        args.output,
        args.vocab_size,
  )
  
