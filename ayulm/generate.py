
import argparse
import torch

from ayulm.model import AyuLM, AyuLMConfig
from ayulm.tokenizer import AyuLMTokenizer


def generate_text(
    checkpoint_path,
    prompt,
    max_new_tokens=100,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    config = AyuLMConfig(**checkpoint["config"])

    model = AyuLM(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    tokenizer = AyuLMTokenizer()

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("Prompt cannot be empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(max_new_tokens):

            context = tokens[
                :, -config.context_length:
            ]

            logits = model(context)

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

    return tokenizer.decode(
        tokens[0].tolist()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ayulm_tiny.pt",
    )

    parser.add_argument(
        "--prompt",
        default="Hello",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    print(
        generate_text(
            args.checkpoint,
            args.prompt,
            args.max_new_tokens,
        )
    )
