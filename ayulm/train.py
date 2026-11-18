import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ayulm.model import AyuLM, AyuLMConfig
from ayulm.tokenizer import AyuLMTokenizer
from ayulm.dataset import AyuLMDataset


def train(config_path, data_path, max_steps):
    with open(config_path, "r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    model_settings = settings["model"]
    training_settings = settings["training"]

    device_name = training_settings.get("device", "auto")

    if device_name == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)
    print("Training device:", device)

    tokenizer = AyuLMTokenizer()

    config = AyuLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=model_settings["context_length"],
        embedding_dim=model_settings["embedding_dim"],
        num_layers=model_settings["num_layers"],
        num_heads=model_settings["num_heads"],
    )

    model = AyuLM(config).to(device)

    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    text = Path(data_path).read_text(encoding="utf-8")

    dataset = AyuLMDataset(
        text,
        tokenizer,
        context_length=config.context_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=training_settings["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_settings["learning_rate"],
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()
    step = 0

    while step < max_steps:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)

            loss = loss_function(
                logits.reshape(-1, tokenizer.vocab_size),
                targets.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            step += 1

            if step % 10 == 0 or step == 1:
                print(
                    f"Step {step}/{max_steps} "
                    f"Loss: {loss.item():.4f}"
                )

            if step >= max_steps:
                break

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "ayulm_tiny.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
            "step": step,
        },
        checkpoint_path,
    )

    print("Training completed.")
    print("Checkpoint saved:", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/tiny.yaml",
    )

    parser.add_argument(
        "--data",
        default="data/train.txt",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    random.seed(42)
    torch.manual_seed(42)

    train(args.config, args.data, args.steps)
