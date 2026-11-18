
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
    runtime_settings = settings.get("runtime", {})
    checkpoint_settings = settings.get("checkpoint", {})

    seed = settings.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    device_name = runtime_settings.get("device", "auto")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

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

    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    text = Path(data_path).read_text(encoding="utf-8")

    split_index = int(len(text) * 0.9)
    train_text = text[:split_index]
    val_text = text[split_index:]

    train_dataset = AyuLMDataset(
        train_text, tokenizer, config.context_length
    )

    val_dataset = AyuLMDataset(
        val_text, tokenizer, config.context_length
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    batch_size = training_settings["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_settings["learning_rate"],
    )

    loss_function = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)

                loss = loss_function(
                    logits.reshape(-1, tokenizer.vocab_size),
                    targets.reshape(-1),
                )

                total_loss += loss.item()
                batches += 1

        model.train()
        return total_loss / batches

    model.train()
    step = 0

    while step < max_steps:
        for inputs, targets in train_loader:
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
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            step += 1

            if step == 1 or step % 10 == 0:
                val_loss = evaluate()

                print(
                    f"Step {step}/{max_steps} "
                    f"Train Loss: {loss.item():.4f} "
                    f"Validation Loss: {val_loss:.4f}"
                )

            if step >= max_steps:
                break

    if checkpoint_settings.get("save_enabled", True):
        checkpoint_dir = Path(
            checkpoint_settings.get("directory", "checkpoints")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "ayulm_tiny.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(config),
                "step": step,
                "seed": seed,
            },
            checkpoint_path,
        )

        print("Checkpoint saved:", checkpoint_path)

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--steps", type=int, default=100)

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    train(args.config, args.data, args.steps)
