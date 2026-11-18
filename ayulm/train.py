
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

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

    device_name = runtime_settings.get("device", "auto")

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(device_name)
    print("Training device:", device)

    seed = settings.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

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

    dataset = AyuLMDataset(
        text,
        tokenizer,
        context_length=config.context_length,
    )

    if len(dataset) < 2:
        raise ValueError("Dataset is too small for training and validation")

    validation_size = max(1, int(len(dataset) * 0.1))
    training_size = len(dataset) - validation_size

    train_dataset, val_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

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

    checkpoint_dir = Path(
        checkpoint_settings.get("directory", "checkpoints")
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    precision = runtime_settings.get("precision", "float32")
    if precision != "float32":
        raise ValueError("This initial training engine supports float32 only")

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
        return total_loss / max(batches, 1)

    model.train()
    step = 0
    max_epochs = training_settings.get("epochs", 1)

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
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            step += 1

            if step % 10 == 0 or step == 1:
                val_loss = evaluate()

                print(
                    f"Step {step}/{max_steps} "
                    f"Train Loss: {loss.item():.4f} "
                    f"Validation Loss: {val_loss:.4f}"
                )

            if step >= max_steps:
                break

        if step >= max_steps:
            break

        max_epochs -= 1
        if max_epochs <= 0:
            break

    if checkpoint_settings.get("save_enabled", True):
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
