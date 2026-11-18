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
