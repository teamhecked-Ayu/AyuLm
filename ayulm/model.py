import torch
import torch.nn as nn


class AyuLMConfig:
    def __init__(
        self,
        vocab_size=256,
        context_length=128,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads


class AyuLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_dim % config.num_heads == 0

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.embedding_dim,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
        )

        self.final_norm = nn.LayerNorm(config.embedding_dim)

        self.output_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError("Sequence exceeds context length")

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.transformer(
            x,
            mask=causal_mask,
            is_causal=True,
        )

        x = self.final_norm(x)

        return self.output_head(x)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
