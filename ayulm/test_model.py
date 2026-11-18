
import torch

from ayulm.model import AyuLM, AyuLMConfig, count_parameters


def create_model():
    config = AyuLMConfig(
        vocab_size=256,
        context_length=128,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
    )

    return AyuLM(config)


def test_model_output_shape():
    model = create_model()

    inputs = torch.randint(0, 256, (2, 32))

    outputs = model(inputs)

    assert outputs.shape == (2, 32, 256)


def test_model_parameters_exist():
    model = create_model()

    assert count_parameters(model) > 0


def test_model_supports_backward():
    model = create_model()

    inputs = torch.randint(0, 256, (2, 32))

    outputs = model(inputs)

    loss = outputs.mean()
    loss.backward()

    assert model.token_embedding.weight.grad is not None


def test_model_causal_attention():
    model = create_model()
    model.eval()

    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 9, 9]])

    with torch.no_grad():
        output_a = model(first)
        output_b = model(second)

    # Future tokens must not affect earlier predictions.
    assert torch.allclose(
        output_a[:, :2, :],
        output_b[:, :2, :],
        atol=1e-5,
  )
  
