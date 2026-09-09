import pytest
from pydantic import ValidationError

from src.schemas import ChatMessage, ModelConfig


def test_valid_message() -> None:
    message = ChatMessage(role="user", content="Hola")
    assert message.role == "user"
    assert message.content == "Hola"


def test_valid_temperature() -> None:
    config = ModelConfig(model="gpt-4o-mini", temperature=1.5, max_tokens=100)
    assert config.temperature == 1.5


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(model="gpt-4o-mini", temperature=temperature, max_tokens=100)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_invalid_max_tokens(max_tokens: int) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(model="gpt-4o-mini", max_tokens=max_tokens)


@pytest.mark.parametrize("model", ["", "   "])
def test_invalid_model(model: str) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(model=model)
