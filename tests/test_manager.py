import pytest

from src.manager import AsyncLLMManager


def test_manager_selects_openai() -> None:
    manager = AsyncLLMManager("openai")
    assert manager.client.provider_name == "openai"


def test_manager_selects_anthropic() -> None:
    manager = AsyncLLMManager("anthropic")
    assert manager.client.provider_name == "anthropic"


def test_manager_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="openai.*anthropic"):
        AsyncLLMManager("otro")
