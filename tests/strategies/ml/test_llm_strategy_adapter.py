from __future__ import annotations

from typing import Any, Mapping

from bot_core.ai.llm_strategy_adapter import LLMStrategyAdapter, PromptTemplate


class DummyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Mapping[str, str]], Mapping[str, Any]]] = []

    def generate(
        self,
        messages: list[Mapping[str, str]],
        *,
        model: str,
        tools: list[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        self.calls.append((messages, {"model": model, "tools": tools or [], **kwargs}))
        return {"content": "stub"}


def test_prompt_template_renders_variables() -> None:
    template = PromptTemplate(
        system_prompt="Zachowuj się jak asystent tradingowy",
        user_template="Oceń {symbol} przy sentymencie {sentiment}",
        default_variables={"sentiment": "neutralny"},
    )
    messages = template.build_messages(symbol="BTC")
    assert messages[0]["role"] == "system"
    assert "BTC" in messages[1]["content"]


def test_llm_adapter_invokes_client_with_tools() -> None:
    client = DummyClient()
    template = PromptTemplate(
        system_prompt="Asystent",
        user_template="Analiza {symbol}",
    )
    adapter = LLMStrategyAdapter(client=client, model="gpt-5", prompt_template=template)
    adapter.generate(variables={"symbol": "ETH"}, temperature=0.2)
    assert client.calls, "Brak wywołań klienta"
    _, call_kwargs = client.calls[0]
    assert call_kwargs["model"] == "gpt-5"
    assert any(tool["name"] == "sentiment_analysis" for tool in call_kwargs["tools"])


def test_local_tools_are_available() -> None:
    template = PromptTemplate(system_prompt="Asystent", user_template=".")
    adapter = LLMStrategyAdapter(client=DummyClient(), model="stub", prompt_template=template)
    sentiment = adapter.analyse_sentiment("Rekordowy wzrost i brak ryzyka")
    assert sentiment["sentiment"] == "bullish"
    news = adapter.interpret_news("Company downgrade due to lawsuit", "XYZ")
    assert news["impact"] == "negative"
