"""Adapter ułatwiający integrację LLM z silnikami strategii."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


class LLMClient(Protocol):
    """Minimalny interfejs klienta modeli językowych."""

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        ...


ToolHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(slots=True)
class PromptTemplate:
    """Reprezentacja promptu wraz z możliwością podmieniania kontekstu."""

    system_prompt: str
    user_template: str
    default_variables: Mapping[str, Any] = field(default_factory=dict)

    def build_messages(self, **variables: Any) -> list[Mapping[str, str]]:
        merged = dict(self.default_variables)
        merged.update({key: value for key, value in variables.items() if value is not None})
        user_prompt = self.user_template.format(**merged)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]


@dataclass(slots=True)
class LLMStrategyAdapter:
    """Adapter zapewniający ujednolnione API LLM i funkcji narzędziowych."""

    client: LLMClient
    model: str
    prompt_template: PromptTemplate
    tools: Mapping[str, ToolHandler] = field(default_factory=dict)
    temperature: float = 0.1

    def __post_init__(self) -> None:
        self.tools = dict(self.tools)
        self.tools.setdefault("sentiment_analysis", self._sentiment_tool)
        self.tools.setdefault("news_interpretation", self._news_tool)

    # ------------------------------------------------------------------ prompt --
    def build_messages(self, **variables: Any) -> list[Mapping[str, str]]:
        """Buduje wiadomości na podstawie szablonu promptu."""

        return self.prompt_template.build_messages(**variables)

    # ------------------------------------------------------------------- tools --
    def list_tools(self) -> list[Mapping[str, Any]]:
        """Zwraca deskryptory funkcji narzędziowych dla klienta LLM."""

        descriptors: list[Mapping[str, Any]] = []
        for name in self.tools:
            descriptors.append(
                {
                    "name": name,
                    "description": {
                        "sentiment_analysis": "Analiza sentymentu wiadomości rynkowych",
                        "news_interpretation": "Interpretacja wpływu newsów na instrument",
                    }.get(name, "Niestandardowa funkcja narzędziowa"),
                }
            )
        return descriptors

    def invoke_tool(self, name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Wykonuje narzędzie lokalnie i zwraca wynik."""

        if name not in self.tools:
            raise KeyError(f"Nieznana funkcja narzędziowa: {name}")
        handler = self.tools[name]
        return handler(payload)

    def _sentiment_tool(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        text = str(payload.get("text", "")).lower()
        positive = sum(text.count(word) for word in ("wzrost", "zysk", "rekord"))
        negative = sum(text.count(word) for word in ("spadek", "strata", "ryzyko"))
        score = positive - negative
        sentiment = "neutral"
        if score > 0:
            sentiment = "bullish"
        elif score < 0:
            sentiment = "bearish"
        return {"score": score, "sentiment": sentiment}

    def _news_tool(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headline = str(payload.get("headline", ""))
        symbol = str(payload.get("symbol", ""))
        impact_words = {"downgrade": "negative", "upgrade": "positive", "lawsuit": "negative"}
        impact = "neutral"
        for word, label in impact_words.items():
            if word in headline.lower():
                impact = label
                break
        summary = f"Nagłówek dotyczący {symbol or 'instrumentu'} oceniony jako {impact}."
        return {"impact": impact, "summary": summary.strip()}

    # --------------------------------------------------------------------- call --
    def generate(self, *, variables: Mapping[str, Any] | None = None, **kwargs: Any) -> Mapping[str, Any]:
        """Generuje odpowiedź LLM z wykorzystaniem zdefiniowanych narzędzi."""

        messages = self.build_messages(**(variables or {}))
        response = self.client.generate(
            messages,
            model=self.model,
            tools=self.list_tools(),
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )
        return response

    # -------------------------------------------------------------- convenience --
    def analyse_sentiment(self, text: str) -> Mapping[str, Any]:
        """Analiza sentymentu bezpośrednio przez lokalne narzędzie."""

        return self.invoke_tool("sentiment_analysis", {"text": text})

    def interpret_news(self, headline: str, symbol: str) -> Mapping[str, Any]:
        """Przygotowuje interpretację wpływu newsów dla strategii."""

        payload = {"headline": headline, "symbol": symbol}
        return self.invoke_tool("news_interpretation", payload)


__all__ = [
    "LLMClient",
    "PromptTemplate",
    "LLMStrategyAdapter",
]
