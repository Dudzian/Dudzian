"""Adaptery dla modeli sekwencyjnych stosowanych w tradingu."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SequentialModelAdapter

try:  # pragma: no cover - import opcjonalny
    import torch
    from torch import nn
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:  # pragma: no cover - import opcjonalny
    from pytorch_forecasting.models import TemporalFusionTransformer
except Exception:  # noqa: BLE001
    TemporalFusionTransformer = None  # type: ignore[assignment]


@dataclass(slots=True)
class LSTMTrainer:
    """Minimalna pętla treningowa LSTM obsługiwana przez adapter."""

    model: Any
    optimizer: Any
    loss_fn: Any
    epochs: int = 1

    def fit(self, features: Any, target: Any) -> None:  # pragma: no cover - wymaga torch
        if torch is None:
            raise ImportError("Wymagany jest PyTorch do treningu LSTM")
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, features: Any) -> Any:  # pragma: no cover - wymaga torch
        if torch is None:
            raise ImportError("Wymagany jest PyTorch do inferencji LSTM")
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32)
            return self.model(tensor).cpu().numpy()


class LSTMAdapter(SequentialModelAdapter):
    """Adapter dla modeli LSTM z możliwością dostarczenia własnego trenera."""

    def __init__(self, trainer: LSTMTrainer | None = None):
        if trainer is None:
            if torch is None or nn is None:
                raise ImportError("Wymagany jest PyTorch do konstrukcji domyślnego LSTM")
            model = nn.Sequential(
                nn.LSTM(input_size=1, hidden_size=16, batch_first=True),
                nn.Flatten(),
                nn.Linear(16, 1),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = LSTMTrainer(model=model, optimizer=optimizer, loss_fn=nn.MSELoss(), epochs=5)
        super().__init__(trainer=trainer, name="lstm", hyperparameters=getattr(trainer, "hyperparameters", {}))


class TemporalFusionTransformerAdapter(SequentialModelAdapter):
    """Adapter dla Temporal Fusion Transformer z biblioteki PyTorch Forecasting."""

    def __init__(self, model: Any | None = None, **hyperparameters: Any):
        if model is None:
            if TemporalFusionTransformer is None:
                raise ImportError("Wymagana jest biblioteka pytorch-forecasting")
            model = TemporalFusionTransformer.from_dataset(**hyperparameters)
        super().__init__(trainer=model, name="temporal_fusion_transformer", hyperparameters=hyperparameters)


__all__ = [
    "LSTMTrainer",
    "LSTMAdapter",
    "TemporalFusionTransformerAdapter",
]
