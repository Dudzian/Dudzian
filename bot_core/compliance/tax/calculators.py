"""Implementacje metod wyceny kosztu uzyskania przychodu."""

from __future__ import annotations

from collections import deque
from typing import Deque, List

from .models import DisposalEvent, MatchedLot, TaxLot


class CostBasisCalculator:
    """Bazowa klasa implementująca logikę gromadzenia partii."""

    def __init__(self) -> None:
        self._lots: List[TaxLot] = []

    def add_lot(self, lot: TaxLot) -> None:
        self._lots.append(lot)

    def dispose(self, event: DisposalEvent) -> DisposalEvent:
        raise NotImplementedError

    def remaining_lots(self) -> List[TaxLot]:
        leftovers: List[TaxLot] = []
        for lot in self._lots:
            remaining = lot.remaining_lot()
            if remaining is not None:
                leftovers.append(remaining)
        return leftovers


class FIFOCostBasisCalculator(CostBasisCalculator):
    """Implementacja metody FIFO."""

    def __init__(self) -> None:
        super().__init__()
        self._queue: Deque[TaxLot] = deque()

    def add_lot(self, lot: TaxLot) -> None:
        super().add_lot(lot)
        self._queue.append(lot)

    def dispose(self, event: DisposalEvent) -> DisposalEvent:
        quantity = event.quantity
        if quantity <= 0:
            raise ValueError("Disposal quantity must be positive")
        matched: List[MatchedLot] = []
        while quantity > 1e-12:
            if not self._queue:
                raise ValueError("Brak partii do rozliczenia (FIFO)")
            lot = self._queue[0]
            available = lot.remaining_quantity
            portion = min(available, quantity)
            matched_lot = lot.consume(portion, disposal_time=event.disposal_time)
            matched.append(matched_lot)
            quantity -= portion
            if lot.remaining_quantity <= 1e-12:
                self._queue.popleft()
        event.matched_lots = matched
        return event


class LIFOCostBasisCalculator(CostBasisCalculator):
    """Implementacja metody LIFO."""

    def __init__(self) -> None:
        super().__init__()

    def dispose(self, event: DisposalEvent) -> DisposalEvent:
        quantity = event.quantity
        if quantity <= 0:
            raise ValueError("Disposal quantity must be positive")
        matched: List[MatchedLot] = []
        while quantity > 1e-12:
            if not self._lots:
                raise ValueError("Brak partii do rozliczenia (LIFO)")
            lot = self._lots[-1]
            available = lot.remaining_quantity
            portion = min(available, quantity)
            matched_lot = lot.consume(portion, disposal_time=event.disposal_time)
            matched.append(matched_lot)
            quantity -= portion
            if lot.remaining_quantity <= 1e-12:
                self._lots.pop()
        event.matched_lots = matched
        return event


class AverageCostBasisCalculator(CostBasisCalculator):
    """Implementacja metody średniego kosztu."""

    def __init__(self) -> None:
        super().__init__()
        self._total_quantity = 0.0
        self._total_cost = 0.0
        self._total_fee = 0.0
        self._asset: str | None = None

    def add_lot(self, lot: TaxLot) -> None:
        super().add_lot(lot)
        self._total_quantity += lot.quantity
        self._total_cost += lot.cost_basis
        self._total_fee += lot.fee
        if not self._asset:
            self._asset = lot.asset

    def dispose(self, event: DisposalEvent) -> DisposalEvent:
        quantity = event.quantity
        if quantity <= 0:
            raise ValueError("Disposal quantity must be positive")
        if self._total_quantity + 1e-12 < quantity:
            raise ValueError("Brak partii do rozliczenia (średni koszt)")
        if self._total_quantity <= 0:
            raise ValueError("Brak zgromadzonych partii dla średniego kosztu")
        cost_per_unit = (self._total_cost + self._total_fee) / self._total_quantity
        cost = cost_per_unit * quantity
        acquisition_time = self._lots[0].acquisition_time if self._lots else event.disposal_time
        holding_days = 0.0
        if event.disposal_time and acquisition_time:
            delta = event.disposal_time - acquisition_time
            holding_days = max(0.0, delta.total_seconds() / 86400.0)
        matched = MatchedLot(
            lot_id="average",
            acquisition_time=acquisition_time,
            quantity=quantity,
            cost_basis=cost,
            fee=0.0,
            venue=None,
            source="average-cost",
            holding_period_days=holding_days,
        )
        self._total_quantity -= quantity
        total_basis = self._total_cost + self._total_fee
        remaining_basis = max(0.0, total_basis - cost)
        if self._total_quantity <= 1e-9:
            self._total_quantity = 0.0
            self._total_cost = 0.0
            self._total_fee = 0.0
        else:
            ratio = 0.0
            denominator = self._total_cost + self._total_fee
            if denominator > 0:
                ratio = self._total_cost / denominator
            self._total_cost = remaining_basis * ratio
            self._total_fee = remaining_basis * (1 - ratio)
        event.matched_lots = [matched]
        return event

    def remaining_lots(self) -> List[TaxLot]:
        if self._total_quantity <= 1e-12:
            return []
        unit_cost = self._total_cost / self._total_quantity if self._total_quantity else 0.0
        unit_fee = self._total_fee / self._total_quantity if self._total_quantity else 0.0
        lot = TaxLot(
            lot_id="average",
            asset=self._asset or "*",
            acquisition_time=self._lots[0].acquisition_time if self._lots else event_time_placeholder(),
            quantity=self._total_quantity,
            cost_basis=unit_cost * self._total_quantity,
            fee=unit_fee * self._total_quantity,
            venue=None,
            source="average-cost",
        )
        return [lot]


def event_time_placeholder():
    from datetime import datetime, timezone

    return datetime.fromtimestamp(0, timezone.utc)
