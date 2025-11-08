"""Narzędzia raportowania podatkowego.

Pakiet udostępnia modele danych oraz generator raportów wykorzystujący wpisy
z dziennika zleceń i lokalne magazyny transakcji.
"""

from .models import (
    AssetBreakdown,
    DisposalEvent,
    MatchedLot,
    PeriodBreakdown,
    TaxLot,
    TaxReport,
    TaxReportTotals,
)
from .fx import FXRateProvider, StaticFXRateProvider
from .generator import TaxReportGenerator
from .calculators import (
    AverageCostBasisCalculator,
    CostBasisCalculator,
    FIFOCostBasisCalculator,
    LIFOCostBasisCalculator,
)

__all__ = [
    "DisposalEvent",
    "MatchedLot",
    "TaxLot",
    "TaxReport",
    "TaxReportTotals",
    "AssetBreakdown",
    "PeriodBreakdown",
    "TaxReportGenerator",
    "FXRateProvider",
    "StaticFXRateProvider",
    "CostBasisCalculator",
    "FIFOCostBasisCalculator",
    "LIFOCostBasisCalculator",
    "AverageCostBasisCalculator",
]
