"""Pakiet adaptera dla rynku spot giełdy Zonda."""

from bot_core.exchanges.zonda.margin import ZondaMarginAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter

__all__ = ["ZondaSpotAdapter", "ZondaMarginAdapter"]
