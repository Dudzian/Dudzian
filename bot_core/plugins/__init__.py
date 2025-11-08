"""API pluginów strategii dla dostawców marketplace."""

from .loader import PluginLoadError, StrategyPluginLoader
from .manifest import PluginAuthor, PluginSignature, SignedStrategyPlugin, StrategyPluginManifest
from .signing import PluginSigner, PluginVerifier
from .review import PluginReviewBoard, PluginReviewFinding, PluginReviewResult, ReviewStatus
from .registry import PluginRegistryError, RegisteredPlugin, StrategyPluginRegistry
from .io import dump_manifest, dump_package, load_manifest, load_package

__all__ = [
    "PluginAuthor",
    "PluginSignature",
    "SignedStrategyPlugin",
    "StrategyPluginManifest",
    "PluginSigner",
    "PluginVerifier",
    "PluginReviewBoard",
    "PluginReviewFinding",
    "PluginReviewResult",
    "ReviewStatus",
    "StrategyPluginLoader",
    "PluginLoadError",
    "StrategyPluginRegistry",
    "RegisteredPlugin",
    "PluginRegistryError",
    "load_manifest",
    "dump_manifest",
    "load_package",
    "dump_package",
]

