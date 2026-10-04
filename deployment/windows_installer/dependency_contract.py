"""Side-effect-free dependency contract for production Windows executables."""

REQUIRED_WIN32_MODULES = (
    "servicemanager",
    "win32api",
    "win32con",
    "win32event",
    "win32job",
    "win32process",
    "win32security",
    "win32service",
    "win32serviceutil",
    "win32timezone",
)

# PyInstaller cannot infer this importlib.resources dependency from the runtime
# lookup in bot_core.ai.config_loader.  Keep the frozen-build and smoke proof on
# one explicit resource contract.
AI_DEFAULTS_PACKAGE = "bot_core.ai._defaults"
AI_DEFAULTS_RESOURCE = "risk_thresholds.yaml"
AI_DEFAULTS_SMOKE_MARKER = "AI_DEFAULT_RESOURCE = PASS"
