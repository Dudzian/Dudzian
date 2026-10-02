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
