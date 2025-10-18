"""Infrastruktura runtime nowej architektury bota."""

import json
import logging
import logging.config
import os
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module, reload
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, cast, runtime_checkable


PathInput = str | bytes | PathLike[str] | PathLike[bytes]


@runtime_checkable
class _LoggerLike(Protocol):
    def log(self, level: int, msg: str, *args: object, **kwargs: object) -> None:
        """Minimalny protokół loggera używany przez helpery logujące."""

_LOGGER = logging.getLogger(__name__)
_OPTIONAL_EXPORT_LOGGER: _LoggerLike = cast(_LoggerLike, _LOGGER)


@dataclass(slots=True)
class OptionalExportsLoggingSpec:
    """Znormalizowana specyfikacja konfiguracji logowania helperów lazy-eksportów."""

    kind: Literal["json", "file", "python"]
    value: str
    attribute: str | None = None
    origin: Literal["prefixed", "inline", "bare"] = "prefixed"

    def __post_init__(self) -> None:
        if self.kind not in ("json", "file", "python"):
            raise ValueError("kind must be one of: 'json', 'file', 'python'")

        if not isinstance(self.value, str):
            raise TypeError("value must be a string")

        if not self.value:
            raise ValueError("value cannot be empty")

        if self.origin not in ("prefixed", "inline", "bare"):
            raise ValueError("origin must be one of: 'prefixed', 'inline', 'bare'")

        if self.kind == "python":
            if self.attribute is not None and not isinstance(self.attribute, str):
                raise TypeError("attribute must be a string when provided")
        elif self.attribute is not None:
            raise ValueError("attribute is only supported for python logging specs")


def parse_optional_exports_logging_spec(
    spec: str,
    *,
    source_description: str = "logging specification",
) -> OptionalExportsLoggingSpec:
    """Znormalizuj specyfikację logowania helperów lazy-eksportów."""

    if not isinstance(spec, str):
        raise TypeError("spec must be a string")

    if not isinstance(source_description, str):
        raise TypeError("source_description must be a string")

    normalized = spec.strip()

    if not normalized:
        raise ValueError(f"{source_description} cannot be empty")

    lowered = normalized.lower()

    if lowered.startswith(("json:", "dict:")):
        prefix, remainder = normalized.split(":", 1)
        payload = remainder.lstrip()

        if not payload:
            raise ValueError(
                f"{source_description} must provide JSON payload after '{prefix}:'"
            )

        return OptionalExportsLoggingSpec(
            kind="json",
            value=payload,
            origin="prefixed",
        )

    if lowered.startswith(("file:", "ini:")):
        prefix, remainder = normalized.split(":", 1)
        path_spec = remainder.strip()

        if not path_spec:
            raise ValueError(
                f"{source_description} must provide file path after '{prefix}:'"
            )

        return OptionalExportsLoggingSpec(
            kind="file",
            value=path_spec,
            origin="prefixed",
        )

    if lowered.startswith("python:"):
        prefix, remainder = normalized.split(":", 1)
        target_spec = remainder.strip()

        if not target_spec:
            raise ValueError(
                f"{source_description} must provide python target after '{prefix}:'"
            )

        module_spec = target_spec
        attribute_spec: str | None = None

        if ":" in target_spec:
            module_part, attribute_part = target_spec.split(":", 1)
            module_spec = module_part.strip()
            attribute_spec = attribute_part.strip() or None
        else:
            module_spec = module_spec.strip()

        if not module_spec:
            raise ValueError(
                f"{source_description} must include module name in python specification"
            )

        return OptionalExportsLoggingSpec(
            kind="python",
            value=module_spec,
            attribute=attribute_spec,
            origin="prefixed",
        )

    if normalized.startswith("{") or normalized.startswith("["):
        return OptionalExportsLoggingSpec(
            kind="json",
            value=normalized,
            origin="inline",
        )

    if ":" in normalized:
        possible_prefix, _ = normalized.split(":", 1)
        if possible_prefix.isalpha() and len(possible_prefix) > 1:
            raise ValueError(
                f"{source_description} uses unsupported prefix '{possible_prefix}'"
            )

    return OptionalExportsLoggingSpec(
        kind="file",
        value=normalized,
        origin="bare",
    )


def get_optional_exports_logger() -> _LoggerLike:
    """Zwróć domyślny logger używany przez helpery logujące lazy-eksporty."""

    return _OPTIONAL_EXPORT_LOGGER


def _configure_optional_exports_logging_from_json_text(
    text: str,
    *,
    logger_name: str | None,
    set_as_default: bool,
    invalid_error_template: str | None = None,
    non_mapping_error_template: str | None = None,
) -> logging.Logger:
    """Wczytaj konfigurację logowania z tekstu JSON."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        template = invalid_error_template or "invalid JSON logging config: {exc}"
        raise ValueError(template.format(exc=exc)) from exc

    if not isinstance(payload, Mapping):
        template = (
            non_mapping_error_template
            or "JSON logging config must decode to a mapping-compatible object"
        )
        raise ValueError(template)

    return configure_optional_exports_logging_from_dict(
        payload,
        logger_name=logger_name,
        set_as_default=set_as_default,
    )


def _configure_optional_exports_logging_from_file_path(
    path_spec: str,
    *,
    logger_name: str | None,
    set_as_default: bool,
    defaults: Mapping[str, Any] | None,
    disable_existing_loggers: bool | None,
    encoding: str,
    empty_error: str | None = None,
    missing_error_template: str | None = None,
    invalid_json_template: str | None = None,
    non_mapping_json_template: str | None = None,
) -> logging.Logger:
    """Wczytaj konfigurację logowania ze ścieżki plikowej."""

    if not path_spec:
        raise ValueError(empty_error or "file specification cannot be empty")

    target_path = Path(path_spec).expanduser()

    if not target_path.exists():
        template = (
            missing_error_template
            or "logging configuration file '{path}' does not exist"
        )
        raise FileNotFoundError(template.format(path=target_path))

    if target_path.suffix.lower() == ".json":
        text = target_path.read_text(encoding=encoding)
        return _configure_optional_exports_logging_from_json_text(
            text,
            logger_name=logger_name,
            set_as_default=set_as_default,
            invalid_error_template=invalid_json_template,
            non_mapping_error_template=non_mapping_json_template,
        )

    kwargs: dict[str, Any] = {}

    if defaults is not None:
        kwargs["defaults"] = defaults

    if disable_existing_loggers is not None:
        kwargs["disable_existing_loggers"] = disable_existing_loggers

    return configure_optional_exports_logging_from_file(
        target_path,
        logger_name=logger_name,
        set_as_default=set_as_default,
        **kwargs,
    )


def _materialize_logging_config(value: Any) -> Any:
    """Zamień zagnieżdżone mapowania na zwykłe słowniki dla konfiguracji logowania."""

    if isinstance(value, Mapping):
        return {key: _materialize_logging_config(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return [_materialize_logging_config(item) for item in value]

    if isinstance(value, list):
        return [_materialize_logging_config(item) for item in value]

    return value


def set_optional_exports_logger(logger: _LoggerLike | None) -> None:
    """Ustaw domyślny logger wykorzystywany przez helpery logujące runtime."""

    global _OPTIONAL_EXPORT_LOGGER

    if logger is None:
        _OPTIONAL_EXPORT_LOGGER = cast(_LoggerLike, _LOGGER)
        return

    if not isinstance(logger, _LoggerLike):
        raise TypeError("logger must provide a log(level, message, *args, **kwargs) method")

    _OPTIONAL_EXPORT_LOGGER = logger


@contextmanager
def temporary_optional_exports_logger(logger: _LoggerLike | None):
    """Tymczasowo nadpisz logger helperów logujących lazy-eksporty."""

    previous = get_optional_exports_logger()
    set_optional_exports_logger(logger)
    try:
        yield
    finally:
        set_optional_exports_logger(previous)


def configure_optional_exports_logging(
    *,
    logger: logging.Logger | None = None,
    logger_name: str | None = None,
    level: int | str = logging.INFO,
    handler: logging.Handler | None = None,
    formatter: logging.Formatter | None = None,
    ensure_handler: bool = False,
    clear_handlers: bool = False,
    propagate: bool | None = None,
    set_as_default: bool = True,
) -> logging.Logger:
    """Skonfiguruj logger używany przez helpery logujące lazy-eksporty runtime.

    Funkcja pozwala przygotować dedykowany logger (z opcjonalnymi handlerami,
    formatterami i ustawieniami propagacji), a następnie – domyślnie – ustawić
    go jako źródło logów dla helperów ``log_optional_*``. Można przekazać
    istniejący obiekt :class:`logging.Logger` lub nazwę loggera do
    zainicjalizowania.
    """

    if logger is not None and not isinstance(logger, logging.Logger):
        raise TypeError("logger must be an instance of logging.Logger")

    if logger_name is not None and not isinstance(logger_name, str):
        raise TypeError("logger_name must be a string")

    if handler is not None and not isinstance(handler, logging.Handler):
        raise TypeError("handler must be an instance of logging.Handler")

    if formatter is not None and not isinstance(formatter, logging.Formatter):
        raise TypeError("formatter must be an instance of logging.Formatter")

    if propagate is not None and not isinstance(propagate, bool):
        raise TypeError("propagate must be a boolean")

    target_logger = logger or logging.getLogger(
        logger_name or f"{__name__}.optional_exports"
    )

    if clear_handlers:
        target_logger.handlers.clear()

    handler_to_add: logging.Handler | None = handler

    if ensure_handler and handler_to_add is None and not target_logger.handlers:
        handler_to_add = logging.StreamHandler()

    if handler_to_add is not None and handler_to_add not in target_logger.handlers:
        target_logger.addHandler(handler_to_add)

    if formatter is not None:
        targets = (
            [handler_to_add]
            if handler_to_add is not None
            else list(target_logger.handlers)
        )
        if not targets:
            raise ValueError(
                "formatter provided but no handlers are configured; "
                "add a handler or use ensure_handler=True"
            )
        for configured_handler in targets:
            if configured_handler is not None:
                configured_handler.setFormatter(formatter)

    target_logger.setLevel(_coerce_log_level(level))

    if propagate is not None:
        target_logger.propagate = propagate

    if set_as_default:
        set_optional_exports_logger(target_logger)

    return target_logger


def configure_optional_exports_logging_from_parsed_spec(
    parsed_spec: OptionalExportsLoggingSpec,
    *,
    set_as_default: bool = True,
    logger_name: str | None = None,
    defaults: Mapping[str, Any] | None = None,
    disable_existing_loggers: bool | None = None,
    encoding: str = "utf-8",
    source_description: str = "logging specification",
) -> logging.Logger:
    """Skonfiguruj logowanie helperów lazy-eksportów na podstawie znormalizowanej specyfikacji."""

    if not isinstance(parsed_spec, OptionalExportsLoggingSpec):
        raise TypeError("parsed_spec must be an OptionalExportsLoggingSpec instance")

    if not isinstance(source_description, str):
        raise TypeError("source_description must be a string")

    if parsed_spec.kind == "json":
        return _configure_optional_exports_logging_from_json_text(
            parsed_spec.value,
            logger_name=logger_name,
            set_as_default=set_as_default,
            invalid_error_template=(
                f"invalid JSON logging config in {source_description}: {{exc}}"
            ),
        )

    if parsed_spec.kind == "file":
        kwargs = {
            "logger_name": logger_name,
            "set_as_default": set_as_default,
            "defaults": defaults,
            "disable_existing_loggers": disable_existing_loggers,
            "encoding": encoding,
            "missing_error_template": (
                "logging configuration file '{path}' referenced by "
                f"{source_description} does not exist"
            ),
            "invalid_json_template": (
                f"invalid JSON logging config in {source_description}: {{exc}}"
            ),
        }

        if parsed_spec.origin == "prefixed":
            kwargs["empty_error"] = (
                f"{source_description} must provide a non-empty file specification"
            )

        return _configure_optional_exports_logging_from_file_path(
            parsed_spec.value,
            **kwargs,
        )

    return configure_optional_exports_logging_from_python(
        parsed_spec.value,
        attribute=parsed_spec.attribute,
        logger_name=logger_name,
        set_as_default=set_as_default,
        defaults=defaults,
        disable_existing_loggers=disable_existing_loggers,
        encoding=encoding,
    )


def configure_optional_exports_logging_from_spec(
    spec: str,
    *,
    set_as_default: bool = True,
    logger_name: str | None = None,
    defaults: Mapping[str, Any] | None = None,
    disable_existing_loggers: bool | None = None,
    encoding: str = "utf-8",
    source_description: str = "logging specification",
) -> logging.Logger:
    """Skonfiguruj logowanie helperów lazy-eksportów na podstawie specyfikacji tekstowej."""

    parsed_spec = parse_optional_exports_logging_spec(
        spec,
        source_description=source_description,
    )

    return configure_optional_exports_logging_from_parsed_spec(
        parsed_spec,
        set_as_default=set_as_default,
        logger_name=logger_name,
        defaults=defaults,
        disable_existing_loggers=disable_existing_loggers,
        encoding=encoding,
        source_description=source_description,
    )



def configure_optional_exports_logging_from_dict(
    config: Mapping[str, Any],
    *,
    logger_name: str | None = None,
    set_as_default: bool = True,
) -> logging.Logger:
    """Skonfiguruj logowanie helperów lazy-eksportów na podstawie słownika."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping compatible with logging.dictConfig")

    if logger_name is not None and not isinstance(logger_name, str):
        raise TypeError("logger_name must be a string")

    materialized = _materialize_logging_config(config)
    logging.config.dictConfig(materialized)

    target_logger = logging.getLogger(
        logger_name or f"{__name__}.optional_exports"
    )

    if set_as_default:
        set_optional_exports_logger(target_logger)

    return target_logger


def configure_optional_exports_logging_from_file(
    path: PathInput,
    *,
    defaults: Mapping[str, Any] | None = None,
    disable_existing_loggers: bool | None = None,
    logger_name: str | None = None,
    set_as_default: bool = True,
) -> logging.Logger:
    """Skonfiguruj logowanie helperów lazy-eksportów na podstawie pliku."""

    if defaults is not None and not isinstance(defaults, Mapping):
        raise TypeError("defaults must be a mapping")

    if disable_existing_loggers is not None and not isinstance(
        disable_existing_loggers, bool
    ):
        raise TypeError("disable_existing_loggers must be a boolean")

    if logger_name is not None and not isinstance(logger_name, str):
        raise TypeError("logger_name must be a string")

    config_path = Path(path)
    kwargs: dict[str, Any] = {}

    if defaults is not None:
        kwargs["defaults"] = dict(defaults)

    if disable_existing_loggers is not None:
        kwargs["disable_existing_loggers"] = disable_existing_loggers

    logging.config.fileConfig(config_path, **kwargs)

    target_logger = logging.getLogger(
        logger_name or f"{__name__}.optional_exports"
    )

    if set_as_default:
        set_optional_exports_logger(target_logger)

    return target_logger


def configure_optional_exports_logging_from_python(
    target: str,
    *,
    attribute: str | None = None,
    call: bool | None = None,
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    logger_name: str | None = None,
    set_as_default: bool = True,
    defaults: Mapping[str, Any] | None = None,
    disable_existing_loggers: bool | None = None,
    encoding: str = "utf-8",
) -> logging.Logger:
    """Skonfiguruj logowanie helperów lazy-eksportów na podstawie modułu Pythona.

    Parametr ``target`` powinien wskazywać moduł (np. ``"pkg.module"``). Jeśli
    w wartości znajduje się dwukropek (``"pkg.module:factory"``), zostanie on
    zinterpretowany jako nazwa atrybutu. W przeciwnym razie należy ją przekazać
    poprzez argument ``attribute``. Wczytany obiekt może być:

    * słownikiem kompatybilnym z ``logging.config.dictConfig`` – zostanie
      przekazany do :func:`configure_optional_exports_logging_from_dict`;
    * ścieżką do pliku konfiguracyjnego – zostanie użyty
      :func:`configure_optional_exports_logging_from_file`, przy czym pliki
      ``.json`` są traktowane jako konfiguracje JSON;
    * instancją :class:`logging.Logger` – ustawianą opcjonalnie jako logger
      domyślny helperów ``log_optional_*``;
    * wywoływalnym obiektem zwracającym dowolny z powyższych typów. Domyślnie
      funkcja zostanie wywołana, o ile jest wywoływalna. Zachowanie to można
      nadpisać flagą ``call``.
    """

    if not isinstance(target, str):
        raise TypeError("target must be a string with module specification")

    if attribute is not None and not isinstance(attribute, str):
        raise TypeError("attribute must be a string if provided")

    if call is not None and not isinstance(call, bool):
        raise TypeError("call must be a boolean if provided")

    if args is not None and not isinstance(args, Sequence):
        raise TypeError("args must be a sequence when provided")

    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping when provided")

    if logger_name is not None and not isinstance(logger_name, str):
        raise TypeError("logger_name must be a string")

    if defaults is not None and not isinstance(defaults, Mapping):
        raise TypeError("defaults must be a mapping")

    if disable_existing_loggers is not None and not isinstance(
        disable_existing_loggers, bool
    ):
        raise TypeError("disable_existing_loggers must be a boolean")

    if not isinstance(encoding, str):
        raise TypeError("encoding must be a string")

    module_name = target
    attribute_name = attribute

    if ":" in target:
        module_name, attr_part = target.split(":", 1)
        if not module_name:
            raise ValueError("module name in target cannot be empty")
        if attr_part:
            attribute_name = attr_part
        elif attribute_name is None:
            raise ValueError("attribute must be provided for python target")

    if not module_name:
        raise ValueError("target must specify a module name")

    if attribute_name is None:
        raise ValueError("attribute must be provided for python target")

    module = import_module(module_name)

    if not hasattr(module, attribute_name):
        raise AttributeError(
            f"module '{module_name}' does not provide attribute '{attribute_name}'"
        )

    obj = getattr(module, attribute_name)
    should_call = call if call is not None else callable(obj)

    args_to_use = tuple(args or ())
    kwargs_to_use = dict(kwargs or {})

    value: Any
    if should_call:
        value = obj(*args_to_use, **kwargs_to_use)
    else:
        value = obj

    if isinstance(value, logging.Logger):
        if set_as_default:
            set_optional_exports_logger(value)
        return value

    if isinstance(value, Mapping):
        return configure_optional_exports_logging_from_dict(
            value,
            logger_name=logger_name,
            set_as_default=set_as_default,
        )

    if isinstance(value, (str, bytes, os.PathLike)):
        path = Path(value).expanduser()

        if path.suffix.lower() == ".json":
            text = path.read_text(encoding=encoding)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON logging config from '{path}' provided by "
                    f"{module_name}:{attribute_name}: {exc}"
                ) from exc

            if not isinstance(payload, Mapping):
                raise ValueError(
                    "JSON logging config produced by python target must decode to a mapping"
                )

            return configure_optional_exports_logging_from_dict(
                payload,
                logger_name=logger_name,
                set_as_default=set_as_default,
            )

        kwargs_file: dict[str, Any] = {}
        if defaults is not None:
            kwargs_file["defaults"] = defaults
        if disable_existing_loggers is not None:
            kwargs_file["disable_existing_loggers"] = disable_existing_loggers

        return configure_optional_exports_logging_from_file(
            path,
            logger_name=logger_name,
            set_as_default=set_as_default,
            **kwargs_file,
        )

    raise TypeError(
        "python logging target must resolve to a mapping, path-like object or logging.Logger"
    )


def configure_optional_exports_logging_from_env(
    *,
    env_var: str = "BOT_CORE_OPTIONAL_EXPORTS_LOGGING",
    set_as_default: bool = True,
    missing_ok: bool = False,
    logger_name: str | None = None,
    defaults: Mapping[str, Any] | None = None,
    disable_existing_loggers: bool | None = None,
    encoding: str = "utf-8",
) -> logging.Logger | None:
    """Skonfiguruj logowanie helperów lazy-eksportów na podstawie zmiennej środowiskowej."""

    if not isinstance(env_var, str):
        raise TypeError("env_var must be a string")

    if logger_name is not None and not isinstance(logger_name, str):
        raise TypeError("logger_name must be a string")

    if defaults is not None and not isinstance(defaults, Mapping):
        raise TypeError("defaults must be a mapping")

    if disable_existing_loggers is not None and not isinstance(
        disable_existing_loggers, bool
    ):
        raise TypeError("disable_existing_loggers must be a boolean")

    raw_value = os.getenv(env_var)
    if raw_value is None or not raw_value.strip():
        if missing_ok:
            return None
        raise KeyError(f"environment variable '{env_var}' is not set or empty")

    spec = raw_value.strip()

    return configure_optional_exports_logging_from_spec(
        spec,
        set_as_default=set_as_default,
        logger_name=logger_name,
        defaults=defaults,
        disable_existing_loggers=disable_existing_loggers,
        encoding=encoding,
        source_description=f"environment variable '{env_var}'",
    )


try:  # pragma: no cover - środowiska testowe mogą nie mieć pełnego runtime
    from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment
    from bot_core.runtime.resource_monitor import (
        ResourceBudgetEvaluation,
        ResourceBudgets,
        ResourceSample,
        evaluate_resource_sample,
    )
except Exception:  # pragma: no cover - brak monitoringu zasobów w tej dystrybucji
    ResourceBudgetEvaluation = None  # type: ignore
    ResourceBudgets = None  # type: ignore
    ResourceSample = None  # type: ignore
    evaluate_resource_sample = None  # type: ignore

try:  # pragma: no cover - testy mogą działać bez modułu load testu
    from bot_core.runtime.scheduler_load_test import (
        LoadTestResult,
        LoadTestSettings,
        execute_scheduler_load_test,
    )
except Exception:  # pragma: no cover - brak modułu load testu
    LoadTestResult = None  # type: ignore
    LoadTestSettings = None  # type: ignore
    execute_scheduler_load_test = None  # type: ignore

try:  # pragma: no cover - zależne od gałęzi
    from bot_core.runtime.stage5_hypercare import (
        Stage5ComplianceConfig,
        Stage5HypercareConfig,
        Stage5HypercareCycle,
        Stage5HypercareResult,
        Stage5HypercareVerificationResult,
        Stage5OemAcceptanceConfig,
        Stage5RotationConfig,
        Stage5SloConfig,
        Stage5TcoConfig,
        Stage5TrainingConfig,
        verify_stage5_hypercare_summary,
    )
except Exception:  # pragma: no cover - brak modułu stage5 w tej dystrybucji
    Stage5ComplianceConfig = None  # type: ignore
    Stage5HypercareConfig = None  # type: ignore
    Stage5HypercareCycle = None  # type: ignore
    Stage5HypercareResult = None  # type: ignore
    Stage5HypercareVerificationResult = None  # type: ignore
    Stage5OemAcceptanceConfig = None  # type: ignore
    Stage5RotationConfig = None  # type: ignore
    Stage5SloConfig = None  # type: ignore
    Stage5TcoConfig = None  # type: ignore
    Stage5TrainingConfig = None  # type: ignore
    verify_stage5_hypercare_summary = None  # type: ignore

try:  # pragma: no cover - moduł full hypercare może być opcjonalny
    from bot_core.runtime.full_hypercare import (
        FullHypercareSummaryBuilder,
        FullHypercareSummaryConfig,
        FullHypercareSummaryResult,
        FullHypercareVerificationResult,
        verify_full_hypercare_summary,
    )
except Exception:  # pragma: no cover - brak modułu full hypercare
    FullHypercareSummaryBuilder = None  # type: ignore
    FullHypercareSummaryConfig = None  # type: ignore
    FullHypercareSummaryResult = None  # type: ignore
    FullHypercareVerificationResult = None  # type: ignore
    verify_full_hypercare_summary = None  # type: ignore

try:  # pragma: no cover - moduł stage6 może nie istnieć
    from bot_core.runtime.stage6_hypercare import (
        Stage6HypercareConfig,
        Stage6HypercareCycle,
        Stage6HypercareResult,
        Stage6HypercareVerificationResult,
        verify_stage6_hypercare_summary,
    )
except Exception:  # pragma: no cover - brak modułu stage6
    Stage6HypercareConfig = None  # type: ignore
    Stage6HypercareCycle = None  # type: ignore
    Stage6HypercareResult = None  # type: ignore
    Stage6HypercareVerificationResult = None  # type: ignore
    verify_stage6_hypercare_summary = None  # type: ignore

# --- Metrics service (opcjonalny – zależy od dostępności gRPC i wygenerowanych stubów) ---
try:
    from bot_core.runtime.metrics_service import (  # type: ignore
        MetricsServer,
        MetricsServiceServicer,
        MetricsSnapshotStore,
        MetricsSink,
        JsonlSink,
        ReduceMotionAlertSink,
        OverlayBudgetAlertSink,
        create_server as create_metrics_server,
        build_metrics_server_from_config,
    )
except Exception:  # pragma: no cover - brak wygenerowanych stubów lub grpcio
    MetricsServer = None  # type: ignore
    MetricsServiceServicer = None  # type: ignore
    MetricsSnapshotStore = None  # type: ignore
    MetricsSink = None  # type: ignore
    JsonlSink = None  # type: ignore
    ReduceMotionAlertSink = None  # type: ignore
    OverlayBudgetAlertSink = None  # type: ignore
    create_metrics_server = None  # type: ignore
    build_metrics_server_from_config = None  # type: ignore

# --- Risk service (opcjonalny – zależy od wygenerowanych stubów) ---
try:
    from bot_core.runtime.risk_service import (  # type: ignore
        RiskExposure,
        RiskServer,
        RiskServiceServicer,
        RiskSnapshot,
        RiskSnapshotBuilder,
        RiskSnapshotPublisher,
        RiskSnapshotStore,
        build_risk_server_from_config,
    )
except Exception:  # pragma: no cover - brak stubów risk service
    RiskExposure = None  # type: ignore
    RiskServer = None  # type: ignore
    RiskServiceServicer = None  # type: ignore
    RiskSnapshot = None  # type: ignore
    RiskSnapshotBuilder = None  # type: ignore
    RiskSnapshotPublisher = None  # type: ignore
    RiskSnapshotStore = None  # type: ignore
    build_risk_server_from_config = None  # type: ignore

try:  # pragma: no cover - eksporter metryk jest opcjonalny
    from bot_core.runtime.risk_metrics import RiskMetricsExporter  # type: ignore
except Exception:  # pragma: no cover - brak zależności opcjonalnych
    RiskMetricsExporter = None  # type: ignore

# --- Kontrolery / pipeline (opcjonalne – różnice między gałęziami) ---
_TRADING_CONTROLLER_IMPORT_ERROR: Exception | None = None
_TRADING_CONTROLLER_IMPORT_TRACEBACK: str | None = None

_TradingController = None  # type: ignore

try:
    from bot_core.runtime.controller import DailyTrendController as _DailyTrendController  # type: ignore
except Exception:
    _DailyTrendController = None  # type: ignore

try:
    from bot_core.runtime.realtime import (  # type: ignore
        DailyTrendRealtimeRunner as _DailyTrendRealtimeRunner
    )
except Exception:  # pragma: no cover - starsze gałęzie mogą nie mieć modułu realtime
    _DailyTrendRealtimeRunner = None  # type: ignore

try:
    from bot_core.runtime.pipeline import (  # type: ignore
        DailyTrendPipeline,
        build_daily_trend_pipeline,
        create_trading_controller,
    )
except Exception:  # pragma: no cover - starsze gałęzie mogą nie mieć modułu pipeline
    DailyTrendPipeline = None  # type: ignore
    build_daily_trend_pipeline = None  # type: ignore
    create_trading_controller = None  # type: ignore

# --- Publiczny interfejs modułu ---
__all__ = [
    "BootstrapContext",
    "bootstrap_environment",
    "ResourceBudgets",
    "ResourceSample",
    "ResourceBudgetEvaluation",
    "evaluate_resource_sample",
    "LoadTestSettings",
    "LoadTestResult",
    "execute_scheduler_load_test",
    "Stage5HypercareCycle",
    "Stage5HypercareConfig",
    "Stage5HypercareResult",
    "Stage5HypercareVerificationResult",
    "Stage5TcoConfig",
    "Stage5RotationConfig",
    "Stage5ComplianceConfig",
    "Stage5TrainingConfig",
    "Stage5SloConfig",
    "Stage5OemAcceptanceConfig",
    "verify_stage5_hypercare_summary",
    "FullHypercareSummaryBuilder",
    "FullHypercareSummaryConfig",
    "FullHypercareSummaryResult",
    "FullHypercareVerificationResult",
    "verify_full_hypercare_summary",
    "Stage6HypercareCycle",
    "Stage6HypercareConfig",
    "Stage6HypercareResult",
    "Stage6HypercareVerificationResult",
    "verify_stage6_hypercare_summary",
]

# Eksport elementów metrics service tylko jeśli są dostępne
if MetricsServer is not None:
    __all__.extend(
        [
            "MetricsServer",
            "MetricsServiceServicer",
            "MetricsSnapshotStore",
            "MetricsSink",
            "JsonlSink",
            "ReduceMotionAlertSink",
            "OverlayBudgetAlertSink",
            "create_metrics_server",
            "build_metrics_server_from_config",
        ]
    )

# Eksport elementów risk service tylko jeśli są dostępne
if RiskServer is not None:
    __all__.extend(
        [
            "RiskExposure",
            "RiskServer",
            "RiskServiceServicer",
            "RiskSnapshot",
            "RiskSnapshotBuilder",
            "RiskSnapshotPublisher",
            "RiskSnapshotStore",
            "build_risk_server_from_config",
        ]
    )

if RiskMetricsExporter is not None:
    __all__.append("RiskMetricsExporter")

# Eksportuj tylko te kontrolery, które są dostępne w danej gałęzi.
if _TradingController is None:
    # Defensywny fallback, gdy bezpośredni import się nie powiódł
    try:  # pragma: no cover
        _TradingController = getattr(
            import_module("bot_core.runtime.controller"), "TradingController", None
        )
    except Exception:  # pragma: no cover
        _TradingController = None  # type: ignore

if _TradingController is not None:
    TradingController = _TradingController  # type: ignore
    __all__.append("TradingController")

if _DailyTrendController is None:
    try:  # pragma: no cover
        _DailyTrendController = getattr(
            import_module("bot_core.runtime.controller"), "DailyTrendController", None
        )
    except Exception:  # pragma: no cover
        _DailyTrendController = None  # type: ignore

if _DailyTrendController is not None:
    DailyTrendController = _DailyTrendController  # type: ignore
    __all__.append("DailyTrendController")

if _DailyTrendRealtimeRunner is None:
    try:  # pragma: no cover
        _DailyTrendRealtimeRunner = getattr(
            import_module("bot_core.runtime.realtime"), "DailyTrendRealtimeRunner", None
        )
    except Exception:  # pragma: no cover
        _DailyTrendRealtimeRunner = None  # type: ignore

if _DailyTrendRealtimeRunner is not None:
    DailyTrendRealtimeRunner = _DailyTrendRealtimeRunner  # type: ignore
    __all__.append("DailyTrendRealtimeRunner")

if DailyTrendPipeline is None or build_daily_trend_pipeline is None:
    try:  # pragma: no cover
        _pipeline_module = import_module("bot_core.runtime.pipeline")
        DailyTrendPipeline = getattr(_pipeline_module, "DailyTrendPipeline", None)
        build_daily_trend_pipeline = getattr(
            _pipeline_module, "build_daily_trend_pipeline", None
        )
        create_trading_controller = getattr(
            _pipeline_module, "create_trading_controller", None
        )
    except Exception:  # pragma: no cover
        DailyTrendPipeline = None  # type: ignore
        build_daily_trend_pipeline = None  # type: ignore
        create_trading_controller = None  # type: ignore

if DailyTrendPipeline is not None and build_daily_trend_pipeline is not None:
    __all__.extend(["DailyTrendPipeline", "build_daily_trend_pipeline"])
    if create_trading_controller is not None:
        __all__.append("create_trading_controller")


_LAZY_OPTIONAL_EXPORTS: dict[str, tuple[str, str]] = {
    "TradingController": ("bot_core.runtime.controller", "TradingController"),
    "DailyTrendController": ("bot_core.runtime.controller", "DailyTrendController"),
    "DailyTrendRealtimeRunner": ("bot_core.runtime.realtime", "DailyTrendRealtimeRunner"),
    "DailyTrendPipeline": ("bot_core.runtime.pipeline", "DailyTrendPipeline"),
    "build_daily_trend_pipeline": ("bot_core.runtime.pipeline", "build_daily_trend_pipeline"),
    "create_trading_controller": ("bot_core.runtime.pipeline", "create_trading_controller"),
}

_BASE_OPTIONAL_EXPORTS: frozenset[str] = frozenset(_LAZY_OPTIONAL_EXPORTS)

_MISSING = object()

# Dodaj nazwy do __all__, aby statyczne narzędzia widziały dostępne symbole nawet przy lazy-loadzie
for _lazy_name in _LAZY_OPTIONAL_EXPORTS:
    if _lazy_name not in __all__:
        __all__.append(_lazy_name)


@dataclass(frozen=True, slots=True)
class OptionalExportStatus:
    """Stan pojedynczego lazy-eksportu runtime."""

    name: str
    module: str
    attribute: str
    available: bool
    cached: bool
    error: str | None = None


def _status_to_dict(status: OptionalExportStatus) -> dict[str, object]:
    """Zamień obiekt statusu lazy-eksportu na słownik."""

    return {
        "name": status.name,
        "module": status.module,
        "attribute": status.attribute,
        "available": status.available,
        "cached": status.cached,
        "error": status.error,
    }


def _status_from_dict(name: str, payload: object) -> OptionalExportStatus:
    """Zbuduj status lazy-eksportu z reprezentacji słownikowej."""

    if not isinstance(payload, dict):
        raise ValueError("status entries must be mappings")

    module = payload.get("module")
    attribute = payload.get("attribute")
    available = payload.get("available")
    cached = payload.get("cached")
    error = payload.get("error")
    entry_name = payload.get("name", name)

    if not isinstance(entry_name, str):
        raise ValueError("status entry must define a string 'name' field")
    if not isinstance(module, str) or not isinstance(attribute, str):
        raise ValueError(
            "status entry must define string 'module' and 'attribute' fields"
        )
    if not isinstance(available, bool) or not isinstance(cached, bool):
        raise ValueError(
            "status entry must define boolean 'available' and 'cached' fields"
        )
    if error is not None and not isinstance(error, str):
        raise ValueError("status entry 'error' field must be a string or None")

    return OptionalExportStatus(
        name=entry_name,
        module=module,
        attribute=attribute,
        available=available,
        cached=cached,
        error=error,
    )


def _target_to_dict(target: tuple[str, str]) -> dict[str, str]:
    """Zamień definicję modułu/atrybutu na słownik."""

    module, attribute = target
    return {"module": module, "attribute": attribute}


def _target_from_dict(payload: object, *, context: str) -> tuple[str, str]:
    """Odtwórz krotkę moduł/atrybut z reprezentacji słownikowej."""

    if not isinstance(payload, dict):
        raise ValueError(f"{context} entries must be mappings")

    module = payload.get("module")
    attribute = payload.get("attribute")
    if not isinstance(module, str) or not isinstance(attribute, str):
        raise ValueError(
            f"{context} entry must define string 'module' and 'attribute' fields"
        )

    return module, attribute


@dataclass(frozen=True, slots=True)
class OptionalExportRegistrySnapshot:
    """Migawka rejestru lazy-eksportów wraz z diagnostyką."""

    registered: dict[str, tuple[str, str]]
    cached_names: frozenset[str]
    statuses: dict[str, OptionalExportStatus]

    @property
    def available_names(self) -> frozenset[str]:
        """Zwróć zbiór nazw, które są dostępne w bieżącym środowisku."""

        if not self.statuses:
            return frozenset()
        return frozenset(
            name for name, status in self.statuses.items() if status.available
        )

    @property
    def missing(self) -> dict[str, OptionalExportStatus]:
        """Zwróć mapę niedostępnych eksportów wraz z diagnostyką."""

        if not self.statuses:
            return {}
        return {
            name: status
            for name, status in self.statuses.items()
            if not status.available
        }


@dataclass(frozen=True, slots=True)
class OptionalExportRegistryDiff:
    """Różnica pomiędzy dwiema migawkami lazy-eksportów runtime."""

    added: dict[str, tuple[str, str]]
    removed: dict[str, tuple[str, str]]
    changed_targets: dict[str, tuple[tuple[str, str], tuple[str, str]]]
    status_changes: dict[
        str, tuple[OptionalExportStatus | None, OptionalExportStatus | None]
    ]
    cache_gained: frozenset[str]
    cache_lost: frozenset[str]

    @property
    def has_changes(self) -> bool:
        """Czy diff zawiera jakiekolwiek zmiany względem bazowej migawki."""

        return bool(
            self.added
            or self.removed
            or self.changed_targets
            or self.status_changes
            or self.cache_gained
            or self.cache_lost
        )


class OptionalExportUnavailableError(RuntimeError):
    """Wyjątek zgłaszany, gdy wymagany lazy-eksport nie jest dostępny."""

    def __init__(self, status: OptionalExportStatus):
        self.status = status
        message = (
            f"Optional export '{status.name}' is not available "
            f"(expected {status.module}.{status.attribute})"
        )
        if status.error:
            message = f"{message}: {status.error}"
        super().__init__(message)


def _load_optional_export(name: str, *, cache: bool = True):
    """Załaduj opcjonalny symbol runtime i opcjonalnie zapamiętaj go w module."""

    target = _LAZY_OPTIONAL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'bot_core.runtime' has no attribute '{name}'")

    module_name, attr_name = target
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - przekazujemy błąd jako AttributeError
        raise AttributeError(
            f"module 'bot_core.runtime' could not load optional attribute '{name}'"
        ) from exc

    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - moduł nie posiada atrybutu
        raise AttributeError(
            f"module '{module_name}' has no attribute '{attr_name}'"
        ) from exc

    if cache:
        globals()[name] = value

    return value


def __getattr__(name: str):  # pragma: no cover - mechanizm defensywny
    """Lazy loader dla opcjonalnych komponentów runtime.

    Niektóre gałęzie repozytorium lub środowiska mogą nie posiadać wszystkich
    modułów runtime. Zamiast podnosić ImportError już na etapie importu pakietu
    ``bot_core.runtime`` staramy się dociągnąć zależność dopiero przy dostępie
    do konkretnego symbolu. Pozwala to zachować kompatybilność z testami,
    które oczekują obecności np. ``TradingController``.
    """

    return _load_optional_export(name, cache=True)


def require_optional_export(name: str, *, cache: bool = True):
    """Wymuś załadowanie zadeklarowanego lazy-eksportu lub zgłoś błąd."""

    if name not in _LAZY_OPTIONAL_EXPORTS:
        raise ValueError(
            f"'{name}' nie jest zarejestrowane jako opcjonalny eksport runtime"
        )

    try:
        return _load_optional_export(name, cache=cache)
    except AttributeError as exc:
        status = _probe_optional_export(name)
        raise OptionalExportUnavailableError(status) from exc


def register_optional_export(
    name: str,
    module: str,
    attribute: str,
    *,
    override: bool = False,
) -> None:
    """Zarejestruj nowy lazy-eksport dostępny przez ``bot_core.runtime``.

    Pozwala to rozszerzać runtime o dodatkowe komponenty bez konieczności
    modyfikowania modułu źródłowego. Jeśli wskazana nazwa już istnieje, domyślnie
    zgłaszany jest błąd, aby uniknąć przypadkowego nadpisania istniejącego
    eksportu. Można to zachowanie zmienić przekazując ``override=True``.
    """

    if not name:
        raise ValueError("Nazwa opcjonalnego eksportu nie może być pusta")

    if name in _LAZY_OPTIONAL_EXPORTS and not override:
        raise ValueError(
            f"Optional export '{name}' is already registered – use override=True to replace it"
        )

    _LAZY_OPTIONAL_EXPORTS[name] = (module, attribute)
    if override:
        globals().pop(name, None)

    if name not in __all__:
        __all__.append(name)


def unregister_optional_export(name: str, *, allow_builtin: bool = False) -> None:
    """Usuń istniejący lazy-eksport runtime."""

    if name not in _LAZY_OPTIONAL_EXPORTS:
        raise ValueError(f"Optional export '{name}' is not registered")

    if name in _BASE_OPTIONAL_EXPORTS and not allow_builtin:
        raise ValueError(
            "Cannot unregister built-in optional export without allow_builtin=True"
        )

    _LAZY_OPTIONAL_EXPORTS.pop(name, None)
    globals().pop(name, None)

    if name not in _BASE_OPTIONAL_EXPORTS and name in __all__:
        __all__.remove(name)


@contextmanager
def temporary_optional_export(
    name: str,
    module: str,
    attribute: str,
    *,
    override: bool = False,
    cache: bool = True,
):
    """Zarejestruj tymczasowy lazy-eksport na czas trwania kontekstu.

    Funkcja przydaje się w testach oraz scenariuszach dynamicznych rozszerzeń,
    gdzie chcemy podmienić lub dodać symbol tylko w obrębie wybranego bloku
    kodu. Gdy ``override`` jest ``False`` i wskazana nazwa istnieje, zgłaszany
    jest błąd – analogicznie do :func:`register_optional_export`.

    Parametr ``cache`` pozwala kontrolować, czy wartość ma zostać zachowana w
    module po pierwszym załadowaniu. Domyślne ``True`` odpowiada zachowaniu
    :func:`require_optional_export`.
    """

    existed = name in _LAZY_OPTIONAL_EXPORTS
    previous_target: tuple[str, str] | None = None
    previous_value = _MISSING

    if existed:
        if not override:
            raise ValueError(
                f"Optional export '{name}' is already registered – set override=True"
            )
        previous_target = _LAZY_OPTIONAL_EXPORTS[name]
        previous_value = globals().get(name, _MISSING)

    register_optional_export(name, module, attribute, override=override)

    try:
        value = require_optional_export(name, cache=cache)
        yield value
    finally:
        if existed and previous_target is not None:
            _LAZY_OPTIONAL_EXPORTS[name] = previous_target
            if previous_value is _MISSING:
                globals().pop(name, None)
            else:
                globals()[name] = previous_value
            if name not in __all__:
                __all__.append(name)
        else:
            if name in _LAZY_OPTIONAL_EXPORTS:
                unregister_optional_export(name)


def _probe_optional_export(name: str) -> OptionalExportStatus:
    """Zwróć informacje diagnostyczne o zadeklarowanym lazy-eksportcie."""

    target = _LAZY_OPTIONAL_EXPORTS.get(name)
    if target is None:
        raise ValueError(f"'{name}' nie jest zarejestrowane jako opcjonalny eksport runtime")

    module_name, attr_name = target

    if name in globals():
        return OptionalExportStatus(
            name=name,
            module=module_name,
            attribute=attr_name,
            available=True,
            cached=True,
            error=None,
        )

    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - środowiska testowe mogą pomijać moduły
        return OptionalExportStatus(
            name=name,
            module=module_name,
            attribute=attr_name,
            available=False,
            cached=False,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    try:
        getattr(module, attr_name)
    except AttributeError as exc:  # pragma: no cover - moduł nie udostępnia symbolu
        return OptionalExportStatus(
            name=name,
            module=module_name,
            attribute=attr_name,
            available=False,
            cached=False,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    return OptionalExportStatus(
        name=name,
        module=module_name,
        attribute=attr_name,
        available=True,
        cached=False,
        error=None,
    )


def is_optional_export_available(name: str) -> bool:
    """Sprawdź, czy zadeklarowany lazy eksport jest dostępny w bieżącym środowisku."""

    return _probe_optional_export(name).available


def list_optional_exports(*, available_only: bool = False) -> list[str]:
    """Zwróć posortowaną listę nazw opcjonalnych eksportów.

    Parametr ``available_only`` pozwala ograniczyć wynik do symboli, które w
    bieżącym środowisku można faktycznie zaimportować.
    """

    names = sorted(_LAZY_OPTIONAL_EXPORTS)
    if not available_only:
        return names

    available: list[str] = []
    for name in names:
        status = _probe_optional_export(name)
        if status.available:
            available.append(name)
    return available


def describe_optional_exports(*, available_only: bool = False) -> list[OptionalExportStatus]:
    """Zwróć szczegółowe informacje o wszystkich lazy-eksportach runtime."""

    statuses: list[OptionalExportStatus] = []
    for name in sorted(_LAZY_OPTIONAL_EXPORTS):
        status = _probe_optional_export(name)
        if available_only and not status.available:
            continue
        statuses.append(status)
    return statuses


def snapshot_optional_exports() -> OptionalExportRegistrySnapshot:
    """Przygotuj migawkę rejestru lazy-eksportów wraz z diagnostyką."""

    registered = dict(_LAZY_OPTIONAL_EXPORTS)
    cached_names = frozenset(name for name in registered if name in globals())
    statuses = {name: _probe_optional_export(name) for name in registered}
    return OptionalExportRegistrySnapshot(
        registered=registered,
        cached_names=cached_names,
        statuses=statuses,
    )


def diff_optional_exports_snapshots(
    previous: OptionalExportRegistrySnapshot,
    current: OptionalExportRegistrySnapshot,
) -> OptionalExportRegistryDiff:
    """Porównaj dwie migawki rejestru lazy-eksportów."""

    if not isinstance(previous, OptionalExportRegistrySnapshot):
        raise TypeError(
            "previous snapshot must be an instance of OptionalExportRegistrySnapshot"
        )
    if not isinstance(current, OptionalExportRegistrySnapshot):
        raise TypeError(
            "current snapshot must be an instance of OptionalExportRegistrySnapshot"
        )

    previous_registered = previous.registered
    current_registered = current.registered

    added = {
        name: current_registered[name]
        for name in current_registered
        if name not in previous_registered
    }

    removed = {
        name: previous_registered[name]
        for name in previous_registered
        if name not in current_registered
    }

    shared_names = set(previous_registered) & set(current_registered)
    changed_targets: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
    for name in sorted(shared_names):
        previous_target = previous_registered[name]
        current_target = current_registered[name]
        if previous_target != current_target:
            changed_targets[name] = (previous_target, current_target)

    previous_statuses = previous.statuses
    current_statuses = current.statuses
    status_changes: dict[
        str, tuple[OptionalExportStatus | None, OptionalExportStatus | None]
    ] = {}
    for name in sorted(set(previous_statuses) | set(current_statuses)):
        previous_status = previous_statuses.get(name)
        current_status = current_statuses.get(name)
        if previous_status != current_status:
            status_changes[name] = (previous_status, current_status)

    previous_cached = set(previous.cached_names)
    current_cached = set(current.cached_names)
    cache_gained = frozenset(current_cached - previous_cached)
    cache_lost = frozenset(previous_cached - current_cached)

    return OptionalExportRegistryDiff(
        added=added,
        removed=removed,
        changed_targets=changed_targets,
        status_changes=status_changes,
        cache_gained=cache_gained,
        cache_lost=cache_lost,
    )


def optional_exports_diff_to_dict(
    diff: OptionalExportRegistryDiff,
) -> dict[str, object]:
    """Zserializuj diff rejestru lazy-eksportów do struktury słownikowej."""

    if not isinstance(diff, OptionalExportRegistryDiff):
        raise TypeError(
            "diff must be an instance of OptionalExportRegistryDiff"
        )

    added = {name: _target_to_dict(target) for name, target in diff.added.items()}
    removed = {
        name: _target_to_dict(target) for name, target in diff.removed.items()
    }
    changed_targets: dict[str, dict[str, dict[str, str]]] = {}
    for name, (before, after) in diff.changed_targets.items():
        changed_targets[name] = {
            "before": _target_to_dict(before),
            "after": _target_to_dict(after),
        }

    status_changes: dict[str, dict[str, dict[str, object] | None]] = {}
    for name, (before, after) in diff.status_changes.items():
        status_changes[name] = {
            "before": None if before is None else _status_to_dict(before),
            "after": None if after is None else _status_to_dict(after),
        }

    return {
        "added": added,
        "removed": removed,
        "changed_targets": changed_targets,
        "status_changes": status_changes,
        "cache_gained": sorted(diff.cache_gained),
        "cache_lost": sorted(diff.cache_lost),
    }


def optional_exports_diff_from_dict(data: object) -> OptionalExportRegistryDiff:
    """Odtwórz diff rejestru lazy-eksportów z reprezentacji słownikowej."""

    if not isinstance(data, dict):
        raise TypeError(
            "data must be a mapping produced by optional_exports_diff_to_dict"
        )

    required_keys = {
        "added",
        "removed",
        "changed_targets",
        "status_changes",
        "cache_gained",
        "cache_lost",
    }
    missing_keys = required_keys - data.keys()
    if missing_keys:
        missing_str = ", ".join(sorted(missing_keys))
        raise ValueError(
            "diff dictionary must contain keys: " + missing_str
        )

    added_raw = data["added"]
    if not isinstance(added_raw, dict):
        raise ValueError("'added' entry must be a mapping")
    added: dict[str, tuple[str, str]] = {}
    for name, payload in added_raw.items():
        if not isinstance(name, str):
            raise ValueError("added keys must be strings")
        added[name] = _target_from_dict(payload, context="added")

    removed_raw = data["removed"]
    if not isinstance(removed_raw, dict):
        raise ValueError("'removed' entry must be a mapping")
    removed: dict[str, tuple[str, str]] = {}
    for name, payload in removed_raw.items():
        if not isinstance(name, str):
            raise ValueError("removed keys must be strings")
        removed[name] = _target_from_dict(payload, context="removed")

    changed_targets_raw = data["changed_targets"]
    if not isinstance(changed_targets_raw, dict):
        raise ValueError("'changed_targets' entry must be a mapping")
    changed_targets: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
    for name, payload in changed_targets_raw.items():
        if not isinstance(name, str):
            raise ValueError("changed_targets keys must be strings")
        if not isinstance(payload, dict):
            raise ValueError("changed_targets entries must be mappings")
        before_payload = payload.get("before")
        after_payload = payload.get("after")
        if "before" not in payload or "after" not in payload:
            raise ValueError(
                "changed_targets entries must define 'before' and 'after' fields"
            )
        before = _target_from_dict(before_payload, context="changed_targets.before")
        after = _target_from_dict(after_payload, context="changed_targets.after")
        changed_targets[name] = (before, after)

    status_changes_raw = data["status_changes"]
    if not isinstance(status_changes_raw, dict):
        raise ValueError("'status_changes' entry must be a mapping")
    status_changes: dict[
        str, tuple[OptionalExportStatus | None, OptionalExportStatus | None]
    ] = {}
    for name, payload in status_changes_raw.items():
        if not isinstance(name, str):
            raise ValueError("status_changes keys must be strings")
        if not isinstance(payload, dict):
            raise ValueError("status_changes entries must be mappings")

        before_payload = payload.get("before")
        after_payload = payload.get("after")
        if "before" not in payload or "after" not in payload:
            raise ValueError(
                "status_changes entries must define 'before' and 'after' fields"
            )

        before_status = (
            None
            if before_payload is None
            else _status_from_dict(name, before_payload)
        )
        after_status = (
            None
            if after_payload is None
            else _status_from_dict(name, after_payload)
        )
        status_changes[name] = (before_status, after_status)

    cache_gained_raw = data["cache_gained"]
    if not isinstance(cache_gained_raw, (list, tuple, set, frozenset)):
        raise ValueError("'cache_gained' entry must be an iterable of strings")
    cache_gained: set[str] = set()
    for item in cache_gained_raw:
        if not isinstance(item, str):
            raise ValueError("cache_gained entries must be strings")
        cache_gained.add(item)

    cache_lost_raw = data["cache_lost"]
    if not isinstance(cache_lost_raw, (list, tuple, set, frozenset)):
        raise ValueError("'cache_lost' entry must be an iterable of strings")
    cache_lost: set[str] = set()
    for item in cache_lost_raw:
        if not isinstance(item, str):
            raise ValueError("cache_lost entries must be strings")
        cache_lost.add(item)

    return OptionalExportRegistryDiff(
        added=added,
        removed=removed,
        changed_targets=changed_targets,
        status_changes=status_changes,
        cache_gained=frozenset(cache_gained),
        cache_lost=frozenset(cache_lost),
    )


def optional_exports_diff_to_json(
    diff: OptionalExportRegistryDiff,
    *,
    indent: int | None = None,
    sort_keys: bool = True,
) -> str:
    """Serializuj diff rejestru lazy-eksportów do łańcucha JSON."""

    if not isinstance(diff, OptionalExportRegistryDiff):
        raise TypeError(
            "diff must be an instance of OptionalExportRegistryDiff"
        )

    payload = optional_exports_diff_to_dict(diff)
    return json.dumps(payload, indent=indent, sort_keys=sort_keys)


def optional_exports_diff_from_json(data: object) -> OptionalExportRegistryDiff:
    """Odtwórz diff rejestru lazy-eksportów na podstawie JSON."""

    if isinstance(data, (bytes, bytearray)):
        try:
            text = data.decode("utf-8")
        except Exception as exc:  # pragma: no cover - defensywnie
            raise ValueError("diff JSON must be UTF-8 decodable") from exc
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("diff JSON must be provided as str or bytes")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid diff JSON: {exc}") from exc

    return optional_exports_diff_from_dict(payload)


def optional_exports_diff_to_file(
    diff: OptionalExportRegistryDiff,
    path: PathInput,
    *,
    indent: int | None = None,
    encoding: str = "utf-8",
) -> None:
    """Zapisz diff lazy-eksportów do pliku JSON."""

    if not isinstance(diff, OptionalExportRegistryDiff):
        raise TypeError(
            "diff must be an instance of OptionalExportRegistryDiff"
        )

    target_path = Path(path)
    payload = optional_exports_diff_to_json(diff, indent=indent)
    target_path.write_text(payload, encoding=encoding)


def optional_exports_diff_from_file(
    path: PathInput,
    *,
    encoding: str = "utf-8",
) -> OptionalExportRegistryDiff:
    """Wczytaj diff lazy-eksportów z pliku JSON."""

    target_path = Path(path)
    contents = target_path.read_text(encoding=encoding)
    return optional_exports_diff_from_json(contents)


def _format_target(target: tuple[str, str]) -> str:
    module, attribute = target
    return f"{module}.{attribute}"


def format_optional_export_status(
    status: OptionalExportStatus,
    *,
    include_error: bool = True,
) -> str:
    """Zwróć tekstową reprezentację statusu lazy-eksportu."""

    if not isinstance(status, OptionalExportStatus):
        raise TypeError("status must be an instance of OptionalExportStatus")

    availability = "available" if status.available else "missing"
    cache_state = "cached" if status.cached else "not cached"
    parts = [
        f"{status.name}: {_format_target((status.module, status.attribute))}",
        f"[{availability}, {cache_state}]",
    ]
    if include_error and status.error:
        parts.append(f"error={status.error}")
    return " ".join(parts)


def _ensure_status_sequence(statuses: Iterable[OptionalExportStatus]) -> Sequence[OptionalExportStatus]:
    sequence = list(statuses)
    for status in sequence:
        if not isinstance(status, OptionalExportStatus):
            raise TypeError(
                "statuses must contain OptionalExportStatus instances only"
            )
    return sequence


def _coerce_log_level(level: int | str) -> int:
    if isinstance(level, int):
        return level

    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
        if isinstance(numeric_level, int):
            return numeric_level

    raise TypeError("level must be an int or a valid logging level name")


def _coerce_log_extra(extra: Mapping[str, object] | None) -> dict[str, object] | None:
    if extra is None:
        return None

    if not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping with string keys")

    for key in extra.keys():
        if not isinstance(key, str):
            raise TypeError("extra mapping keys must be strings")

    return dict(extra)


def _coerce_stacklevel(stacklevel: int | None) -> int | None:
    if stacklevel is None:
        return None

    if not isinstance(stacklevel, int):
        raise TypeError("stacklevel must be an integer")

    if stacklevel < 1:
        raise ValueError("stacklevel must be greater than or equal to 1")

    return stacklevel


def _emit_log_message(
    message: str,
    *,
    logger: _LoggerLike | None,
    level: int | str,
    extra: Mapping[str, object] | None,
    stacklevel: int | None,
) -> None:
    target_logger = (
        cast(_LoggerLike, logger)
        if logger is not None
        else get_optional_exports_logger()
    )

    if not isinstance(target_logger, _LoggerLike):  # pragma: no cover - walidacja typu w runtime
        raise TypeError(
            "logger must provide a log(level, message, *args, **kwargs) method"
        )

    kwargs: dict[str, object] = {}
    extra_payload = _coerce_log_extra(extra)
    if extra_payload is not None:
        kwargs["extra"] = extra_payload

    stacklevel_value = _coerce_stacklevel(stacklevel)
    if stacklevel_value is not None:
        kwargs["stacklevel"] = stacklevel_value

    target_logger.log(_coerce_log_level(level), message, **kwargs)


def format_optional_exports_summary(
    statuses: Iterable[OptionalExportStatus],
    *,
    include_header: bool = True,
    include_errors: bool = True,
) -> str:
    """Przygotuj zwięzły raport statusów lazy-eksportów."""

    ordered = sorted(
        _ensure_status_sequence(statuses),
        key=lambda status: status.name,
    )

    if not ordered:
        return "Optional exports: (none)" if include_header else ""

    lines: list[str] = []
    if include_header:
        lines.append("Optional exports:")

    for status in ordered:
        line = format_optional_export_status(status, include_error=include_errors)
        lines.append(f"  - {line}")

    return "\n".join(lines)


def format_optional_exports_diff(
    diff: OptionalExportRegistryDiff,
    *,
    include_header: bool = True,
    include_errors: bool = True,
) -> str:
    """Zwróć czytelne podsumowanie diffu lazy-eksportów."""

    if not isinstance(diff, OptionalExportRegistryDiff):
        raise TypeError("diff must be an instance of OptionalExportRegistryDiff")

    sections: list[str] = []

    if diff.added:
        added_lines = "\n".join(
            f"  - {name}: {_format_target(target)}"
            for name, target in sorted(diff.added.items())
        )
        sections.append(f"Added:\n{added_lines}")

    if diff.removed:
        removed_lines = "\n".join(
            f"  - {name}: {_format_target(target)}"
            for name, target in sorted(diff.removed.items())
        )
        sections.append(f"Removed:\n{removed_lines}")

    if diff.changed_targets:
        changed_lines = "\n".join(
            "  - {name}: {before} -> {after}".format(
                name=name,
                before=_format_target(before),
                after=_format_target(after),
            )
            for name, (before, after) in sorted(diff.changed_targets.items())
        )
        sections.append(f"Changed targets:\n{changed_lines}")

    if diff.status_changes:
        status_lines = []
        for name, (before, after) in sorted(diff.status_changes.items()):
            before_repr = (
                format_optional_export_status(before, include_error=include_errors)
                if before is not None
                else "<missing>"
            )
            after_repr = (
                format_optional_export_status(after, include_error=include_errors)
                if after is not None
                else "<missing>"
            )
            status_lines.append(f"  - {name}: {before_repr} -> {after_repr}")
        sections.append("Status changes:\n" + "\n".join(status_lines))

    if diff.cache_gained:
        gained_lines = "\n".join(
            f"  - {name}" for name in sorted(diff.cache_gained)
        )
        sections.append(f"Cache gained:\n{gained_lines}")

    if diff.cache_lost:
        lost_lines = "\n".join(f"  - {name}" for name in sorted(diff.cache_lost))
        sections.append(f"Cache lost:\n{lost_lines}")

    if not sections:
        return "Optional export diff: no changes" if include_header else "no changes"

    report = "\n\n".join(sections)

    if include_header:
        return f"Optional export diff:\n{report}"
    return report


def log_optional_export_status(
    status: OptionalExportStatus,
    *,
    include_error: bool = True,
    logger: _LoggerLike | None = None,
    level: int | str = logging.INFO,
    extra: Mapping[str, object] | None = None,
    stacklevel: int | None = None,
) -> str:
    """Zaloguj status pojedynczego lazy-eksportu i zwróć sformatowaną wiadomość."""

    if not isinstance(status, OptionalExportStatus):
        raise TypeError("status must be an instance of OptionalExportStatus")

    message = format_optional_export_status(status, include_error=include_error)
    _emit_log_message(
        message,
        logger=logger,
        level=level,
        extra=extra,
        stacklevel=stacklevel,
    )
    return message


def log_optional_exports_summary(
    statuses: Iterable[OptionalExportStatus] | None = None,
    *,
    include_errors: bool = True,
    available_only: bool = False,
    logger: _LoggerLike | None = None,
    level: int | str = logging.INFO,
    extra: Mapping[str, object] | None = None,
    stacklevel: int | None = None,
) -> str:
    """Zaloguj zbiorcze podsumowanie lazy-eksportów i zwróć wygenerowany raport."""

    if statuses is None:
        statuses = describe_optional_exports(available_only=available_only)
    else:
        statuses = _ensure_status_sequence(statuses)

    message = format_optional_exports_summary(
        statuses,
        include_errors=include_errors,
    )
    _emit_log_message(
        message,
        logger=logger,
        level=level,
        extra=extra,
        stacklevel=stacklevel,
    )
    return message


def log_optional_exports_diff(
    diff: OptionalExportRegistryDiff,
    *,
    include_errors: bool = True,
    logger: _LoggerLike | None = None,
    level: int | str = logging.INFO,
    extra: Mapping[str, object] | None = None,
    stacklevel: int | None = None,
) -> str:
    """Zaloguj diff rejestru lazy-eksportów i zwróć jego reprezentację tekstową."""

    if not isinstance(diff, OptionalExportRegistryDiff):
        raise TypeError("diff must be an instance of OptionalExportRegistryDiff")

    message = format_optional_exports_diff(
        diff,
        include_errors=include_errors,
    )
    _emit_log_message(
        message,
        logger=logger,
        level=level,
        extra=extra,
        stacklevel=stacklevel,
    )
    return message


def optional_exports_snapshot_to_dict(
    snapshot: OptionalExportRegistrySnapshot,
) -> dict[str, object]:
    """Zserializuj migawkę rejestru lazy-eksportów do struktury słownikowej."""

    if not isinstance(snapshot, OptionalExportRegistrySnapshot):
        raise TypeError(
            "snapshot must be an instance of OptionalExportRegistrySnapshot"
        )

    registered = {name: _target_to_dict(target) for name, target in snapshot.registered.items()}

    statuses = {name: _status_to_dict(status) for name, status in snapshot.statuses.items()}

    return {
        "registered": registered,
        "cached_names": sorted(snapshot.cached_names),
        "statuses": statuses,
    }


def optional_exports_snapshot_from_dict(
    data: object,
) -> OptionalExportRegistrySnapshot:
    """Odtwórz migawkę rejestru lazy-eksportów z reprezentacji słownikowej."""

    if not isinstance(data, dict):
        raise TypeError(
            "data must be a mapping produced by optional_exports_snapshot_to_dict"
        )

    required_keys = {"registered", "cached_names", "statuses"}
    missing_keys = required_keys - data.keys()
    if missing_keys:
        missing_str = ", ".join(sorted(missing_keys))
        raise ValueError(
            "snapshot dictionary must contain keys: " + missing_str
        )

    registered_raw = data["registered"]
    if not isinstance(registered_raw, dict):
        raise ValueError("'registered' entry must be a mapping")

    registered: dict[str, tuple[str, str]] = {}
    for name, payload in registered_raw.items():
        if not isinstance(name, str):
            raise ValueError("optional export names must be strings")
        registered[name] = _target_from_dict(payload, context="registered")

    cached_raw = data["cached_names"]
    if not isinstance(cached_raw, (list, tuple, set, frozenset)):
        raise ValueError("'cached_names' entry must be an iterable of strings")

    cached_names: set[str] = set()
    for item in cached_raw:
        if not isinstance(item, str):
            raise ValueError("cached name entries must be strings")
        cached_names.add(item)

    statuses_raw = data["statuses"]
    if not isinstance(statuses_raw, dict):
        raise ValueError("'statuses' entry must be a mapping")

    statuses: dict[str, OptionalExportStatus] = {}
    for name, payload in statuses_raw.items():
        if not isinstance(name, str):
            raise ValueError("status keys must be strings")
        statuses[name] = _status_from_dict(name, payload)

    return OptionalExportRegistrySnapshot(
        registered=registered,
        cached_names=frozenset(cached_names),
        statuses=statuses,
    )


def optional_exports_snapshot_to_json(
    snapshot: OptionalExportRegistrySnapshot,
    *,
    indent: int | None = None,
    sort_keys: bool = True,
) -> str:
    """Serializuj migawkę rejestru lazy-eksportów do łańcucha JSON."""

    if not isinstance(snapshot, OptionalExportRegistrySnapshot):
        raise TypeError(
            "snapshot must be an instance of OptionalExportRegistrySnapshot"
        )

    payload = optional_exports_snapshot_to_dict(snapshot)
    return json.dumps(payload, indent=indent, sort_keys=sort_keys)


def optional_exports_snapshot_from_json(data: object) -> OptionalExportRegistrySnapshot:
    """Odtwórz migawkę rejestru lazy-eksportów na podstawie JSON."""

    if isinstance(data, (bytes, bytearray)):
        try:
            text = data.decode("utf-8")
        except Exception as exc:  # pragma: no cover - defensywna ochrona
            raise ValueError("snapshot JSON must be UTF-8 decodable") from exc
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("snapshot JSON must be provided as str or bytes")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid snapshot JSON: {exc}") from exc

    return optional_exports_snapshot_from_dict(payload)


def optional_exports_snapshot_to_file(
    snapshot: OptionalExportRegistrySnapshot,
    path: PathInput,
    *,
    indent: int | None = None,
    encoding: str = "utf-8",
) -> None:
    """Zapisz migawkę lazy-eksportów do pliku JSON."""

    if not isinstance(snapshot, OptionalExportRegistrySnapshot):
        raise TypeError(
            "snapshot must be an instance of OptionalExportRegistrySnapshot"
        )

    target_path = Path(path)
    payload = optional_exports_snapshot_to_json(snapshot, indent=indent)
    target_path.write_text(payload, encoding=encoding)


def optional_exports_snapshot_from_file(
    path: PathInput,
    *,
    encoding: str = "utf-8",
) -> OptionalExportRegistrySnapshot:
    """Wczytaj migawkę lazy-eksportów z pliku JSON."""

    target_path = Path(path)
    contents = target_path.read_text(encoding=encoding)
    return optional_exports_snapshot_from_json(contents)


def restore_optional_exports(
    snapshot: OptionalExportRegistrySnapshot,
    *,
    reload_cached: bool = False,
) -> None:
    """Przywróć stan rejestru lazy-eksportów na podstawie migawki."""

    if not isinstance(snapshot, OptionalExportRegistrySnapshot):
        raise TypeError(
            "snapshot must be an instance of OptionalExportRegistrySnapshot"
        )

    target_registry = dict(snapshot.registered)
    target_names = set(target_registry)

    missing_builtin = sorted(
        name for name in _BASE_OPTIONAL_EXPORTS if name not in target_names
    )
    if missing_builtin:
        raise ValueError(
            "Snapshot is missing builtin optional exports: "
            + ", ".join(missing_builtin)
        )

    current_names = set(_LAZY_OPTIONAL_EXPORTS)
    names_to_remove = current_names - target_names

    for name in names_to_remove:
        _LAZY_OPTIONAL_EXPORTS.pop(name, None)
        globals().pop(name, None)
        if name in __all__ and name not in target_names:
            try:
                __all__.remove(name)
            except ValueError:  # pragma: no cover - defensywna ochrona
                pass

    _LAZY_OPTIONAL_EXPORTS.clear()
    _LAZY_OPTIONAL_EXPORTS.update(target_registry)

    for name in target_names:
        if name not in __all__:
            __all__.append(name)

    cached_target = set(snapshot.cached_names)

    for name in target_names:
        if name not in cached_target:
            globals().pop(name, None)

    if reload_cached:
        for name in cached_target:
            try:
                _load_optional_export(name, cache=True)
            except AttributeError as exc:
                status = _probe_optional_export(name)
                raise OptionalExportUnavailableError(status) from exc


def probe_optional_export(name: str) -> OptionalExportStatus:
    """Zwróć diagnostykę pojedynczego lazy-eksportu."""

    return _probe_optional_export(name)


def __dir__() -> list[str]:  # pragma: no cover - prosta introspekcja
    """Zwraca listę dostępnych symboli wraz z lazy eksportami.

    Dzięki temu narzędzia interaktywne oraz autouzupełnianie zobaczą nazwy,
    które mogą zostać dociągnięte przez :func:`__getattr__`, nawet jeśli nie
    zostały jeszcze zmaterializowane w module.
    """

    current = set(globals())
    current.update(_LAZY_OPTIONAL_EXPORTS)
    return sorted(current)


__all__.extend(
    [
        "is_optional_export_available",
        "list_optional_exports",
        "describe_optional_exports",
        "probe_optional_export",
        "OptionalExportStatus",
        "OptionalExportUnavailableError",
        "require_optional_export",
        "get_optional_export",
        "refresh_optional_export",
        "is_optional_export_cached",
        "evict_optional_export",
        "ensure_optional_exports",
        "OptionalExportRegistryDiff",
        "OptionalExportRegistrySnapshot",
        "diff_optional_exports_snapshots",
        "optional_exports_diff_to_dict",
        "optional_exports_diff_from_dict",
        "optional_exports_diff_to_json",
        "optional_exports_diff_from_json",
        "optional_exports_diff_to_file",
        "optional_exports_diff_from_file",
        "format_optional_export_status",
        "format_optional_exports_summary",
        "format_optional_exports_diff",
        "log_optional_export_status",
        "log_optional_exports_summary",
        "log_optional_exports_diff",
        "configure_optional_exports_logging",
        "configure_optional_exports_logging_from_parsed_spec",
        "configure_optional_exports_logging_from_spec",
        "configure_optional_exports_logging_from_dict",
        "configure_optional_exports_logging_from_file",
        "configure_optional_exports_logging_from_python",
        "configure_optional_exports_logging_from_env",
        "get_optional_exports_logger",
        "set_optional_exports_logger",
        "temporary_optional_exports_logger",
        "OptionalExportsLoggingSpec",
        "parse_optional_exports_logging_spec",
        "optional_exports_snapshot_to_dict",
        "optional_exports_snapshot_from_dict",
        "optional_exports_snapshot_to_json",
        "optional_exports_snapshot_from_json",
        "optional_exports_snapshot_to_file",
        "optional_exports_snapshot_from_file",
        "snapshot_optional_exports",
        "restore_optional_exports",
        "register_optional_export",
        "unregister_optional_export",
        "temporary_optional_export",
    ]
)


def get_optional_export(
    name: str,
    default: object = _MISSING,
    *,
    cache: bool = True,
):
    """Spróbuj załadować lazy-eksport zwracając wartość domyślną, gdy jest niedostępny.

    Funkcja działa podobnie do :func:`require_optional_export`, ale zamiast
    zgłaszać błąd przy niedostępnej zależności, może zwrócić podany `default`.
    Przydatne w ścieżkach, gdzie komponent runtime jest opcjonalny i można
    kontynuować działanie z innym mechanizmem.
    """

    if name not in _LAZY_OPTIONAL_EXPORTS:
        raise ValueError(
            f"'{name}' nie jest zarejestrowane jako opcjonalny eksport runtime"
        )

    try:
        return _load_optional_export(name, cache=cache)
    except AttributeError as exc:
        status = _probe_optional_export(name)
        if default is not _MISSING:
            return default
        raise OptionalExportUnavailableError(status) from exc


def refresh_optional_export(name: str, *, reload_module: bool = False):
    """Odśwież zcache'owany lazy-eksport i opcjonalnie przeładuj moduł źródłowy."""

    if name not in _LAZY_OPTIONAL_EXPORTS:
        raise ValueError(
            f"'{name}' nie jest zarejestrowane jako opcjonalny eksport runtime"
        )

    module_name, _ = _LAZY_OPTIONAL_EXPORTS[name]
    previous_value = globals().pop(name, _MISSING)

    try:
        if reload_module:
            try:
                module = import_module(module_name)
            except Exception as exc:
                status = _probe_optional_export(name)
                if previous_value is not _MISSING:
                    globals()[name] = previous_value
                raise OptionalExportUnavailableError(status) from exc

            try:
                reload(module)
            except Exception as exc:
                status = _probe_optional_export(name)
                if previous_value is not _MISSING:
                    globals()[name] = previous_value
                raise OptionalExportUnavailableError(status) from exc

        value = _load_optional_export(name, cache=True)
    except AttributeError as exc:
        status = _probe_optional_export(name)
        if previous_value is not _MISSING:
            globals()[name] = previous_value
        raise OptionalExportUnavailableError(status) from exc

    return value


def ensure_optional_exports(
    names: Iterable[str],
    *,
    require_all: bool = False,
    cache: bool = True,
) -> tuple[dict[str, object], dict[str, OptionalExportStatus]]:
    """Spróbuj załadować wiele lazy-eksportów jednocześnie.

    Funkcja zwraca dwie mapy: pierwsza zawiera poprawnie załadowane symbole,
    druga – statusy brakujących lub niedostępnych eksportów. Gdy ``require_all``
    jest ustawione na ``True``, napotkanie niedostępnego eksportu powoduje
    natychmiastowe zgłoszenie :class:`OptionalExportUnavailableError`.

    Parametr ``cache`` jest przekazywany do :func:`_load_optional_export` i
    pozwala kontrolować, czy uzyskane wartości mają pozostać zcache'owane w
    module.
    """

    loaded: dict[str, object] = {}
    missing: dict[str, OptionalExportStatus] = {}

    for name in names:
        if name not in _LAZY_OPTIONAL_EXPORTS:
            raise ValueError(
                f"'{name}' nie jest zarejestrowane jako opcjonalny eksport runtime"
            )

        try:
            loaded[name] = _load_optional_export(name, cache=cache)
        except AttributeError as exc:
            status = _probe_optional_export(name)
            missing[name] = status
            if require_all:
                raise OptionalExportUnavailableError(status) from exc

    return loaded, missing


def is_optional_export_cached(name: str) -> bool:
    """Sprawdź, czy zadany lazy-eksport jest obecnie zcache'owany w module."""

    if name not in _LAZY_OPTIONAL_EXPORTS:
        raise ValueError(
            f"'{name}' nie jest zarejestrowane jako opcjonalny eksport runtime"
        )

    return name in globals()


def evict_optional_export(name: str) -> bool:
    """Usuń zcache'owaną wartość lazy-eksportu, jeśli istnieje."""

    if name not in _LAZY_OPTIONAL_EXPORTS:
        raise ValueError(
            f"'{name}' nie jest zarejestrowane jako opcjonalny eksport runtime"
        )

    previous = globals().pop(name, _MISSING)
    return previous is not _MISSING
