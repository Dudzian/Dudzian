"""Silnik optymalizacji parametrów strategii oraz harmonogram."""
from __future__ import annotations

import itertools
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from bot_core.config.models import StrategyOptimizationTaskConfig
from bot_core.strategies.base import StrategyEngine
from bot_core.strategies.catalog import StrategyCatalog, StrategyDefinition

try:  # pragma: no cover - opcjonalna zależność
    import optuna
except Exception:  # pragma: no cover - środowiska bez optuna
    optuna = None  # type: ignore[assignment]


StrategyEvaluator = Callable[[StrategyEngine, Mapping[str, Any]], tuple[float, Mapping[str, Any]] | float]


@dataclass(slots=True)
class OptimizationTrial:
    """Pojedyncza próba optymalizacji z wynikiem."""

    parameters: Mapping[str, Any]
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyOptimizationReport:
    """Podsumowanie sesji optymalizacji strategii."""

    strategy: str
    engine: str
    algorithm: str
    objective: str
    goal: str
    started_at: datetime
    completed_at: datetime
    trials: Sequence[OptimizationTrial]
    best: OptimizationTrial
    dataset: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.completed_at - self.started_at).total_seconds())


@dataclass(slots=True)
class _ScheduledTask:
    """Opis zarejestrowanego zadania optymalizacji."""

    config: StrategyOptimizationTaskConfig
    definition: StrategyDefinition
    evaluator: StrategyEvaluator
    algorithm: str
    objective: str
    goal: str
    search_grid: Mapping[str, Sequence[Any]]
    search_bounds: Mapping[str, tuple[float, float, float | None]]
    max_trials: int
    random_seed: int | None
    cadence_seconds: float
    jitter_seconds: float
    next_run: float
    dataset: str | None
    tags: tuple[str, ...]
    last_report: StrategyOptimizationReport | None = None
    running: bool = False


class StrategyOptimizer:
    """Przeprowadza optymalizację parametrów strategii korzystając z katalogu."""

    def __init__(self, catalog: StrategyCatalog, *, logger: logging.Logger | None = None) -> None:
        self._catalog = catalog
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        *,
        base_definition: StrategyDefinition,
        algorithm: str,
        objective: str,
        goal: str,
        search_grid: Mapping[str, Sequence[Any]] | None,
        search_bounds: Mapping[str, tuple[float, float, float | None]] | None,
        max_trials: int,
        evaluator: StrategyEvaluator,
        dataset: str | None = None,
        tags: Sequence[str] = (),
        random_seed: int | None = None,
    ) -> StrategyOptimizationReport:
        """Uruchamia optymalizację i zwraca raport końcowy."""

        normalized_algorithm = (algorithm or "grid").strip().lower()
        normalized_goal = (goal or "maximize").strip().lower()
        if normalized_goal not in {"maximize", "minimize"}:
            normalized_goal = "maximize"

        trials: list[OptimizationTrial] = []
        started_at = datetime.now(timezone.utc)
        rng = random.Random(random_seed)
        best_trial: OptimizationTrial | None = None
        best_score: float | None = None
        maximize = normalized_goal == "maximize"

        candidates = list(
            self._iter_candidates(
                base_definition,
                normalized_algorithm,
                search_grid=search_grid or {},
                search_bounds=search_bounds or {},
                max_trials=max(1, int(max_trials)),
                rng=rng,
            )
        )
        if not candidates:
            candidates.append({})

        for candidate in candidates:
            parameters = dict(base_definition.parameters)
            parameters.update(candidate)
            try:
                definition = self._clone_definition(base_definition, parameters)
                engine = self._catalog.create(definition)
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                self._logger.debug("Nie udało się zainicjalizować silnika strategii", exc_info=True)
                score = -math.inf if maximize else math.inf
                metadata = {"error": str(exc), "stage": "instantiate"}
            else:
                score, metadata = self._safe_evaluate(
                    engine,
                    parameters,
                    evaluator,
                    maximize=maximize,
                )
            trial = OptimizationTrial(parameters=parameters, score=score, metadata=metadata)
            trials.append(trial)
            if best_score is None:
                best_trial = trial
                best_score = score
            else:
                if maximize and score > best_score:
                    best_trial = trial
                    best_score = score
                elif not maximize and score < best_score:
                    best_trial = trial
                    best_score = score

        completed_at = datetime.now(timezone.utc)
        if best_trial is None:
            best_trial = trials[-1]
        report = StrategyOptimizationReport(
            strategy=base_definition.name,
            engine=base_definition.engine,
            algorithm=normalized_algorithm,
            objective=objective,
            goal=normalized_goal,
            started_at=started_at,
            completed_at=completed_at,
            trials=tuple(trials),
            best=best_trial,
            dataset=dataset,
            tags=tuple(tags),
        )
        return report

    # ------------------------------------------------------------------
    # Wewnętrzne helpery
    # ------------------------------------------------------------------

    def _clone_definition(
        self, base: StrategyDefinition, parameters: Mapping[str, Any]
    ) -> StrategyDefinition:
        metadata = dict(base.metadata)
        return StrategyDefinition(
            name=base.name,
            engine=base.engine,
            license_tier=base.license_tier,
            risk_classes=tuple(base.risk_classes),
            required_data=tuple(base.required_data),
            parameters=dict(parameters),
            risk_profile=base.risk_profile,
            tags=tuple(base.tags),
            metadata=metadata,
        )

    def _safe_evaluate(
        self,
        engine: StrategyEngine,
        parameters: Mapping[str, Any],
        evaluator: StrategyEvaluator,
        *,
        maximize: bool,
    ) -> tuple[float, Mapping[str, Any]]:
        try:
            result = evaluator(engine, parameters)
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            self._logger.debug("Wyjątek podczas ewaluacji kandydata", exc_info=True)
            score = -math.inf if maximize else math.inf
            metadata: Mapping[str, Any] = {"error": str(exc), "stage": "evaluate"}
            return score, metadata
        if isinstance(result, tuple):
            score, metadata = result
            metadata = dict(metadata)
        else:
            score = float(result)
            metadata = {}
        if not math.isfinite(score):
            score = -math.inf if maximize else math.inf
            metadata.setdefault("warning", "score not finite")
        return float(score), metadata

    def _iter_candidates(
        self,
        base_definition: StrategyDefinition,
        algorithm: str,
        *,
        search_grid: Mapping[str, Sequence[Any]],
        search_bounds: Mapping[str, tuple[float, float, float | None]],
        max_trials: int,
        rng: random.Random,
    ) -> Iterable[Mapping[str, Any]]:
        base_params = dict(base_definition.parameters)
        grid_items = {
            key: tuple(dict.fromkeys(value))
            for key, value in search_grid.items()
            if value
        }
        if algorithm == "grid":
            # Rozszerz zakresy ciągłe na siatkę jeśli zdefiniowano krok.
            for key, (low, high, step) in search_bounds.items():
                if step in (None, 0):
                    continue
                count = max(1, int(round((high - low) / float(step))))
                values = [low + idx * float(step) for idx in range(count + 1)]
                grid_items.setdefault(key, tuple(values))
            if not grid_items:
                yield {}
                return
            keys = list(grid_items.keys())
            for combo in itertools.product(*(grid_items[key] for key in keys)):
                yield {key: value for key, value in zip(keys, combo)}
            return

        # Algorytm bayesowski / adaptacyjny.
        continuous_items = list(search_bounds.items())
        categorical_items = list(grid_items.items())
        if algorithm == "bayesian" and optuna is not None:
            study = optuna.create_study(direction="maximize")

            def _objective(trial: "optuna.Trial") -> float:
                candidate: dict[str, Any] = {}
                for key, (low, high, step) in continuous_items:
                    if step in (None, 0):
                        candidate[key] = trial.suggest_float(key, low, high)
                    else:
                        count = max(1, int(round((high - low) / float(step))))
                        candidate[key] = trial.suggest_float(key, low, high, step=abs(float(step)))
                for key, choices in categorical_items:
                    candidate[key] = trial.suggest_categorical(key, list(choices))
                # Optuna sama wywołuje ewaluację, tu jedynie zwracamy placeholder.
                # Faktyczna ewaluacja następuje poza optuna w pętli optimize().
                return 0.0

            # Próbki generowane przez optuna – wykorzystamy sampler do wylosowania kandydatów.
            sampler = study.sampler
            for index in range(max_trials):
                trial_id = study._storage.create_new_trial(study._study_id)  # type: ignore[attr-defined]
                trial = optuna.trial.Trial(study, trial_id)
                candidate: dict[str, Any] = {}
                for key, (low, high, step) in continuous_items:
                    if step in (None, 0):
                        value = sampler.sample_float(study, trial, key, low, high)
                    else:
                        value = sampler.sample_float(study, trial, key, low, high, step=abs(float(step)))
                    candidate[key] = value
                for key, choices in categorical_items:
                    if choices:
                        candidate[key] = sampler.sample_categorical(study, trial, key, list(choices))
                yield candidate
            return

        # Fallback adaptacyjny: początkowe próby losowe, potem zawężanie wokół najlepszego.
        for idx in range(max_trials):
            candidate: dict[str, Any] = {}
            for key, (low, high, step) in continuous_items:
                if step not in (None, 0):
                    count = max(1, int(round((high - low) / float(step))))
                    index = idx % (count + 1)
                    candidate[key] = low + index * float(step)
                else:
                    fraction = (idx + 0.5) / max(1, max_trials)
                    candidate[key] = low + fraction * (high - low)
            for key, choices in categorical_items:
                if choices:
                    candidate[key] = choices[idx % len(choices)]
                else:
                    candidate[key] = base_params.get(key)
            yield candidate

    # ------------------------------------------------------------------


class OptimizationScheduler:
    """Lekki harmonogram do uruchamiania zadań optymalizacyjnych."""

    def __init__(
        self,
        optimizer: StrategyOptimizer,
        *,
        clock: Callable[[], float] | None = None,
        time_factory: Callable[[], datetime] | None = None,
        report_directory: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._clock = clock or time.monotonic
        self._time_factory = time_factory or (lambda: datetime.now(timezone.utc))
        self._logger = logger or logging.getLogger(__name__)
        self._tasks: MutableMapping[str, _ScheduledTask] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._report_directory = Path(report_directory).expanduser() if report_directory else None

    # ------------------------------------------------------------------

    def add_task(
        self,
        *,
        config: StrategyOptimizationTaskConfig,
        definition: StrategyDefinition,
        evaluator: StrategyEvaluator,
        default_algorithm: str | None = None,
    ) -> None:
        algorithm = (config.algorithm or default_algorithm or "grid").strip().lower()
        objective_metric = getattr(config.objective, "metric", "score")
        goal = getattr(config.objective, "goal", "maximize")
        max_trials = max(1, int(getattr(config, "max_trials", 0) or 0))
        random_seed = getattr(config, "random_seed", None)
        schedule = getattr(config, "schedule", None)
        cadence_seconds = 0.0
        jitter_seconds = 0.0
        run_immediately = True
        if schedule is not None:
            cadence_seconds = float(getattr(schedule, "cadence_seconds", 0.0) or 0.0)
            jitter_seconds = max(0.0, float(getattr(schedule, "jitter_seconds", 0.0) or 0.0))
            run_immediately = bool(getattr(schedule, "run_immediately", True))
            delay = max(0.0, float(getattr(schedule, "start_delay_seconds", 0.0) or 0.0))
        else:
            delay = 0.0
        next_run = self._clock() + delay
        if run_immediately:
            next_run = self._clock()
        grid = {
            key: tuple(dict.fromkeys(values))
            for key, values in getattr(config.search_space, "grid", {}).items()
            if values
        }
        bounds: dict[str, tuple[float, float, float | None]] = {}
        for key, bound in getattr(config.search_space, "bounds", {}).items():
            try:
                low = float(getattr(bound, "lower", getattr(bound, "min", getattr(bound, "minimum"))))
                high = float(getattr(bound, "upper", getattr(bound, "max", getattr(bound, "maximum"))))
            except Exception:
                continue
            step_value = getattr(bound, "step", None)
            if step_value in (None, ""):
                step = None
            else:
                try:
                    step = float(step_value)
                except Exception:
                    step = None
            if high < low:
                low, high = high, low
            bounds[str(key)] = (low, high, step)
        dataset = getattr(config.evaluation, "dataset", None)
        tags = tuple(getattr(config, "tags", ()) or ())
        task = _ScheduledTask(
            config=config,
            definition=definition,
            evaluator=evaluator,
            algorithm=algorithm,
            objective=objective_metric,
            goal=goal,
            search_grid=grid,
            search_bounds=bounds,
            max_trials=max_trials,
            random_seed=random_seed,
            cadence_seconds=cadence_seconds,
            jitter_seconds=jitter_seconds,
            next_run=next_run,
            dataset=dataset,
            tags=tags,
        )
        with self._lock:
            self._tasks[config.name] = task

    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if not any(task.cadence_seconds > 0 for task in self._tasks.values()):
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="optimization-scheduler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=5.0)

    # ------------------------------------------------------------------

    def trigger(self, name: str | None = None) -> Sequence[StrategyOptimizationReport]:
        with self._lock:
            if name is not None:
                tasks = [self._tasks[name]] if name in self._tasks else []
            else:
                tasks = list(self._tasks.values())
        reports: list[StrategyOptimizationReport] = []
        for task in tasks:
            report = self._execute_task(task)
            if report is not None:
                reports.append(report)
        return tuple(reports)

    def get_last_report(self, name: str) -> StrategyOptimizationReport | None:
        with self._lock:
            task = self._tasks.get(name)
            return task.last_report if task else None

    # ------------------------------------------------------------------

    def has_tasks(self) -> bool:
        with self._lock:
            return bool(self._tasks)

    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = self._clock()
            executed = False
            with self._lock:
                tasks = list(self._tasks.values())
            for task in tasks:
                if task.cadence_seconds <= 0:
                    continue
                if task.running:
                    continue
                if now < task.next_run:
                    continue
                if self._execute_task(task) is not None:
                    executed = True
            if not executed:
                self._stop_event.wait(0.5)

    def _execute_task(self, task: _ScheduledTask) -> StrategyOptimizationReport | None:
        with self._lock:
            if task.running:
                return None
            task.running = True
        report: StrategyOptimizationReport | None = None
        try:
            report = self._optimizer.optimize(
                base_definition=task.definition,
                algorithm=task.algorithm,
                objective=task.objective,
                goal=task.goal,
                search_grid=task.search_grid,
                search_bounds=task.search_bounds,
                max_trials=task.max_trials,
                evaluator=task.evaluator,
                dataset=task.dataset,
                tags=task.tags,
                random_seed=task.random_seed,
            )
        except Exception:  # pragma: no cover - logika awaryjna
            self._logger.exception(
                "Błąd podczas wykonywania zadania optymalizacji %s",
                task.config.name,
            )
        else:
            if report is not None and self._report_directory is not None:
                self._export_report(report)
        finally:
            with self._lock:
                task.running = False
                if task.cadence_seconds > 0:
                    jitter = random.uniform(-task.jitter_seconds, task.jitter_seconds)
                    task.next_run = self._clock() + max(0.0, task.cadence_seconds + jitter)
                if report is not None:
                    task.last_report = report
        return report

    def _export_report(self, report: StrategyOptimizationReport) -> None:
        try:
            from bot_core.reporting.optimization import export_report  # import lokalny

            export_report(report, self._report_directory)
        except Exception:  # pragma: no cover - raportowanie nie powinno zatrzymywać harmonogramu
            self._logger.debug("Nie udało się zapisać raportu optymalizacji", exc_info=True)

__all__ = [
    "OptimizationScheduler",
    "OptimizationTrial",
    "StrategyOptimizationReport",
    "StrategyOptimizer",
]
