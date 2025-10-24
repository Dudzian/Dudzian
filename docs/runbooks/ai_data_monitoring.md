# Runbook: Monitoring danych AI

Runbook opisuje monitorowanie jakości danych oraz dryfu cech dla pipeline'u AI. Monitoring opiera się na helperach z modułu `bot_core.ai.monitoring` oraz raportach audytu zapisywanych w `audit/ai_decision/{data_quality,drift}`.

> **Wybór modułu:**
> - Użyj `bot_core.ai.monitoring`, gdy działasz w pipeline'ie treningowym/batchowym i pracujesz na ramkach Pandas – eksporty `DataCompletenessWatcher` oraz `FeatureBoundsValidator` pochodzą bezpośrednio z tego modułu.
> - Użyj `bot_core.ai.data_monitoring` (lub aliasów `bot_core.ai.InferenceDataCompletenessWatcher`/`bot_core.ai.InferenceFeatureBoundsValidator`), gdy monitorujesz cechy w trakcie inference on-line. Ta wersja zapisuje raporty JSON, potrafi wymuszać blokadę scoringu (`policy.enforce`) i pracuje na słownikach cech.
> - Jeśli preferujesz bezpośrednie importy modułu inference, użyj `from bot_core.ai.data_monitoring import DataCompletenessWatcher as InferenceDataCompletenessWatcher` (analogicznie dla `FeatureBoundsValidator`).

## 1. Szybka checklista

1. **Kompletność świec** – uruchom `DataCompletenessWatcher` (częstotliwość np. `1min`) na ramce OHLCV z ostatniej doby. Sprawdź `assessment.summary.status` oraz `summary.total_gaps`. Jeśli `status == "critical"`, eskaluj do zespołu danych i wstrzymaj inference dla symbolu.
2. **Dryf cech** – porównaj zbiór treningowy z oknem produkcyjnym za pomocą `FeatureDriftAnalyzer`. Zweryfikuj `assessment.summary.triggered_features`, `metrics.feature_drift.psi` oraz `distribution_summary.max_ks`. Trigger > progów PSI/KS wymaga wpisu w decision journalu oraz przeglądu risk/compliance.
3. **Zakres cech inference** – podczas obsługi alertu sprawdź `InferenceFeatureBoundsValidator.observe()` dla ostatnich próbek inference. Jeśli którekolwiek `feature_out_of_bounds`, zatrzymaj scoring i potwierdź wznowienie z risk.
4. **Automatyczny audyt pipeline'u** – zweryfikuj, że `AIManager.get_last_data_quality_report_path()` wskazuje świeży raport (`pipeline:<symbol>:<kontrola>`) po ostatnim `run_pipeline`. Brak raportu oznacza, że kontrola nie została zarejestrowana.
5. **Bramka podpisów** – przed promocją modelu do DecisionOrchestratora uruchom `ensure_compliance_sign_offs(...)`, aby potwierdzić, że ostatnie alerty `data_quality` i `drift` mają status `approved/waived` dla ról Risk i Compliance. Jeśli wymagany jest inny zestaw odpowiedzialności, ustaw go wcześniej przez `AIManager.set_compliance_sign_off_roles(("risk",))` lub podobną konfigurację i włącz wymóg podpisów (`AIManager.set_compliance_sign_off_requirement(True)`).

## 2. Procedura operacyjna

### 2.1 DataCompletenessWatcher

```python
from datetime import timedelta

import pandas as pd

from bot_core.ai.monitoring import DataCompletenessWatcher

df = load_ohlcv("BTCUSDT", hours=24)
watcher = DataCompletenessWatcher(timedelta(minutes=1))
assessment = watcher.assess(df)

if assessment.status != "ok":
    manager.record_data_quality_issues(
        assessment,
        dataset=build_feature_dataset(df),
        job_name="btc-data-quality",
        source="ohlcv-monitor",
    )
```

Najważniejsze pola raportu: `summary.total_gaps`, `summary.missing_ratio`, `issues[*].details.missing_bars`. Eskaluj, gdy `missing_ratio >= 0.05`.

### 2.2 FeatureDriftAnalyzer

```python
from bot_core.ai.monitoring import FeatureDriftAnalyzer

baseline = load_training_frame("BTCUSDT", span_days=30)
production = load_recent_frame("BTCUSDT", span_days=3)
analyzer = FeatureDriftAnalyzer(psi_threshold=0.1, ks_threshold=0.1)
assessment = analyzer.compare(baseline, production)

if assessment.triggered:
    manager.record_data_quality_issues(
        {"code": "feature_drift_alert", "features": assessment.summary["triggered_features"]},
        job_name="btc-feature-monitor",
        source="drift-monitor",
        tags=("drift", "psi"),
        summary=assessment.summary,
    )
```

Raport dryfu z pipeline'u (`_persist_drift_report`) zawiera te same dane pod kluczami `metrics.feature_drift.{score,psi,ks}` oraz `metrics.features`.

### 2.3 FeatureBoundsValidator

```python
from bot_core.ai.monitoring import FeatureBoundsValidator

validator = FeatureBoundsValidator(sigma_multiplier=3.0)
issues = validator.validate(latest_features, latest_scalers)

if issues:
    raise RuntimeError("Feature bounds exceeded", issues)
```

Wpisz incydent, jeśli którekolwiek odchylenie przekracza `3σ` i zarchiwizuj wynik w `audit/ai_decision/data_quality/`.

### 2.4 Integracja z AIManager

`AIManager` może automatycznie uruchamiać kontrole jakości danych po każdym `run_pipeline`. Zarejestruj kontrolę za pomocą `DataQualityCheck`:

```python
from bot_core.ai import AIManager, DataQualityCheck
from bot_core.ai.monitoring import DataCompletenessWatcher

manager = AIManager(audit_root="audit/ai_decision")
watcher = DataCompletenessWatcher("1min", warning_gap_ratio=0.0)

manager.register_data_quality_check(
    DataQualityCheck(
        name="completeness",
        callback=lambda frame: watcher.assess(frame),
        tags=("completeness", "pipeline"),
        source="ohlcv-monitor",
    )
)

await manager.run_pipeline("BTCUSDT", df, ["alpha"], seq_len=3, folds=2)
```

Po wykonaniu pipeline'u raport znajdziesz w `audit/ai_decision/data_quality/<timestamp>.json` pod nazwą `pipeline:btcusdt:completeness`. Helper `manager.get_data_quality_checks()` zwraca aktualnie aktywne kontrole, a `clear_data_quality_checks()` pozwala szybko wyłączyć monitoring podczas testów.

Do walidacji podpisów compliance użyj helperów inference:

```python
from bot_core.ai.data_monitoring import (
    ComplianceSignOffError,
    ensure_compliance_sign_offs,
    load_recent_data_quality_reports,
    load_recent_drift_reports,
)

try:
    pending = ensure_compliance_sign_offs(
        data_quality_reports=load_recent_data_quality_reports(limit=5),
        drift_reports=load_recent_drift_reports(limit=5),
        roles=("risk", "compliance"),  # możesz ograniczyć wymagane role
    )
except ComplianceSignOffError as exc:
    # exc.missing zawiera tylko role z brakami podpisów
    for role, entries in exc.missing.items():
        print("Brak podpisu", role, "dla raportów:")
        for entry in entries:
            print(" -", entry.get("category"), entry.get("report_path"))
    raise
else:
    assert all(not entries for entries in pending.values())
```
`ComplianceSignOffError` sygnalizuje brak wymaganych podpisów – aktualizuj `docs/compliance/ai_pipeline_signoff.md` i eskaluj do właścicieli raportu.

Włącz wymóg podpisów i – opcjonalnie – zawęź wymagane role przed aktywacją inference:

```python
manager.set_compliance_sign_off_requirement(True)
manager.set_compliance_sign_off_roles(("risk",))  # wymaga tylko podpisu zespołu Risk
```

Ustawienie `None` przywraca domyślny zestaw (`risk`, `compliance`), a `False` w `set_compliance_sign_off_requirement` ponownie wyłącza bramkę.

Jeśli musisz wymusić blokadę w samych podsumowaniach, helpery `summarize_data_quality_reports(..., require_sign_off=True)` oraz
`summarize_drift_reports(..., require_sign_off=True)` zgłoszą ten sam wyjątek, gdy dla wskazanych ról pozostaną otwarte zadania.

### 2.5 Monitoring inference

Helpery inference działają na słownikach cech i raportach JSON z `audit/ai_decision`. Korzystają z aliasów eksportowanych przez `bot_core.ai`.

```python
from bot_core.ai import (
    DecisionModelInference,
    InferenceDataCompletenessWatcher,
    InferenceFeatureBoundsValidator,
    score_with_data_monitoring,
)

watcher = InferenceDataCompletenessWatcher()
watcher.configure(["alpha", "beta"])

bounds = InferenceFeatureBoundsValidator()
bounds.configure({"ratio": {"min": 0.0, "max": 1.0}})

report = watcher.observe({"alpha": 1.0, "beta": None}, context={"run": "nightly"})

inference = DecisionModelInference(repository)
inference.load_weights("latest.json")  # konfiguruje watchery na podstawie metadanych artefaktu

score_with_data_monitoring(
    inference,
    latest_features,
    context={"run": "nightly"},
)
```

Raporty inference zawierają pola `policy.enforce`, `sign_off` oraz ścieżkę do pliku (`report_path`). W przypadku alertu `DataQualityException` obejmuje jednocześnie raporty kompletności i zakresów.

### 2.6 Decision journal

Każdy zapis raportu (`save_data_quality_report`, `save_drift_report`) generuje zdarzenie w `TradingDecisionJournal`:

- `event="ai_data_quality_report"` – metadane: `report_path`, `issues_count`, `status`, `tags`, `schedule`.
- `event="ai_drift_report"` – metadane: `report_path`, `feature_drift`, `volatility_shift`, `threshold`, `triggered`, `triggered_features`.

Ustaw `AIManager(..., decision_journal=JsonlTradingDecisionJournal(...))`, aby logi trafiały do pliku JSONL lub wykorzystaj `InMemoryTradingDecisionJournal` w testach. Zespół operacyjny sprawdza, czy w dzienniku pojawił się wpis z aktualnym `report_path` oraz poprawnym `environment/portfolio` zgodnie z kontekstem (`decision_journal_context`). Brak zdarzenia blokuje eskalację alertu i wymaga ręcznej weryfikacji pipeline'u.

## 3. Eskalacja

- **Status critical** w `DataCompletenessWatcher` lub `FeatureDriftAnalyzer` → zgłoszenie do #sec-alerts, wstrzymanie inference oraz aktualizacja decision journalu.
- **Potwierdzony dryf danych / incydent compliance** → wykonaj procedurę [`docs/runbooks/ai_model_rollback.md`](ai_model_rollback.md) przed ponowną aktywacją inference.
- **Feature out of bounds** → blokada scoringu do czasu przywrócenia stabilności cech, podpis risk przed wznowieniem.

## 4. Artefakty

- Raporty data-quality: `audit/ai_decision/data_quality/<timestamp>.json` (pola `issues`, `summary.status`, `summary.total_gaps`).
- Raporty dryfu: `audit/ai_decision/drift/<timestamp>.json` (`metrics.feature_drift.psi`, `metrics.features`, `distribution_summary`).
- Helpery API: `bot_core.ai.{DataCompletenessWatcher, FeatureDriftAnalyzer, FeatureBoundsValidator, AIManager.record_data_quality_issues}` dla pipeline'u oraz `bot_core.ai.{InferenceDataCompletenessWatcher, InferenceFeatureBoundsValidator, score_with_data_monitoring}` dla inference.

