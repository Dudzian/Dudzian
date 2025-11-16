# Runtime UI

PySide6 klient korzysta z `ui/backend/runtime_service.py`, aby udostępnić QML
aktualny stan decyzji AI, blokad ryzyka, alertów i kanałów gRPC. Serwis
kapsułkuje komunikację z `bot_core.runtime` oraz zapewnia sygnały Qt, które można
podpinać w widokach blur/FontAwesome.

## RuntimeService – najważniejsze funkcje

* `refreshDecisions()` i `decisionsChanged` pobierają najnowsze decyzje AI z
  backendu i udostępniają je panelowi Explainability oraz kreatorom trybów.
* `requestUnblock()` i `requestFreeze()` umożliwiają operatorowi zarządzanie
  blokadami/override’ami bezpośrednio z UI – kontroler samodzielnie loguje
  akcje oraz odświeża stan modeli.【F:ui/backend/runtime_service.py†L1458-L1625】
* `activitySummary()` normalizuje liczbę blokad, zamrożeń i override’ów dla
  poszczególnych profili i udostępnia dane w strukturze przyjaznej QML. Widok
  ryzyka może dzięki temu kolorować karty zgodnie z aktywnością profilu.【F:ui/backend/runtime_service.py†L385-L570】
* `RuntimeService` emituje sygnały `alertsChanged`, `riskStateChanged`,
  `licensingChanged` i `cloudRuntimeStatusChanged`, co upraszcza spójne
  aktualizacje paneli statusu i powiadomień.

## Telemetria i pojedyncza instancja

UI przekazuje naruszenia polityki single-instance do backendu poprzez
`bot_core.security.fingerprint.report_single_instance_event`. Helper dostępny w
CLI (`python -m bot_core.security.fingerprint report-single-instance ...`)
wysyła zdarzenie `ui_single_instance_conflict`, które trafia do
`logs/security_admin.log`. Dzięki temu operatorzy wiedzą, kiedy druga kopia UI
próbuje wystartować na tym samym hostcie.【F:bot_core/security/fingerprint.py†L1439-L1497】

## Integracja z testami

* `tests/ui/test_single_instance.py` sprawdza, że funkcja
  `report_single_instance_event` zapisuje poprawny payload i jest wywoływana z
  CLI.
* `tests/ui_pyside/test_app_bootstrap.py` startuje PySide6 w trybie offscreen i
  potwierdza, że `RuntimeService` oraz pozostałe kontrolery zostały poprawnie
  zarejestrowane w QML.
* `tests/ui_pyside/test_ai_decisions_view.py` weryfikuje rendering widoku
  „Decyzje AI” bez potrzeby łączenia się z backendem gRPC; snapshot danych
  obejmuje rekomendowane tryby, historię decyzji oraz telemetry, dzięki czemu
  test wychwytuje regresje modeli QML.【F:tests/ui_pyside/test_ai_decisions_view.py†L1-L74】

## Widok „Decyzje AI”

Nowy panel Qt Quick (`ui/pyside_app/qml/views/AiDecisionsView.qml`) prezentuje
ostatnią decyzję governora, rekomendowane tryby i timeline historii wraz z
confidence/telemetry. Widok wykorzystuje karty blur (MultiEffect), ikony
FontAwesome przypisane do trybów oraz filtry rekomendacji. Dane pochodzą z
`RuntimeService.aiGovernorSnapshot`, który subskrybuje kanał gRPC
`AutoTraderAIGovernor` i aktualizuje właściwości `history`, `telemetry` oraz
`recommendedModes`.【F:ui/pyside_app/qml/views/AiDecisionsView.qml†L1-L236】【F:ui/backend/runtime_service.py†L855-L999】

### Procedura QA (manualna)

1. Uruchom `python -m ui.pyside_app --demo` aby wystartować aplikację w trybie
   demo – `RuntimeService` zasili widok snapshotem referencyjnym.
2. Przejdź do panelu „Decyzje AI”; sprawdź, że nagłówek prezentuje bieżący tryb
   oraz przycisk „Odśwież” wywołuje `reloadAiGovernorSnapshot()`.
3. Klikaj kapsuły rekomendowanych trybów i potwierdź, że timeline filtruje się
   względem wybranego tokena (właściwość `selectedModeFilter`).
4. Zweryfikuj, że telemetry (`confidence`, `riskScore`, `cycle_latency_p95_ms`)
   aktualizują się po zmianie snapshotu – wartości powinny odpowiadać danym z
   feedu gRPC lub demo snapshotu.

### Artefakty

- `docs/benchmark/cryptohopper_comparison.md` zawiera opis przewagi UI względem
  CryptoHopper i link do widoku „Decyzje AI”.
- `docs/ui/images/pyside6_blur_dashboard.svg` można wykorzystać jako
  placeholder w materiałach wizualnych do czasu przygotowania nowego zrzutu
  przedstawiającego timeline AI.
