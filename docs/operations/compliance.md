# Operacyjna obsługa audytu zgodności

Niniejszy dokument opisuje sposób uruchamiania lokalnego audytu zgodności
(KYC/AML) oraz integrację z harmonogramem w runtime. Audyt działa w oparciu o
moduł `core.compliance.compliance_auditor.ComplianceAuditor` i generuje
raporty Markdown/JSON przy użyciu `core.reporting.compliance_reporter`.

## Ręczne uruchomienie audytu (CLI)

Do jednorazowego audytu służy skrypt `scripts/run_compliance_audit.py`:

```bash
python scripts/run_compliance_audit.py \
  --strategy data/strategy.yaml \
  --transactions data/transactions.json \
  --kyc-profile data/kyc.yaml \
  --data-sources data/sources.yml \
  --report-dir reports/compliance/manual
```

Najważniejsze opcje:

- `--audit-config` – ścieżka do pliku `config/compliance/audit.yml` z regułami.
- `--strategy`, `--kyc-profile`, `--transactions`, `--data-sources` – opcjonalne
  pliki JSON/YAML. Brak któregokolwiek powoduje użycie bezpiecznych wartości
  domyślnych.
- `--fail-on-findings` – jeśli ustawione, skrypt zwraca kod wyjścia `2` przy
  wykryciu naruszeń.

Artefakty raportu trafiają do katalogu wskazanego parametrem `--report-dir` i
zawierają zarówno plik `.json`, jak i `.md` z tym samym znacznikiem czasu.

## Harmonogram automatyczny

Okresowe uruchamianie audytu zapewnia moduł
`core.runtime.compliance_scheduler.ComplianceScheduler`. Konfiguracja znajduje
się w `config/compliance/schedule.yml` i udostępnia parametry:

```yaml
enabled: true
interval_hours: 12
window:
  start: "06:00"
  end: "22:00"
```

- `enabled` – globalne włączenie/wyłączenie harmonogramu.
- `interval_hours`, `interval_minutes`, `interval_seconds` – odstęp pomiędzy
  kolejnymi audytami.
- `window.start`, `window.end` – dozwolone okno czasowe (w formacie HH:MM).

Scheduler publikuje zdarzenie `ComplianceAuditCompleted` oraz aktualizuje
metryki guardrail’i, dzięki czemu panel telemetry UI prezentuje wyniki audytu.

## Interpretacja raportu

Raport Markdown zawiera trzy sekcje:

1. **Podsumowanie** – status audytu, kontekst strategii i liczba naruszeń per
   poziom istotności.
2. **Naruszenia** – tabela reguł (`rule_id`) wraz z opisem i metadanymi.
3. **Rekomendacje** – lista sugerowanych działań korygujących.

Brak naruszeń oznacza, że strategia spełnia lokalne wymagania KYC/AML i można
przejść do kolejnych etapów (paper/live). W przypadku naruszeń zaleca się
wykorzystanie panelu runbooków oraz dokumentu `docs/compliance/audit.md`, który
opisuje znaczenie poszczególnych reguł.
