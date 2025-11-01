# Audyt zgodności (KYC/AML)

Moduł `core.compliance.compliance_auditor.ComplianceAuditor` wykonuje audyt konfiguracji
strategii, źródeł danych i historii transakcji zgodnie z regułami zdefiniowanymi w pliku
`config/compliance/audit.yml`. Audyt działa lokalnie i służy do wychwytywania
najczęstszych naruszeń polityk KYC/AML, zanim strategia przejdzie do trybu live.

## Zakres audytu

Audyt obejmuje trzy grupy reguł:

1. **Kompletność KYC** – sprawdza, czy w profilu użytkownika znajdują się wszystkie
   wymagane pola (`kyc.required_fields`). Braki zgłaszane są jako naruszenie
   `KYC_MISSING_FIELDS` z poziomem istotności określonym w `kyc.severity`.
2. **Reguły AML** – weryfikują, czy profil lub transakcje nie są powiązane z krajami
   objętymi blokadą (`aml.blocked_countries`), jurysdykcjami wysokiego ryzyka
   (`aml.high_risk_jurisdictions`), zakazanymi źródłami danych czy tagami strategii
   (`aml.suspicious_tags`). Dodatkowo można ustawić maksymalny wolumen dla profili
   niezweryfikowanych (`aml.max_unverified_volume_usd`). Wszystkie naruszenia publikują
   alert `ComplianceViolation` dla guardrail’i.
3. **Limity transakcyjne** – kontrolują maksymalną wartość pojedynczej transakcji
   (`transaction_limits.max_single_trade_usd`) oraz skumulowany wolumen w zadanym oknie
   (`transaction_limits.max_daily_volume_usd`, `lookback_days`). Wykryte przekroczenia
   generują naruszenia `TX_SINGLE_LIMIT_EXCEEDED` lub
   `TX_DAILY_LIMIT_EXCEEDED`.

## Integracja z guardrailami

Każde naruszenie o randze `warning` lub wyższej powoduje publikację zdarzenia
`ComplianceViolation`. Guardrail’e rejestrują zdarzenia w metryce
`compliance_violation_total`, zapisują je w `logs/guardrails/events.log` oraz emitują
powiadomienia do UI (zdarzenie `compliance_violation`). Dzięki temu panel runbooków i
telemetrii natychmiast pokazuje problemy zgodności.

## Konfiguracja

Przykładowy plik `config/compliance/audit.yml`:

```yaml
kyc:
  required_fields:
    - full_name
    - address
    - country
    - id_document
  severity: high
aml:
  blocked_countries:
    - IR
    - KP
  high_risk_jurisdictions:
    - RU
    - IR
  suspicious_tags:
    - sanctioned
    - mixer
  forbidden_data_sources:
    - darkpool
  max_unverified_volume_usd: 10000
  severity: critical
transaction_limits:
  max_single_trade_usd: 25000
  max_daily_volume_usd: 100000
  lookback_days: 1
  severity: warning
```

Wartości progowe można dostosowywać dla poszczególnych dystrybucji OEM, a brak pliku
konfiguracyjnego powoduje użycie bezpiecznych ustawień domyślnych.

## Interpretacja wyników

Metoda `audit()` zwraca obiekt `ComplianceAuditResult`, który zawiera:

- znacznik czasu audytu (`generated_at`),
- listę znalezionych naruszeń (`findings`),
- zwięzłe podsumowanie kontekstu (strategia, liczba transakcji, status KYC).

Raport można serializować do słownika (`to_dict`) i włączyć do istniejących pipeline’ów
raportowania. Brak naruszeń (`passed=True`) oznacza, że strategia spełnia minimalne
wymagania zgodności lokalnej instalacji.

## Checklisty LIVE i podpisy kryptograficzne

Od wersji z weryfikacją guardrail’i LIVE każdy raport compliance używany w konfiguracji
`environment.live_readiness` musi być podpisany kluczem HMAC i udostępniony lokalnie.
Klucze przechowujemy w katalogu `secrets/hmac/`, a podpisy są walidowane przy
uruchamianiu `bootstrap_environment`. Brak podpisu lub klucza blokuje start środowiska
`Environment.LIVE`, dlatego proces publikacji raportów compliance powinien zawsze
obejmować wygenerowanie podpisu (`bot_core.security.signing.build_hmac_signature`) oraz
zapis metadanych (`signature_path`) w konfiguracji środowiska.
