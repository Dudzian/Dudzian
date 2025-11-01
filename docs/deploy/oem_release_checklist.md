# Checklista wydania OEM

Dokument opisuje obowiązkowe kroki techniczne przed dostarczeniem pakietu OEM
klientowi. Listę należy wypełnić dla każdej wersji dystrybucji i dołączyć do
pakietu przekazywanego partnerowi.

## Licencje i fingerprint sprzętowy
- [ ] Zweryfikowano, że wszystkie klucze licencyjne są wygenerowane w
      `scripts/manage_license.py` i przypisane do docelowych HWID.
- [ ] Porównano fingerprinty z `probe_keyring.py` oraz raportami audytu licencji.
- [ ] Przetestowano aktywację licencji na maszynie referencyjnej przy użyciu
      kreatora onboarding (`LicenseWizard`).
- [ ] Wygenerowano raport audytu licencji (`scripts/manage_license.py audit`) i
      dołączono go do pakietu dokumentacji.

## Aktualizacje i dystrybucja
- [ ] Zbudowano instalator przy użyciu `deploy/packaging/desktop_installer.py`
      oraz potwierdzono obecność hooków HWID.
- [ ] Przygotowano paczkę aktualizacji offline `.kbot` i zweryfikowano podpisy
      (`scripts/package_update.py verify`).
- [ ] Zweryfikowano proces aktualizacji offline w UI (`UpdateDialog`).
- [ ] Oznaczono instalator oraz paczki aktualizacji numerem wersji i sumami
      SHA-256 zgodnymi z manifestem.

## Zgodność i testy
- [ ] Uruchomiono audyt zgodności (`scripts/run_compliance_audit.py`) i brak
      naruszeń krytycznych.
- [ ] Przeprowadzono retrening modeli wraz z walidacją danych (`scripts/run_retraining_cycle.py`).
- [ ] Wykonano scenariusz smoke (`scripts/run_smoke_tests.py`) oraz testy E2E
      demo → paper.
- [ ] Zarchiwizowano raporty: licencyjny, compliance, retrening, smoke tests w
      katalogu `reports/oem/<wersja>/`.

## Materiały dla partnera
- [ ] Dołączono instrukcję instalacji oraz aktywacji (wyciąg z dokumentacji).
- [ ] Przekazano kontakt do wsparcia technicznego i runbook reagowania na
      incydenty krytyczne.
- [ ] Zweryfikowano, że wszystkie artefakty posiadają podpisy cyfrowe (jeśli
      wymagane przez SLA).

Wygenerowaną checklistę można zautomatyzować przy pomocy
`scripts/manage_release.py generate-checklist`, które tworzy dokument z danymi
konkretnego wydania.
