# Dokumentacja wdrożeniowa

Ten katalog zawiera procedury budowy i dystrybucji KryptoŁowcy w różnych
modelach wdrożeniowych. Najważniejsze dokumenty:

- `desktop_installer.md` – proces budowy instalatora desktopowego.
- `offline_updates.md` – format paczek aktualizacji offline.
- `release_process.md` – kanały wydań i wymagane testy.
- `installer_build.md` – szczegóły walidacji hooka HWID (`--verify-hook`).

## OEM release

Wydanie OEM łączy przygotowanie artefaktów (instalator, paczki `.kbot`), raportów
kontrolnych oraz dokumentacji dla partnera. Do planowania prac używaj
`docs/deploy/oem_release_checklist.md`, a generację dokumentów zautomatyzuj
skryptem:

```bash
python scripts/manage_release.py generate-checklist \
  --version 1.4.0 \
  --release-tag v1.4.0-rc1 \
  --owner "Zespół OEM" \
  --output reports/oem/1.4.0/checklist.md
```

Szablony raportów (licencje, compliance, testy) znajdują się w
`docs/deploy/templates/` i mogą być renderowane za pomocą poleceń
`generate-license-report`, `generate-compliance-report` oraz
`generate-test-report`. Wszystkie wygenerowane dokumenty przechowuj w katalogu
`reports/oem/<wersja>/` obok raportów automatycznych (audyt licencji, smoke).
Zanim przekażesz pakiet partnerowi, uruchom `deploy/packaging/desktop_installer.py`
z opcją `--verify-hook`, aby potwierdzić poprawną integrację hooka HWID i
keyringu. Dzięki temu partner otrzymuje spójny pakiet OEM.
