# SBOM w CI

Pipeline CI generuje SBOM w formacie CycloneDX JSON jako osobny job `sbom`
w workflow `.github/workflows/ci.yml`.

## Gdzie powstaje SBOM

- Job: `Generate SBOM`
- Narzędzie: `anchore/sbom-action`
- Plik wynikowy: `sbom.cdx.json`
- Artefakt GitHub Actions: `sbom-cyclonedx`

## Jak pobrać

1. Otwórz run workflow `CI` w GitHub Actions.
2. Wejdź w sekcję artefaktów.
3. Pobierz artefakt `sbom-cyclonedx` zawierający `sbom.cdx.json`.
