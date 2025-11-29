# Marketing bundle parity report (bootstrap)

Ten plik jest automatycznie nadpisywany przez job CI `marketing-bundle-parity`.
Zawiera podsumowanie porównania hashy artefaktów `stress-lab-report` oraz
`signal_quality/index.csv` względem lustrzanej kopii w S3/Git. Raport jest
zachowywany jako artefakt audytowy i może być cytowany w `docs/audit/paper_trading_log.md`.

Siostrzany plik JSON (`docs/audit/marketing_parity_report.json`) przechowuje
te same dane w formacie maszynowym i jest dołączany do artefaktu CI
`marketing-parity-report`.

*Ostatnia aktualizacja w repozytorium*: raport wygenerowany ręcznie podczas
wdrażania joba CI; kolejne wpisy będą pochodziły z uruchomień pipeline.
