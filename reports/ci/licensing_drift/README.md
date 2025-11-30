# Raport kompatybilności HWID/licencji

Artefakt `compatibility.json` generowany przez `scripts/generate_hwid_drift_report.py` opisuje tolerancję dryfu (MAC/CPU/dysk/TPM) zgodnie z dokumentacją bezpieczeństwa.

Raport jest publikowany w nightly CI (`deploy/ci/github_actions_security_nightly.yml`) jako artefakt `licensing-drift-<run_id>` i powinien być dołączany do przeglądów bezpieczeństwa lub zgłoszeń rebind/appeal.
