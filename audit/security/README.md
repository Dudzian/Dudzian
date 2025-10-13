# Raporty audytu bezpieczeństwa

Pliki w tym katalogu przechowują artefakty audytu RBAC/mTLS generowane
przez `scripts/audit_security_baseline.py`.  W repozytorium utrzymujemy
przykładowy raport `security_baseline.json`, który służy operatorom jako
referencja struktury JSON oraz zestaw wymaganych scope'ów dla
scheduler-a multi-strategy.

Aby odświeżyć raport, uruchom:

```bash
PYTHONPATH=. python scripts/audit_security_baseline.py \
  --config config/core.yaml \
  --json-output audit/security/security_baseline.json \
  --pretty
```

Przed generowaniem raportu uzupełnij zmienne środowiskowe tokenów RBAC i
lokacje materiału TLS, tak aby wynik odpowiadał produkcyjnej
konfiguracji.
