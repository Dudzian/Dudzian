# CryptoHunter Product Architecture Contract — M0

Ten katalog jest wersjonowanym źródłem prawdy dla bloku M0 „Product Architecture Contract”. Starsze dokumenty w `docs/`, `archive/`, workflowach i testach pozostają materiałem źródłowym, ale nie są nadrzędne wobec finalnego kontraktu produktu powstającego w M0.

## Status M0.1

M0.1 jest audytem aktualnego stanu repozytorium. Nie implementuje docelowej architektury, nie zmienia runtime'u, nie uruchamia Live i nie wybiera jeszcze modułów do usunięcia. Inwentaryzacja bazuje na kodzie, testach, konfiguracji, workflowach i packagingu istniejących w commicie bazowym zapisanym w JSON.

## Artefakty M0.1

- [current_state_inventory.md](current_state_inventory.md) — opisowa inwentaryzacja obecnego stanu.
- [current_state_inventory.json](current_state_inventory.json) — maszynowo walidowany kontrakt inwentaryzacji.

## Planowane elementy M0.1–M0.14

1. M0.1 — current-state inventory i audyt dowodów.
2. M0.2 — słownik statusów, środowisk i granic bezpieczeństwa.
3. M0.3 — kontrakt procesu desktop/core/background.
4. M0.4 — kontrakt frontendów i migracji UI.
5. M0.5 — kontrakt execution, routing i kont.
6. M0.6 — kontrakt exchange/testnet/live gates.
7. M0.7 — kontrakt trading universe.
8. M0.8 — kontrakt strategii, instancji i AI.
9. M0.9 — kontrakt risk, safety i kill switch.
10. M0.10 — kontrakt persistence, journal, ledger i recovery.
11. M0.11 — kontrakt operator shell, tray i reconnect.
12. M0.12 — kontrakt packaging, updater, rollback i CI.
13. M0.13 — macierz migracji/legacy bez decyzji kasujących.
14. M0.14 — finalny Product Architecture Contract i bramki walidacyjne.
