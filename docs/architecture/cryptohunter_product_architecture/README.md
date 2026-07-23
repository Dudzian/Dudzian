# CryptoHunter Product Architecture Contract — M0

Ten katalog jest wersjonowanym źródłem prawdy dla bloku M0 „Product Architecture Contract”. Starsze dokumenty w `docs/`, `archive/`, workflowach i testach pozostają materiałem źródłowym, ale nie są nadrzędne wobec finalnego kontraktu produktu powstającego w M0.

## Status M0.1 — closed

M0.1 jest zamkniętym audytem aktualnego stanu repozytorium. Nie implementuje docelowej architektury, nie zmienia runtime'u, nie uruchamia Live i nie wybiera jeszcze modułów do usunięcia. Inwentaryzacja bazuje na kodzie, testach, konfiguracji, workflowach i packagingu istniejących w commicie bazowym zapisanym w JSON.

## Artefakty M0.1

- [current_state_inventory.md](current_state_inventory.md) — opisowa inwentaryzacja obecnego stanu.
- [current_state_inventory.json](current_state_inventory.json) — maszynowo walidowany kontrakt inwentaryzacji.


## Status M0.2 — closed

M0.2 jest kolejną warstwą źródła prawdy po zamkniętym M0.1: definiuje kanoniczny słownik domeny, publiczne środowiska tradingowe, niezależne osie stanu, politykę trwałych identyfikatorów, granice bezpieczeństwa oraz konflikty legacy. M0.2 nie zmienia runtime'u, produkcyjnych enumów ani konfiguracji środowisk.

## Artefakty M0.2

- [canonical_domain_vocabulary.md](canonical_domain_vocabulary.md) — opisowy kanoniczny słownik domeny M0.2.
- [canonical_domain_vocabulary.json](canonical_domain_vocabulary.json) — maszynowo walidowany kontrakt słownika domeny M0.2.


## Status M0.3 — under audit

M0.3 definiuje kontrakt topologii procesów i lifecycle’u aplikacji: role CoreHost/TrayAgent/DesktopShell/Bootstrapper, zasady IPC/discovery, background po zamknięciu GUI, shutdown intents, first-run readiness, autostart oraz failure/restart policy. M0.3 pozostaje under audit i nie implementuje osobnych procesów, tray, QML, proto ani Windows Service.

## Artefakty M0.3

- [process_topology_and_lifecycle.md](process_topology_and_lifecycle.md) — opisowy kontrakt topologii procesów i lifecycle’u M0.3.
- [process_topology_and_lifecycle.json](process_topology_and_lifecycle.json) — maszynowo walidowany kontrakt topologii procesów i lifecycle’u M0.3.

## Planowane elementy M0.1–M0.14

1. M0.1 — current-state inventory i audyt dowodów.
2. M0.2 — kanoniczny słownik domeny, środowisk, stanów bezpieczeństwa i trwałych identyfikatorów.
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
