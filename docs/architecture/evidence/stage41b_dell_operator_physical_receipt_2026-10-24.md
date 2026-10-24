# Stage 41B — raport operatora z fizycznej kwalifikacji (TEST ONLY)

> **Pochodzenie i ograniczenie dowodu:** poniższe wartości są transkrypcją wyników terminala przekazanych przez operatora fizycznego Dell Latitude 5490 z Ubuntu 24.04 dnia 2026-10-24. Asystent nie miał dostępu do tego hosta ani do bajtów czterech plików raportu i nie obliczał ich SHA256 niezależnie. Zapis w repozytorium pozwala porównać później dostarczone pliki z podanymi digestami; nie jest kryptograficznie podpisaną, niezależną atestacją hosta, produkcyjnego custody ani approval GO.

## Tożsamość kwalifikacji

- Zakres: **Stage 41B**, fizyczna separacja dwóch syntetycznych signerów, `41B_TEST_ONLY`.
- Sprzęt i OS (według operatora): **Dell Latitude 5490**, **Ubuntu 24.04**.
- Testowany dokładny commit: `782816540373d7b31f426dab645ad115320d1234` (PR [#3203](https://github.com/Dudzian/Dudzian/pull/3203)).
- Identyfikator fizycznego wykonania: `1fafdf40f9f3c858fc69ab5f2925a618`.
- Instalacja źródłowa: `/opt/cryptohunter-stage41b/782816540373d7b31f426dab645ad115320d1234`.
- Oryginalne raporty pozostają lokalnie w `~/stage41b-reports-782816540373d7b31f426dab645ad115320d1234/`. Nie załączono ich do tego commita.

## Zgłoszony przebieg

| Etap | Wynik z terminala |
| --- | --- |
| Unprivileged bundle | `BUNDLE_READY`, `physical_executed=false` |
| Offline installation | `PREPARED_ENVIRONMENT_VERIFIED`, `physical_executed=false` |
| Weryfikacja instalacji | `PREPARED_ENVIRONMENT_VERIFIED` |
| Read-only preflight | `host_preflight=PASS`; ogólny `BLOCKED` i exit 2 są oczekiwane w `--check` |
| Explicit physical `--execute` | Exit 0; `STAGE41B_TEST_ONLY_PASS` |
| 19 required boundaries | **19 PASS, 0 FAIL, 0 BLOCKED** |
| Cleanup boundary | `PASS / OWNED_SCOPE_REMOVED` |
| Read-only `--inspect-cleanup` | Exit 0; `PASS / ALL_EXPECTED_RESOURCES_ABSENT` |
| Startup diagnostics | `{}` |
| Cleanup issues | `[]` |

W fizycznym `--execute` raportowano `PASS` dla wszystkich nazw wymaganych w `REQUIRED_BOUNDARIES`:
`host_preflight`, `unit_collision_guard`, `systemd_process_boundaries`,
`own_signature`, `private_material_separation`, `foreign_store_dac`,
`foreign_keyring_files`, `foreign_mount`, `foreign_dbus`,
`foreign_keyring`, `foreign_custody`, `foreign_custody_store`,
`foreign_ipc`, `own_liveness`, `identity_binding`,
`unauthorized_ipc`, `artifact_secrecy`, `service_liveness`, `cleanup`.

## Zgłoszone digests raportów

Operator wykonał lokalnie `sha256sum -c SHA256SUMS`; terminal potwierdził `DOBRZE` dla każdego z czterech plików.

```text
5687a9f3e05683dc04f8402280f41f5d34ee4701c1c1ced9f6068ffd450bc590  installation.json
97ae625c51215dfc6ab0776cdc43a92f6ea79848a05e8718c8d98caedb9357cf  preflight.json
deabb330c6a8242c458f1af877bed5f36f8232578f972437f9b1eff42bcebdf1  execution.json
989417559004f875d6f00945a151e60b001097fa30f59d1d5e1cd193b2cc9b82  cleanup.json
```

Ten zapis nie przenosi dowodu z hosta: aby zweryfikować zawartość raportów, należy pozyskać je bezpiecznym kanałem z Della, ponownie obliczyć ich SHA256 oraz sprawdzić wartości, schemat, powiązanie commita/run-id i pochodzenie. Samo lokalne `SHA256SUMS` nie dowodzi niezależnej autentyczności przebiegu.

## Granice wniosku i status produkcji

Uzyskany, **operator-reported `STAGE41B_TEST_ONLY_PASS`**, dotyczy jednego fizycznego przebiegu z nowymi kluczami syntetycznymi. Nie dowodzi, że wszystkie produkcyjne ścieżki dostępu, lifecycle, recovery, PostgreSQL, trust activation, provisioning i kontrola rzeczywistych produkcyjnych kluczy zostały zakwalifikowane. Nie autoryzuje produkcyjnego `SIGN`, `BIND`, enrollment ani Stage 10.

Nie modyfikować `deployment/stage9_current_status.json`, frozen kontraktów, progów CI ani bramek produkcyjnych na podstawie tego raportu. **Stage 9: `IN_PROGRESS / NO-GO`**; `private_control_status=BLOCKED` i niezależna atestacja pozostają wymagane.

Szczegóły granic dowodu: [architektura fizycznej kwalifikacji Stage 41B](../stage9_stage41b_physical_signer_isolation.md).
