# Stage 9 V5d6 — wyniki walidacji

Przebiegi lokalne wykonano na zmianach V5d6 opartych na wskazanym commicie
`ad85f19a15b6b1acffd9146a8b08760a3208ed7a`. Przed publikacją draftu
zdalny `main` wskazywał `db927164b5203964538a3810a7c1c2e305857b11`.
Zachowano wskazaną bazę commita. Zmiany, które weszły w międzyczasie na
`main`, nie nakładały się na zmieniane fragmenty kodu ani instrukcje `AGENTS.md`.

| Zakres | Zaobserwowany wynik |
| --- | --- |
| V5d6, pełne V5d5, V5d4 i V5d3, Linux CPython 3.11.16 | 798 PASS, 42 SKIP, exit 0. Obejmuje regresje dwóch pominięć PYC i pełnego zbioru 62 ścieżek PySide6. |
| Wszystkie nowe przypadki V5d6, Linux CPython 3.12.14 | 91 PASS, exit 0. Dwa dodatkowe przypadki sprawdzają ochronę wheel przed nadpisaniem przez wyjście inspektora. |
| Przypadki składni parsera V5d4 i V5d3, CPython 3.12.14 | 20 PASS, exit 0. |
| V5d2 inventory | 59 PASS, 4 timeouty po 20 s w pierwszym przebiegu. Powtórzenie dokładnie tych czterech przypadków po zakończeniu innych przebiegów dało 4 PASS z tym samym limitem. |
| Kontrakty CI | V5d4 history 2 PASS, V5d5/V5d6 inventory i agregacja 14 PASS. |
| Wcześniejsze Observer V1–V5d1 | Pełna próba 25 plików zatrzymała się na 2 błędach zbierania zestawów V5d1 z powodu braku `defusedxml`. Zakres 23 plików bez tej zależności dał 2163 PASS, 32 SKIP. Wśród pominięć 31 wymagało preprovisioned `gpgv`, 1 zależało od hosta. |
| Lokalne pokrycie z gałęziami | 90% w dziewięciu modułach V5d4–V5d6. W macierzy CI nadal obowiązuje próg pokrycia zmienionego kodu 70% po połączeniu platform. |
| Ruff 0.14.11 | `ruff check .` PASS. Jawna kontrola formatu zmienionych plików PASS. Bezpośredni Ruff z zakresem hooka i wykluczeniami z konfiguracji PASS dla 2561 plików. |
| Dokładny hook z `AGENTS.md` | `python -m pre_commit run ruff-format --all-files` zatrzymał się na `No module named pre_commit`. Osobny workflow Style preflight sprawdza ten hook. |
| Mypy Linux i Win32 | Zakres dziewięciu modułów PASS na obu platformach mypy. Pełny skonfigurowany zakres 428 źródeł zgłosił po 2 błędy importu `defusedxml` w istniejącym module V5d1. |
| Semgrep, Betterleaks, pip-audit i diff-cover | Brak narzędzi lokalnie. Wyniki tych kontroli pozostają zależne od CI. |

Zestawy testów uruchomiono z `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
i odpowiednim `--confcutdir`. Przebieg z pokryciem odpowiada zakresowi
V5d5/V5d6 w `platform-deployment.yml`. Dla CPython 3.12 użyto dostępnego
lokalnie pytest przez `PYTHONPATH`. Żaden z tych przebiegów nie wymagał
instalacji pakietów, odtworzenia wheelhouse ani uruchomienia VM lub libvirt.
Wyniki GitHub Actions dla commita publikacji są dostępne w sekcji Checks
draft PR. Lokalny `mypy --platform win32` sprawdza typy dla tej platformy.
Wyniki uruchomienia na natywnym Windows i macOS pochodzą z macierzy CI.

Statyczna inspekcja oryginalnego greenlet potwierdziła SHA-256 zgodny
z metadanymi PyPI, integralność ZIP i 98 wpisów `RECORD` wiążących hash
oraz rozmiar. Piny nagłówka są zgodne z pomiarem. Pełny inspektor
zależności zatrzymał się na następnym kodzie `WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED`
dla `greenlet/platform/setup_switch_x64_masm.cmd`. Zachowano dowód
tej odmowy oraz obserwację zgodności nagłówka z polityką.
[Inspekcja archiwum](evidence/stage9_v5d6_greenlet_inspection.json),
[kontrola polityki](evidence/stage9_v5d6_original_policy_check.json),
[dowód odmowy](evidence/stage9_v5d6_next_refusal.json).

Pełny oryginalny wheelhouse 69 wheels ma status `NOT_TESTED_IN_CLOUD`.
Jego bajty i `origins.json` są na laptopie operatora. Polecenie do statycznego
przebiegu po scaleniu i projekt kontraktu opisuje dokument V5d6.
[Kontrakt i polecenie dla operatora](stage9_v5d6_greenlet_header.md).

Odrębne aprobaty PYC, współdzielenia, nagłówka, pochodzenia i native/ABI
pozostają wymagane. Wystawienie receipt do rzeczywistego środowiska wymaga
również dowodu chronionego widoku runtime oraz niezależnego digestu receipt.

## Powrót do przeglądu PR #3294

Do gałęzi draftu włączono `origin/main` z commita
`db927164b5203964538a3810a7c1c2e305857b11`, zachowując oba ciągi historii.
Zmiany Security oraz inwentarz #3293 pochodzą wprost z `main`. Względem
`main` konfiguracja mypy dodaje trzy moduły V5d6 do istniejącego zakresu.
Próg 70% oraz trzyplatformowa agregacja pokrycia pozostają w mocy.

Dla HEAD PR sprzed synchronizacji, `46228f346df4da735cbe428463053c884ed8e3a4`,
zakończyło się osiem workflow. Siedem przeszło, w tym [dokładny hook
formatowania](https://github.com/Dudzian/Dudzian/actions/runs/45384094569)
i [pełne mypy Linux/Win32](https://github.com/Dudzian/Dudzian/actions/runs/45384094727)
po 430 źródeł. [Job V5d1](https://github.com/Dudzian/Dudzian/actions/runs/45384094441/job/138410391409)
wykonał oba wcześniej zablokowane zestawy, 292 PASS w `kvm_domain` i 62 PASS
w `kvm_preflight`. Całe V5d1 dało 458 PASS i 1 SKIP.
Syntetyczny V5d2 dał 63 PASS z dotychczasowym limitem 20 s na wywołanie.
Macierz V5d5/V5d6 i agregacja pokrycia przeszły. Dwie blokady tego workflow
wystąpiły na odrębnych bramkach dowodowych:

- Linux zgłosił `MISSING_CURRENT_EVIDENCE` dla `SYSTEMD_INTEGRATION`,
  `CLEAN_INSTALL` i `REBOOT_LIFECYCLE_UPDATE` oraz status
  `LINUX_PRODUCTION_READY=NOT_READY`.
  [Log zadania](https://github.com/Dudzian/Dudzian/actions/runs/45384094441/job/138449693512)
- Windows zgłosił `WINDOWS_BETA_RELEASE_READY=NOT_READY`, w tym brak
  aktualnych dowodów cyklu, restartu i aktualizacji.
  [Log zadania](https://github.com/Dudzian/Dudzian/actions/runs/45384094441/job/138522382410)

Wyniki te obowiązują wyłącznie dla powyższego SHA. W podsumowaniu draftu
osobno zapisuje się stan [kontroli aktualnego HEAD](https://github.com/Dudzian/Dudzian/pull/3294/checks).
Pełny wheelhouse i niezależne aprobaty nadal są odrębnymi warunkami.

Po synchronizacji 108 lokalnych testów kontraktów CI i inwentarza platform
przeszło. Na CPython 3.12 ukierunkowane regresje inspektora, zgodności
schematów, budowy i V5d4/V5d5 dały 338 PASS i 19 SKIP dla testów
wymagających Windows. Ruff oraz mypy dla dziewięciu modułów przeszły na
skonfigurowanym Linux i Win32. Lokalny pełny mypy 430 źródeł oraz dwa
zestawy V5d1 nadal wymagają brakującego `defusedxml`, a dokładny hook
formatowania wymaga `pre_commit`. Wyniki Actions rozliczają te kontrole
oddzielnie od lokalnych ograniczeń narzędzi.

Przebieg diagnostyki z jednym oryginalnym greenlet zachował 1 przypiętego
kandydata nagłówka jako `PARTIAL` i obie stwierdzone odmowy. Zbadano
621404 bajty archiwum, a domknięcie pozostało `NOT_TESTED`. To pomiar
pojedynczego wheel. [Surowy JSON v2](evidence/stage9_v5d6_greenlet_diagnostic_v2.json).

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
