# Audyt bezpieczeństwa CryptoHunter — punkt wznowienia

Data: 2026-11-18. Issue: [#3280](https://github.com/Dudzian/Dudzian/issues/3280).
Gałąź raportów: `security/audit-checkpoint-20261117`.

## Aktualny stan

- Zweryfikowany `main`: `3462968720e858f4a8557454b5342375840f1f71`.
- `MERGED`: #3279 (`b143b570...`), #3282 (`10e1f854...`) oraz #3284 (`34629687...`).
- Siedem poprawek poniżej: `AVAILABLE_ON_REMOTE_BRANCH`, wszystkie PR-y draft.
  Nie scalono żadnej z nich i nie włączono auto-merge.
- Poprzednie środowisko i niezapisane artefakty: `UNAVAILABLE`.
- Codebase Memory: `UNAVAILABLE`; przegląd oparty na źródłach i testach.
- Dawne liczby 318/838/149 testów i stary raport 25 sygnałów/10 błędów:
  `EVIDENCE_ONLY`, nie traktowano ich jako nowych wyników.
- Nie odnaleziono osobnych wcześniejszych patchy poza scalonym cloud guard.
  Brakujące poprawki odtworzono, zamiast powtarzać cały audyt od początku.

## Trwałe poprawki i checkpointy

| Obszar / gałąź `security/…` | Commit kodu | Końcowy zdalny HEAD | Draft PR |
| --- | --- | --- | --- |
| XML / `xml-dtd-20261117` | `b2e7a99df028d26ae22340f81f70ea5ff8a1a37f` | `815cbb6e0d81c3a6dd2def3601460d2521e40a5e` | [#3283](https://github.com/Dudzian/Dudzian/pull/3283) |
| Semgrep / `semgrep-completeness-20261117` | `7c50af1146c3d7133e88d2cad90e1e7a9e472dac` | `03cca4b3f62df4dc70a494a2c31be20f510646db` | [#3286](https://github.com/Dudzian/Dudzian/pull/3286) |
| Auth / `cloud-auth-replay-20261117` | `2a15911a51c76474d0420e10f636799b486a7dc2` | `495ae69c01162e6b9010ebc185a4c1a5a8e5a2f6` | [#3288](https://github.com/Dudzian/Dudzian/pull/3288) |
| Git index / `git-index-scan-20261117` | `da715c6167fd667e26a51df84a3d3b5fa9a7d256` | `dc784fef8b9aca7c13ab73f913ca77b2ee2182ad` | [#3287](https://github.com/Dudzian/Dudzian/pull/3287) |
| Update / `update-recovery-20261117` | `dbdc1c595ee2be49aa183f2d0dc59db40b696dcd` | `c32237c52290d98807cdb10cd75b5b1929afe791` | [#3285](https://github.com/Dudzian/Dudzian/pull/3285) |
| npm / `npm-electron-20261117` | `1c60786e64a6aa94d33eed72e40d122a86556f87`, docs `b05492d6` | `2fcb635fc2cdeb83c26abaca36f67a6db770f36d` | [#3289](https://github.com/Dudzian/Dudzian/pull/3289) |
| Runtime gRPC / `grpc-loopback-plaintext-20261117` | `af32e24d69442deecabd4177cebd8787299e4a6a` | `a9fbe5c236ab3e78ce58df2934a9616f17228236` | [#3290](https://github.com/Dudzian/Dudzian/pull/3290) |

Końcowe gałęzie zawierają aktualny main. Integracja nie używała force push.
Lokalne drzewo łącznej walidacji: `be9bc20f9b6a7a1db00a801ac6fbb37c8c23fc04`.
Nie tworzono z niego zbiorczego PR. Można je odtworzyć z powyższych przejrzanych
commitów; konflikt workflow wymaga zachowania obu nowych list testów.

## Potwierdzona walidacja

- Regresja łączna na wskazanym drzewie: **569 PASS, 1 SKIP**, exit 0.
- Smoke property/gate/gRPC z `-n 4` i coverage jak w CI: **224 PASS**, exit 0;
  to podzbiór powyższych testów, bez sumowania.
- Diff coverage mierzonych modułów: **94% (472/498 zmienionych linii)**, exit 0,
  niezmieniony próg 70%. Nie jest to pokrycie całego repozytorium.
- Pełny pre-commit, dokładny format i Ruff: PASS. Configured mypy Linux/Win32:
  PASS, 426 plików. Win32 jest analizą statyczną, bez natywnego Windows.
- Dodatkowy targeted mypy auth: FAIL, 6 identycznych istniejących błędów na
  bazie i po poprawce, 0 nowych. Brak wyłączeń i suppressions.
- Working tree Betterleaks integracji: exit 0, 117,80 MB; baseline bez zmian.
- Nowy skan historii `origin/main`: exit 1, **84 nierozstrzygnięte kandydaty**.
  Nie są automatycznie rzeczywistymi sekretami i nie otrzymują statusu PASS.
- Odświeżony 18 XI pip-audit kanonicznego lockfile: exit 0, **104 zależności,
  0 znanych podatności**, bez wyjątków.
- npm: 18 forms i 5 Chromium renderer E2E PASS, build/tsc PASS.
  `npm audit --omit=dev`: exit 0 / 0 podatności. Pełny audit: exit 1 / 4 pakiety
  high powiązane z publicznym GHSA-fc5q-39xc-vjpf; Electron native NOT_TESTED.
- Rzeczywiste skany Semgrep: REFUSED (timeouty/partial parsing). Gate poprawnie
  odrzuca niepełne raporty. Przejrzano 25 bieżących sygnałów: 25 false positive;
  nie jest to odzyskany stary raport ani dowód kompletności skanu.
- Actions końcowych SHA pozostają częściowo queued/in_progress; nie zastępują
  ich historyczne wyniki ani lokalne testy. Przed merge potrzebny nowy odczyt.

Pełny zakres, polecenia, SHA, rejestr ustaleń i macierz 14 obszarów:
[security-audit-final-report.md](security-audit-final-report.md).

## Otwarte blokady i następne działania

1. Dokończyć wymagane checks każdego końcowego SHA, bez automatycznego merge.
2. Usunąć przyczyny niekompletności Semgrep bez ukrywania błędów lub redukcji scope.
3. Prywatnie rozstrzygnąć kandydatów historii i zweryfikować potrzebne odwołanie
   lub rotację. Nie publikować surowych danych i nie przepisywać historii automatycznie.
4. Domknąć podatność narzędziową npm po dostępnej poprawce upstream; nie maskować audit.
5. Zrealizować opisane braki autoryzacji, end-to-end update, PG i kwalifikacji
   natywnej w odrębnych zakresach i testach. Nie uznawać niesprawdzonych obszarów za PASS.

`SECURITY_ADVISORIES=UNAVAILABLE` dla konektora. Jednorazowy odczyt REST był
możliwy, uprawnienia zapisu nie zostały potwierdzone. Pełne nierozwiązane dowody
wymagają prywatnego kanału właściciela; nie zapisano ich w publicznym repo.

Nie zamykać #3280 na podstawie tej transzy. Status wymagany:

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
