# PR #3235 — końcowa remediacja po niezależnym review

Data: 2026-11-02. Gałąź pozostaje `codex/stage9-linux-installer-safety`.
Przegląd bazował na `006466ed6b22f66b148db99726fb692a3992a59c`;
poprawka testów to `00b891fad30e5c406ebeaf0d119fd00058263a51`.
Ten raport jest źródłowym zapisem analizy. Bieżący HEAD oraz późniejsze wyniki
GitHub Actions są podane oddzielnie w opisie istniejącego PR i raporcie końcowym.
Nie przenosimy wyników CI starszego SHA na późniejsze commity.

## 1. Regresja Windows i naprawa

Job [Python tests Windows 125099399133](https://github.com/Dudzian/Dudzian/actions/runs/41389772192/job/125099399133)
zakończył się błędem testu single-byte: zamiast odmowy digest mismatch wystąpiła
wcześniejsza odmowa native Linux. Fixture `installed_source` zastępowała lokalne
`qualification.os`, ale nie nową kontrolę `qualification.sys.platform`.

Fixture modeluje teraz oba warunki wyłącznie w przestrzeni kwalifikatora:
`qualification.os=SimpleNamespace(name="posix")` i
`qualification.sys=SimpleNamespace(platform="linux")`. Globalny `sys.platform`
procesu pytest ani natywny `os.name` nie są zmieniane. To samo skorygowano
w drugim modelu POSIX: negatywnym teście delegacji service identity.
W pozostałych miejscach kwalifikator nie ma takiej fixture.

Nie zmieniono produkcyjnej bramki Linux ani reviewed source pins. Test nadal
czyta rzeczywisty plik, potwierdza jego hash względem oryginalnego pinu, zmienia
jeden bajt albo wprowadza CRLF i wymaga dokładnej odmowy SHA-256. Nowy spy
sprawdza, że po odmowie nie są wywoływane obserwacja procesu ani PostgreSQL.
Test nie pomija pętli integralności i nie nadaje PASS na podstawie innego błędu.

Dodano osobny model odmowy Windows, kontrolujący niezmienność globalnego
`sys.platform`, oraz test natywnego wywołania na Windows bez zastępowania
platformy lub baz kont. Ten ostatni jest pomijany wyłącznie poza Windows;
na Windows musi potwierdzić rzeczywistą odmowę przed dostępem do kont POSIX.

## 2. Testy wieloplatformowe

Lokalny zestaw reviewed source pins, service identity, deployment i checkout:
130 PASS, 1 SKIP na Linux. Pominięcie dotyczy właśnie natywnego Windows,
nie testów single-byte/CRLF. Zestaw service identity zawiera 112 przypadków.
Wcześniejszy bazowy przebieg 006466ed miał jedną rzeczywistą regresję Windows;
Python Ubuntu i macOS zakończyły się sukcesem. Nowy job Windows wymaga osobnego
odczytu po push; wynik pending/in_progress nigdy nie jest PASS.

Ruff i dokładny hook `python -m pre_commit run ruff-format --all-files` przeszły.
Zmiana nie dotyczy implementacji, progów, wyjątków ani klas dowodów. Kontrole
mypy i kolejne wyniki CI są odnotowane w końcowym raporcie PR.

## 3. GitGuardian — kompletne uzasadnienie dla właściciela

Wszystkie trzy historyczne incydenty wskazują
`tests/deployment/test_production_local_installer.py` w commicie
`a5c4e882091a2ce3daaed0943ec1177fd88c17ff`:

| Incydent | Occurrence | Linia | API / cel fixture |
| --- | --- | --- | --- |
| 37919812 | 309397301 | 557 | getpwnam: zwrot syntetycznego konta po nazwie |
| 39396249 | 309397302 | 555 | getpwall: enumeracja dwóch syntetycznych kont |
| 39396250 | 309397300 | 978 | getpwuid: celowo niezgodna nazwa reverse lookup |

Przeczytano pełne bloki oraz callpath starego i aktualnego kodu. Standardowy
moduł POSIX `pwd` służy tu wyłącznie do modelowania NSS. SimpleNamespace zawiera
nazwę, UID/GID, katalog `/nonexistent` i shell `/usr/sbin/nologin`; nie zawiera
pola hasła. Funkcje fixture zwracają wyłącznie te obiekty. Słowo `foreign`
jest nazwą obcego konta w negatywnym teście, nie poświadczeniem. W tych trzech
lokalizacjach nie ma rzeczywistych haseł, tokenów ani innych danych logowania;
nie są odczytywane shadow ani zewnętrzne usługi uwierzytelniania.

Od 006466ed używany jest alias `account_database`. Stary wzorzec lokalnej
zmiennej modułu został usunięty; diagnostyczne dane i odmowy pozostały identyczne.
Nie oznacza to zamknięcia historycznego incydentu przez skaner. Ostatni check
[125098359105](https://github.com/Dudzian/Dudzian/runs/125098359105) dla bazowego
HEAD nadal zgłasza trzy historyczne wystąpienia w
[komentarzu bota](https://github.com/Dudzian/Dudzian/pull/3235#issuecomment-6565374973).

Uzasadnienie ręcznego triage: właściciel powinien porównać dokładny occurrence,
commit i linię z tabelą, przeczytać cały blok i potwierdzić brak materiału
uwierzytelniającego oraz syntetyczny charakter NSS. Zakres proponowanej
klasyfikacji false positive obejmuje tylko te trzy occurrence, nie inne
znaleziska w repozytorium. Nie ma tu materiału wymagającego rotacji.

Analiza triage jest kompletna. Incydenty nie zostały administracyjnie zamknięte.
Nie wyłączono GitGuardiana, nie dodano wyjątków i nie przepisano historii.
Bieżący kod, historyczne incydenty i status checka są oddzielnymi faktami;
check failure nie został nazwany PASS.

## 4. Tożsamość systemd

**SERVICE_IDENTITY_CODE_REVIEW=PASS** dla źródłowego zakresu dziesięciu kontroli.
Nie wykryto konkretnego problemu uzasadniającego zmianę implementacji.
Przeanalizowano także dokładne źródła upstream systemd v255 z przypiętego
commita; format GET-property i renderowanie porównywanych właściwości są zgodne
z parserem. Szczegóły i źródła znajdują się w dodatku.

Nie wykonano natywnej kwalifikacji zainstalowanej usługi na Ubuntu 24.04.
Podano odtwarzalną, odczytową procedurę operatorską. Jest ona podporządkowana
blokadzie runtime i nie uprawnia do automatycznej instalacji lub uruchomienia.

## 5. CI i bramki produkcyjne

Sprawdzane są osobno CI, Quality and Security, CodeQL, Style preflight,
Platform deployment i GitGuardian. Dokładne nowe run/job IDs oraz ich stan
są aktualizowane w raporcie końcowym. Na bazowym 006466ed awaria Windows jest
rzeczywistą regresją testu; brakujące dowody systemd/clean-install/lifecycle
Linux oraz lifecycle Windows są oczekiwanymi odmowami produkcyjnymi.
Nie osłabiamy tych bramek, by uzyskać zielony wynik. Stan alertów CodeQL #54/#58
pozostaje niepotwierdzony z powodu Forbidden; sam sukces analizy nie dowodzi
ich zamknięcia.

## 6. Cztery niezależne poziomy gotowości

| Poziom | Znaczenie i aktualne ograniczenie |
| --- | --- |
| Kod instalatora | Odczytowy preflight, kontrolowana publikacja, integralność i odmowy potwierdzone testami; nie jest to instalacja fizyczna |
| Środowisko runtime | Standardowy Noble ma za stare cryptography/psycopg; osobny podpisany runtime nadal niezaimplementowany |
| Rzeczywista instalacja | Faza dodatnia na standardowym Ubuntu 24.04 pozostaje BLOCKED; procedura VM nie była wykonywana |
| Produkcja Stage 9 | Pełna kompozycja runtime, piny i dowody fizyczne/lifecycle nadal blokują gotowość |

Nie obniżono wymagań bibliotek. Nie użyto sudo pip ani .pth/PYTHONPATH jako
obejścia. [Procedura VM](../runbooks/STAGE9_LINUX_UBUNTU2404_VM_TRIAL.md)
pozostaje instrukcją do odrębnego ręcznego wykonania, z jawnym STOP.

## 7. Pozostałe ograniczenia

Oczekują osobny podpisany runtime, rozstrzygnięcie historycznych incydentów
GitGuardiana przez właściciela, wszystkie wymagane wyniki nowego HEAD, wcześniejszy
drift AccountGenesis i właściwe dowody funkcjonalne/fizyczne. Żadnej próby
uprzywilejowanej nie wykonano automatycznie. Nie modyfikowano Della, kont,
klastra PostgreSQL, kluczy, GitHub Environments ani historycznych TEST_ONLY.
Nie wykonywano restartu ani merge.

## 8. Rekomendacja

Pozostawić draft. Sukces poprawionego joba Windows potwierdza usunięcie konkretnej
regresji testu, a nie gotowość runtime Ubuntu lub całego produktu. Wymagane są
wyniki aktualnego SHA i triage bezpieczeństwa; nie zmieniać Stage 9 na COMPLETE
ani LINUX_PRODUCTION_READY na PASS. STAGE9_PRODUCTION_READY pozostaje NO.

## Dodatek — kontrola tożsamości i systemd 255

### Dziesięć kontroli

| Warunek | Wniosek przeglądu |
| --- | --- |
| Typowany D-Bus ExecStart | Dokładny typ a(sasbttttuii), jeden rekord, 10 pól, brak duplikatów kluczy JSON; wiele komend i opakowanie metody CALL odrzucane |
| Program i argumenty | Osobne dokładne porównanie programu i listy argv; ścieżka jako argument echo/shell nie wystarcza |
| Efektywny systemd | Pełna lista właściwości i dokładna zgodność, fragment, brak DropInPaths, NeedDaemonReload=no, środowisko i sandbox; niepełne/niejednoznaczne odpowiedzi odmawiają |
| MainPID/ExecMainPID | Dodatnie ograniczone PID, identyczne wartości w snapshotcie; PID rekordu ExecStart musi odpowiadać MainPID |
| PIDFD | os.pidfd_open po obserwacji MainPID, /proc/self/fdinfo potwierdza Pid, poll odmawia zakończenia procesu; uchwyt zamykany również przy błędzie |
| /proc/PID/exe | Jądrowy obraz jest porównywany z rozwiązanym /usr/bin/python3, a nie z argv/comm; alternatywny obraz odmawia |
| Metadane i SHA-256 | Device/inode, mode, uid/gid, size, mtime/ctime; wymagany root:root i brak zapisu 022; deskryptory obrazu i referencji muszą wskazywać ten sam plik przed/po haszowaniu |
| UID/GID/cgroup | Istniejący capture_process_identity sprawdza wszystkie aliasy credentials, start ticks, grupy, obraz i cgroup; dodatkowe grupy/obca cgroup odmawiają |
| Powtórne obserwacje | PIDFD po zbieraniu, ponowne capture/proc image/argv/ExecStart/systemd; ostatnia kontrola PIDFD po końcowym snapshotcie |
| Niedostępność/drift | Błąd dostępu/polecenia, timeout, brak obserwacji, zmiana konfiguracji/procesu/obrazu kończą się odmową; brak fallbacku akceptującego niepotwierdzoną tożsamość |

Kontrola jest ograniczoną obserwacją konkretnego żywego procesu. Nie jest
zdalną atestacją hosta, dowodem całego wykonania Pythona ani gwarancją przyszłego
życia usługi. Root/manager i niezależnie zaufana instalacja systemowego interpretera
pozostają granicą zaufania. Te ograniczenia nie są zastępowane testami syntetycznymi.

## Zgodność źródeł upstream systemd v255

Tag v255 wskazuje commit `db11bab38ccf1ed257f310d29070843d4c58ea01`
(obiekt tagu `0966fe6734417ce07d5eb82a245fdec3789714da`). Odczytano dokładne
źródła upstream, zapisane do `/workspace/work/systemd255-review`.

- `src/core/dbus-execute.c`, append_exec_command: struktura `sasbttttuii`
  zawiera path, argv, ignore-failure, cztery timestamps, PID, code, status.
  bus_property_get_exec_command_list wystawia tablicę `(sasbttttuii)`.
- `src/busctl/busctl.c`, get_property i json_transform_variant: Properties.Get
  rozpakowuje wariant i tworzy obiekt `{type: contents, data: value}`.
  Array zawiera bezpośrednio listę rekordów, bez dodatkowego wrappera CALL.
  Parser bieżącego helpera odpowiada temu formatowi.
- `src/systemctl/systemctl.c`, case p: --property automatycznie ustawia
  BUS_PRINT_PROPERTY_SHOW_EMPTY. Brak --all nie pomija pustych EnvironmentFiles,
  DropInPaths ani capability sets przy jawnej liście właściwości.
- `src/shared/bus-print-properties.c`: RestrictNamespaces przy masce 0 drukuje
  yes; puste capability sets dają pusty tekst; Mode i UMask używają %04o.
- `src/systemctl/systemctl-show.c`: allowlist RestrictAddressFamilies drukuje
  AF_UNIX bez prefiksu ~; denylist miałaby ~ i zostałaby odrzucona.
- `src/core/dbus-execute.c`: PrivateTmp w tej wersji jest boolean, więc yes
  odpowiada jednostce. `SystemCallArchitectures` zachowuje native przez mapowanie
  SCMP_ARCH_NATIVE z `src/shared/seccomp-util.c`.

Źródła i linie:
- https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/src/busctl/busctl.c#L2150
- https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/src/core/dbus-execute.c#L1130
- https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/src/systemctl/systemctl.c#L591
- https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/src/shared/bus-print-properties.c#L118
- https://github.com/systemd/systemd/blob/db11bab38ccf1ed257f310d29070843d4c58ea01/src/systemctl/systemctl-show.c#L1151

To potwierdza kontrakt upstream v255; nie zastępuje obserwacji konkretnego
pakietu Ubuntu z poprawkami dystrybucji. Lokalny busctl executora ma wersję257,
a nie255, więc nie został przedstawiony jako natywny dowód Ubuntu255.

## Wykonane testy

`/workspace/work/venv/bin/python -m pytest -q tests/security/test_freshness_service_identity.py`

Wynik: **112 passed in 0.79s**. Testy są modelowe i ograniczają się do własnych
procesów/pidfd, bez usług i PostgreSQL. Fixture pozytywnego obrazu jawnie modeluje
root ownership, ponieważ interpreter executora jest nobody:nogroup. Nie należy
z tego robić pozytywnej kwalifikacji interpretera hosta.

## Odtwarzalny odczyt w jednorazowej VM — do wykonania operatora

Poniższe kroki nie zostały wykonane. Są wyłącznie dodatkiem do istniejącej
procedury VM i jej bramki STOP: standardowe Ubuntu24.04 ma za stare biblioteki,
więc nie wolno uruchamiać dodatniej fazy ani podstawiać venv/drop-inu dla zaliczenia.
Próba właściwej działającej usługi wymaga uprzednio zatwierdzonego zgodnego
profilu runtime oraz osobnego przygotowania schematu/kluczy. Nie używać atrap.

1. Na ręcznie przygotowanej VM zachować /etc/os-release i dokładne wersje
   `systemctl --version`, `busctl --version`, `dpkg-query -W systemd`.
   Wymagana natywna instancja systemd PID1; brak warunku oznacza STOP.
2. Z uprzednio zaufanego narzędzia z tej samej zatwierdzonej rewizji uzyskać
   literalną listę SYSTEMD_PROPERTIES (bez importowania niezaufanego checkoutu).
   Uruchomić odczyt `systemctl show cryptohunter-freshness-verifier.service
   --no-pager --property=<ta dokładna lista>`. Zachować stdout oraz return code
   lokalnie w pliku0600; nie publikować surowego Environment ani stderr.
3. Odczytać dokładny typowany rekord:

```sh
busctl --system --json=short get-property org.freedesktop.systemd1 \
  /org/freedesktop/systemd1/unit/cryptohunter_2dfreshness_2dverifier_2eservice \
  org.freedesktop.systemd1.Service ExecStart
```

4. Operator porównuje typ, jeden rekord, path, pełne argv, PID i brak exit status
   z kontraktem. MainPID i ExecMainPID muszą być równe. Odczyt /proc/PID/cmdline,
   readlink /proc/PID/exe oraz stat/hash referencji ma dotyczyć tego samego
   przypiętego PIDFD i być powtórzony. Same niepowiązane polecenia shell nie
   zastępują helpera — zatwierdzony helper wykonuje właściwą sekwencję.
5. Uruchomić wyłącznie odczytowy helper kwalifikacji z niezależnie zaufanego,
   chronionego narzędzia i zatwierdzonego interpretera. Zarchiwizować hash narzędzia,
   pełną rewizję, stały kod wyniku i metadane wersji VM. Nie zmieniać źródłowych
   pinów, by dojść do PASS pełnej kwalifikacji; istniejący AccountGenesis drift
   nadal ma odmówić w kwalifikatorze całości.
6. Kontrolowane negatywne odmowy sprawdzić na oddzielnych snapshotach tylko po
   odrębnej zgodzie na mutacje VM: obcy drop-in, alternatywny program z oczekiwanym
   skryptem jedynie jako argument, dodatkowa/brakująca flaga, inna cgroup/UID/GID,
   zakończenie procesu podczas obserwacji. Nie wykonywać tego w zwykłym CI ani
   na Dellu. Nie uruchamiać restartów w zakresie obecnego zadania.

Wynik źródłowego review PASS nie otwiera bramek physical/runtime/Stage9.

SHA-256 przejrzanego helpera: `75b8db762362baf95e99e7ae4fd15e94f81041c6a19eb8e3cb7f68885f612a33`.
