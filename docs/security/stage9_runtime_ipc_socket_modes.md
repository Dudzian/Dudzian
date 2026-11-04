# Stage 9 — audyt trybu socketów testowych i CodeQL #63–65

Przegląd dotyczy trzech wywołań `chmod(0o660)` w
`tests/security/test_freshness_runtime_ipc.py` przy HEAD
`e893984d880d5dab01c16e2ae3dff88625b5c6ac`: pierwotnego listenera,
celowo podstawianego socketu oraz fixture testującej odmowy symlinków.
Nie dotyczy instalacji ani kwalifikacji rzeczywistej usługi.

## Stan alertów i dostęp do dowodów GitHuba

Odczyt każdego z alertów przez GitHub REST API zwrócił HTTP 403
`Resource not accessible by integration`. Także sprawdzenie drugiego
skonfigurowanego poświadczenia nie udostępniło odczytu alertu. Nie
zmieniano ani nie odrzucano alertów i nie ujawniano poświadczeń.

| Alert | Link | Potwierdzony wynik odczytu |
| --- | --- | --- |
| #63 | [CodeQL #63](https://github.com/Dudzian/Dudzian/security/code-scanning/63) | HTTP 403; stan `open`/`fixed` niezweryfikowany |
| #64 | [CodeQL #64](https://github.com/Dudzian/Dudzian/security/code-scanning/64) | HTTP 403; stan `open`/`fixed` niezweryfikowany |
| #65 | [CodeQL #65](https://github.com/Dudzian/Dudzian/security/code-scanning/65) | HTTP 403; stan `open`/`fixed` niezweryfikowany |

Poniższe uzasadnienie wynika z bezpośredniego przeglądu wskazanych
źródeł i rzeczywistych testów jądrowych. Nie jest potwierdzeniem zamknięcia
alertów przez CodeQL. Ponowna analiza GitHuba nowego HEAD pozostaje osobnym
wynikiem CI; nie czekamy na nią w aktywnej sesji.

## Wymagany kontrakt AF_UNIX

Obiekt po `socket(AF_UNIX, SOCK_STREAM).bind()` jest inode typu socket,
a nie regularnym plikiem z hasłem lub kluczem. Tryb `0660` jest wymaganym
trybem istniejącej usługi: właściciel i grupa IPC mogą nawiązać połączenie,
pozostali nie mają praw. Grupowe prawo zapisu socketu oznacza możliwość
połączenia AF_UNIX; samo w sobie nie zastępuje autoryzacji UID przez serwer.

`freshness_verifier_service.py` wybiera stałą grupę
`freshness_verifier_ipc` i katalog `0750`; niezmieniony
`serve_local_unix_socket()` ustawia socket `0660`. Niezmieniony
`deployment.linux_physical_ipc._pinned_verifier_socket()` wymaga dokładnie
tego trybu oraz właściwych UID/GID, typu socketu i pojedynczego linku.
Zmiana fixture na `0600` lub `0640` osłabiłaby zgodność testu z kontraktem.

Testy działają jako jeden nieuprzywilejowany UID/GID. Nie przypisują mu
produkcyjnych kont: właścicielem i grupą socketu są rzeczywiste effective
UID/GID procesu testowego. Produkcyjny katalog i nazwy tożsamości pozostają
jawnym modelem; wynik nie dowodzi izolacji dwóch produkcyjnych UID.

Przed poprawką ścieżki znajdowały się wewnątrz prywatnego katalogu pytest;
nawet zagnieżdżony katalog symlink fixture był osłonięty przez prywatnego
przodka. Samo wymagane `0660` nie udostępniało zatem obiektu wszystkim
użytkownikom. Brakowało jednak lokalnego, jawnego potwierdzenia tych założeń
w miejscu tworzenia oraz atomowego nadania docelowego trybu.

## Najwęższa poprawka fixture bez wyciszania skanera

Trzy miejsca używają teraz jednego testowego helpera
`_bind_private_test_socket()`. Helper:

- odmawia wykonania w głównym procesie pytest;
- otwiera bezpośredni katalog przez `O_DIRECTORY | O_NOFOLLOW` i wymaga
  rzeczywistych UID/GID procesu, dokładnego `0700` oraz braku access/default ACL;
- tworzy socket przez przypięty deskryptor katalogu w `/proc/self/fd`,
  pod krótką `umask(0o117)`, więc obiekt od razu powstaje jako `0660`;
- zawsze przywraca wcześniejszą umask w `finally`, również po błędzie `bind`;
- po utworzeniu sprawdza typ socketu, dokładny tryb, rzeczywiste UID/GID,
  pojedynczy link i brak dodatkowych ACL.

Nie ma okna `bind` z szerszym trybem, po którym następuje pathname `chmod`.
Nie zmienia się wymagany tryb socketu. Umask jest globalna dla procesu,
dlatego wszystkie te bindy, w tym fixture symlinków, wykonuje oddzielny
kontrolowany proces potomny; umask głównego pytest nie jest zmieniana.
To dodatkowe zabezpieczenie rzeczywistych założeń fixture, a nie zastąpienie
reguły skanera ani algebraiczne ukrycie stałej `0660`.

Nie dodano `noqa`, `nosemgrep`, wykluczenia CodeQL, zmiany konfiguracji
skanera ani zmiany stanu alertów. Pięć zamrożonych źródeł usługi, piny,
manifesty, podpisy i status Stage 9 pozostają bez zmian.

## Regresje

Testy `test_real_fixture_socket_dac_and_umask_restore` sprawdzają rzeczywisty
inode socketu `0660`, UID/GID i link count oraz przywrócenie restrykcyjnej
umask po sukcesie i błędzie. Odmowy obejmują katalog widoczny dla grupy,
symlink katalogu i podstawiony regularny plik zamiast socketu.
`test_fixture_socket_bind_refuses_to_change_main_process_umask` sprawdza
odmowę przed operacją w procesie głównym. Dotychczasowe testy symlinków,
podmiany socketu i rzeczywistego połączenia korzystają z tego samego helpera.

```bash
python -m pytest -q tests/security/test_freshness_runtime_ipc.py -k 'fixture_socket or real_symlink or real_socket_replacement or real_local_unix'
```

Pozytywny wynik tych regresji nie nadaje pełnej kwalifikacji IPC produktu,
nie rozwiązuje ograniczenia procfs między odrębnymi UID i nie zmienia
`STAGE9_PRODUCTION_READY=NO`.
