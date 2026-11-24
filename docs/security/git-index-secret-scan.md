# Skanowanie sekretów z indeksu Git — #3280

Stan bazowy: `b143b570a8159426020ddaac8c31d74059667c70`.
Gałąź: `security/git-index-scan-20261117`.

## Zakres poprawki

Hook staged skanuje niezmienne bloby wskazane przez indeks Git. Jedno wywołanie
Betterleaks otrzymuje dotychczasowe ścieżki względne w prywatnym katalogu
roboczym. Rozpoznaje dodanie, modyfikację, rename jako nowy cel oraz zmianę typu.
Usunięcie nie wprowadza nowej zawartości do commita. Blob dowiązania jest
skanowany jako tekst jego celu; hook nie tworzy ani nie śledzi dowiązania.

Ustawienia `diff.ignoreSubmodules` i konfiguracja submodułów nie mogą ukryć
gitlinku: odczyt indeksu jawnie wymusza `--ignore-submodules=none`, a nieobsługiwany
typ nadal kończy się odmową. Hook nie zmienia zawartości indeksu ani working tree.

Nierozstrzygnięty indeks, brak obiektu, niekompletny odczyt, niebezpieczna
ścieżka, awaria materializacji lub zmiana indeksu podczas skanu powodują odmowę.
Brak kopii w working tree nie pomija bloba z indeksu. Testy korzystają z
rzeczywistych jednorazowych repozytoriów i wyłącznie syntetycznych danych.
Przed odczytem blobów hook odrzuca aliasy wielkości liter (także katalogów),
końcowe kropki/spacje, nazwy urządzeń i znaki niedozwolone na Windows.
Zapis wyłączny (`xb`) dodatkowo uniemożliwia nadpisanie istniejącego celu.

Jednorazowy katalog tymczasowy jest prywatny i usuwany także przy odmowie.
Nie jest artefaktem CI. Komenda skanera zachowuje `--redact=100`; przechwycone
komunikaty obu strumieni skanera nie są przekazywane do logów, ponieważ mogą zawierać
konfigurację i wrażliwe ścieżki. Zamiast tego hook zgłasza kategorię odmowy.
Komunikaty o niepełnym skanie, pominiętych binariach/archiwach i błędach
uniemożliwiają uznanie kodu 0 skanera za kompletny wynik.

## Granice

- Blob większy niż 32 MiB, łączny snapshot większy niż 128 MiB oraz przekroczenie
  istniejącego limitu długości komendy powodują kod 2, bez dzielenia zakresu
  ani pomijania plików. Limity chronią zasoby lokalnego hooka.
- NUL, tekst inny niż UTF-8, gitlink i nierozpoznane typy wymagają osobnego
  przeglądu; nie otrzymują cichego PASS. Zmiana może zablokować dodanie
  uzasadnionego zasobu binarnego, dopóki nie zostanie ustalona jego kontrola.
- Zachowano baseline i nazwy względne. Kontrola z prawdziwym Betterleaks 1.9.0
  sprawdza tożsamość baseline oraz odmowę nowego znaleziska poza nim.
- Konfiguracja TOML i pliki ignore nadal pochodzą z katalogu repozytorium;
  względne zmienne `BETTERLEAKS_CONFIG`/`GITLEAKS_CONFIG` są zamieniane na
  ścieżki bezwzględne. Zachowano pierwszeństwo konfiguracji ze środowiska.
  Jawny wybór domyślnych reguł zapobiega samoczynnemu załadowaniu konfiguracji
  istniejącej wyłącznie w indeksie. Względne zależności konfiguracji (`extend.path`)
  mogą wymagać ścieżki bezwzględnej; błąd ich ładowania blokuje commit.
- Jawne argumenty CLI nadal wybierają working tree, lecz nieczytelne,
  niebezpieczne lub inne niż zwykły plik cele powodują odmowę.
- To kontrola staged, odrębna od skanu working tree w CI i od skanu historii.
  Nie dowodzi braku sekretów w logach, zewnętrznych artefaktach ani całej historii.
  Historyczne kandydaty opisane w istniejącym przeglądzie baseline pozostają
  nierozstrzygnięte do czasu odrębnego potwierdzenia właściciela.
- Lokalny hook nie jest granicą zaufania przeciw osobie mogącej zmieniać sam
  hook, baseline lub konfigurację skanera. Niezależne bramki CI pozostają wymagane.

## Walidacja

Negatywne testy uruchomione na niezmienionym kodzie bazowym: **5 FAILED**,
kod 1. Dowodzą braków kontroli zawartości indeksu i typów plików, a nie
wycieku rzeczywistego sekretu. Wyniki po poprawce i dokładne SHA zapisuje
zbiorczy checkpoint audytu oraz opis draft PR.

Polecenie komponentu:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 \
python -m pytest tests/quality/test_index_secret_scan.py \
  tests/quality/test_secret_hygiene.py -q
```

Zakres sąsiedni: pre-commit, lokalne ścieżki i interpretacja wyniku Betterleaks.
Brak zmian protokołów cloud, risk gate, licencji, update i kwalifikacji Stage 9/10.
Natywne zachowanie Windows wymaga osobnych wyników CI. Stan pozostaje:

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
