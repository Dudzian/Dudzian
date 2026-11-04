# Stage 9 — ACCEPT/REJECT startu Freshness na izolowanej Ubuntu 24.04 VM

Stan nadrzędny: **IN_PROGRESS / PRODUCTION_NO_GO**. Dokument definiuje przyszłą,
odrębnie zatwierdzoną próbę TEST_ONLY; nie jest zgodą na jej wykonanie.
Nie wykonywano instalacji uprzywilejowanej, restartu ani operacji na Dellu.
Obowiązują istniejące bramki
[harnessu VM](STAGE9_LINUX_VM_HARNESS.md), izolacji i zatwierdzenia operatora.
Wynik dotyczy wyłącznie startu i odmów minimalnej usługi. Nie otwiera pełnej
kwalifikacji produktu, AccountGenesis, enrollment ani Stage 10.

## Co dowodzą testy PR

| Odcinek | Dowód automatyczny | Granica dowodu |
| --- | --- | --- |
| Manifest aplikacji v2 → instalator | Rzeczywiste podpisy Ed25519, bajty i inwentarz; instalacja pod tymczasowym rootem; odmowy podpisanych podmian trzech helperów | Tożsamość hosta, użytkownicy, systemd i runtime są modelowane |
| Instalator → zachowane zatwierdzenie | Dokładne manifest/signature/authority, binding publikowany na końcu; brak adopcji częściowej instalacji i odmowy driftu | Nie jest to uprzywilejowana instalacja VM |
| Zatwierdzenie → launcher → FreshnessVerifier | Test pomostowy wykonuje rzeczywistą weryfikację aplikacji, launcher, import zainstalowanych modułów i konstruktory usługi; malformed request jest rzeczywiście odrzucane | Runtime, kanoniczne ścieżki/root, NSS, połączenie DB i transport socketu są modelowane; brak dodatniego dowodu działającej usługi |
| Podpisany runtime | Oddzielne testy sprawdzają prawdziwe podpisy, kompletny inwentarz, odmowy i publikację do tymczasowego rootu; osobny workflow wykonuje offline build/importy na Noble | Nie łączą same z siebie pełnej instalacji v2, startu systemd, żywego PostgreSQL i transportu |
| `ExecStartPre` | Regresje wymagają kontroli runtime i aplikacji przed pozwoleniem na start; odmowa zatwierdzenia zatrzymuje ścieżkę | Modele wykonania nie potwierdzają zachowania rzeczywistego managera usług |

Nie wolno sumować oddzielnych pozytywnych modeli jako dowodu pełnego E2E.
Poniższa próba ma zebrać brakujące, połączone dowody.

## Warunki wejścia — każdy wymagany przed próbą

1. Odrębna zgoda na jednorazową Ubuntu 24.04/amd64 VM z systemd jako PID 1.
   Zgoda wiąże dokładny commit, machine/boot ID, obraz i snapshot, wersje
   pakietów oraz identyfikator środowiska. Kontener, fizyczny Dell, aktywna
   sieć, współdzielony katalog hosta lub inna maszyna oznaczają REJECT.
2. Niezależny reviewer akceptuje dokładne bajty istniejącego bootstrapu,
   manifest aplikacji i osobny manifest runtime. Operator otrzymuje
   fingerprinty publicznych autorytetów odrębnym kanałem. Klucze prywatne
   TEST_ONLY pozostają poza VM, repozytorium i nośnikiem. Klucze produkcyjne
   nie są używane. Sam podpis kandydata bez review nie jest akceptacją.
3. Zaufany systemowy OpenSSL sprawdza podpisy dokładnych manifestów i
   `bootstrap.sha256`. Operator porównuje wszystkie wpisy sum przed pierwszym
   uruchomieniem kodu bootstrapu oraz po skopiowaniu do chronionego źródła.
   Zapisuje wyniki, rewizję, fingerprinty, typy, właścicieli i tryby plików.
   Zastane/nieznane kopie, symlinki, hardlinki lub zapis grupy/innych oznaczają
   REJECT. Samo `root:root/0644` nie potwierdza pochodzenia.
4. Runtime spełnia istniejący kontrakt Noble CPython 3.12, ABI, pełnego locka,
   wheelhouse, stdlib/native `external_files`, braku user/system-site,
   dodatkowych `.pth`, cache i pakietów. Publiczne autorytety są poza payloadem.
   Binding runtime odpowiada dokładnie rewizji i manifestowi aplikacji.
5. Wszystkie sześć pełnych `REVIEWED_CODE_DIGESTS` pozostaje bez zmian.
   Obecny AccountGenesis nadal nie spełnia zamrożonego pinu; ten fakt nie może
   być ukryty, automatycznie naprawiony ani zastąpiony pozytywnym wynikiem VM.

## ACCEPT — wszystkie obserwacje muszą być rzeczywiste i zgodne

| Obszar | Wymagane obserwacje |
| --- | --- |
| Preflight i instalacja | Rzeczywisty instalator przyjmuje dokładny podpisany manifest v2 i osobno zatwierdzony runtime. Preflight nie zmienia mierzonego hosta. Instalacja publikuje komplet plików, manifest/signature/authority i końcowy binding; raportowany SHA-256 odpowiada oryginalnym bajtom manifestu. Trzy helpery payloadu odpowiadają zatwierdzonym siblingom instalatora. |
| Kolejność startu | Dziennik właściwej jednostki pokazuje pomyślny `ExecStartPre`: kontrolę runtime i aplikacji przed `ExecStart`. Jednostka odpowiada dokładnemu istniejącemu kontraktowi, bez lokalnych override lub zamiany komendy. Sam kod wyjścia preflight nie zastępuje tej obserwacji. |
| Proces i interpreter | `MainPID`/boot/lifetime są spójne. Proces działa jako właściwe istniejące konto/grupa usługi, nie root. Ścieżka, urządzenie/inode i SHA-256 rzeczywistego `/proc/<pid>/exe` odpowiadają zatwierdzonemu runtime. Importy pochodzą z podpisanego inwentarza, bez zapisu cache i obcego kodu. |
| PostgreSQL | Rzeczywisty izolowany klaster i konfiguracja odpowiadają kontraktowi. Usługa wykonuje wymagany odczyt przez właściwy peer/rolę/bazę. Zbierz stałe kategorie i wyniki kontroli uprawnień bez haseł, kluczy i wrażliwych wierszy. Model połączenia nie zalicza tej pozycji. |
| FreshnessVerifier i transport | Rzeczywisty AF_UNIX ma właściwe uprawnienia i tożsamość peer. Rzeczywista usługa obsługuje ograniczone, poprawne żądanie z uprzednio zatwierdzonym materiałem TEST_ONLY i zwraca dokładny oczekiwany wynik istniejącego protokołu. Wadliwe/podmienione/replay żądanie daje właściwą odmowę bez nieautoryzowanego zapisu. Sam żywy PID lub socket nie wystarcza. |
| Ponowny start i dowody | Ponowienie startu nie zmienia ani nie adoptuje podpisanych bajtów/bindingów. Raport wiąże wszystkie wyniki z tą samą rewizją, boot ID i manifestami; zachowuje surowe kody wyjścia oraz jawne pominięcia. Cleanup i stan guard danych odpowiadają zatwierdzonemu planowi VM. |

Żaden etap nie może wymagać zmiany pinu AccountGenesis, progu coverage,
wyłączenia skanera lub ominięcia produkcyjnego kwalifikatora. Brak wejściowego
materiału pozwalającego wykonać poprawne żądanie oznacza BLOCKED/NOT_EXECUTED,
a nie ACCEPT na podstawie samych odmów.

## REJECT — macierz negatywna na odtwarzalnym snapshotcie VM

Każdą zmianę wykonać tylko zgodnie z odrębnie zatwierdzonym planem jednorazowej
VM. Przywracać snapshot pomiędzy przypadkami; nie rekonstruować dowodów przez
automatyczne podmienianie hashy lub kluczy.

| Przypadek | Wymagana odmowa |
| --- | --- |
| Nieprawidłowy podpis bootstrapu lub inny fingerprint | Operator zatrzymuje próbę przed wykonaniem jego Pythona; plik sum z bundle nie może zatwierdzić własnego autora |
| Podpisany payload z innymi bajtami jednego z trzech helperów | Instalator odmawia przed publikacją; brak nowego zatwierdzenia i zmian mierzonego hosta |
| Zły podpis aplikacji/runtime, rewizja, scope lub runtime binding | Instalator albo kontrola startowa odmawia; brak aktywnej usługi |
| Podmiana/usunięcie launchera, helpera, manifestu lub retained bindingu po instalacji | `ExecStartPre` odmawia przed wykonaniem launchera; brak procesu właściwego verifiera, dostępu DB i żądania socketu |
| Dodatkowy niepodpisany plik/cache, obcy symlink/hardlink, błędny właściciel lub zapis grupy/innych | Odmowa inwentarza/ochrony ścieżek; brak adopcji i automatycznego naprawiania |
| Interpreter lub jego podpisane zależności/`external_files` różnią się od manifestu | Odmowa runtime przed `ExecStart`; bez fallbacku do systemowych bibliotek |
| Częściowa instalacja albo przerwany binding/journal | Odmowa startu/adopcji; wymagane ręczne recovery zgodne z istniejącym kontraktem |
| Nieprawidłowy peer/rola/baza/socket, timeout, malformed lub replay request | Istniejąca konkretna odmowa; brak autoryzacji i nieuprawnionych efektów ubocznych |

## Format rozstrzygnięcia

Raport musi oddzielić `VM_FRESHNESS_SERVICE_START=ACCEPT/REJECT/BLOCKED` od
`SIGNED_FRESHNESS_SUBSTRATE`, które nadal zwraca
`production_qualification=WITHHELD`. Brak któregoś wymaganego dowodu oznacza
BLOCKED; zaobserwowana niezgodność oznacza REJECT. Pełne
`PRODUCTION_LOCAL` i AccountGenesis pozostają BLOCKED, jeżeli nie spełniają
własnych niezależnych wymagań. Niezależnie od wyniku tej próby Stage 9 zachowuje
**IN_PROGRESS / PRODUCTION_NO_GO**; nie deklarować uruchomienia produktu.
