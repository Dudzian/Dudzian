# Przewodnik użytkownika

## Wprowadzenie
Ten przewodnik opisuje kompletny proces przygotowania, konfiguracji i obsługi lokalnego bota handlowego. Dokument stanowi podstawową referencję dla użytkowników końcowych dystrybucji desktopowej.

## Wymagania wstępne
- System operacyjny Windows 11, macOS 13+ lub Linux (Ubuntu 22.04 LTS / Fedora 39).
- Procesor x86_64 lub Apple Silicon, co najmniej 16 GB RAM.
- Połączenie internetowe wymagane jedynie do pobierania danych giełdowych lub aktualizacji.
- Aktywna licencja powiązana z odciskiem sprzętowym.

## Instalacja
1. Pobierz instalator odpowiedni dla systemu z pakietu dystrybucyjnego.
2. Uruchom instalator z uprawnieniami administratora i postępuj zgodnie z kreatorem.
3. Przy pierwszym uruchomieniu aplikacja poprosi o rejestrację licencji – przygotuj klucz aktywacyjny.
4. W systemach macOS/Linux upewnij się, że aplikacja ma uprawnienia do dostępu do keychaina / menedżera sekretów.

Szczegóły budowy i dystrybucji instalatorów opisano w `docs/deployment/installer_build.md`.

## Aktywacja licencji
1. Uruchom aplikację i przejdź do zakładki **Kreator**.
2. Wprowadź klucz licencji oraz identyfikator klienta.
3. Kreator wygeneruje odcisk sprzętowy i zapisze podpisany token licencyjny.
4. Po pomyślnej weryfikacji licencji zostanie odblokowana konfiguracja giełd i strategii.

Informacje szczegółowe znajdują się w `docs/security/runbook.md`.

## Konfiguracja giełd i kluczy API
1. W kreatorze wybierz docelowe giełdy (Binance, Coinbase, Kraken, OKX, Bitget, Bybit).
2. Podaj klucze API i sekrety – zostaną zaszyfrowane w systemowym keyringu.
3. Uruchom test połączenia, aby potwierdzić autoryzację i synchronizację zegara.
4. Zapisz profil połączenia, który będzie wykorzystywany przez runtime.

## Konfiguracja strategii
1. W zakładce **Konfigurator strategii** wybierz interesujące profile (np. trend following, options income, cross-exchange hedge).
2. Dostosuj parametry, korzystając z walidacji po stronie UI oraz podpowiedzi.
3. Zapisz konfigurację do `config/runtime.yaml` lub utwórz nowy preset dla marketplace.
4. Opcjonalnie uruchom optymalizację parametrów z poziomu sekcji **Optymalizacja**.

## Wykorzystanie kreatora konfiguracji (Setup Wizard)
Kreator prowadzi użytkownika krok po kroku:
1. Walidacja licencji.
2. Dodanie giełd i test połączeń.
3. Wybór strategii i przypisanie do portfeli.
4. Ustawienia alertów i personalizacji interfejsu.
5. Podsumowanie konfiguracji i uruchomienie runtime.

## Dashboard portfela
Zakładka **Dashboard** udostępnia:
- Krzywe P&L w ujęciu dziennym i godzinowym.
- Ekspozycję per giełda, para handlowa i strategia.
- Stan portfela oraz aktywne zlecenia.
- Ostatnie alerty i rekomendacje zarządzania ryzykiem.

Wykresy i dane pochodzą z lokalnego serwera gRPC oraz eksportera Prometheus.

## Alerty i powiadomienia
- Alerty krytyczne (np. utrata połączenia, ryzyko przekroczenia limitów) pojawiają się w nakładce toast oraz w logach.
- Można skonfigurować powiadomienia e-mail lub systemowe poprzez kreator.
- Historia alertów dostępna jest w panelu bocznym Dashboardu.

## Marketplace presetów
1. Przejdź do zakładki **Marketplace**.
2. Wybierz lokalny katalog z paczkami presetów lub podłącz zewnętrzny dysk.
3. Importuj preset – aplikacja zweryfikuje podpis i wersję.
4. Aktywuj preset, aby został dodany do katalogu strategii.
5. Eksportuj własne ustawienia, aby udostępnić je innym użytkownikom.

Więcej informacji: `docs/marketplace/offline_usage.md`.

## Optymalizacja strategii
1. Z menu narzędzi wybierz **Optymalizacja** i wskaż strategię oraz zakres parametrów.
2. Wybierz tryb: grid search (deterministyczny) lub Bayesian (adaptacyjny).
3. Uruchom zadanie – postęp widać w panelu monitoringu i w logach.
4. Raporty wyników (HTML/PDF/JSON) zapisują się w `var/reports/optimization/`.
5. Zastosuj najlepszy preset bezpośrednio z raportu lub marketplace.

Szczegóły techniczne opisano w `docs/reporting/optimization.md` oraz `docs/training/`.

## Tryb runtime
- Do uruchamiania środowiska lokalnego służy `scripts/run_local_bot.py`.
- Logi runtime zapisywane są w `logs/` oraz eksportowane do Prometheusa.
- W przypadku pracy offline aplikacja korzysta z lokalnych cache danych.

## Bezpieczeństwo i kopie zapasowe
- Klucze API są przechowywane wyłącznie w systemowym keyringu.
- Licencja i fingerprint znajdują się w `secrets/license/` – wykonuj regularne kopie.
- Raporty i modele przechowywane są w `var/` – zachowaj politykę backupu.

## Aktualizacje aplikacji
- Aktualizacje dystrybuowane są jako nowe wersje instalatora lub paczki różnicowe.
- Przed aktualizacją zatrzymaj runtime i wykonaj kopię konfiguracji.
- Po instalacji uruchom kreator w trybie „Sprawdź konfigurację”, aby potwierdzić integralność ustawień.

## Gdzie szukać pomocy?
- Sekcja troubleshooting: `docs/user_manual/troubleshooting.md`.
- Plan wsparcia i procedura kontaktu: `docs/support/plan.md`.
- Kanały społecznościowe i FAQ znajdują się na stronie partnerów OEM.

