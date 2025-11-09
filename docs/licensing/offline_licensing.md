# Licencjonowanie offline

Ten moduł wprowadza obsługę licencji offline podpisanych kluczem Ed25519. Pakiet
licencyjny (`.lic`) zawiera dwa pola: `payload_b64` oraz `signature_b64`. Po
zdekodowaniu i weryfikacji podpisu generowane są *capabilities* opisujące dostępne
funkcje, limity i moduły w ramach aplikacji bot_core oraz towarzyszących launcherów
CLI.

## Kluczowe komponenty

| Ścieżka | Opis |
| --- | --- |
| `bot_core/security/capabilities.py` | Definicje struktur danych opisujących moduły, limity i statusy trial/maintenance. |
| `bot_core/security/license_service.py` | Ładowanie pakietu `.lic`, weryfikacja Ed25519, budowa `LicenseCapabilities` oraz zapis migawki do `var/security/license_status.json` i loga JSONL `logs/security_admin.log`. |
| `bot_core/security/clock.py` | Monotoniczny zegar zapisujący ostatnią datę użycia licencji (anti-rollback). |
| `bot_core/security/hwid.py` | Dostawca fingerprintu sprzętowego zgodnego z `bot_core.security.fingerprint`. |
| `bot_core/security/guards.py` | Runtime'owe strażniki do egzekwowania limitów i wymagań edycji/modułów. |
| `python scripts/generate_license.py` | CLI do podpisywania payloadów JSON i generowania pakietów `.lic`. |

## Integracja runtime

`bot_core/runtime/bootstrap.py` wczytuje licencję offline, jeżeli zdefiniowano
zmienne środowiskowe:

```bash
export BOT_CORE_LICENSE_PATH=/opt/dudzian/licenses/acme.lic
export BOT_CORE_LICENSE_PUBLIC_KEY=0123ABCD...
```

Po pozytywnej weryfikacji `LicenseValidationResult.capabilities` zawiera
przetworzone uprawnienia. Moduły i launchery mogą korzystać z
`CapabilityGuard`, aby egzekwować dostęp do giełd, strategii, czy limitów
instancji. `ClockService` utrwala ostatnią datę użycia licencji, dzięki czemu
cofnięcie zegara systemowego nie przywróci wygasłych uprawnień, a
`HwIdProvider` dopasowuje pole `hwid` z payloadu do lokalnego fingerprintu
hosta. Migawka licencji (edycja, moduły, status trial/maintenance, skrót
payloadu) jest zapisywana w `var/security/license_status.json`, a zdarzenie
z weryfikacji trafia jako wpis JSONL do `logs/security_admin.log`, co
ułatwia audyt działań operatorów OEM.

## Integracja z interfejsami GUI

*Trading GUI* (`bot_core.ui.trading`) prezentuje podsumowanie licencji w
nagłówku okna i automatycznie blokuje niedozwolone funkcje:

- combobox sieci usuwa opcję **Live**, jeśli licencja nie dopuszcza środowiska
  produkcyjnego lub edycji Pro;
- tryb **Futures** jest ukrywany, gdy moduł `futures` nie został aktywowany;
- przycisk **Start** pozostaje nieaktywny, jeśli runtime `auto_trader` jest
  niedostępny w licencji.
- status licencji i komunikaty (np. konieczność rozszerzenia modułu) są
  aktualizowane w czasie rzeczywistym i powiązane z błędami strażnika
  `CapabilityGuard`.

Panel administracyjny Qt prezentuje edycję, aktywne moduły, środowiska,
utrzymanie/trial oraz dane posiadacza licencji. Listy modułów i runtime są
pobierane z `LicenseCapabilities`, co pozwala operatorom OEM szybko
zidentyfikować brakujące rozszerzenia bez zaglądania do plików `.lic`.

Kontroler sesji handlowej rezerwuje slot licencyjny (`paper_controller` lub
`live_controller`) w momencie startu i zwalnia go po zatrzymaniu. W przypadku
braku uprawnień użytkownik otrzymuje komunikat z instrukcją kontaktu z opiekunem
licencji, zgodnie z wymaganiami OEM.

## Launcher AutoTrader (headless)

`bot_core.auto_trader.app` stanowi kanoniczny runtime, a
`bot_core.auto_trader.paper_app.PaperAutoTradeApp` ładuje `LicenseCapabilities`
przy uruchomieniu. Strażnik licencji blokuje start, gdy brakuje modułu
`auto_trader`, odpowiedniego środowiska (`paper/demo` lub `live`) albo
przekroczono limity instancji. Wymagane zachowania:

- tryb **Live** wymaga edycji co najmniej *Pro* oraz dostępu do środowiska `live`;
- środowiska futures (np. `binance_futures`, `kraken_futures`) wymagają modułu
  `futures`;
- przed uruchomieniem rezerwowane są sloty licencyjne `paper_controller`/`live_controller`
  oraz `bot`; przekroczenie limitów skutkuje `LicenseCapabilityError` i alertem
  `license_restriction`.

Launcher CLI (wykorzystujący `PaperAutoTradeApp.main()`) kończy działanie kodem
wyjścia różnym od zera przy naruszeniu restrykcji, a logi zawierają komunikat o
blokadzie licencyjnej. Wywołanie deleguje do runtime `bot_core.auto_trader.app`,
dlatego scenariusze licencyjne należy testować również na importerach
korzystających bezpośrednio z pakietu `bot_core`.

## Generator licencji

Przykład użycia CLI do podpisania licencji:

```bash
python scripts/generate_license.py payload.json <private_key_hex> -o acme.lic --verify <public_key_hex>
```

Po wygenerowaniu narzędzie zapisuje pakiet `.lic` i wyświetla sumę SHA-256
payloadu.
