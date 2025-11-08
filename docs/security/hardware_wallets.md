# Hardware wallets w środowisku Dudzian

## Przegląd

Warstwa egzekucji live wymaga sprzętowego podpisu wypłat (withdrawal/payout)
jeżeli licencja OEM zawiera flagę `security.require_hardware_wallet_for_outgoing`
lub konfiguracja runtime wymusza podpis urządzeniem. Obsługiwane są portfele
Ledger (ECDSA/secp256k1) oraz Trezor (EdDSA/Ed25519). W środowiskach
deweloperskich i podczas testów dostępny jest symulator oparty na bibliotece
`cryptography` — aktywowany poprzez ustawienie zmiennej
`BOT_CORE_HW_SIMULATOR=1`.

Do komunikacji z urządzeniami wymagane są biblioteki `ledgerblue` (Ledger) oraz
`trezor` (Trezor). W środowisku produkcyjnym instalują je playbooki deployowe,
natomiast lokalnie można skorzystać z:

```bash
pip install ledgerblue trezor
```

Po podłączeniu urządzenia upewnij się, że odpowiednia aplikacja (np. Ethereum
na Ledgerze) jest aktywna, a system udostępnia dostęp do interfejsu HID/USB
aktualnemu użytkownikowi.  Klasy `LedgerSigner` oraz `TrezorSigner` obsługują
metodę `close()` i mogą być używane jako kontekst menedżera:

```python
from bot_core.security.hardware_wallets import LedgerSigner

with LedgerSigner(use_simulator=True) as signer:
    signature = signer.sign({"operation": "withdrawal"})
```

W trybie produkcyjnym wywołanie `close()` zwalnia uchwyty HID/USB.

## Konfiguracja runtime

Sekcja `runtime.execution.live.signers` pozwala przypisać podpisujących do
konkretnych kont giełdowych. Konfiguracja może zawierać klucz `default` oraz
mapę `accounts` z nadpisaniami per identyfikator konta. Przykład:

```yaml
runtime:
  execution:
    live:
      enabled: true
      default_route: ["primary"]
      signers:
        default:
          type: ledger
          derivation_path: "m/44'/60'/0'/0/0"
          key_id: ledger-prod
        accounts:
          trezor-backup:
            type: trezor
            simulate: true  # użyj symulatora w środowisku testowym
          paper-hmac:
            type: hmac
            key_env: PAPER_WITHDRAWAL_KEY
            algorithm: HMAC-SHA256
```

Dozwolone pola konfiguracji:

| Pole                | Typ     | Opis                                                                 |
|---------------------|---------|----------------------------------------------------------------------|
| `type`              | str     | `ledger`, `trezor` lub `hmac`                                         |
| `derivation_path`   | str     | Ścieżka BIP-32 dla urządzeń hardware                                 |
| `key_id`            | str     | Opcjonalny identyfikator klucza (zapisywany w metadanych podpisu)     |
| `simulate`          | bool    | Wymusza tryb symulatora niezależnie od zmiennej środowiskowej         |
| `seed` / `seed_hex` | bytes/str | Opcjonalne ziarno symulatora (powtarzalne testy)                    |
| `key_env`/`key_value`/`key_path` | str | Źródła klucza HMAC (analogicznie do decision logu)            |

Ścieżki plików (`key_path`) są rozwijane względem `environment.data_cache_path`.

Klasa `TransactionSignerSelector` udostępnia metodę `describe_signers()`, która
zwraca metadane wszystkich podpisujących (w tym domyślnego).  Dane te są
logowane na poziomie DEBUG podczas budowy routera live, co ułatwia audyt
konfiguracji bez ujawniania sekretów HMAC.  Każdy `TransactionSigner`
implementuje również metodę `verify(payload, signature)`, a selektor udostępnia
narzędzie `verify(account_id, payload, signature)` umożliwiające centralną
walidację podpisów.  Dodatkowo metoda `resolve_by_key_id(key_id)` zwraca krotkę
podpisujących posiadających wskazany identyfikator klucza, dzięki czemu
mechanizmy audytowe mogą w prosty sposób zweryfikować, czy konfiguracja
uwzględnia wszystkie oczekiwane urządzenia.  W przypadku gdy podpis zawiera pole
`key_id`, selektor potrafi dobrać właściwego podpisującego nawet wtedy, gdy
podpis pochodzi z innej konfiguracji konta niż aktualnie używana w routerze.
Metoda `describe_key_index()` grupuje informacje po `key_id`, wskazując listę
powiązanych kont, liczbę unikalnych signerów oraz używane algorytmy. Dane te
są logowane na poziomie DEBUG podczas startu routera, co ułatwia audyt
konfiguracji i wykrywanie niespójności.  Uzupełniająca metoda
`describe_hardware_requirements()` agreguje informacje per konto, wskazując
które trasy wymagają podpisu sprzętowego, które nadal korzystają z podpisów
software'owych oraz gdzie brakuje `key_id`.  Pozwala to szybko zweryfikować
zgodność konfiguracji z wymogami licencyjnymi i procesami operacyjnymi.

Nowo dodana metoda `describe_audit_bundle()` zwraca skonsolidowany raport
zawierający sekcje `signers`, `key_index`, `hardware_requirements` oraz listę
`issues`.  Każdy problem posiada pole `severity` (`warning` albo `critical`),
co umożliwia szybkie oszacowanie priorytetu reagowania.  Wykrywane problemy
obejmują m.in. konta korzystające z podpisów software'owych, brakujące
identyfikatory `key_id`, a także konflikty (`key_id_algorithm_conflict`) oraz
niespójności wymagań sprzętowych (`key_id_hardware_mismatch`) pomiędzy
podpisującymi dzielącymi ten sam `key_id`.  Raport jest logowany podczas startu
routera live – brak problemów powoduje pojawienie się wpisu `Konfiguracja
podpisów nie zgłasza problemów audytowych.`, natomiast znalezione niespójności
są wypisane w komunikacie `Wykryte problemy konfiguracji podpisów`.

## Integracja z licencją

`LicenseService` zapamiętuje wymóg sprzętowego podpisu wypłat w statusie
(`var/security/license_status.json`) oraz logu audytowym. Jeżeli licencja
ustawia flagę `require_hardware_wallet_for_outgoing`, uruchomienie routera live
bez skonfigurowanego `transaction_signers` zakończy się błędem.

Funkcja `build_live_execution_service` dodatkowo sprawdza, czy każdy wskazany
podpisujący (domyślny i per konto) deklaruje `requires_hardware=True`. Dzięki
temu mis-konfiguracja (np. pozostawienie podpisu HMAC) jest wychwytywana już na
etapie startu runtime, zanim jakiekolwiek zlecenie wyjdzie poza system.

## Metadane wypłat

Pomocnicza funkcja `bot_core.portfolio.payouts.require_hardware_wallet_metadata`
dodaje do metadanych zlecenia atrybuty:

- `operation: "withdrawal"`,
- `account` – identyfikator konta,
- `requires_hardware_wallet: true`.

Podczas egzekucji router zapisuje podpis w polu `hardware_wallet_signature`
oraz dodatkowe informacje (`hardware_wallet_algorithm`, `..._signed_at`).
Podpis zawiera publiczny klucz urządzenia (`device_public_x/y` dla Ledgera lub
`device_public_key` dla Trezora), dzięki czemu metoda `verify()` może zostać
wykorzystana do potwierdzenia ważności podpisu bez dostępu do fizycznego
urządzenia.  Router ponownie użyje istniejącego, poprawnego podpisu znajdującego
się w metadanych zlecenia (np. gdy wypłata była przygotowana offline).  Jeżeli
weryfikacja się powiedzie, podpis nie jest nadpisywany — uzupełniany jest
jedynie brakujący kontekst (operacja, `requires_hardware_wallet`,
identyfikator konta).  Jeżeli metadane wypłaty przechowują `hardware_wallet_key_id`
(np. podpis został zanonimizowany przed wysłaniem), router odtworzy to pole w
samym podpisie przed weryfikacją oraz w zapisanych metadanych.
