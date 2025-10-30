# Marketplace offline – import i dystrybucja presetów

Ten dokument opisuje obsługę lokalnego marketplace presetów strategii w środowisku
offline. Marketplace działa całkowicie na komputerze użytkownika i wykorzystuje
nowe serwisy gRPC udostępniane przez `run_local_bot.py`.

## Struktura i weryfikacja presetów

* Pliki presetów to dokumenty JSON lub YAML zawierające sekcję `metadata.id`,
  `metadata.version` oraz konfigurację strategii i ryzyka.
* Integralność zapewniają podpisy Ed25519. Podpis można osadzić wraz z kluczem
  publicznym lub zweryfikować poprzez zaufane klucze w `runtime.yaml`:

```yaml
marketplace:
  enabled: true
  presets_path: config/marketplace/presets
  signing_keys:
    authors/main: "BASE64_PUBLIC_KEY"
```

## CLI – generowanie i weryfikacja presetów

Do pracy z presetami służy zaktualizowane narzędzie
`scripts/marketplace_cli.py`:

| Polecenie | Opis |
|-----------|------|
| `package` | Buduje podpisany preset na podstawie specyfikacji JSON/YAML. |
| `validate` | Weryfikuje podpisy oraz fingerprint licencyjny paczek. |
| `list`/`show` | Przegląda lokalny katalog marketplace. |

Przykład generowania podpisanego presetu:

```bash
python scripts/marketplace_cli.py package presets/mean_reversion.yaml \
  --key-id authors/main \
  --private-key secrets/marketplace_author_ed25519.pem \
  --issuer "Marketplace Guild" \
  --format json
```

Wygenerowany plik zostanie zapisany obok specyfikacji z rozszerzeniem `.json`
(lub `.yaml` przy wyborze formatu YAML).

## Serwis gRPC Marketplace

Serwer lokalnego runtime udostępnia nowy serwis `MarketplaceService` z metodami:

* `ListPresets` – zwraca dostępne presetów wraz z metadanymi i statusem podpisu,
* `ImportPreset` – importuje podpisany dokument (JSON/YAML) do katalogu,
* `ExportPreset` – eksportuje wybrany preset w żądanym formacie,
* `RemovePreset` – usuwa preset z repozytorium,
* `ActivatePreset` – odświeża katalog strategii i zwraca aktualne metadane.

Warstwa UI korzysta z tych metod poprzez nowy widok `Marketplace.qml`, dzięki
czemu zarządzanie presetami jest dostępne bezpośrednio z aplikacji desktopowej.

## Interfejs użytkownika

Zakładka **Marketplace** w aplikacji desktopowej pozwala na:

* podgląd dostępnych presetów (nazwa, wersja, profil, tagi, status podpisu),
* import plików JSON/YAML poprzez przycisk **Importuj…**,
* eksport aktywnych presetów do wybranego formatu (`yaml` lub `json`),
* aktywację oraz usuwanie presetów jednym kliknięciem.

Widok komunikuje się bezpośrednio z `TradingClient` korzystając z metody
`MarketplaceService`, dzięki czemu każde działanie odświeża katalog strategii
w czasie rzeczywistym.

## Ścieżki plików i konfiguracja

Domyślny katalog na pliki to `config/marketplace/presets`. Ścieżkę można
zmienić w `config/runtime.yaml`:

```yaml
marketplace:
  enabled: true
  presets_path: /mnt/secure/presets
  signing_keys:
    authors/main: "BASE64_PUBLIC_KEY"
  allow_unsigned: false
```

* `signing_keys` – mapa zaufanych kluczy publicznych (base64 lub hex).
* `allow_unsigned` – gdy ustawione na `true`, pozwala na import presetów bez
  podpisów (niezalecane).

Po imporcie lub usunięciu presetu runtime automatycznie odświeża katalog
strategii, dzięki czemu nowe konfiguracje są natychmiast dostępne w UI oraz w
silniku strategii.
