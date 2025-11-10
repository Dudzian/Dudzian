# Panel Marketplace w aplikacji desktopowej

Niniejszy dokument opisuje sposób korzystania z nowej warstwy backendowej
odpowiedzialnej za zakładkę **Marketplace** w aplikacji desktopowej bota.
Moduł jest w pełni offline – katalog presetów znajduje się lokalnie w
`bot_core/strategies/marketplace/`, a instalacja wykorzystuje repozytorium
plików w formacie JSON.

## Struktura katalogu

```
bot_core/strategies/marketplace/
├── catalog.yaml          # Manifest presetów wraz z metadanymi
├── licenses/             # Opcjonalne licencje OEM (powiązane z fingerprintem)
└── presets/              # Fizyczne presety podpisane kluczem Ed25519
```

Każdy wpis w `catalog.yaml` definiuje identyfikator (`id`), nazwę, wersję,
autora, wymagane giełdy (`required_exchanges`) oraz ścieżkę do pliku presetu.
Artefakty są podpisane kluczem Ed25519 – podpis przechowywany jest bezpośrednio
w pliku JSON.

## Backend Marketplace

Nowy moduł `bot_core.strategies.installer.MarketplacePresetInstaller`
realizuje walidację podpisu, kontrolę licencji oraz dopasowanie fingerprintu
sprzętowego. Do jego obsługi z perspektywy UI służy klasa
`bot_core.ui.api.MarketplaceService`, która udostępnia spójne API:

```python
from pathlib import Path

from bot_core.marketplace import PresetRepository
from bot_core.security.hwid import HwIdProvider
from bot_core.strategies.installer import MarketplacePresetInstaller
from bot_core.ui.api import MarketplaceService

marketplace_dir = Path("bot_core/strategies/marketplace")
licenses_dir = marketplace_dir / "licenses"
repository = PresetRepository(Path("var/marketplace_installed"))
installer = MarketplacePresetInstaller(
    repository,
    catalog_path=marketplace_dir,
    licenses_dir=licenses_dir,
    hwid_provider=HwIdProvider(fingerprint_reader=lambda: "OEM-TEST-DEVICE"),
)
service = MarketplaceService(installer, repository)
```

### Dostępne operacje

* `service.list_presets()` – zwraca listę obiektów `MarketplacePresetView`,
  zawierających metadane, status podpisu oraz wynik weryfikacji fingerprintu.
* `service.list_presets_payload()` – wariant gotowy do serializacji do JSON
  (w tym flagi `installed` i `installedVersion`).
* `service.install_from_catalog(preset_id)` – instaluje preset korzystając z
  wpisu w katalogu.
* `service.install_from_file(path)` – importuje preset z wybranej lokalizacji
  (np. plik wskazany przez użytkownika w UI).
* `service.remove_preset(preset_id)` – usuwa preset z lokalnego repozytorium.
* `service.export_preset(preset_id, format="yaml")` – eksportuje zainstalowany
  preset do formatu JSON/YAML (przydatne przy backupach lub diagnostyce).
* `service.plan_installation(preset_ids)` – oblicza plan instalacji dla
  wskazanych presetów, uwzględniając zależności oraz sugerowane aktualizacje.
  Wariant `plan_installation_payload()` zwraca gotowy do serializacji słownik z
  polami `installOrder`, `requiredDependencies`, `issues` oraz `upgrades`.

Każda instalacja zwraca `MarketplaceInstallResult`, który informuje o statusie
podpisu (`signature_verified`), wyniku dopasowania fingerprintu oraz liście
problemów (`issues`) i ostrzeżeń (`warnings`). UI może wykorzystać te informacje
do prezentacji alertów i komunikatów doradczych (np. wygasająca subskrypcja
czy brak przydziału stanowiska).

Widok kart presetu prezentuje również sekcję **Ostrzeżenia licencji**, w której
wyświetlane są znormalizowane komunikaty wygenerowane przez walidator (np.
zapełniona pula seatów, pauza w subskrypcji). Bezpośrednio poniżej znajduje się
panel **Licencja** z podsumowaniem przydzielonych urządzeń, dostępnych miejsc
oraz statusem subskrypcji – dane te pochodzą z `license.validation` oraz
sekcji `seat_summary`/`subscription_summary` wygenerowanych przez backend.

## Integracja z QML

Warstwa QML (`ui/qml/views/Marketplace.qml`) powinna wywoływać metody
`MarketplaceService` poprzez kontroler aplikacji. Przykładowy adapter:

```python
def marketplaceListPresets(self):
    return self._marketplace_service.list_presets_payload()

def marketplaceImportPreset(self, url):
    result = self._marketplace_service.install_from_file(url.toLocalFile())
    return {
        "success": result.success,
        "issues": list(result.issues),
        "warnings": list(result.warnings),
    }
```

Po każdej operacji instalacji lub usunięcia warto ponownie wywołać
`list_presets_payload()`, aby odświeżyć widok.

## Testy

Zestaw testów jednostkowych obejmuje zarówno moduł instalatora, jak i warstwę
UI:

* `tests/strategies/test_marketplace_installer.py`
* `tests/ui/test_marketplace_flow.py`

Przed uruchomieniem testów należy zainstalować zależności developerskie i
wygenerować stuby gRPC zgodnie z instrukcją w `README.md`.
