# Marketplace preset-signing key remediation

## Klasyfikacja

`config/marketplace/keys/dev-presets-ed25519.key` jest śledzonym przez Git,
deterministycznym fixture'em **DEV/TEST ONLY**. Nie jest zaufany do żadnego przyszłego
podpisu produkcyjnego, a jego publiczna tożsamość nie może zostać wybrana jako nowa
production authority.

`config/marketplace/keys/dev-hmac.key` jest również fixture'em **DEV/TEST ONLY**.
Wspólna polityka signing material odrzuca w production zarówno ten plik, jak i każdy
inny HMAC/Ed25519 secret fizycznie znajdujący się w repozytorium albo wskazany przez
alias ścieżki. Production nie ma fallbacku dla żadnego z tych sekretów.

Przed korektą udokumentowana ścieżka danych wyglądała następująco:

```text
„Rollout presetów (produkcyjny)”
→ scripts/build_marketplace_catalog.py --private-key
→ Path(...).resolve()
→ _load_ed25519_private_key / build_catalog
→ config/marketplace/keys/dev-presets-ed25519.key
```

Dodatkowo `scripts/sign_marketplace_presets.py` miał bezpośredni domyślny argument do
tego pliku, a `scripts/reconcile_exchange_presets.py` stałą `DEFAULT_PRIVATE_KEY`
wskazującą ten sam plik. Joby CI przekazują fixture jawnie i są obecnie oznaczone jako
środowisko `test`.

Po korekcie każdy signing CLI rozróżnia `dev`/`test` od `production`. Production bez
jawnego external Ed25519 i HMAC source kończy się fail-closed. Każdy HMAC entry jest
kwalifikowany przed odczytem; dotyczy to również catalog HMAC i plików review-signing.
Fizyczna ścieżka jest rozwiązywana przed odczytem materiału; production odrzuca cały
checkout repozytorium, `.git`, pliki śledzone oraz aliasy przez traversal, symlink albo
junction/reparse point. Produkcyjne review signing zabrania sekretu inline, ponieważ
nie da się wtedy potwierdzić jego zewnętrznego źródła custody.

Production wymaga również jawnych identity labels właściwych dla nowej authority.
Identyfikatory `dev-hmac`, `dev-presets`, `dev-presets-ed25519` oraz issuer
`marketplace-ci` są zarezerwowane dla DEV/TEST i odrzucane niezależnie od źródła
materiału kryptograficznego. Zapobiega to oznaczeniu nowego external production key
starą tożsamością fixture'a.

Wąski audyt `ui_marketplace_bridge.py sync-reviews` potwierdził, że mapa kluczy służy
tam wyłącznie do `verify_hmac_signature()` istniejących recenzji. Wynikiem jest zwykły,
niepodpisywany plik agregatu `.meta/reviews.json`; jedyną ścieżką tworzącą podpis HMAC
jest `submit-review`, które waliduje wybrany `--review-key-id`. Z tego powodu IDs mapy
verification w `sync-reviews` nie są kwalifikowane jako nowe signing authorities.

## Istniejące artefakty z publiczną tożsamością starego klucza

Skan publicznej tożsamości wyprowadzonej lokalnie z fixture'a (bez ujawniania private
key material) wykrył poniższe artefakty. Metadane nie pozwalają rozstrzygnąć, które z
nich zostały faktycznie opublikowane jako production. Z uwagi na dawną osiągalność z
instrukcji produkcyjnej wszystkie wymagają późniejszej, jawnej decyzji migration /
re-signing; ta korekta ich nie usuwa ani nie podpisuje ponownie.

| Ścieżka | Typ | Liczba |
| --- | --- | ---: |
| `config/marketplace/catalog.json.sig` | podpis katalogu JSON | 1 |
| `config/marketplace/catalog.md.sig` | podpis katalogu Markdown | 1 |
| `config/marketplace/packages/strategies/*.json` | podpisana paczka presetu strategii | 15 |
| `config/marketplace/packages/exchanges/*.json` | podpisana paczka presetu giełdy | 16 |
| `config/marketplace/presets/strategies/*.json.sig` | podpis źródłowego presetu JSON | 15 |
| `config/marketplace/presets/strategies/*.md.sig` | podpis źródłowego presetu Markdown | 15 |
| `config/marketplace/presets/exchanges/*.json` | osadzony podpis presetu giełdy | 16 |
| `config/marketplace/presets/exchanges/*.json.sig` | podpis źródłowego presetu JSON | 16 |
| `config/marketplace/presets/exchanges/*.md.sig` | podpis źródłowego presetu Markdown | 16 |
| `config/marketplace/presets/.meta/reviews.json.sig` | podpis metadanych recenzji JSON | 1 |
| `config/marketplace/presets/.meta/reviews.md.sig` | podpis metadanych recenzji Markdown | 1 |

Łącznie: **113 artefaktów** zawierających publiczną tożsamość starego klucza.

Status pozostaje bez zmian: Stage 9 `IN_PROGRESS`, production ceremony tooling
`READY`, production root material `NOT_PROVISIONED`, Windows production
`NOT_READY`, Stage 10 `NOT_STARTED`.
