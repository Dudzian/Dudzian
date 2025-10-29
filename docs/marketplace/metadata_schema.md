# Schemat metadanych paczek konfiguracyjnych

Niniejszy dokument opisuje pola oraz zasady rozszerzania schematu metadanych
zdefiniowanego w module `bot_core.config_marketplace.schema`.

## Podstawowe pola

| Pole                | Typ                            | Opis |
|---------------------|--------------------------------|------|
| `schema_version`    | string (semver)                | Wersja schematu metadanych. Aktualne implementacje powinny używać `1.0.0`. |
| `config_id`         | string                         | Unikalny identyfikator paczki w obrębie marketplace. Dozwolone litery, cyfry, `_`, `-`, `.`. |
| `config_version`    | string (semver)                | Wersja samej konfiguracji. |
| `title`             | string                         | Czytelna nazwa konfiguracji. |
| `description`       | string                         | Szczegółowy opis działania i przeznaczenia paczki. |
| `author`            | string                         | Imię i nazwisko lub nazwa zespołu odpowiedzialnego. |
| `author_contact`    | string                         | Kanał kontaktu (e-mail, URL, itp.). |
| `license`           | [`LicenseInfo`](#informacje-o-licencji) | Informacje o licencji paczki. |
| `data_requirements` | Lista [`DataRequirement`](#wymagania-danych) | Opis potrzebnych danych wejściowych/wyjściowych. |
| `component_dependencies` | Lista [`ComponentDependency`](#zależności-komponentów) | Zależności od innych elementów ekosystemu. |
| `integrity`         | [`IntegrityInfo`](#integralność-paczki) | Informacje o sumach kontrolnych, podpisach i whitelistach fingerprintów. |
| `created_at`        | datetime (ISO 8601)            | Data utworzenia metadanych. |
| `updated_at`        | datetime (ISO 8601)            | Data ostatniej aktualizacji. |
| `tags`              | Lista string                   | Niepowtarzalne tagi opisujące paczkę. |

## Informacje o licencji

Struktura `LicenseInfo` obejmuje nazwę licencji, URL prowadzący do jej treści
oraz opcjonalny identyfikator SPDX. W przypadku licencji niestandardowych należy
zapewnić publicznie dostępny dokument opisujący warunki.

## Wymagania danych

Każdy element `DataRequirement` określa nazwę kanału danych, oczekiwany format,
krótki opis oraz flagę `required` informującą czy paczka zadziała bez tych danych.
Można podać `schema_uri` z odnośnikiem do formalnej specyfikacji (np. JSONSchema,
Protobuf, Avro).

## Zależności komponentów

`ComponentDependency` pozwala zdefiniować minimalne oraz opcjonalne maksymalne
wersje komponentów. Wszystkie wersje stosują semver. Jeśli w przyszłości dodamy
nowe pola (np. warunkowe zależności), powinny być one opcjonalne, aby zachować
wsteczną kompatybilność.

## Integralność paczki

`IntegrityInfo` przechowuje sumę kontrolną paczki (`checksum`), opcjonalny
podpis kryptograficzny (`signature`) wraz z identyfikatorem klucza (`signing_key_id`)
i nazwą algorytmu (`signature_algorithm`), a także listę dozwolonych fingerprintów
sprzętowych (`fingerprint_whitelist`). Pola te są ze sobą powiązane:

* jeśli `signature` jest ustawiony, należy obowiązkowo podać zarówno `signing_key_id`,
  jak i `signature_algorithm`;
* jeżeli podpis nie jest dostarczony, powiązane pola muszą pozostać puste;
* fingerprinty muszą być unikalne – duplikaty są odrzucane na etapie walidacji.

Weryfikator marketplace musi potwierdzić zgodność tych informacji przed instalacją paczki.

## Restrykcje modeli

Wszystkie modele schematu dziedziczą po wspólnej klasie bazowej, która odrzuca
nieznane pola oraz przycina białe znaki na końcach łańcuchów znaków. Dzięki temu
ładujące dane komponenty mogą liczyć na deterministyczną walidację – wszelkie
literówki w nazwach pól lub nadmiarowe informacje zostaną zgłoszone jako błąd
`ValidationError`.

## Rozszerzanie schematu

1. Podnieś `schema_version` zgodnie z zasadami semver (np. `major` dla zmian
   niekompatybilnych, `minor` dla dodania opcjonalnych pól, `patch` dla poprawek).
2. Dodawaj nowe pola jako opcjonalne, aby starsze paczki zachowały kompatybilność.
3. Rozszerzenia wymagające nowych typów pomocniczych umieszczaj w module
   `schema.py`, eksportując je w `__all__`.
4. Zaktualizuj testy jednostkowe w `tests/bot_core/test_config_marketplace_schema.py`,
   aby obejmowały nowe przypadki.
5. Opublikuj dokumentację zmian w tym pliku, zachowując sekcję changelog (jeżeli
   zostanie utworzona).

## Walidacja

Metody walidujące (np. poprawność semver, unikalność tagów, relacje czasowe)
zapewniają spójność danych. W przypadku błędów walidacji zgłaszany jest
`pydantic.ValidationError`. Integracje (CLI, UI) powinny przechwytywać ten
wyjątek i prezentować użytkownikowi zrozumiałą informację zwrotną.
