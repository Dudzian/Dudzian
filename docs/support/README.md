# Centrum pomocy – przewodnik dla autorów

Ten katalog zawiera artykuły pomocowe prezentowane w module **Support Center** aplikacji desktopowej. Artykuły są ładowane lokalnie, dzięki czemu użytkownik końcowy może korzystać z FAQ i poradników bez połączenia z Internetem.

## Struktura katalogów

```
docs/support/
├── README.md                # ten plik
├── articles/                # treści prezentowane w UI
│   ├── getting_started.md
│   └── troubleshooting_network.md
└── translations/            # (opcjonalnie) dodatkowe zasoby językowe
```

Każdy artykuł to plik Markdown zakończony rozszerzeniem `.md`. Wczytywanie odbywa się tylko z katalogu `docs/support/articles`.

## Format artykułu

Każdy plik powinien rozpoczynać się blokiem *front matter* zakończonym linią `---`. Dostępne pola:

| Pole        | Wymagane | Opis                                                                 |
|-------------|----------|----------------------------------------------------------------------|
| `id`        | tak      | Unikalny identyfikator (używany w wyszukiwarce i QML).                |
| `title`     | tak      | Tytuł artykułu wyświetlany w UI.                                     |
| `summary`   | tak      | Krótki opis pokazywany na liście FAQ.                                |
| `category`  | nie      | Nazwa kategorii (np. `onboarding`, `wsparcie`).                      |
| `tags`      | nie      | Lista tagów oddzielonych przecinkami, np. `sieć, diagnostyka`.       |
| `runbooks`  | nie      | Lista ścieżek do powiązanych runbooków (oddzielonych przecinkami).   |

Przykładowy plik:

```markdown
---
id: troubleshooting-network
title: Rozwiązywanie problemów z siecią
summary: Kroki diagnostyczne dla zrywanego połączenia z giełdą.
category: wsparcie
tags: sieć, diagnostyka
runbooks: docs/operations/runbooks/network_diagnostics.md
---
# Rozwiązywanie problemów z siecią
1. Sprawdź zaporę sieciową.
2. Zweryfikuj, czy porty wymagane przez giełdę są otwarte.
```

Treść po bloku `---` może zawierać pełny Markdown (nagłówki, listy, kod). W UI jest renderowana jako tekst Markdown.

## Dodawanie nowych artykułów

1. Utwórz plik w `docs/support/articles/` zgodny z powyższym formatem.
2. Jeśli artykuł odwołuje się do runbooków, upewnij się, że ścieżki wskazują istniejące pliki w `docs/operations/runbooks/`.
3. (Opcjonalnie) dopisz testy jednostkowe, jeżeli artykuł wprowadza nowe scenariusze diagnostyczne, które powinny być odzwierciedlone w automatyzacji.

## Tłumaczenia interfejsu

Teksty interfejsu Support Center są przechowywane w `ui/i18n/pl.ts`. Aby dodać nowy napis:

1. Dodaj identyfikator w QML przy użyciu `qsTrId("supportCenter.nazwaId")`.
2. Dopisz odpowiadające tłumaczenie w pliku `ui/i18n/pl.ts`.
3. Uruchom `lupdate`/`linguist`, jeśli pracujesz nad dodatkowymi językami.

## Paczki diagnostyczne i zgłoszenia

Instrukcja przygotowania paczki ZIP dla zespołu wsparcia znajduje się w dokumencie [docs/support/diagnostics.md](diagnostics.md). Opisuje on zarówno przepływ w UI, jak i narzędzie `scripts/generate_diagnostics.py` do generowania archiwów z linii poleceń.

## Testy

Automatyczne testy kontrolują parser oraz komponent QML:

- `tests/support/test_article_loader.py` – weryfikuje format plików Markdown i indeks wyszukiwania.
- `tests/ui/test_support_center.py` – sprawdza poprawne ładowanie UI (test jest pomijany, jeżeli w środowisku nie ma bibliotek Qt).

Utrzymuj strukturę katalogu i format front matter, aby zachować kompatybilność z aplikacją desktopową.
