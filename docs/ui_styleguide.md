# Przewodnik stylu interfejsu

## Cel
Ujednolicenie wyglądu nowych widoków (Dashboard wyników, panel aktualizacji, kreatory) oraz zapewnienie spójności z centralnym motywem `AppTheme`.

## Kolorystyka
| Rola                     | Klucz w `AppTheme`             | Wartość HEX |
|--------------------------|--------------------------------|-------------|
| Tło główne               | `backgroundPrimary`            | `#0E1320`   |
| Powierzchnia mocna       | `surfaceStrong`                | `#1F2536`   |
| Powierzchnia subtelna    | `surfaceSubtle`                | `#2C3448`   |
| Tekst podstawowy         | `textPrimary`                  | `#F5F7FA`   |
| Tekst drugorzędny        | `textSecondary`                | `#A4ACC4`   |
| Tekst pomocniczy         | `textTertiary`                 | `#7C86A4`   |
| Akcent                   | `accent` / `accentMuted`       | `#4FA3FF` / `#3577D4` |
| Pozytywny (sukces)       | `positive`                     | `#3FD0A4`   |
| Negatywny (błąd)         | `negative`                     | `#FF6B6B`   |
| Ostrzeżenie              | `warning`                      | `#F8C572`   |

## Typografia
- Czcionka bazowa: `Inter`
- Czcionka monospace: `Fira Code`
- Rozmiary tekstu:
  - Nagłówki: `fontSizeHeadline` (24 px)
  - Tytuły sekcji: `fontSizeTitle` (20 px)
  - Podtytuły: `fontSizeSubtitle` (16 px)
  - Tekst główny: `fontSizeBody` (14 px)
  - Podpisy: `fontSizeCaption` (11 px)

## Siatka i odstępy
- Marginesy główne i padding sekcji: `spacingLg` (18 px)
- Odstępy między kartami i komponentami: `spacingMd` (12 px)
- Odstępy wewnątrz kart: `spacingSm` (8 px)
- Drobne odstępy lub separatory: `spacingXs` (4 px)

## Promienie zaokrągleń
- Karty/sekcje: `radiusLarge` (14 px)
- Pane/ramki: `radiusMedium` (10 px)
- Wnętrza / drobne elementy: `radiusSmall` (6 px)

## Ikonografia
- Ikony ładujemy przez `AppTheme.iconSource("nazwa")`.
- Styl ikon: wektorowe (SVG), kolor domyślny biały z półprzezroczystością `0.72`.
- Ikony statusów korzystają z kolorów `accent`, `positive`, `negative` oraz `warning`.

## Zasady stosowania motywu
1. Każdy widok importuje `"../styles" as Styles` i korzysta z właściwości `AppTheme`.
2. Wszystkie tła kart i grup ustawiamy poprzez `Styles.AppTheme.cardBackground(opacity)` zamiast ręcznych wartości RGBA.
3. Kolory tekstu wybieramy z `textPrimary`, `textSecondary` lub `textTertiary`; unikamy bezpośredniego używania `palette.*`.
4. Nagłówki i tytuły otrzymują `font.family: Styles.AppTheme.fontFamily` oraz `font.pixelSize` zgodnie z tabelą typografii.
5. Każdy komponent udostępnia `objectName` dla kluczowych sekcji, co umożliwia testy wizualne z `qmltestrunner`.
6. Przy tworzeniu nowych kart zachowujemy hierarchię: tło (`surfaceStrong`), wewnętrzne panele (`cardBackground(0.9)`), akcenty (`accentMuted`).

## Testy wizualne
- Testy w `ui/tests/qml/tst_style_components.qml` uruchamiamy przez `ctest -R ui_tst_style_components` lub podczas pełnego pipeline'u CI.
- Test sprawdza, czy główne komponenty stosują `AppTheme` dla odstępów i kolorystyki kart.
- Nowe widoki należy rozszerzać o `objectName` oraz dodać asercje w testach QML.

## Checklist przy dodawaniu widoku
- [ ] Import `"../styles" as Styles`
- [ ] Wykorzystanie właściwości `AppTheme` dla kolorów, typografii i spacingu
- [ ] Zdefiniowane `objectName` dla głównych sekcji
- [ ] Dodany przypadek testowy w `tst_style_components.qml`
- [ ] Dokumentacja zaktualizowana, jeśli wprowadzono nowe role kolorystyczne lub warianty ikon
