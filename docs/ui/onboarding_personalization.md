# Onboarding interfejsu strategii

Nowy moduł **Strategie 360°** w powłoce Qt zapewnia gotowe ekrany startowe oraz personalizację doświadczenia użytkownika zarządzającego katalogiem strategii.

## Szybki start

1. Otwórz kartę **Strategie 360°** w głównym oknie aplikacji.
2. W sekcji **Ulubione strategie** dodaj wpisy z katalogu – lista jest synchronizowana z kontrolerem `UserProfileController` i pozostaje spójna pomiędzy sesjami.
3. Przejdź przez **Kreator konfiguracji**, aby aktywować licencję, wprowadzić połączenia z giełdami i zapisać parametry automatyzacji dla aktywnego profilu.
4. W panelu **Personalizacja motywu** wybierz styl dopasowany do preferencji operatora. Wybór natychmiast aktualizuje motyw QML (`AppTheme.applyPalette`) oraz systemową paletę Qt.

## Profile użytkownika

- Panel **Zarządzanie profilami użytkownika** pozwala dodawać, usuwać i przełączać profile bezpośrednio z widoku „Strategie 360°”.
- Nowe profile automatycznie stają się aktywne, a zmiany nazw są natychmiast propagowane do pozostałych komponentów QML.
- Operator może szybko duplikować istniejący profil (np. aby przetestować wariant motywu) oraz przywrócić ustawienia domyślne pojedynczego profilu jednym kliknięciem.
- Kontroler `UserProfileController` obsługuje wiele profili z zapisem w `config/ui_profiles.json`.
- Każdy profil przechowuje nazwę, motyw oraz listę ulubionych strategii synchronizowaną z katalogiem `StrategyCatalog`.
- Operatorzy mogą zapisać niestandardowe kolory akcentów (`paletteOverrides`) w ramach profilu – wartości są scalane z bazowym
  motywem i automatycznie zapisywane na dysku.
- Zmiana profilu (np. z panelu API QML `appController.userProfiles`) przełącza motyw, ulubione i rekomendacje.
- Panel progressu kreatora pokazuje ile kroków konfiguracji zostało ukończonych dla profilu; postęp jest automatycznie
  aktualizowany przy duplikowaniu, resetowaniu oraz ukończeniu kreatora i pozwala na ręczne oznaczanie zakończenia lub wyczyszczenie
  stanu z poziomu interfejsu.

## Podsumowanie kreatora konfiguracji

- Karta **Kreator konfiguracji** zawiera pasek postępu dla aktywnego profilu wraz z listą ukończonych kroków.
- Wskaźnik synchronizuje się z kontrolerem `ConfigurationWizardController` – uruchomienie, przechodzenie kroków oraz zakończenie
  kreatora aktualizują stan profilu w `UserProfileController`.
- Operator może ręcznie oznaczyć kreator jako ukończony lub zresetować postęp dla profilu, co automatycznie odświeża podsumowanie
  i panel zarządzania profilami.

## Integracja z katalogiem strategii

- `UserProfileController` nasłuchuje zmian `StrategyWorkbenchController::catalogChanged` i aktualizuje rekomendacje w sekcji „Rekomendacje katalogu”.
- Rekomendacje sortowane są według metadanych (`metadata.popularity` lub `metadata.score`), dzięki czemu operator natychmiast widzi najciekawsze presety.

## Testy i jakość

- Nowe scenariusze QML w `ui/tests/qml/tst_strategy_personalization.qml` pokrywają dashboard, kreator, personalizację motywów oraz
  edycję niestandardowych kolorów.
- Jednostkowy test C++ `ui/tests/UserProfileControllerTest.cpp` weryfikuje obsługę profili, ulubionych strategii, zmian motywu oraz
  zapisywanie i czyszczenie nadpisanych kolorów.

Te elementy zapewniają spójny onboarding użytkownika oraz szybsze wdrożenie strategów do pracy z katalogiem.
