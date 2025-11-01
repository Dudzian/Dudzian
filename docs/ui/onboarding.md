# Kreator aktywacji licencji

Nowy kreator **LicenseWizard** zapewnia operatorom prowadzenie krok po kroku przez proces aktywacji licencji OEM i przygotowania środowiska przed startem bota.

## Przepływ pięciu kroków
1. **Powitanie** – streszcza zakres kreatora i przypomina o konieczności przygotowania pliku licencyjnego.
2. **Sprawdzenie fingerprintu** – odczytuje HWID urządzenia, pozwala go odświeżyć i sygnalizuje problemy (np. brak TPM) poprzez komunikaty tłumaczone w `ui/i18n/pl.ts`.
3. **Wprowadzenie licencji** – umożliwia wklejenie treści JSON lub wybór pliku, wywołując logikę `LicensingController` z walidacją podpisu, fingerprintu i obsługą błędów.
4. **Konfiguracja strategii i kluczy API** – komponent `StrategySetupStep.qml` prezentuje katalog dostępnych strategii (dostarczany przez `OnboardingService`) i pozwala zapisać klucze API giełd poprzez `SecretStore`.
5. **Podsumowanie** – prezentuje status wraz z identyfikatorem licencji, wybraną strategią, ostatnią skonfigurowaną giełdą oraz szczegółami weryfikacji.

## Warstwa backendowa
- Moduł `ui.backend.licensing_controller.LicensingController` integruje się z `core.security.license_verifier.LicenseVerifier`, zapewniając spójne kody statusu i sygnały Qt dla QML.
- `OnboardingService` udostępnia strategiom dane z `bot_core.strategies.public.list_available_strategies`, przechowuje wybrany wariant i zapisuje klucze API przez `core.security.secret_store.SecretStore`.
- `LicenseVerifier` oraz `SecretStore` mapują komunikaty na identyfikatory tłumaczeń, aby w UI można było prezentować lokalizowane treści.

## Testy automatyczne
- Scenariusz `tests/ui/test_license_wizard.py` wykorzystuje PySide6 (pytest-qt) do uruchomienia QML-a, symulując poprawną i błędną weryfikację licencji.
- Test `tests/ui/test_onboarding_strategy.py` waliduje przepływ wyboru strategii oraz zapis kluczy API (z wykorzystaniem atrap `SecretStore`), dzięki czemu integracja QML ↔ backend pozostaje pokryta regresją.
- Testy gwarantują, że zmiana kroków, aktualizacja fingerprintu, wybór strategii oraz generacja komunikatów statusu działają bez potrzeby odpalania całej aplikacji desktopowej.

## Lokalizacja
- Nowe identyfikatory tłumaczeń zebrano w `ui/i18n/pl.ts`. QML wykorzystuje helper `trId` z fallbackiem, aby zapewnić czytelne komunikaty nawet w środowiskach bez wczytanego `QTranslator`.

## Integracja z istniejącym UI
- `ui/qml/qml.qrc` zawiera wpisy dla `onboarding/LicenseWizard.qml` oraz `onboarding/StrategySetupStep.qml`, dzięki czemu komponenty są dostępne z poziomu głównego silnika QML.
- Kreator można osadzić w istniejących widokach (np. pierwszego uruchomienia) poprzez ustawienie kontekstowych właściwości `licensingController` oraz `onboardingService` na instancje backendowych klas.
