# Zależności bundla desktopowego w trybie offline

Sprawdzenie potwierdza, że repozytorium zawiera pełną listę bibliotek wymaganych do
zbudowania pakietu instalacyjnego Qt w środowiskach odciętych od sieci.

## Lista pakietów

Zależności są zebrane w pliku `deploy/packaging/requirements-desktop.txt` wraz z
pionowanymi wersjami zapewniającymi deterministyczne buildy:

- `numpy==1.26.4`
- `pandas==2.2.2`
- `joblib==1.4.2`
- `pyinstaller==6.5.0`
- `briefcase==0.3.18`
- `PySide6==6.7.0`
- `packaging==25.0`
- `cryptography==42.0.8`

Plik zawiera także komentarz opisujący cel zestawu zależności, co ułatwia
synchronizację artefaktów na mirrorach PyPI wykorzystywanych w trybie offline.

## Instalacja w hermetycznym środowisku

Do przygotowania bundla należy użyć dedykowanego wirtualnego środowiska Pythona
(np. `.venv-desktop`) i zainstalować powyższe pakiety poleceniem:

```bash
pip install -r deploy/packaging/requirements-desktop.txt
```

Wszystkie paczki są już zdefiniowane w repozytorium; dodatkowe zależności nie są
wymagane do zbudowania bundla w trybie offline.
