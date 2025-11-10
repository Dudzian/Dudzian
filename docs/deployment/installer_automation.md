# Zautomatyzowane budowanie instalatorów

Nowy skrypt `scripts/build/build_cross_platform_installers.py` upraszcza przygotowanie
instalatorów dla systemów Windows, macOS i Linux. Narzędzie łączy istniejące
skrypty platformowe oraz zapewnia spójne opcje wersjonowania i katalogu wyjściowego.

## Wymagania

* Python 3.10+
* Dostęp do zależnych narzędzi kompilacyjnych (np. Xcode/`codesign` dla macOS,
  `pwsh` lub `powershell` dla Windows jeśli korzystamy z natywnego środowiska).
* Istniejące profile instalatorów w `deploy/packaging/profiles/` (domyślne profile
  są używane automatycznie).

## Przykłady użycia

```bash
# Zbuduj wszystkie instalatory z domyślnymi profilami
python scripts/build/build_cross_platform_installers.py --version 1.6.0 --output-dir dist/installers

# Pomiń kompilację Windows i podaj własny profil macOS
python scripts/build/build_cross_platform_installers.py \
  --skip-windows \
  --profile-macos deploy/packaging/profiles/custom_macos.toml
```

Parametr `--output-dir` pozwala kierować wszystkie artefakty do jednego katalogu,
co ułatwia dalsze pakowanie i archiwizację. Dodatkowe argumenty podawane po `--`
(`argparse.REMAINDER`) są przekazywane bezpośrednio do skryptów platformowych.

## Integracja z CI/CD

Skrypt może być uruchamiany z poziomu pipeline'ów CI. Warto ustawić zmienną
środowiskową `VERSION`, aby zachować spójność numeracji buildów z metadanymi
wydania. W przypadku buildów dla Windows poza systemem Windows wykorzystywany jest
fallback na `build_installer_windows.sh`, który generuje paczkę w środowisku
zgodnym z POSIX.
