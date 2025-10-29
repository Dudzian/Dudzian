# Aktualizacje offline

Ten dokument opisuje proces przygotowania oraz instalacji pakietów offline z modelami
i strategiami bota handlowego. Zestaw narzędzi obejmuje dwa skrypty:

- `scripts/package_offline_release.py` – buduje paczkę `.tar.gz`, tworzy manifest
  `manifest.json` i opcjonalnie podpisuje go kluczem HMAC.
- `scripts/offline_update.py` – udostępnia polecenia `prepare-release`,
  `verify-release` i `install-release` do późniejszej dystrybucji oraz instalacji
  paczek na stanowiskach bez dostępu do sieci.

## Budowanie paczki offline

1. Przygotuj katalogi `data/models/` i `data/strategies/` zawierające artefakty,
   które mają zostać dostarczone.
2. Wygeneruj paczkę i manifest:

   ```bash
   python scripts/package_offline_release.py \
     --version 2024.10.0 \
     --output var/releases/offline-2024.10.0.tar.gz \
     --models-dir data/models \
     --strategies-dir data/strategies \
     --signing-key SERVICE=U2VjcmV0S21leQ==
   ```

   Skrypt wypisze na standardowe wyjście manifest i, jeżeli podano
   `--manifest-output`, zapisze go także do wskazanego pliku.

## Weryfikacja paczki

Na stacji air-gapped można zweryfikować integralność i podpis:

```bash
python scripts/offline_update.py verify-release \
  --archive var/releases/offline-2024.10.0.tar.gz \
  --signing-key SERVICE=U2VjcmV0S21leQ==
```

Wynikiem jest manifest JSON zawierający listę artefaktów wraz z sumami SHA-384.

## Instalacja na środowisku docelowym

1. Skopiuj paczkę `.tar.gz` na stanowisko docelowe.
2. Uruchom instalację, wskazując docelowe katalogi modeli i strategii.

   ```bash
    python scripts/offline_update.py install-release \
      --archive /mnt/usb/offline-2024.10.0.tar.gz \
      --models-dir data/models \
      --strategies-dir data/strategies \
      --backup-dir var/offline_updates/backups \
      --signing-key SERVICE=U2VjcmV0S21leQ== \
      --require-signature
   ```

   Skrypt automatycznie tworzy kopie zapasowe w katalogu `--backup-dir`, weryfikuje
   podpis (jeżeli został przekazany klucz) oraz kopiuje artefakty do katalogów
   roboczych.

## Dobre praktyki

- Przechowuj klucze HMAC w sejfie offline i przekazuj je do skryptów tylko w
  kontrolowanym środowisku.
- Po każdej instalacji zachowaj manifest i raport z `install-release` w repozytorium
  audytowym (np. `var/audit/offline_updates/`).
- Przed dystrybucją nowego pakietu wykonaj testy jednostkowe
  `pytest tests/test_offline_updates.py`, aby upewnić się, że proces wykrywa
  manipulacje manifestem i poprawnie odtwarza kopie zapasowe.
