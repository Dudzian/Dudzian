# FAQ wsparcia OEM

## Jak zweryfikować, że bundel został poprawnie znotaryzowany?

1. Sprawdź raport w `var/dist/notary/<nazwa>.json`. Pole `status` powinno mieć
   wartość `Accepted`.
2. Jeśli pipeline działał w trybie `dry-run`, raport zawiera klucz `mode=dry-run`
   i listę komend, które należy uruchomić ręcznie.
3. W przypadku błędów uruchom `xcrun notarytool history --keychain-profile
   oem-notary` i dołącz log do zgłoszenia.

## Co oznacza błąd "fingerprint-mismatch" podczas instalacji?

* Sprawdź `config/fingerprint.expected.json` w bundlu – czy zawiera prawidłowy
  fingerprint oraz podpis (`signature.algorithm=HMAC-SHA384`).
* Uruchom `python secrets/licensing/offline_portal.py status --read-local` na
  docelowej maszynie. Porównaj fingerprint z raportem.
* Jeśli fingerprint się zmienił (np. wymiana sprzętu), wykonaj procedurę
  odzyskiwania licencji (`offline_portal.py recover`).

## Jak zastosować aktualizację delta?

1. Zweryfikuj wersję bazową (pole `base_version` w `delta.json`).
2. Rozpakuj archiwum delta do katalogu bundla (`tar -xf delta.tar.gz`).
3. Uruchom `bootstrap/verify_fingerprint.py`, aby upewnić się, że fingerprint nie
   został naruszony.
4. Po aktualizacji zrestartuj usługi i wykonaj smoke-test (`python
   scripts/local_orchestrator.py launch live --run-once`).

## Co zrobić, gdy magazyn licencji nie otwiera się po migracji?

* Błąd `LicenseStoreDecryptionError` oznacza najczęściej zmianę fingerprintu
  sprzętu. Użyj `offline_portal.py recover --old-fingerprint-file
  backups/fingerprint.old --read-local-new`.
* Jeżeli błąd utrzymuje się, zweryfikuj obecność wpisu licencji w
  `secrets/license_store.json` i eskaluj do L3 z raportem wygenerowanym przez
  `offline_portal.py verify`.

## Gdzie zgłaszać problemy z delta update?

* Utwórz ticket w `docs/support/articles/` z opisem problemu, wersjami bazową i
  docelową oraz załączonym `delta.json`.
* Jeśli delta pomija pliki krytyczne, wykonaj pełny rebuild bundla i poinformuj
  zespół release o konieczności dystrybucji nowej wersji.
