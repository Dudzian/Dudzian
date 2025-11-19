# Stage5 – lista kontrolna rotacji kluczy API

## Cel
Zapewnić cykliczną rotację kluczy API w środowiskach Dudziana, z aktualizacją
rejestru i podpisanym raportem audytowym, aby spełnić wymagania hypercare oraz
polityki bezpieczeństwa.

## Przygotowanie
1. Upewnij się, że nowe klucze API zostały wygenerowane i zweryfikowane w
   panelach giełdowych.
2. Przygotuj klucz HMAC (≥32 bajty) do podpisu raportów w
   `var/audit/stage5/key_rotation` – wartość musi być rotowana zgodnie z
   polityką bezpieczeństwa.
3. Zweryfikuj uprawnienia nowych kluczy (RBAC) i zapisz je w magazynie sekretów
   (`keyring`/magazyn szyfrowany) przed uruchomieniem checklisty.

## Kroki
1. Uruchom rotację dla wybranych środowisk:
   ```bash
   python -m scripts.rotate_keys \
     --config config/core.yaml \
     --environment paper \
     --environment live \
     --operator "SecOps" \
     --notes "Planowa rotacja Q2" \
     --signing-key-env STAGE5_ROTATION_HMAC
   ```
2. Zweryfikuj komunikat CLI – powinien wskazywać ścieżkę wygenerowanego raportu
   oraz listę środowisk objętych rotacją.
3. Otwórz `var/audit/stage5/key_rotation/*.json` i sprawdź pola:
   - `records[].next_due_at` – kolejny termin rotacji,
   - `records[].was_overdue` – potwierdzenie, że przed rotacją nie było długu,
   - `signature` – obecność podpisu HMAC.
4. Upewnij się, że rejestr rotacji w `var/<ENV>/security/rotation_log.json`
   został zaktualizowany bieżącą datą.
5. Zaktualizuj decision log hypercare lub tablicę operacyjną informacją o
   wykonanej rotacji (kto, kiedy, jakie środowiska).

## Artefakty / Akceptacja
- Raport JSON w `var/audit/stage5/key_rotation/` zawierający pola `records` oraz
  podpis `signature` (`HMAC-SHA256`).
- Zaktualizowany rejestr rotacji `security/rotation_log.json` dla każdego
  środowiska objętego checklistą.
- Notatka w decision logu hypercare (lub innym rejestrze operacyjnym) z datą,
  operatorem i listą środowisk.

> **Tip:** Status bundla mTLS możesz sprawdzić osobno poleceniem
> `python scripts/rotate_keys.py --status --bundle core-oem`
> (również w skróconej formie `--status core-oem` lub `status core-oem`). Raport JSON zawiera
> sekcję `summary` (liczby `ok/warning/due/overdue`) oraz listę `entries` z
> polami `state`, `days_since_rotation` i `due_in_days` dla kluczy `ca/server/client`,
> co pozwala błyskawicznie ocenić aktualność pakietu bez uruchamiania pełnej rotacji.
