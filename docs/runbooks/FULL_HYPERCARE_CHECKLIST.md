# Pełny przegląd hypercare (Stage5 + Stage6)

## Cel
Zapewnić, że wszystkie artefakty hypercare Stage5 i Stage6 zostały wygenerowane,
zweryfikowane oraz zebrane w jeden podpisany raport zbiorczy, który można
zarchiwizować w repozytorium audytowym.

## Kroki
1. Upewnij się, że zakończono indywidualne cykle hypercare:
   - Stage5: `python scripts/run_stage5_hypercare_cycle.py`
   - Stage6: `python scripts/run_stage6_hypercare_cycle.py`
2. Zweryfikuj oba raporty przy pomocy:
   - `python scripts/verify_stage5_hypercare_summary.py`
   - `python scripts/verify_stage6_hypercare_summary.py`
3. Uruchom agregację raportów:
   ```bash
   python scripts/run_full_hypercare_summary.py \
     --stage5-summary var/audit/stage5/hypercare/summary.json \
     --stage5-signature var/audit/stage5/hypercare/summary.json.sig \
     --stage5-signing-key-file secrets/stage5_hmac.key \
     --stage6-summary var/audit/stage6/hypercare_summary.json \
     --stage6-signature var/audit/stage6/hypercare_summary.sig \
     --stage6-signing-key-file secrets/stage6_hmac.key \
     --signing-key-file secrets/full_hypercare.key \
     --output var/audit/hypercare/full_hypercare_summary.json
   ```
   > Jeśli korzystasz z domyślnej konfiguracji skryptu Stage6 (bez własnego pliku
   > konfiguracyjnego), ścieżka raportu może zawierać znacznik czasu w nazwie
   > `stage6_hypercare_summary_<TS>.json`. Dostosuj argumenty `--stage6-summary`
   > oraz `--stage6-signature` odpowiednio do wygenerowanych plików.
4. Opcjonalnie dołącz dodatkowe metadane (np. identyfikator sprintu) poprzez
   parametr `--metadata` wskazujący na plik JSON.
5. Zweryfikuj raport zbiorczy:
   ```bash
   python scripts/verify_full_hypercare_summary.py \
     var/audit/hypercare/full_hypercare_summary.json \
     --signature var/audit/hypercare/full_hypercare_summary.json.sig \
     --require-signature \
     --hmac-key-file secrets/full_hypercare.key \
     --revalidate-stage5 --stage5-hmac-key-file secrets/stage5_hmac.key \
     --stage5-require-signature \
     --revalidate-stage6 --stage6-hmac-key-file secrets/stage6_hmac.key \
     --stage6-require-signature
   ```

## Artefakty / Akceptacja
- `var/audit/hypercare/full_hypercare_summary.json` – podpisany raport zbiorczy.
- `var/audit/hypercare/full_hypercare_summary.json.sig` – podpis HMAC raportu.
- Raporty Stage5 i Stage6 wraz z podpisami oraz log z weryfikacji.
- Wynik komendy weryfikacyjnej (`overall_status`) równy `ok` lub `warn` (wraz z
  uzasadnieniem w decision logu).

