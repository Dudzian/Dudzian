# STAGE5 – Checklist: Raport TCO

## Cel
Przygotować i zarchiwizować raport Total Cost of Ownership (TCO) na potrzeby
cykli hypercare Stage5, obejmujący miesięczne koszty infrastruktury, operacji
i szkoleń wraz z podpisem HMAC.

## Prerekwizyty
- Aktualny plik z pozycjami kosztowymi w formacie JSON (np. `data/stage5/tco.json`).
- Zweryfikowane wartości miesięcznej liczby transakcji i wolumenu obrotu.
- Klucz HMAC do podpisu raportu zapisany lokalnie (np. `secrets/hmac/tco.key`).

## Procedura
1. Zweryfikuj/uzupełnij plik wejściowy z kosztami (kategorie: infrastruktura,
   operacje, szkolenia). Każda pozycja powinna zawierać `name`, `category` oraz
   `monthly_cost` w tej samej walucie.
2. Uruchom skrypt generujący raport TCO wraz z podpisem HMAC:
   ```bash
   python scripts/run_tco_analysis.py \
       --input data/stage5/tco.json \
       --artifact-root var/audit/tco \
       --monthly-trades 200 \
       --monthly-volume 450000 \
       --signing-key-file secrets/hmac/tco.key \
       --signing-key-id stage5-tco \
       --tag weekly-cycle \
       --print-summary
   ```
   Parametry `--monthly-trades` i `--monthly-volume` dostosuj do aktualnych
   metryk operacyjnych. Opcja `--print-summary` pozwala zapisać podsumowanie w
   logu operacyjnym.
3. Zweryfikuj wygenerowane pliki w katalogu `var/audit/tco/<TS>/` (gdzie `<TS>`
   odpowiada znacznikowi czasu). Upewnij się, że podpis HMAC został zapisany i
   odpowiada raportowi JSON.
4. Dodaj informację o wykonanym raporcie do decision logu hypercare wraz z
   linkiem do artefaktów (`var/audit/tco/...`).

## Artefakty/Akceptacja
| Artefakt | Lokalizacja | Kryteria akceptacji |
| --- | --- | --- |
| Raport TCO (JSON) | `var/audit/tco/<TS>/tco_summary.json` | Zawiera `monthly_total`, `usage.cost_per_trade`, `tag` cyklu |
| Rozbicie kosztów (CSV) | `var/audit/tco/<TS>/tco_breakdown.csv` | Wszystkie pozycje z inputu, sumy miesięczne w dwóch miejscach po przecinku |
| Podpis raportu | `var/audit/tco/<TS>/tco_summary.signature.json` | HMAC zgodny z kluczem `--signing-key-id`, walidacja `bot_core.security.signing.build_hmac_signature` |
