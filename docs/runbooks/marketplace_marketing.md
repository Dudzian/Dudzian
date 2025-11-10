# Runbook marketingowy – wydania Marketplace

Ten runbook opisuje obowiązki marketingowe po przygotowaniu nowego presetu Marketplace.

## 1. Artefakty wejściowe

- Zatwierdzony preset w `config/marketplace/presets/...` oraz wpis w `catalog.json` (schemat 1.1).
- Raport QA / Stress Lab (link w `catalog.release_notes`).
- Wynik workflow CI `marketplace-catalog` – artefakt `marketplace-packages` (podpisane paczki).

## 2. Checklist publikacji

| Krok | Odpowiedzialny | Szczegóły |
| --- | --- | --- |
| Weryfikacja katalogu | Marketplace Guild | `python scripts/marketplace_cli.py validate --key dev-hmac:config/marketplace/keys/dev-hmac.key` |
| Audyt marketingowy | Zespół marketingu | Sprawdź `docs/benchmark/cryptohopper_comparison.md` – sekcja Marketplace powinna zawierać nowy preset i aktualny SLA. |
| Przygotowanie komunikacji | Marketing + CS | Draft wpisu na blog, newsletteru i komunikatu do klientów OEM. Załącz parametry `release.summary`, `tags`, `exchange_compatibility`. |
| Aktualizacja materiałów sprzedażowych | Marketing | Zaktualizuj slajdy ofertowe i bazę Q&A (w tym informację o kanałach release: public/beta). |
| Synchronizacja katalogu | Customer Success | Wyślij instrukcję `python scripts/marketplace_cli.py sync` partnerom korzystającym z katalogu offline. |

## 3. Harmonogram komunikacji

1. **T-1 dzień** – draft komunikatu marketingowego + wstępne grafiki (marketplace@example.com → marketing@example.com).
2. **T+0** – publikacja wpisu na blogu, newsletter do partnerów, aktualizacja notki na portalu wsparcia.
3. **T+2** – raport adopcji: ilu klientów zainstalowało preset (`MarketplaceAnalytics` w Grafanie).
4. **T+7** – przegląd metryk (Sharpe, drawdown, guardraile) z logów Stress Lab. Wynik dopisz do `docs/benchmark/cryptohopper_comparison.md`.

## 4. Materiały obowiązkowe

- **Release note** – format Markdown, linkowany w `catalog.release_notes`.
- **Slajd sprzedażowy** – uzupełnij w repozytorium marketingu (`marketing/assets/marketplace/`).
- **QA FAQ** – wpis w bazie wiedzy `docs/support/plan.md` (sekcja presetów).

## 5. Eskalacje

- W przypadku regresji metryk (alert w Grafanie) → eskalacja do `oncall-marketplace` i aktualizacja `catalog.release.review_status` na `in_review`.
- Problemy z podpisami → uruchom `python scripts/marketplace_cli.py validate --allow-fingerprint-mismatch` aby zdiagnozować klucz; zgłoś do Security Guild.

## 6. Odniesienia

- Dokument procesowy: `docs/marketplace/README.md`.
- Strategiczny benchmark: `docs/benchmark/cryptohopper_comparison.md`.
- Runbook wsparcia: `docs/runbooks/STAGE6_HYPERCARE_CHECKLIST.md` (sekcja Marketplace/strategii).
- Pakiet Stress Lab: `docs/marketing/stage6_stress_lab_whitepaper.md`, `docs/marketing/stage6_stress_lab_case_studies.md` oraz artefakty workflow `stress-lab-report`.
