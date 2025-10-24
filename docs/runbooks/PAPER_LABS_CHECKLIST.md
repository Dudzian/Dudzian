# Paper Labs – checklista symulacji profili ryzyka

## Cel
Zapewnić, że przed przejściem z trybu paper do live wszystkie profile ryzyka (konserwatywny, zbalansowany, agresywny, manualny) zostały
zweryfikowane w ramach scenariuszy baseline oraz stres testów flash crash / dry liquidity / latency spike.

## Wymagane wejścia
- [ ] Zatwierdzona wersja konfiguracji `config/core.yaml` (commit SHA / tag).
- [ ] Dostęp do bundla danych Parquet lub włączenie trybu syntetycznego (`--synthetic-fallback`).
- [ ] Klucz HMAC do podpisu raportów (jeśli raport trafia do rejestru compliance).
- [ ] Ścieżka wyjściowa na artefakty (`var/paper_labs/<data>` lub dedykowany katalog CI).

## Kroki
1. [ ] Uruchom komendę:

       ```bash
       python scripts/run_risk_simulation_lab.py \
         --config config/core.yaml \
         --output-dir reports/paper_labs \
         --environment <env> \
         --symbols <lista_symboli> \
         --fail-on-breach
       ```

       gdzie `--config` domyślnie wskazuje `config/core.yaml`, a `--output-dir` – katalog `reports/paper_labs`.
2. [ ] Zweryfikuj, że raport JSON zawiera wszystkie profile oraz pola `breach_count == 0`, `stress_failures == 0`.
3. [ ] Otwórz PDF i potwierdź brak naruszeń oraz poprawne podpisy sekcji stres testów.
4. [ ] Zabezpiecz artefakty: JSON, PDF, log CLI oraz (opcjonalnie) podpis HMAC → katalog audytowy.
5. [ ] Uaktualnij decision log (`audit/decisions`) wpisem z wynikiem Paper Labs (status PASS/FAIL, operator, timestamp).

## Artefakty / Akceptacja
| Artefakt | Lokalizacja | Odpowiedzialny | Akceptacja |
| --- | --- | --- | --- |
| `risk_simulation_report.json` | katalog wyjściowy CLI | Risk Lead | [ ]
| `risk_simulation_report.pdf` | katalog wyjściowy CLI | Risk Lead | [ ]
| Log z CLI (`run_risk_simulation_lab.log`) | katalog wyjściowy CLI | Ops | [ ]
| Wpis w decision log (`audit/decisions`) | repozytorium audytowe | Compliance | [ ]

## Uwagi operacyjne
- W trybie syntetycznym raport oznaczony jest flagą `synthetic_data: true`; akceptacja wymaga zgody Compliance i planu pozyskania
  pełnych danych historycznych.
- W przypadku wykrycia naruszeń (breach/stress failure) należy wygenerować task w systemie ticketowym i zablokować przejście do etapu live.
- Raporty powinny być przechowywane co najmniej 730 dni w repozytorium audytowym (zgodnie z runbookiem OEM Licensing).
