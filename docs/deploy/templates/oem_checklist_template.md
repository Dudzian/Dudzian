# Checklista wydania OEM – wersja {version}

Data wygenerowania: {timestamp}
Właściciel wydania: {owner}
Tag wydania: {release_tag}

## Licencje i HWID
- [ ] Zweryfikuj kompletność licencji dla wszystkich klientów docelowych.
- [ ] Porównaj fingerprinty HWID z raportami audytu (`license_audit`).
- [ ] Potwierdź aktywację licencji na maszynie referencyjnej.

## Aktualizacje i dystrybucja
- [ ] Zbuduj instalator desktopowy i potwierdź hooki HWID.
- [ ] Wygeneruj paczkę aktualizacji offline `.kbot` oraz zweryfikuj podpis.
- [ ] Przetestuj scenariusz aktualizacji offline oraz powrót do poprzedniej wersji.

## Zgodność i testy
- [ ] Uruchom audyt zgodności (`run_compliance_audit.py`).
- [ ] Zweryfikuj raport retreningu i walidacji danych.
- [ ] Przeprowadź smoke testy i scenariusz demo → paper.

## Raporty dołączenia
- Raport licencyjny: {license_report}
- Raport zgodności: {compliance_report}
- Raport testów: {test_report}

Uwagi dodatkowe:
{notes}
