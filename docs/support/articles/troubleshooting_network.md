---
id: troubleshooting-network
title: Rozwiązywanie problemów z siecią
summary: Kroki diagnostyczne dla zrywanego połączenia z giełdą.
category: wsparcie
tags: sieć, diagnostyka, łączność
runbooks: docs/operations/runbooks/network_diagnostics.md, docs/operations/runbooks/incident_response.md
---
# Rozwiązywanie problemów z siecią
1. Zweryfikuj, czy zegar systemowy jest zsynchronizowany (NTP).
2. Sprawdź zaporę sieciową i upewnij się, że porty API giełd są otwarte.
3. Upewnij się, że adresy IP giełd nie są blokowane przez dostawcę internetu.
4. Jeżeli problem trwa nadal, uruchom scenariusz diagnostyczny opisany w runbooku `network_diagnostics.md`.
