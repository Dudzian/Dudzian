# Stage 9 V5d11 — dowody lokalne do przeglądu

Bazą jest `f4082ff4d9ea1f2c5aafd9c3611e5edc834c19a6`. W porównaniu z rewizją
materializacji V5d10 w tej bazie zmienił się workflow marketingowy i jego test.
Launcher, builder, kontrakt i plik `pyproject.toml` zachowują bajty z rewizji
materializacji. V5d11 dodaje własny zakres walidacji syntetycznej.

## Dane wejściowe V5d10

Sprawdzono 30 wewnętrznych sum kontrolnych w trzech dostarczonych archiwach.
Archiwum fizyczne ma 263 591 bajtów i oczekiwany SHA-256. Jego receipt,
stdout buildera i kandydatura V2 mają identyczne bajty. Stderr jest pusty.
Przypięcia aprobaty, żądania oraz oczekiwań ABI zgadzają się z bajtami wejść.
Przypięcia planu i domknięcia źródeł zgadzają się po kanonicznej serializacji
z kontraktu. Zakres sprawdzenia obejmował pliki dowodowe, bez wykonania ich
kodu i bez materializacji A.

## V5d11

- [Instrukcja operatora i protokół sesji](../stage9_v5d11_observer_isolation.md)
- [Wersjonowane polityki i skróty](../../../deployment/stage9_v5d11/README.md)
- [Kwalifikacja libvirt i loadera](../stage9_v5d11_libvirt_loader_qualification.md)
- [Niezgodności integracji profilu A z V5d1](../stage9_v5d11_v5d1_compatibility.md)

| Kontrola | Wynik lokalny |
| --- | --- |
| Syntetyczne testy V5d11 | 107 zaliczonych. Pokrycie z gałęziami 89%, pokrycie zmienionych linii 91% przy progu 70%. |
| Regresje wdrożeniowe Stage 9 do V5d10 | 3 597 zaliczonych, 54 pominięte przez warunki testów. |
| Regresje skryptów, fixture i normalizacji ścieżek coverage | 109 zaliczonych. |
| Ruff oraz mypy dla Linux i Win32 | `PASS` dla zakresu repozytorium i osobnego zakresu V5d11. |
| Semgrep i Betterleaks | Reguły blokujące i nowe źródła bez ustaleń. Betterleaks bez wycieków. |
| pip-audit | 104 przypięte zależności locku desktop, zero znanych podatności. |
| Kod C i polityki | Kompilacja statyczna, porównanie bajtów BPF, syntetyczne odmowy seccomp i kompilacja AppArmor offline zaliczone. |

Rzeczywisty pomiar przestrzeni nazw, wymuszenie polityk na Ubuntu operatora,
proces qemu-img, protokół libvirt i rzeczywisty loader mają stan `NOT_TESTED`.
W tej rewizji kod workera nie zawiera ścieżki wykonania Observera. Tryb pomiaru
wystawia kandydaturę, a podpis niezależnego recenzenta może potwierdzić tę samą
żywą sesję przy utrzymaniu `authorization=NOT_AUTHORIZED`.

`STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`, `STAGE_3B=NOT_AUTHORIZED`,
`PRODUCTION=NO-GO`.
