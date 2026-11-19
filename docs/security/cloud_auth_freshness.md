# Świeżość uwierzytelniania cloud

Warstwa cloud z wymaganym handshake dopuszcza klienta po podpisaniu
jednorazowego challenge wydanego przez bieżący proces serwera. Challenge ma
krótki termin ważności, jest przypisany do tożsamości klienta i jest zużywany
atomowo. Ponowne użycie, brak challenge lub przekroczenie terminu kończą się
odmową. Limity prób obejmują cały proces, źródło połączenia i dozwoloną
tożsamość; liczba oczekujących challenge i sesji jest ograniczona.

Sesje wygasają również według zegara monotonicznego. Usunięcie klienta albo
zmiana jego sekretu lub ścieżki pakietu licencyjnego odwołuje dotychczasowe
sesje i oczekujące challenge. Skonfigurowany pakiet licencyjny jest sprawdzany
przy każdej nowej autoryzacji; brak pliku lub weryfikatora powoduje odmowę.
Wymagany handshake z pustą allowlistą blokuje uruchomienie usługi.

Kontrakt dodaje RPC `CloudAuthService.IssueChallenge`. Klienta, serwer i stuby
z `proto/trading.proto` należy aktualizować razem. Starszy klient bez obsługi
challenge nie przejdzie autoryzacji na nowym serwerze. Nowy klient odrzuca
serwer bez tego RPC (w tym odpowiedź `UNIMPLEMENTED`) i nie przechodzi na
handshake bez potwierdzenia świeżości. Wdrożenie wymaga skoordynowanej
aktualizacji; nie ma automatycznego trybu zgodności ze starszym protokołem.

Stan challenge i sesji jest lokalny dla procesu. Restart oraz zatrzymanie lub
nieudane uruchomienie usługi odrzucają poprzednie dowody. Dla wielu procesów
klient musi wykonać oba kroki handshake w tej samej instancji; replikacja stanu
autoryzacji nie jest częścią tej zmiany.

Testy obejmują odmowy, wygaśnięcie, współbieżność, odwołanie, limity oraz
integrację klienta z RPC. Wyniki lokalne dotyczą wyłącznie tego komponentu.
Stage 9, Stage 10 i production pozostają **NO-GO** do czasu osobnej
kwalifikacji i przejścia wymaganych bramek; ta zmiana nie stanowi zgody na
wdrożenie ani na testy fizycznego sprzętu.
