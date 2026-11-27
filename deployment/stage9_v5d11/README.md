# Stage 9 V5d11 — polityki profilu A

Wykonaj operacje na zaufanej, przejrzanej kopii kodu. Utrzymuj pliki kandydujące,
wejścia i dowody w prywatnym katalogu operatora. Weryfikuj źródła wersji
poleceniem `sha256sum -c deployment/stage9_v5d11/SHA256SUMS` z katalogu repozytorium.

## AppArmor

Renderer wiąże literalne ścieżki, zatwierdzone aliasy ABI, jedno gniazdo
libvirt i ewentualne pliki danych z tym samym wejściem co bootstrap. Odczytuje
receipt podany z niezależnym skrótem i weryfikuje skrót ABI request z receipt.
Szablon wymaga ABI AppArmor 4.0. Parser kompiluje według cech bieżącego jądra,
z pominięciem cache i zapisu polityki do jądra. Ostrzeżenia o utraconych lub
zdegradowanych regułach, w tym degradacji ABI, kończą kompilację odmową.

W przejrzanym katalogu repozytorium ustaw absolutne ścieżki `A_RECEIPT`,
`A_ABI_REQUEST`, `PRIVATE`, niezależny `A_RECEIPT_SHA256` oraz dokładny endpoint
`A_SOCKET` użyty w manifeście operatora. Dla opcjonalnych ścieżek danych dodaj
odpowiadające wpisy `--data-path` do tablicy `policy_inputs`. Kompilacja tworzy
prywatne materiały do niezależnego przeglądu.

```bash
umask 077
policy_inputs=(--receipt "$A_RECEIPT" --receipt-sha256 "$A_RECEIPT_SHA256"
               --abi-request "$A_ABI_REQUEST" --socket "$A_SOCKET")
policy() {
    /usr/bin/env -i LANG=C LC_ALL=C PATH=/usr/bin:/bin \
        /usr/bin/python3 -I -S -B "$PWD/scripts/stage9_v5d11_policy.py" \
        "$1" "${policy_inputs[@]}" "${@:2}"
}
policy render > "$PRIVATE/profile-a.apparmor"
policy compile > "$PRIVATE/profile-a.apparmor.bin"
sha256sum "$PRIVATE/profile-a.apparmor" "$PRIVATE/profile-a.apparmor.bin"
```

Po niezależnym przeglądzie ustaw `AA_SOURCE_SHA256` i `AA_COMPILED_SHA256`
według otrzymanych przypięć. `load` korzysta z atomowego `--binary --add`,
sprawdza dostępność eksportu binarnej polityki i weryfikuje bajty z jądra.
Jeśli dokładne etykiety są już załadowane, użyj `check` zamiast `load`.
`check` porównuje tryb `enforce`, eksport dokładnie tych dwóch profili i
stabilną rewizję AppArmor. Bootstrap sprawdza etykietę i skuteczną odmowę
odczytu `/proc/self/environ` dla procesu w przygotowanym widoku.

```bash
sudo /usr/bin/env -i LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    /usr/bin/python3 -I -S -B "$PWD/scripts/stage9_v5d11_policy.py" \
    load "${policy_inputs[@]}" --source-sha256 "$AA_SOURCE_SHA256" \
    --compiled-sha256 "$AA_COMPILED_SHA256"
```

Zapisz otrzymane skróty źródła i kompilacji w manifeście operatora. Eksport
z jądra musi odpowiadać dokładnym zatwierdzonym bajtom obu etykiet. Polityka
`qemu-img` nakłada ograniczenie na profil rodzica, zachowując `NoNewPrivs`.
Dopuszcza mapowanie przypiętych bibliotek i odczyt dokładnych obrazów z
wejścia. Odmawia sieci, socketów UNIX i wykonania innych programów.

## Seccomp

Źródło C zawiera tablicę classic BPF dla Linux x86-64 LP64. Tryb
`bootstrap --dump-seccomp` zwraca dokładne bajty tej tablicy. Zweryfikuj ich
SHA-256 względem `seccomp.bpf.sha256` i przypięcia w manifeście operatora.
Bootstrap instaluje filtr z `TSYNC` po `NoNewPrivs`, wykonuje bezskutkowe
negatywne sondy syscalls i pozostawia filtr aktywny na ścieżce potomnej.
Filtr dopuszcza komunikację przez gniazdo UNIX typu stream, a AppArmor i
dokładny widok montowań ograniczają ścieżkę gniazda. Polityka serwera libvirt
wymaga własnej kwalifikacji. AppArmor wiąże prawo mapowania kodu z konkretną
ścieżką pliku. Seccomp blokuje anonimowe mapowania wykonywalne i ustawienie
`PROT_EXEC` przez `mprotect`.

Testy syntetyczne sprawdzają renderowanie polityki, błędy przypięć, zmianę
trybu i rewizji oraz rozbieżność eksportu jądra. Oddzielny proces testowy na
Linux x86-64 instaluje rzeczywisty filtr i sprawdza odmowy. Fizyczne
potwierdzenie AppArmor, loadera i zachowania potomka `qemu-img` pozostaje
`NOT_TESTED`.
