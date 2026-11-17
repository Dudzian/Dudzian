#!/bin/bash
# Przygotowanie materiałów do przeglądu; bez instalacji i autoryzacji wykonania.
set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
unset CDPATH ENV BASH_ENV PYTHONPATH PYTHONHOME

usage() {
    echo 'Użycie: bash scripts/stage9_v5d3_prepare_release.sh [--repository PATH] --output-parent PATH [--resume BUNDLE]' >&2
    exit 64
}

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository=$(cd -- "$script_directory/.." && pwd -P)
output_parent=
previous_bundle=
seen_repository=0
seen_output=0
seen_resume=0
while (($#)); do
    case "$1" in
        --repository)
            (($# >= 2 && seen_repository == 0)) || usage
            repository=$2; seen_repository=1; shift 2 ;;
        --output-parent)
            (($# >= 2 && seen_output == 0)) || usage
            output_parent=$2; seen_output=1; shift 2 ;;
        --resume)
            (($# >= 2 && seen_resume == 0)) || usage
            previous_bundle=$2; seen_resume=1; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$output_parent" ]] || usage

# Interpreter przygotowujący jest narzędziem hosta. Nigdy nie wybieramy Pythona
# z pakietu ani z badanego obrazu; -I -S wyłącza wpływ site/PYTHONPATH/cwd.
helper="$script_directory/stage9_v5d3_release.py"
verifier="$script_directory/stage9_v5d3_verify_bundle.sh"
trusted_python=/usr/bin/python3
[[ -x "$trusted_python" && -f "$helper" && ! -L "$helper" ]] || {
    echo 'Brak zaufanego narzędzia przygotowania.' >&2; exit 2;
}
run_helper() {
    env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
        "$trusted_python" -I -S -B "$helper" "$@"
}

attempt=$(run_helper init "$repository" "$output_parent" "$previous_bundle")
bundle="$attempt/bundle"
finished=0
on_exit() {
    local result=$?
    if ((finished == 0)); then
        run_helper interrupted "$attempt" || true
        echo "Zachowano nieukończoną próbę: $attempt" >&2
    fi
    return "$result"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Każdy etap pozostawia log i stan. Odmowa jednego etapu nie ukrywa wyników
# pozostałych niezależnych kontroli. Nie uruchamiamy regresji ani instalatora.
for stage in A B C D E F G H; do
    echo "Etap $stage: przygotowanie materiałów do przeglądu"
    if ! run_helper stage "$attempt" "$stage" > "$bundle/logs/$stage.log" 2>&1; then
        echo "Etap $stage zakończony odmową; szczegóły w logs/$stage.log" >&2
    fi
done

# I: jawne statusy i zamknięcie inwentarza. Nigdy nie nadpisujemy starej próby.
echo 'Etap I: podsumowanie oraz manifest integralności'
if ! run_helper finalize "$attempt" > "$bundle/logs/I.log" 2>&1; then
    echo 'Nie można zamknąć pakietu. Zachowano raport częściowy.' >&2
    exit 2
fi
# I.log musi być zamknięty PRZED pomiarem manifestu.
run_helper seal "$attempt"
verification_result=0
if [[ ! -f "$verifier" || -L "$verifier" ]]; then
    echo 'Brak niezależnego weryfikatora.' > "$attempt/verification.stderr"
    : > "$attempt/verification.json"
    verification_result=2
elif bash "$verifier" "$bundle" > "$attempt/verification.json" 2> "$attempt/verification.stderr"; then
    verification_result=0
else
    verification_result=$?
fi
printf '%s\n' "$verification_result" > "$attempt/verification.exit"
if ((verification_result == 0)); then
    run_helper handoff "$attempt"
fi
if ((verification_result != 0)); then
    run_helper reject "$attempt"
fi
finished=1
cat -- "$bundle/SUMMARY.tsv"
echo "BUNDLE=$bundle"
echo "Wynik weryfikacji: $attempt/verification.json"
echo "Materiał przekazania: $attempt/HANDOFF.json oraz oryginalne verification.json/.stderr/.exit i bundle/"
echo 'Przygotowanie pakietu nie stanowi autoryzacji wykonania.'
exit "$verification_result"
