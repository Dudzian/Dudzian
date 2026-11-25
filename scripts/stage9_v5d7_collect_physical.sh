#!/usr/bin/env bash
# Collect and package static data on the original Stage 9 operator host.
set -euo pipefail
umask 077
export LC_ALL=C
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset TAR_OPTIONS GZIP

die() {
    printf '%s\n' "$1" >&2
    exit 2
}

if [[ $# != 2 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
    die 'Usage: bash stage9_v5d7_collect_physical.sh SOURCE_COMMIT_SHA OUTPUT_PARENT'
fi
expected_revision=$1
output_parent=$(cd -P -- "$2" && pwd -P)
script_directory=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -P -- "$script_directory/.." && pwd -P)
actual_revision=$(git -C "$source_root" rev-parse --verify 'HEAD^{commit}')
[[ $actual_revision == "$expected_revision" ]] || die 'V5D7_SOURCE_REVISION_MISMATCH'
worktree_status=$(git -C "$source_root" status --porcelain=v1 --untracked-files=all)
[[ -z $worktree_status ]] || die 'V5D7_SOURCE_WORKTREE_CHANGED'

python=/usr/bin/python3.12
[[ -x $python ]] || die 'V5D7_SYSTEM_PYTHON_3_12_MISSING'
version=$(
    "$python" -I -B -S -c 'import sys; print(sys.implementation.name, ".".join(map(str, sys.version_info[:3])), sys.version_info.releaselevel)'
)
[[ $version == 'cpython 3.12.3 final' ]] || die "V5D7_SYSTEM_PYTHON_VERSION_MISMATCH: $version"

base=/home/dudzian/stage9-v5d4-wheelhouse-candidate-NTUwn8B7
wheelhouse=$base/wheelhouse
origins=$base/origins.json
reference=/home/dudzian/stage9-v5d6-audit-PFvqXhAF/stage9-v5d6-full-evidence.tar.gz
[[ -d $wheelhouse && -f $origins && -f $reference ]] || die 'V5D7_ORIGINAL_INPUTS_MISSING'

scratch=$(mktemp -d "${TMPDIR:-/tmp}/stage9-v5d7-collect.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
contents=$scratch/contents
verification=$scratch/verification
mkdir -- "$contents" "$verification"

printf '%s\n' "$actual_revision" > "$contents/revision.txt"
printf 'interpreter: %s\nversion: %s\nwheelhouse: %s\norigins: %s\nreference: %s\n' \
    "$python" "$version" "$wheelhouse" "$origins" "$reference" > "$contents/inputs.txt"

collector_exit=0
"$python" -I -B -S "$script_directory/stage9_v5d7_collect_evidence.py" \
    --reference "$reference" --wheelhouse "$wheelhouse" --origins "$origins" \
    --measure-host-abi --proposal-output "$contents/v5d7-proposal.json" \
    > "$contents/wheelhouse-evidence.json" 2> "$contents/stderr.txt" || collector_exit=$?
printf '%s\n' "$collector_exit" > "$contents/exit-code.txt"

members=(revision.txt inputs.txt wheelhouse-evidence.json stderr.txt exit-code.txt)
if [[ -f $contents/v5d7-proposal.json ]]; then
    members+=(v5d7-proposal.json)
fi
(
    cd -- "$contents"
    sha256sum -- "${members[@]}" > SHA256SUMS
    sha256sum --check --strict SHA256SUMS
)

result=$(mktemp -d "$output_parent/stage9-v5d7-evidence.XXXXXXXX")
archive=stage9-v5d7-physical-evidence.tar.gz
tar -czf "$result/$archive" -C "$contents" -- SHA256SUMS "${members[@]}"
(
    cd -- "$result"
    sha256sum -- "$archive" > "$archive.sha256"
    sha256sum --check --strict "$archive.sha256"
)
tar -xzf "$result/$archive" -C "$verification"
(
    cd -- "$verification"
    sha256sum --check --strict SHA256SUMS
)
printf 'ARCHIVE=%s\nSHA256_FILE=%s\nCOLLECTOR_EXIT=%s\n' \
    "$result/$archive" "$result/$archive.sha256" "$collector_exit"
exit "$collector_exit"
