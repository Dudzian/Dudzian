#!/usr/bin/env bash
# Diagnostic observations only. Never run the controller or a candidate Python.
# Requires Bash; GNU stat is used only on Linux. stdout is the sole output sink.
set -u
export LC_ALL=C
export GIT_OPTIONAL_LOCKS=0
export GIT_TERMINAL_PROMPT=0

EXPECTED_BASELINE=213226f6222f8d1ece05987afe038c82ed61923d
script_directory=${BASH_SOURCE[0]%/*}
if [[ $script_directory == "${BASH_SOURCE[0]}" ]]; then script_directory=.; fi
repository=$script_directory/..
if (( $# )); then
    if [[ $# != 2 || $1 != --repository || -z $2 ]]; then
        printf 'Usage: bash %s [--repository PATH]\n' "${BASH_SOURCE[0]}" >&2
        exit 64
    fi
    repository=$2
fi

emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; }

boundaries() {
    emit inventory_collection PRESENT COLLECTION_COMPLETE_NOT_DEPLOYMENT_READINESS
    emit independent_approval UNVERIFIED EXTERNAL_APPROVAL_NOT_ESTABLISHED_BY_METADATA
    emit independent_launch UNVERIFIED LOADED_CODE_AND_STARTUP_NOT_ATTESTED
    emit policy_signature NOT_TESTED POLICY_CONTENT_AND_KEYS_NOT_READ
    emit python_dependencies NOT_TESTED CANDIDATE_INTERPRETER_AND_IMPORTS_NOT_EXECUTED
    emit domain_qualification NOT_TESTED NO_DOMAIN_SELECTED_OR_INSPECTED
    emit immutable_images NOT_TESTED NO_IMAGES_OPENED_OR_MEASURED
    emit host_qualification BLOCKED INDEPENDENT_ACCEPTANCE_REQUIRED
    printf '%s\n' \
        'Stage 9 = IN_PROGRESS' \
        'Observer 3A = BLOCKED' \
        'Stage 3B = NOT AUTHORIZED' \
        'Production = NO-GO' \
        'HOST_PREFLIGHT_VERIFIED = NOT_CLAIMED' \
        'EXECUTION_AUTHORIZATION = NOT_AUTHORIZED'
}

has_command() { command -v "$1" >/dev/null 2>&1; }

# lstat each component: never traverse a symlink or unsafe protected ancestor.
# This is a racy diagnostic snapshot, NOT V5d1's held-FD protection verifier.
inspect_path() {
    local path=$1 kind=$2 protected=$3 remaining component current=/
    local metadata mode uid gid links size extra bits expected_type detail
    if ! has_command stat; then
        emit "$path" NOT_TESTED STAT_UNAVAILABLE
        return
    fi
    remaining=${path#/}
    while :; do
        if ! metadata=$(stat -c '%f %u %g %h %s' -- "$current" 2>&1); then
            case $metadata in
                *'No such file or directory'*) detail=MISSING ;;
                *'Permission denied'*|*'Operation not permitted'*) detail=INACCESSIBLE ;;
                *) detail=UNVERIFIED ;;
            esac
            emit "$path" "$detail" METADATA_UNAVAILABLE
            return
        fi
        read -r mode uid gid links size extra <<< "$metadata"
        if [[ ! $mode =~ ^[0-9a-fA-F]{1,8}$ || ! $uid =~ ^[0-9]{1,10}$ ||
              ! $gid =~ ^[0-9]{1,10}$ || ! $links =~ ^[0-9]{1,10}$ ||
              ! $size =~ ^[0-9]{1,20}$ || -n $extra ]]; then
            emit "$path" UNVERIFIED METADATA_FORMAT_UNSUPPORTED
            return
        fi
        bits=$((16#$mode))
        if (( (bits & 0170000) == 0120000 )); then
            emit "$path" BLOCKED SYMLINK_PATH
            return
        fi
        expected_type=0040000
        if [[ $current == "$path" ]]; then
            case $kind in
                file) expected_type=0100000 ;;
                device) expected_type=0020000 ;;
            esac
        fi
        if (( (bits & 0170000) != expected_type )); then
            emit "$path" BLOCKED UNEXPECTED_FILE_TYPE
            return
        fi
        if [[ $protected == yes ]]; then
            if [[ $uid != 0 ]] || (( (bits & 0022) != 0 )); then
                emit "$path" BLOCKED OWNER_OR_WRITE_PERMISSIONS_UNSAFE
                return
            fi
            if [[ $current == "$path" && $kind == file && $links != 1 ]]; then
                emit "$path" BLOCKED HARDLINKED_ARTIFACT
                return
            fi
        fi
        if [[ $current == "$path" ]]; then
            emit "$path" PRESENT "mode_hex=$mode uid=$uid gid=$gid links=$links size=$size METADATA_ONLY"
            return
        fi
        component=${remaining%%/*}
        if [[ $remaining == */* ]]; then remaining=${remaining#*/}; else remaining=; fi
        if [[ $current == / ]]; then current=/$component; else current=$current/$component; fi
    done
}

printf 'ITEM\tSTATUS\tDETAIL\n'
emit inventory_profile PRESENT STAGE9_OPERATOR_V5D2_DIAGNOSTIC_ONLY
if ! has_command uname; then
    emit host_platform UNVERIFIED UNAME_UNAVAILABLE
    emit host_checks NOT_TESTED PLATFORM_NOT_ESTABLISHED
    boundaries
    exit 0
fi
host_system=$(uname -s 2>/dev/null) || host_system=
host_machine=$(uname -m 2>/dev/null) || host_machine=
host_kernel=$(uname -r 2>/dev/null) || host_kernel=
if [[ $host_system != Linux || $host_machine != x86_64 ]]; then
    emit host_platform BLOCKED OUTSIDE_RETAINED_LINUX_X86_64_PROFILE
    emit host_checks NOT_TESTED UNSUPPORTED_OR_UNKNOWN_PLATFORM
    boundaries
    exit 0
fi
printf -v host_detail 'system=%q machine=%q kernel=%q OBSERVATION_ONLY' "$host_system" "$host_machine" "$host_kernel"
emit host_platform PRESENT "$host_detail"
emit host_distribution UNVERIFIED UBUNTU_RELEASE_NOT_INDEPENDENTLY_ESTABLISHED

for dependency in stat env git timeout virsh qemu-img qemu-system-x86_64 dpkg-query; do
    if has_command "$dependency"; then
        emit "dependency:$dependency" PRESENT EXECUTABLE_DISCOVERABLE_NOT_APPROVED
    else
        emit "dependency:$dependency" MISSING EXECUTABLE_NOT_DISCOVERABLE
    fi
done

inspect_path /dev/kvm device no
if [[ -r /dev/kvm && -w /dev/kvm ]]; then
    emit kvm_access PRESENT SHELL_READ_WRITE_ACCESS_OBSERVED_NO_IOCTL_TEST
else
    emit kvm_access BLOCKED READ_WRITE_ACCESS_NOT_ESTABLISHED
fi
inspect_path /sys/fs/cgroup/cgroup.controllers file no

for path in /opt/cryptohunter /opt/cryptohunter/observer-v5d1 \
    /opt/cryptohunter/observer-v5d1/venv /opt/cryptohunter/observer-v5d1/venv/bin \
    /etc/cryptohunter /etc/cryptohunter/observer-v5d1; do
    inspect_path "$path" directory yes
done
for path in /opt/cryptohunter/observer-v5d1/venv/bin/python \
    /etc/cryptohunter/observer-v5d1/operator-anchors.json \
    /etc/cryptohunter/observer-v5d1/approved-policy.json \
    /etc/cryptohunter/observer-v5d1/domain-requirements.json \
    /usr/bin/qemu-img /usr/bin/qemu-system-x86_64; do
    inspect_path "$path" file yes
done
for relative in deployment/canonical_json.py \
    deployment/stage9_observer_collector_identity_v5d1.py \
    deployment/stage9_observer_kvm_domain_v5d1.py \
    deployment/stage9_observer_kvm_image_v5d1.py \
    deployment/stage9_observer_kvm_preflight_v5d1.py \
    scripts/stage9_observer_kvm_preflight_v5d1.py; do
    inspect_path "/opt/cryptohunter/observer-v5d1/$relative" file yes
done

if has_command git && has_command env; then
    # Drop inherited trace destinations and disable configured Trace2 targets.
    # No global/system Git configuration, hooks or subsidiary submodule probes.
    git_read=(env -i "PATH=$PATH" LC_ALL=C GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0
        GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
        GIT_TRACE2=0 GIT_TRACE2_EVENT=0 GIT_TRACE2_PERF=0
        git --no-pager --no-optional-locks -c core.fsmonitor=false -c core.untrackedCache=false -C "$repository")
    if revision=$("${git_read[@]}" rev-parse --verify HEAD 2>/dev/null) &&
        [[ $revision =~ ^[0-9a-f]{40}$ ]]; then
        if [[ $revision == "$EXPECTED_BASELINE" ]]; then
            emit repository_revision PRESENT "$revision EXPECTED_BASELINE_OBSERVED_NOT_RELEASE_APPROVAL"
        else
            emit repository_revision UNVERIFIED "$revision BASELINE_DIFFERS_REQUIRES_INDEPENDENT_REVIEW"
        fi
    else
        emit repository_revision INACCESSIBLE REVISION_NOT_OBSERVED
    fi
    if worktree=$("${git_read[@]}" status --porcelain=v1 --untracked-files=normal --ignore-submodules=all 2>/dev/null); then
        if [[ -z $worktree ]]; then
            emit repository_worktree PRESENT CLEAN_SNAPSHOT_NOT_TRUSTED_INSTALLATION
        else
            emit repository_worktree UNVERIFIED LOCAL_CHANGES_PRESENT
        fi
    else
        emit repository_worktree INACCESSIBLE WORKTREE_NOT_OBSERVED
    fi
else
    emit repository_revision NOT_TESTED GIT_OR_ENV_UNAVAILABLE
    emit repository_worktree NOT_TESTED GIT_OR_ENV_UNAVAILABLE
fi

# Only metadata from the host package database; never import Python modules.
# Ubuntu package names are diagnostic hints, not an approved version matrix.
if has_command dpkg-query && has_command timeout && has_command env; then
    for package in python3-libvirt libvirt-clients libvirt-daemon-system qemu-system-x86 qemu-utils; do
        if package_status=$(env -i "PATH=$PATH" LC_ALL=C timeout --signal=TERM --kill-after=2s 10s \
            dpkg-query -W '-f=${db:Status-Status}' "$package" 2>/dev/null); then
            if [[ $package_status == installed ]]; then
                emit "package:$package" PRESENT DATABASE_RECORD_NOT_VERSION_APPROVAL
            else
                emit "package:$package" UNVERIFIED NO_INSTALLED_DATABASE_RECORD
            fi
        else
            emit "package:$package" UNVERIFIED PACKAGE_QUERY_UNAVAILABLE
        fi
    done
else
    emit host_package_database NOT_TESTED DPKG_QUERY_TIMEOUT_OR_ENV_UNAVAILABLE
fi

# No default URI, user-provided URI, domain lookup or lifecycle command.
# Noninteractive mode prevents authentication prompts; never broaden access.
if has_command virsh && has_command timeout && has_command env; then
    # No inherited libvirt file logging, authentication file or virsh history.
    if observed_uri=$(env -i "PATH=$PATH" LC_ALL=C LIBVIRT_AUTH_FILE=/dev/null \
        LIBVIRT_LOG_OUTPUTS=4:stderr VIRSH_HISTFILE=/dev/null \
        timeout --signal=TERM --kill-after=2s 10s \
        virsh --readonly --connect qemu:///system uri </dev/null 2>/dev/null); then
        if [[ $observed_uri == qemu:///system ]]; then
            emit libvirt_readonly PRESENT FIXED_URI_REACHABLE_NOT_DOMAIN_QUALIFICATION
        else
            emit libvirt_readonly BLOCKED RETURNED_URI_MISMATCH
        fi
    else
        emit libvirt_readonly INACCESSIBLE READ_ONLY_PROBE_FAILED_OR_TIMED_OUT
    fi
else
    emit libvirt_readonly NOT_TESTED VIRSH_TIMEOUT_OR_ENV_UNAVAILABLE
fi

boundaries
exit 0
