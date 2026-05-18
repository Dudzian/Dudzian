#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <error-title> <summary-title> <secret-name> [<secret-name> ...]" >&2
  exit 2
fi

error_title="$1"
summary_title="$2"
shift 2

dr_checks_enabled="${DR_CHECKS_ENABLED:-}"
dr_report_dir="${DR_REPORT_DIR:-}"

write_status_artifact() {
  local status="$1"
  local reason="$2"
  local enabled="$3"
  local required_secrets_configured="$4"

  if [[ -z "$dr_report_dir" ]]; then
    return 0
  fi

  mkdir -p "$dr_report_dir"
  cat > "${dr_report_dir}/summary.md" <<EOF
## ${summary_title}
- status: **${status}**
- reason: ${reason}
EOF
  cat > "${dr_report_dir}/dr_status.json" <<EOF
{
  "status": "${status}",
  "reason": "${reason}",
  "enabled": ${enabled},
  "requiredSecretsConfigured": ${required_secrets_configured}
}
EOF
  cp "${dr_report_dir}/dr_status.json" "${dr_report_dir}/probe_status.json"
}

set_output_status() {
  local status="$1"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "dr_status=${status}" >> "$GITHUB_OUTPUT"
  fi
}

escape_actions_cmd() {
  local s="$1"
  s="${s//'%'/'%25'}"
  s="${s//$'\n'/'%0A'}"
  s="${s//$'\r'/'%0D'}"
  echo "$s"
}

if [[ "$dr_checks_enabled" != "true" ]]; then
  message="DR checks skipped: DR infrastructure is not enabled. Set vars.DR_CHECKS_ENABLED=true to run real DR probes."
  echo "$message"
  set_output_status "skipped"
  write_status_artifact "skipped" "$message" "false" "false"

  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "## ${summary_title}"
      echo "- status: **skipped**"
      echo "- reason: ${message}"
    } >> "$GITHUB_STEP_SUMMARY"
  fi
  exit 0
fi

missing=()
for secret_name in "$@"; do
  if [[ -z "${!secret_name:-}" ]]; then
    missing+=("$secret_name")
  fi
done

if (( ${#missing[@]} > 0 )); then
  message="Missing required secrets: ${missing[*]}. Configure them under Settings -> Secrets and variables -> Actions."
  echo "::error title=$(escape_actions_cmd "$error_title")::$(escape_actions_cmd "$message")" >&2

  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo "## ${summary_title}"
      echo "- status: **fail**"
      echo "- reason: ${message}"
    } >> "$GITHUB_STEP_SUMMARY"
  fi
  set_output_status "missing-required"
  write_status_artifact "missing-required" "$message" "true" "false"
  exit 1
fi

set_output_status "configured"
