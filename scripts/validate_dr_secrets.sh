#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <error-title> <summary-title> <secret-name> [<secret-name> ...]" >&2
  exit 2
fi

error_title="$1"
summary_title="$2"
shift 2

escape_actions_cmd() {
  local s="$1"
  s="${s//'%'/'%25'}"
  s="${s//$'\n'/'%0A'}"
  s="${s//$'\r'/'%0D'}"
  echo "$s"
}

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
  exit 1
fi
