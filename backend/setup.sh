#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="$SCRIPT_DIR/.env"
VENV_DIR="$SCRIPT_DIR/.venv"

choose_python() {
  local candidates=("python3.13" "python3.12" "python3")
  local py=""
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1; then
      py="$c"
      break
    fi
  done
  if [[ -z "$py" ]]; then
    echo "Error: Python 3.12+ is required but no python3 executable was found."
    exit 1
  fi
  echo "$py"
}

check_python_version() {
  local py="$1"
  if ! "$py" - <<'PY'
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major, minor) >= (3, 12) else 1)
PY
  then
    echo "Error: OpenSift requires Python 3.12 or newer."
    "$py" -V || true
    exit 1
  fi
}

mask_value() {
  local value="$1"
  local len="${#value}"
  if [[ -z "$value" ]]; then
    printf "(not set)"
  elif (( len <= 8 )); then
    printf "********"
  else
    printf "%s...%s" "${value:0:4}" "${value: -4}"
  fi
}

get_env_value() {
  local key="$1"
  if [[ ! -f "$ENV_FILE" ]]; then
    return 0
  fi
  local line
  line="$(grep -E "^${key}=" "$ENV_FILE" | head -n 1 || true)"
  if [[ -n "$line" ]]; then
    printf "%s" "${line#*=}"
  fi
}

upsert_env_value() {
  local key="$1"
  local value="$2"
  touch "$ENV_FILE"
  awk -v k="$key" -v v="$value" '
    BEGIN { done = 0 }
    $0 ~ ("^" k "=") {
      if (!done) {
        print k "=" v
        done = 1
      }
      next
    }
    { print }
    END {
      if (!done) print k "=" v
    }
  ' "$ENV_FILE" > "${ENV_FILE}.tmp"
  mv "${ENV_FILE}.tmp" "$ENV_FILE"
}

remove_env_value() {
  local key="$1"
  [[ -f "$ENV_FILE" ]] || return 0
  awk -v k="$key" '$0 !~ ("^" k "=") { print }' "$ENV_FILE" > "${ENV_FILE}.tmp"
  mv "${ENV_FILE}.tmp" "$ENV_FILE"
}

prompt_secret_key() {
  local key="$1"
  local label="$2"
  local current
  current="$(get_env_value "$key")"
  local masked
  masked="$(mask_value "$current")"

  echo
  echo "$label"
  echo "Current: $masked"
  echo "Enter to keep, type 'none' to clear."
  read -r -s -p "New value: " value
  echo

  value="${value//$'\n'/}"
  if [[ -z "$value" ]]; then
    return 0
  fi
  if [[ "${value,,}" == "none" ]]; then
    remove_env_value "$key"
  else
    upsert_env_value "$key" "$value"
  fi
}

prompt_plain_key() {
  local key="$1"
  local label="$2"
  local current
  current="$(get_env_value "$key")"

  echo
  echo "$label"
  echo "Current: ${current:-"(not set)"}"
  echo "Enter to keep, type 'none' to clear."
  read -r -p "New value: " value

  value="${value//$'\n'/}"
  if [[ -z "$value" ]]; then
    return 0
  fi
  if [[ "${value,,}" == "none" ]]; then
    remove_env_value "$key"
  else
    upsert_env_value "$key" "$value"
  fi
}

main() {
  echo "OpenSift Bootstrap Setup"
  echo "================================================================"

  local PY
  PY="$(choose_python)"
  check_python_version "$PY"
  echo "Using Python: $("$PY" -V 2>&1)"

  if [[ ! -d "$VENV_DIR" ]]; then
    echo
    echo "Creating virtual environment at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  echo
  echo "Installing Python dependencies..."
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install openai
  python -m pip install anthropic
  python -m pip install sentence-transformers
  python -m pip install -r requirements.txt

  touch "$ENV_FILE"

  echo
  echo "Configure API keys and tokens (.env)"
  echo "================================================================"
  prompt_secret_key "OPENAI_API_KEY" "OpenAI API key (optional)"
  prompt_secret_key "ANTHROPIC_API_KEY" "Anthropic API key (optional)"
  prompt_secret_key "CLAUDE_CODE_OAUTH_TOKEN" "Claude Code OAuth token (optional)"
  prompt_secret_key "CHATGPT_CODEX_OAUTH_TOKEN" "ChatGPT Codex OAuth token (optional)"
  prompt_plain_key "OPENSIFT_CLAUDE_CODE_CMD" "Claude Code command (default: claude)"
  prompt_plain_key "OPENSIFT_CLAUDE_CODE_ARGS" "Claude Code extra args (optional)"
  prompt_plain_key "OPENSIFT_CODEX_CMD" "Codex command (default: codex)"
  prompt_plain_key "OPENSIFT_CODEX_ARGS" "Codex extra args (optional)"
  prompt_plain_key "OPENSIFT_CODEX_AUTH_PATH" "Codex auth path (default: ~/.codex/auth.json)"

  echo
  echo "Saved configuration to: $ENV_FILE"
  echo "Launching OpenSift setup wizard..."
  echo "================================================================"
  python opensift.py setup --skip-key-prompts
}

main "$@"
