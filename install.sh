#!/usr/bin/env bash
# install.sh — install the recall agentic-memory CLI and its hook scripts.
#
# Two modes:
#   1. Repo-local: invoked as `./install.sh` from a checkout of
#      j0yen/recall.
#   2. Curl-piped: invoked as `curl ... | bash`. No checkout exists;
#      script self-clones into ~/.local/share/recall/ (full clone,
#      since we need to build from source).
#
# Steps in either mode:
#   1. Build + install the `recall` binary via `cargo install --path .`
#      (lands in ~/.cargo/bin/recall).
#   2. Symlink the three braid hook scripts into ~/.claude/scripts/.
#   3. Print the JSON snippet to paste into ~/.claude/settings.json
#      to wire the hooks.
#
# install.sh never auto-edits settings.json — the snippet is yours to
# review and merge. Re-running install.sh is safe: hooks are atomic
# symlinks (`ln -snf`); the binary is replaced by cargo's normal
# install-overwrite behavior.

set -euo pipefail

# --- Mode detection ---------------------------------------------------
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=""
if [ -f "$SCRIPT_PATH" ]; then
  SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
fi

if [ -z "$SCRIPT_DIR" ] || [ ! -f "$SCRIPT_DIR/Cargo.toml" ] \
   || ! grep -q '^name = "recall"' "$SCRIPT_DIR/Cargo.toml" 2>/dev/null; then
  # Mode 2: curl|bash. Self-clone.
  echo "→ no local checkout detected; self-cloning j0yen/recall..."
  command -v git >/dev/null 2>&1 || { echo "fatal: git not found"; exit 1; }

  CLONE_ROOT="${RECALL_CLONE_ROOT:-$HOME/.local/share/recall}"
  mkdir -p "$(dirname "$CLONE_ROOT")"

  if [ -d "$CLONE_ROOT/.git" ]; then
    echo "→ existing clone at $CLONE_ROOT — refreshing"
    git -C "$CLONE_ROOT" fetch --depth 1 origin main
    git -C "$CLONE_ROOT" reset --hard origin/main
  else
    git clone --depth 1 https://github.com/j0yen/recall.git "$CLONE_ROOT"
  fi

  SCRIPT_DIR="$CLONE_ROOT"
fi

cd "$SCRIPT_DIR"

# --- Build + install binary ------------------------------------------
command -v cargo >/dev/null 2>&1 || {
  echo "fatal: cargo not found. Install Rust: https://rustup.rs/"
  exit 1
}

echo "→ building + installing recall binary (this can take a few minutes)..."
cargo install --path . --locked

if ! command -v recall >/dev/null 2>&1; then
  echo
  echo "! recall installed but not on PATH. Add ~/.cargo/bin to PATH, or:"
  echo "    export PATH=\"\$HOME/.cargo/bin:\$PATH\""
fi

INSTALLED_VERSION="$(recall --version 2>/dev/null || echo unknown)"
echo "→ installed: $INSTALLED_VERSION"

# --- Symlink hook scripts --------------------------------------------
HOOK_DIR="$HOME/.claude/scripts"
mkdir -p "$HOOK_DIR"

declare -a HOOKS=(
  "post-tool-use.sh:recall-post-tool-use.sh"
  "user-prompt-submit.sh:recall-user-prompt.sh"
  "stop.sh:recall-stop.sh"
)

echo "→ symlinking hooks into $HOOK_DIR/"
for spec in "${HOOKS[@]}"; do
  src_name="${spec%%:*}"
  dst_name="${spec##*:}"
  src="$SCRIPT_DIR/hooks/$src_name"
  dst="$HOOK_DIR/$dst_name"

  if [ ! -f "$src" ]; then
    echo "  ! skipping $dst_name — source not found at $src"
    continue
  fi
  ln -snf "$src" "$dst"
  echo "  $dst → $src"
done

# --- Print settings.json snippet -------------------------------------
echo
echo "═══════════════════════════════════════════════════════════════════"
echo " NEXT STEP — wire the hooks into Claude Code"
echo "═══════════════════════════════════════════════════════════════════"
echo
echo "Add these entries to ~/.claude/settings.json under the .hooks key."
echo "install.sh does NOT auto-edit settings.json. Snapshot the file"
echo "first if you want a rollback target."
echo

USER_HOME="$HOME"
cat <<JSON
{
  "hooks": {
    "UserPromptSubmit": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "$USER_HOME/.claude/scripts/recall-user-prompt.sh",
          "timeout": 1, "async": true }
      ]}
    ],
    "PostToolUseFailure": [
      { "matcher": "*", "hooks": [
        { "type": "command", "command": "$USER_HOME/.claude/scripts/recall-post-tool-use.sh",
          "timeout": 5, "async": true }
      ]}
    ],
    "Stop": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "$USER_HOME/.claude/scripts/recall-stop.sh",
          "timeout": 30, "async": true }
      ]}
    ]
  }
}
JSON
echo
echo "After merging, restart your Claude Code session. The braid"
echo "correlator parks reflective draft memories at:"
echo "   ~/.claude/recall/proposals/"
echo "Review them with:  recall proposals"
echo
echo "✓ recall installed."
