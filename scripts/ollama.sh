#!/usr/bin/env bash
# Git Bash helper to run the Ollama CLI (the exe is not on PATH).
# Usage:  source scripts/ollama.sh   then:  ollama list
OLLAMA_BIN="${OLLAMA_BIN:-}"
if [ -z "$OLLAMA_BIN" ]; then
    for cand in \
        "$LOCALAPPDATA/Programs/Ollama/ollama.exe" \
        "$USERPROFILE/AppData/Local/Programs/Ollama/ollama.exe" \
        "/c/Program Files/Ollama/ollama.exe"; do
        if [ -x "$cand" ]; then OLLAMA_BIN="$cand"; break; fi
    done
fi
if [ -z "$OLLAMA_BIN" ]; then
    echo "ERROR: ollama.exe not found. Install Ollama first." >&2
    return 1
fi
ollama() { "$OLLAMA_BIN" "$@"; }
echo "ollama() helper ready -> $OLLAMA_BIN"
echo "examples:  ollama list   |   ollama pull qwen3.5:4b"
