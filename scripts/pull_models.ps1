# ============================================================
# KrishiSetu AI — pull Ollama models (run in PowerShell/CMD, NOT Git Bash)
# Why PowerShell? Ollama is a Windows desktop app whose binary lives at
# $env:LOCALAPPDATA\Programs\Ollama\ollama.exe and is NOT on the Git Bash PATH.
# Ensure the Ollama app is running first (system tray icon) — the server must
# be up because `ollama pull` talks to the local daemon on 127.0.0.1:11434.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\pull_models.ps1
#
# Model picks (Sept 2026, best open-source/free, fit RTX 4050 6GB / 24GB RAM,
# multilingual Hindi+English):
#   qwen3.5:4b       Tier-4 paraphraser — fits VRAM, strong Hindi (Q4 ~3.4GB)
#   qwen3.5:2b       light fallback LLM, fast (~2.7GB)
#   bge-m3:567m      multilingual embedding, 100+ langs, 1024-d
# ============================================================

$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
    Write-Error "ollama.exe not found at $ollama. Is Ollama installed?"
    exit 1
}

$models = @(
    @{ name = "qwen3.5:4b";          why = "Tier-4 paraphrasing LLM (Q4, strong Hindi, fits 6GB VRAM)" },
    @{ name = "qwen3.5:2b";          why = "small fast fallback LLM" },
    @{ name = "bge-m3:567m";         why = "multilingual embedding model (1024-d, 100+ languages)" }
)

foreach ($m in $models) {
    Write-Host ""
    Write-Host "Pulling $($m.name) — $($m.why)" -ForegroundColor Cyan
    & $ollama pull $m.name
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "pull of $($m.name) failed (exit $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "Listing installed models:" -ForegroundColor Cyan
& $ollama list
Write-Host ""
Write-Host "Optional upgrade (best local quality, slower on a 6GB GPU):"
Write-Host "  & `"$ollama`" pull qwen3.5:9b"
Write-Host ""
Write-Host "Done. Verify end-to-end with:"
Write-Host "  .venv\Scripts\python scripts\verify_setup.py"
