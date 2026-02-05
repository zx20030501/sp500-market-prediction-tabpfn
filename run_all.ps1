param(
    [string]$DataRoot = "",
    [string]$OutDir = "outputs",
    [switch]$InstallDeps,
    [switch]$SkipEda,
    [string]$CkptPath = "",
    [switch]$Offline,
    [int]$MaxTrainingRows = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if ($InstallDeps) {
    Write-Host "[INFO] Installing dependencies..."
    python -m pip install -r requirements.txt
}

if ($DataRoot -ne "") {
    $env:HULL_DATA_ROOT = (Resolve-Path $DataRoot)
}

$resolvedOutDir = Join-Path $repoRoot $OutDir
New-Item -ItemType Directory -Force -Path $resolvedOutDir | Out-Null

if (-not $SkipEda) {
    Write-Host "[INFO] Running EDA reports..."
    $edaArgs = @("src/eda_tabpfn.py", "--output-dir", (Join-Path $resolvedOutDir "eda"))
    if ($DataRoot -ne "") {
        $edaArgs += @("--data-root", $DataRoot)
    }
    python @edaArgs

    Write-Host "[INFO] Generating EDA plots..."
    python src/eda_make_plots.py
}

Write-Host "[INFO] Running TabPFN pipeline..."
$pipelineArgs = @("src/run_tabpfn_pipeline.py", "--out-dir", $resolvedOutDir)
if ($DataRoot -ne "") {
    $pipelineArgs += @("--data-root", $DataRoot)
}
if ($CkptPath -ne "") {
    $pipelineArgs += @("--ckpt-path", $CkptPath)
}
if ($Offline) {
    $pipelineArgs += "--offline"
}
if ($MaxTrainingRows -gt 0) {
    $pipelineArgs += @("--max-training-rows", $MaxTrainingRows)
}
python @pipelineArgs

Write-Host "[DONE] Outputs written to $resolvedOutDir"
