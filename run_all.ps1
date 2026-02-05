param(
    [string]$DataRoot = "",
    [string]$OutDir = "outputs",
    [switch]$InstallDeps,
    [switch]$DownloadData,
    [string]$KaggleCompetition = "hull-tactical-market-prediction",
    [switch]$SkipEda,
    [string]$CkptPath = "",
    [switch]$Offline,
    [int]$MaxTrainingRows = 0,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if ($InstallDeps) {
    Write-Host "[INFO] Installing dependencies..."
    python -m pip install -r requirements.txt
}

function Ensure-DataRoot([string]$root) {
    if ($root -eq "") {
        return $null
    }
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    return (Resolve-Path $root).Path
}

$resolvedOutDir = Join-Path $repoRoot $OutDir
New-Item -ItemType Directory -Force -Path $resolvedOutDir | Out-Null

if ($DownloadData -and $DataRoot -eq "") {
    $DataRoot = Join-Path $repoRoot "data\hull-tactical-market-prediction"
}

$resolvedDataRoot = Ensure-DataRoot $DataRoot
if ($resolvedDataRoot) {
    $env:HULL_DATA_ROOT = $resolvedDataRoot
}

function Download-KaggleData([string]$targetDir, [string]$competition) {
    if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
        throw "Kaggle CLI not found. Install via 'pip install kaggle' and ensure it's on PATH."
    }
    $kaggleConfig = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
    if (-not (Test-Path $kaggleConfig)) {
        throw "Kaggle credentials not found. Place kaggle.json in $kaggleConfig"
    }
    Write-Host "[INFO] Downloading Kaggle competition data: $competition"
    kaggle competitions download -c $competition -p $targetDir

    Get-ChildItem -Path $targetDir -Filter *.zip | ForEach-Object {
        Write-Host "[INFO] Extracting $($_.Name)"
        Expand-Archive -Force -Path $_.FullName -DestinationPath $targetDir
    }
}

if ($DownloadData) {
    if (-not $resolvedDataRoot) {
        throw "DataRoot is required for download."
    }
    Download-KaggleData $resolvedDataRoot $KaggleCompetition
}

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

function Validate-Submission([string]$submissionPath, [string]$testPath) {
    if (-not (Test-Path $submissionPath)) {
        throw "submission.csv not found at $submissionPath"
    }
    $header = (Get-Content $submissionPath -TotalCount 1).Split(",") | ForEach-Object { $_.Trim() }
    $hasPrediction = $header -contains "prediction"
    $hasId = ($header -contains "row_id") -or ($header -contains "id")
    if (-not $hasPrediction -or -not $hasId) {
        throw "Invalid submission header. Expect columns: row_id/prediction or id/prediction. Got: $($header -join ', ')"
    }
    if ($testPath -ne "" -and (Test-Path $testPath)) {
        $testCount = (Get-Content $testPath | Measure-Object -Line).Lines - 1
        $subCount = (Get-Content $submissionPath | Measure-Object -Line).Lines - 1
        if ($testCount -ne $subCount) {
            Write-Host "[WARN] Row count mismatch: test=$testCount submission=$subCount"
        } else {
            Write-Host "[INFO] Submission row count matches test rows: $subCount"
        }
    } else {
        Write-Host "[WARN] test.csv not found. Skipping row count check."
    }
    $nanHits = Select-String -Path $submissionPath -Pattern ",NaN" -SimpleMatch
    if ($nanHits) {
        Write-Host "[WARN] Found NaN values in submission. Check your pipeline output."
    }
}

if (-not $SkipValidation) {
    $submissionCsv = Join-Path $resolvedOutDir "submission.csv"
    $testCsv = if ($resolvedDataRoot) { Join-Path $resolvedDataRoot "test.csv" } else { "" }
    Write-Host "[INFO] Validating submission..."
    Validate-Submission $submissionCsv $testCsv
}

Write-Host "[DONE] Outputs written to $resolvedOutDir"
