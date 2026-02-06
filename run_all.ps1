param(
    [string]$DataRoot = "",
    [string]$OutDir = "outputs",
    [string]$PythonExe = "",
    [switch]$InstallDeps,
    [switch]$DownloadData,
    [string]$KaggleCompetition = "hull-tactical-market-prediction",
    [switch]$SkipEda,
    [string]$CkptPath = "",
    [switch]$Offline,
    [int]$MaxTrainingRows = 0,
    [switch]$SkipValidation,
    [bool]$RequireGPU = $true,
    [int]$GpuMinUtil = 5,
    [int]$GpuMinMemoryMB = 200,
    [int]$GpuCheckDelaySec = 30,
    [int]$GpuCheckWindowSec = 120,
    [int]$GpuCheckIntervalSec = 10,
    [string]$LogPath = "",
    [switch]$DisableLog
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonCmd = "python"
if ($PythonExe -ne "") {
    $pythonCmd = (Resolve-Path $PythonExe).Path
    $pythonDir = Split-Path $pythonCmd
    $env:Path = "$pythonDir;$env:Path"
}

function Get-NvidiaSmiPath() {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Path
    }
    $fallback = Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path $fallback) {
        return $fallback
    }
    return $null
}

function Ensure-DataRoot([string]$root) {
    if ($root -eq "") {
        return $null
    }
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    return (Resolve-Path $root).Path
}

function Download-KaggleData([string]$targetDir, [string]$competition) {
    try {
        & $pythonCmd -m kaggle --version | Out-Null
    } catch {
        throw "Kaggle CLI not found. Install via '$pythonCmd -m pip install kaggle' and ensure it's on PATH."
    }
    $kaggleConfig = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
    if (-not (Test-Path $kaggleConfig)) {
        throw "Kaggle credentials not found. Place kaggle.json in $kaggleConfig"
    }
    Write-Host "[INFO] Downloading Kaggle competition data: $competition"
    & $pythonCmd -m kaggle competitions download -c $competition -p $targetDir

    Get-ChildItem -Path $targetDir -Filter *.zip | ForEach-Object {
        Write-Host "[INFO] Extracting $($_.Name)"
        Expand-Archive -Force -Path $_.FullName -DestinationPath $targetDir
    }
}

function Assert-GpuReady([string]$pythonCmd, [bool]$requireGpu) {
    $nvidiaSmi = Get-NvidiaSmiPath
    if (-not $nvidiaSmi) {
        if ($requireGpu) {
            throw "nvidia-smi not found. Install NVIDIA driver or set -RequireGPU:$false."
        }
        Write-Host "[WARN] nvidia-smi not found. GPU monitoring disabled."
        return $false
    }
    $cudaOk = $false
    try {
        & $pythonCmd -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $cudaOk = $true
        }
    } catch {
        $cudaOk = $false
    }
    if (-not $cudaOk) {
        if ($requireGpu) {
            throw "Torch CUDA is not available for $pythonCmd. Install CUDA-enabled Torch or set -RequireGPU:$false."
        }
        Write-Host "[WARN] Torch CUDA not available. GPU monitoring disabled."
        return $false
    }
    return $true
}

function Get-GpuStats() {
    $nvidiaSmi = Get-NvidiaSmiPath
    if (-not $nvidiaSmi) {
        return $null
    }
    $lines = & $nvidiaSmi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    if (-not $lines) {
        return $null
    }
    $maxUtil = 0
    $maxMem = 0
    foreach ($line in $lines) {
        $parts = $line.Split(",") | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 2) {
            $util = 0
            $mem = 0
            [int]::TryParse($parts[0], [ref]$util) | Out-Null
            [int]::TryParse($parts[1], [ref]$mem) | Out-Null
            if ($util -gt $maxUtil) { $maxUtil = $util }
            if ($mem -gt $maxMem) { $maxMem = $mem }
        }
    }
    return [pscustomobject]@{
        Util = $maxUtil
        MemMB = $maxMem
    }
}

function Wait-ProcessWithGpuMonitor(
    [System.Diagnostics.Process]$Process,
    [bool]$EnableMonitor,
    [int]$MinUtil,
    [int]$MinMemMB,
    [int]$DelaySec,
    [int]$WindowSec,
    [int]$IntervalSec
) {
    if (-not $EnableMonitor) {
        $Process.WaitForExit()
        $Process.Refresh()
        $exitCode = $Process.ExitCode
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
        return $exitCode
    }

    if ($IntervalSec -le 0) {
        throw "GpuCheckIntervalSec must be > 0."
    }

    $samplesNeeded = [math]::Max(1, [math]::Ceiling([double]$WindowSec / $IntervalSec))
    Write-Host "[INFO] GPU monitor: delay=$DelaySec sec window=$WindowSec sec interval=$IntervalSec sec util>=$MinUtil% or mem>=$MinMemMB MB"

    if ($DelaySec -gt 0) {
        Start-Sleep -Seconds $DelaySec
    }

    $lowCount = 0
    while (-not $Process.HasExited) {
        $stats = Get-GpuStats
        if (-not $stats) {
            $lowCount++
            Write-Host "[WARN] Unable to read GPU stats. ($lowCount/$samplesNeeded)"
        } else {
            $active = ($stats.Util -ge $MinUtil) -or ($stats.MemMB -ge $MinMemMB)
            Write-Host "[INFO] GPU stats: util=$($stats.Util)% mem=$($stats.MemMB)MB"
            if ($active) {
                $lowCount = 0
            } else {
                $lowCount++
            }
        }

        if ($lowCount -ge $samplesNeeded) {
            Write-Host "[ERROR] GPU usage below threshold for ${WindowSec}s. Terminating pipeline..."
            try {
                $Process.Kill()
            } catch {
                Write-Host "[WARN] Failed to kill pipeline process: $($_.Exception.Message)"
            }
            throw "GPU usage below threshold. Pipeline terminated."
        }

        Start-Sleep -Seconds $IntervalSec
    }

    $Process.WaitForExit()
    $Process.Refresh()
    $exitCode = $Process.ExitCode
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    return $exitCode
}

$resolvedOutDir = Join-Path $repoRoot $OutDir
New-Item -ItemType Directory -Force -Path $resolvedOutDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $resolvedOutDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if ($LogPath -eq "") {
    $LogPath = Join-Path $logDir "run_all_$timestamp.log"
} elseif (-not [System.IO.Path]::IsPathRooted($LogPath)) {
    $LogPath = Join-Path $repoRoot $LogPath
}

$logStarted = $false
if (-not $DisableLog) {
    try {
        Start-Transcript -Path $LogPath -Append | Out-Null
        $logStarted = $true
        Write-Host "[INFO] Log file: $LogPath"
    } catch {
        Write-Host "[WARN] Failed to start transcript logging: $($_.Exception.Message)"
    }
}

try {
    if ($InstallDeps) {
        Write-Host "[INFO] Installing dependencies..."
        & $pythonCmd -m pip install -r requirements.txt
    }

    if ($DownloadData -and $DataRoot -eq "") {
        $DataRoot = Join-Path $repoRoot "data\hull-tactical-market-prediction"
    }

    $resolvedDataRoot = Ensure-DataRoot $DataRoot
    if ($resolvedDataRoot) {
        $env:HULL_DATA_ROOT = $resolvedDataRoot
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
        & $pythonCmd @edaArgs

        Write-Host "[INFO] Generating EDA plots..."
        & $pythonCmd src/eda_make_plots.py
    }

    $gpuMonitorEnabled = $false
    if ($RequireGPU) {
        $gpuMonitorEnabled = Assert-GpuReady $pythonCmd $true
        if ($GpuCheckWindowSec -le 0) {
            Write-Host "[WARN] GpuCheckWindowSec <= 0. GPU monitoring disabled."
            $gpuMonitorEnabled = $false
        }
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

    $pipelineStdout = Join-Path $logDir "pipeline_$timestamp.log"
    $pipelineStderr = Join-Path $logDir "pipeline_$timestamp.err.log"
    Write-Host "[INFO] Pipeline stdout: $pipelineStdout"
    Write-Host "[INFO] Pipeline stderr: $pipelineStderr"

    $proc = Start-Process -FilePath $pythonCmd -ArgumentList $pipelineArgs -NoNewWindow -PassThru -RedirectStandardOutput $pipelineStdout -RedirectStandardError $pipelineStderr
    $exitCode = Wait-ProcessWithGpuMonitor -Process $proc -EnableMonitor $gpuMonitorEnabled -MinUtil $GpuMinUtil -MinMemMB $GpuMinMemoryMB -DelaySec $GpuCheckDelaySec -WindowSec $GpuCheckWindowSec -IntervalSec $GpuCheckIntervalSec
    if ($exitCode -ne 0) {
        throw "Pipeline failed with exit code $exitCode. See $pipelineStdout and $pipelineStderr"
    }

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
} finally {
    if ($logStarted) {
        Stop-Transcript | Out-Null
    }
}
