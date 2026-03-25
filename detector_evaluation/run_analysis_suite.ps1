param(
    [string]$Train = "data/splits/train.csv",
    [string]$Val = "data/splits/val.csv",
    [string]$Test = "data/splits/test.csv",
    [string]$ModelDir = "results/roberta_model",
    [string]$Device = "cpu",
    [int]$DetectGptPerturb = 2,
    [string]$OutputRoot = "results/analysis_suite",
    [switch]$SkipExtraInsights,
    [string]$LatencyDevice = "cpu"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

Push-Location $PSScriptRoot
try {
    $argsList = @(
        "-m", "evaluation.analysis_suite",
        "--train", $Train,
        "--val", $Val,
        "--test", $Test,
        "--model-dir", $ModelDir,
        "--device", $Device,
        "--detectgpt-perturb", "$DetectGptPerturb",
        "--output-root", $OutputRoot,
        "--latency-device", $LatencyDevice
    )

    if ($SkipExtraInsights) {
        $argsList += "--skip-extra-insights"
    }

    & $pythonExe @argsList
}
finally {
    Pop-Location
}
