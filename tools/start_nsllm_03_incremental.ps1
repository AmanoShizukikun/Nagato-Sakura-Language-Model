param(
    [string]$BaseCheckpoint = "NS-LLM-0.2/checkpoint-epoch-46",
    [string]$OutputDir = "NS-LLM-0.3",
    [string]$TrainingData = "data/train",
    [string]$EvalData = "data/eval",
    [double]$EvalSplitRatio = 0.01,
    [int]$NumEpochs = 120,
    [int]$SchedulerTargetEpochs = 120,
    [double]$LearningRate = 8e-5,
    [double]$ResumeLrScale = 0,
    [int]$BatchSize = 3,
    [int]$GradientAccumulationSteps = 8,
    [switch]$DisablePretokenize,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv/Scripts/python.exe"
$trainPy = Join-Path $repoRoot "train.py"
$baseCheckpointPath = Join-Path $repoRoot $BaseCheckpoint
$outputDirPath = Join-Path $repoRoot $OutputDir
$baseTokenizerPath = Join-Path $baseCheckpointPath "tokenizer.json"
$outputTokenizerPath = Join-Path $outputDirPath "tokenizer.json"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (-not (Test-Path $trainPy)) {
    throw "train.py not found: $trainPy"
}

if (-not (Test-Path $baseCheckpointPath)) {
    throw "Base checkpoint not found: $baseCheckpointPath"
}

if (-not (Test-Path $outputTokenizerPath)) {
    if (-not (Test-Path $baseTokenizerPath)) {
        throw "Tokenizer not found in base checkpoint: $baseTokenizerPath"
    }
    New-Item -ItemType Directory -Path $outputDirPath -Force | Out-Null
    Copy-Item -Path $baseTokenizerPath -Destination $outputTokenizerPath -Force
    Write-Host "Copied tokenizer to output dir: $outputTokenizerPath" -ForegroundColor Cyan
}

$argList = @(
    $trainPy,
    "--output_dir", $OutputDir,
    "--resume_checkpoint", $BaseCheckpoint,
    "--training_data_file", $TrainingData,
    "--eval_data_file", $EvalData,
    "--eval_split_ratio", "$EvalSplitRatio",
    "--num_epochs", "$NumEpochs",
    "--scheduler_target_epochs", "$SchedulerTargetEpochs",
    "--learning_rate", "$LearningRate",
    "--resume_lr_scale", "$ResumeLrScale",
    "--batch_size", "$BatchSize",
    "--gradient_accumulation_steps", "$GradientAccumulationSteps",
    "--precision", "auto",
    "--lr_scheduler_type", "cosine",
    "--eval_interval_epochs", "1",
    "--save_interval_epochs", "2",
    "--save_best_k", "2",
    "--save_latest_k", "2"
)

if ($DisablePretokenize) {
    $argList += "--disable_pretokenize"
}

$displayCommand = @($pythonExe) + $argList
Write-Host "\nPrepared command:" -ForegroundColor Cyan
Write-Host ($displayCommand -join " ")

if ($DryRun) {
    Write-Host "\nDryRun enabled. Command was not executed." -ForegroundColor Yellow
    exit 0
}

& $pythonExe @argList
if ($LASTEXITCODE -ne 0) {
    throw "Training process failed with exit code: $LASTEXITCODE"
}

Write-Host "\nTraining finished successfully." -ForegroundColor Green
