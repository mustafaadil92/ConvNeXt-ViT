param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [int]$Seed = 42,

    [string]$Checkpoint = "",

    [switch]$Deterministic
)

# Assumes:
# - Current directory is project root
# - Correct venv is already activated

$cmd = @("python", "-m", "src.evaluate_cli", "--device", $Device, "--seed", "$Seed")

if ($Checkpoint -ne "") {
    $cmd += @("--checkpoint", $Checkpoint)
}

if ($Deterministic) {
    $cmd += "--deterministic"
}

Write-Host "Running:" ($cmd -join " ")
& $cmd[0] $cmd[1] $cmd[2] $cmd[3] $cmd[4] $cmd[5] $cmd[6] $cmd[7] $cmd[8]

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}