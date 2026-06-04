param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [int]$Seed = 42,

    [switch]$Deterministic
)

# Assumes:
# - You are running this from the project root, OR
# - You call it with the project root as current directory in VS Code terminal
# - The correct venv is already activated

$cmd = @("python", "-m", "src.train_cli", "--device", $Device, "--seed", "$Seed")

if ($Deterministic) {
    $cmd += "--deterministic"
}

Write-Host "Running:" ($cmd -join " ")

& $cmd[0] $cmd[1] $cmd[2] $cmd[3] $cmd[4] $cmd[5] $cmd[6]
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}