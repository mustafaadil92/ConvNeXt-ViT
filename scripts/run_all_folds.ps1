param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",

    [ValidateSet("convnext-vit", "vit-s", "convnext")]
    [string]$Model = "convnext-vit",

    [int]$Seed = 42,

    [int]$StartFold = 1,

    [int]$EndFold = 5,

    [switch]$Deterministic,

    [switch]$SkipTrain,

    [switch]$SkipEval,

    [switch]$Resume,

    [switch]$Binary,

    [switch]$ContinueOnError
)

function Invoke-StatusBeep {
    param(
        [ValidateSet("Error", "FoldComplete", "TotalComplete")]
        [string]$Type
    )

    try {
        switch ($Type) {
            "Error" {
                [console]::Beep(400, 350)
                [console]::Beep(300, 350)
            }
            "FoldComplete" {
                [console]::Beep(900, 150)
                [console]::Beep(1100, 200)
            }
            "TotalComplete" {
                [console]::Beep(700, 180)
                [console]::Beep(900, 180)
                [console]::Beep(1200, 300)
            }
        }
    } catch {
        # Ignore audio failures (e.g., unsupported host/device)
    }
}

# Assumes:
# - Current directory is project root
# - The correct virtual environment is activated

if ($StartFold -lt 1 -or $EndFold -gt 5 -or $StartFold -gt $EndFold) {
    Invoke-StatusBeep -Type Error
    throw "Invalid fold range. Use StartFold/EndFold between 1 and 5."
}

for ($fold = $StartFold; $fold -le $EndFold; $fold++) {
    $configRoot = if ($Binary) { "configs/binary" } else { "configs" }

    if ($Model -eq "vit-s") {
        $config = "$configRoot/vit-s/breakhis_vits_fold$fold.yaml"
    } elseif ($Model -eq "convnext") {
        $config = "$configRoot/convnext/breakhis_convnext_fold$fold.yaml"
    } else {
        $config = "$configRoot/convnext-vit/breakhis_fold$fold.yaml"
    }

    if (-not (Test-Path $config)) {
        $msg = "Config not found: $config"
        Invoke-StatusBeep -Type Error
        if ($ContinueOnError) {
            Write-Warning $msg
            continue
        }
        throw $msg
    }

    $cfgMetaJson = & python -c "import json,sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')) or {}; o=c.get('output', {}) or {}; print(json.dumps({'experiment_name': c.get('experiment_name', 'experiment'), 'checkpoint_dir': o.get('checkpoint_dir', 'outputs/checkpoints')}))" $config
    if ($LASTEXITCODE -ne 0 -or -not $cfgMetaJson) {
        $msg = "Failed to read checkpoint_dir/experiment_name from config: $config"
        Invoke-StatusBeep -Type Error
        if ($ContinueOnError) {
            Write-Warning $msg
            continue
        }
        throw $msg
    }
    $cfgMeta = $cfgMetaJson | ConvertFrom-Json
    $expName = [string]$cfgMeta.experiment_name
    $checkpointDir = [string]$cfgMeta.checkpoint_dir
    $checkpoint = Join-Path $checkpointDir "${expName}_last.pt"

    Write-Host ""
    Write-Host "========== Fold $fold =========="

    if (-not $SkipTrain) {
        $trainArgs = @(
            "-m", "src.train_cli",
            "--config", $config,
            "--device", $Device,
            "--seed", "$Seed",
            "--smoke-train"
        )
        if ($Deterministic) {
            $trainArgs += "--deterministic"
        }
        if ($Resume) {
            if (Test-Path $checkpoint) {
                $trainArgs += @("--resume", $checkpoint)
            } else {
                Write-Host "Resume requested but checkpoint not found for fold ${fold}: $checkpoint"
                Write-Host "Starting fresh training for this fold."
            }
        }

        Write-Host "Running: python $($trainArgs -join ' ')"
        & python @trainArgs
        if ($LASTEXITCODE -ne 0) {
            $msg = "Training failed for fold $fold (exit code $LASTEXITCODE)."
            Invoke-StatusBeep -Type Error
            if ($ContinueOnError) {
                Write-Warning $msg
                continue
            }
            exit $LASTEXITCODE
        }
    }

    if (-not $SkipEval) {
        if (-not (Test-Path $checkpoint)) {
            $msg = "Checkpoint not found for fold ${fold}: $checkpoint"
            Invoke-StatusBeep -Type Error
            if ($ContinueOnError) {
                Write-Warning $msg
                continue
            }
            throw $msg
        }

        $evalArgs = @(
            "-m", "src.evaluate_cli",
            "--config", $config,
            "--device", $Device,
            "--seed", "$Seed",
            "--checkpoint", $checkpoint
        )
        if ($Deterministic) {
            $evalArgs += "--deterministic"
        }

        Write-Host "Running: python $($evalArgs -join ' ')"
        & python @evalArgs
        if ($LASTEXITCODE -ne 0) {
            $msg = "Evaluation failed for fold $fold (exit code $LASTEXITCODE)."
            Invoke-StatusBeep -Type Error
            if ($ContinueOnError) {
                Write-Warning $msg
                continue
            }
            exit $LASTEXITCODE
        }
    }

    Invoke-StatusBeep -Type FoldComplete
    Write-Host "Fold $fold completed."
}

Write-Host ""
Write-Host "All requested folds completed."
Invoke-StatusBeep -Type TotalComplete
