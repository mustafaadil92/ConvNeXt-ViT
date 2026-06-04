param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",

    [ValidateSet("convnext-vit", "vit-s", "convnext")]
    [string[]]$Models = @("convnext-vit", "vit-s", "convnext"),

    [int]$Seed = 42,

    [int]$StartFold = 1,

    [int]$EndFold = 5,

    [switch]$Deterministic,

    [switch]$SkipTrain,

    [switch]$SkipEval,

    [int]$KeepEpochCheckpoints = 2,

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

function Get-Lc25000Config {
    param(
        [ValidateSet("convnext-vit", "vit-s", "convnext")]
        [string]$Model,

        [int]$Fold
    )

    if ($Model -eq "vit-s") {
        return "configs/LC25000/vit-s/lc25000_vit_s_fold$Fold.yaml"
    }
    if ($Model -eq "convnext") {
        return "configs/LC25000/convnext/lc25000_convnext_fold$Fold.yaml"
    }
    return "configs/LC25000/convnext-vit/lc25000_convnext_vit_fold$Fold.yaml"
}

function Remove-OldEpochCheckpoints {
    param(
        [string]$CheckpointDir,

        [string]$ExperimentName,

        [int]$Keep = 2
    )

    if ($Keep -lt 0) {
        return
    }
    if (-not (Test-Path $CheckpointDir)) {
        return
    }

    $pattern = "${ExperimentName}_epoch*.pt"
    $epochCheckpoints = Get-ChildItem -Path $CheckpointDir -Filter $pattern -File |
        ForEach-Object {
            if ($_.BaseName -match "^$([regex]::Escape($ExperimentName))_epoch(\d+)$") {
                [PSCustomObject]@{
                    Path = $_.FullName
                    Epoch = [int]$Matches[1]
                }
            }
        } |
        Sort-Object -Property Epoch -Descending

    $toDelete = $epochCheckpoints | Select-Object -Skip $Keep
    foreach ($checkpointToDelete in $toDelete) {
        Remove-Item -LiteralPath $checkpointToDelete.Path -Force
    }

    $deletedCount = @($toDelete).Count
    if ($deletedCount -gt 0) {
        Write-Host "Deleted $deletedCount old epoch checkpoints for $ExperimentName; kept latest $Keep."
    }
}

# Assumes:
# - Current directory is project root
# - The correct virtual environment is activated

if ($StartFold -lt 1 -or $EndFold -gt 5 -or $StartFold -gt $EndFold) {
    Invoke-StatusBeep -Type Error
    throw "Invalid fold range. Use StartFold/EndFold between 1 and 5."
}

foreach ($model in $Models) {
    Write-Host ""
    Write-Host "################ LC25000 Model: $model ################"

    for ($fold = $StartFold; $fold -le $EndFold; $fold++) {
        $config = Get-Lc25000Config -Model $model -Fold $fold

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
        Write-Host "========== LC25000 $model Fold $fold =========="

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
            if (Test-Path $checkpoint) {
                $trainArgs += @("--resume", $checkpoint)
            } else {
                Write-Host "Resume checkpoint not found for fold ${fold}: $checkpoint"
                Write-Host "Starting fresh training for this fold."
            }

            Write-Host "Running: python $($trainArgs -join ' ')"
            & python @trainArgs
            if ($LASTEXITCODE -ne 0) {
                $msg = "Training failed for $model fold $fold (exit code $LASTEXITCODE)."
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
                $msg = "Checkpoint not found for ${model} fold ${fold}: $checkpoint"
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
                $msg = "Evaluation failed for $model fold $fold (exit code $LASTEXITCODE)."
                Invoke-StatusBeep -Type Error
                if ($ContinueOnError) {
                    Write-Warning $msg
                    continue
                }
                exit $LASTEXITCODE
            }
        }

        if (-not $SkipTrain) {
            Remove-OldEpochCheckpoints -CheckpointDir $checkpointDir -ExperimentName $expName -Keep $KeepEpochCheckpoints
        }

        Invoke-StatusBeep -Type FoldComplete
        Write-Host "LC25000 $model fold $fold completed."
    }
}

Write-Host ""
Write-Host "All requested LC25000 models and folds completed."
Invoke-StatusBeep -Type TotalComplete
