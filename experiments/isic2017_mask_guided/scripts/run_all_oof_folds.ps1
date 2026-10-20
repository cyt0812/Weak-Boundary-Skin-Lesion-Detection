$ErrorActionPreference = 'Stop'

$projectRoot = 'C:\Users\28031\.codex\.chatgpt-projects\g-p-6ab0c73e54108191b66c885edbb00564'
$pythonExe = 'D:\Desktop\MALUNet-main\MALUNet-main\.venv\Scripts\python.exe'
$runner = Join-Path $projectRoot 'isic2017_malunet_work\run_oof_malunet_fold.py'
$logRoot = Join-Path $projectRoot 'isic2017_malunet_work\oof_mask_plan\run_logs'
$statusPath = Join-Path $logRoot 'status.json'
$resultsRoot = 'D:\Desktop\MALUNet-ISIC2017\results'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Save-Status([string]$stage, [int]$fold, [string]$detail) {
    $status = [ordered]@{
        stage = $stage
        fold = $fold
        detail = $detail
        updated_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

try {
    for ($fold = 0; $fold -lt 5; $fold++) {
        $resultDir = Join-Path $resultsRoot ("oof5_fold{0}_epochs105" -f $fold)
        $checkpointDir = Join-Path $resultDir 'checkpoints'
        $completed = @(Get-ChildItem -LiteralPath $checkpointDir -Filter 'best-epoch*-loss*.pth' -File -ErrorAction SilentlyContinue)
        if ($completed.Count -eq 1) {
            Save-Status 'fold_complete' $fold 'Completed checkpoint already exists; preserving it.'
            continue
        }
        if ($completed.Count -gt 1) {
            throw "Fold $fold has multiple completed checkpoints."
        }

        $arguments = @($runner, '--fold', [string]$fold, '--epochs', '105', '--execute')
        if (Test-Path -LiteralPath $resultDir) {
            if (-not (Test-Path -LiteralPath (Join-Path $checkpointDir 'latest.pth'))) {
                throw "Fold $fold result directory exists without a resumable latest.pth."
            }
            $arguments += '--resume'
        }

        Save-Status 'training' $fold 'MALUNet training in progress.'
        $stdoutPath = Join-Path $logRoot ("fold{0}.stdout.log" -f $fold)
        $stderrPath = Join-Path $logRoot ("fold{0}.stderr.log" -f $fold)
        $process = Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        if ($process.ExitCode -ne 0) {
            throw "Fold $fold exited with code $($process.ExitCode). See $stderrPath and $stdoutPath."
        }
        $completed = @(Get-ChildItem -LiteralPath $checkpointDir -Filter 'best-epoch*-loss*.pth' -File -ErrorAction SilentlyContinue)
        if ($completed.Count -ne 1) {
            throw "Fold $fold exited successfully but did not leave exactly one completed best checkpoint."
        }
        Save-Status 'fold_complete' $fold 'Completed checkpoint verified.'
    }
    Save-Status 'complete' 4 'All five MALUNet fold checkpoints are complete. OOF mask generation has not yet been started.'
} catch {
    $failedFold = if ($null -ne $fold) { [int]$fold } else { -1 }
    Save-Status 'failed' $failedFold $_.Exception.Message
    throw
}

