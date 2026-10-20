$ErrorActionPreference = 'Stop'

$projectRoot = 'C:\Users\28031\.codex\.chatgpt-projects\g-p-6ab0c73e54108191b66c885edbb00564'
$pythonExe = 'D:\Desktop\MALUNet-main\MALUNet-main\.venv\Scripts\python.exe'
$generator = Join-Path $projectRoot 'isic2017_malunet_work\generate_oof_classification_masks.py'
$logRoot = Join-Path $projectRoot 'isic2017_malunet_work\oof_mask_plan\run_logs'
$foldStatusPath = Join-Path $logRoot 'status.json'
$postStatusPath = Join-Path $logRoot 'mask_generation_status.json'

function Save-Status([string]$stage, [string]$detail) {
    [ordered]@{
        stage = $stage
        detail = $detail
        updated_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    } | ConvertTo-Json | Set-Content -LiteralPath $postStatusPath -Encoding utf8
}

try {
    Save-Status 'waiting_for_folds' 'Waiting for all five fold checkpoints.'
    while ($true) {
        if (Test-Path -LiteralPath $foldStatusPath) {
            $foldStatus = Get-Content -LiteralPath $foldStatusPath -Raw | ConvertFrom-Json
            if ($foldStatus.stage -eq 'failed') {
                throw "Five-fold training failed at fold $($foldStatus.fold): $($foldStatus.detail)"
            }
            if ($foldStatus.stage -eq 'complete') {
                break
            }
        }
        Start-Sleep -Seconds 30
    }

    Save-Status 'generating_masks' 'Generating OOF masks for all 21,768 classification-training crops.'
    $stdoutPath = Join-Path $logRoot 'generate_masks.stdout.log'
    $stderrPath = Join-Path $logRoot 'generate_masks.stderr.log'
    $process = Start-Process -FilePath $pythonExe -ArgumentList @($generator, '--epochs', '105', '--execute') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    if ($process.ExitCode -ne 0) {
        throw "OOF mask generation exited with code $($process.ExitCode). See $stderrPath and $stdoutPath."
    }
    Save-Status 'complete' 'Five fold checkpoints and 21,768 OOF train-crop masks are complete. Classification comparison has not been run.'
} catch {
    Save-Status 'failed' $_.Exception.Message
    throw
}

