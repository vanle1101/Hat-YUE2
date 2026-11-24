$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$DefaultEnv = @{
    "YUE2_DATA_DIR"             = Join-Path $Root "outputs\service"
    "YUE2_MODEL"                = "m-a-p/YuE2-3B"
    "YUE2_VAE"                  = "m-a-p/YuE2-Vae"
    "YUE2_LOCAL_FILES_ONLY"     = "false"
    "YUE2_DEVICE"               = "cpu"
    "YUE2_BACKEND"              = "torch"
    "YUE2_RESIDENT_MODELS"      = "true"
    "YUE2_AR_CONCURRENCY"       = "1"
    "YUE2_NAR_BATCH_SIZE"       = "1"
    "YUE2_AR_NAR_OVERLAP"       = "false"
    "YUE2_MEMORY_BUDGET_GIB"    = "24"
    "YUE2_VAE_CORE_FRAMES"      = "512"
    "YUE2_WARMUP"               = "false"
    "YUE2_TASK_TIMEOUT_SECONDS" = "7200"
    "YUE2_SHEETSAGE_DEVICE"     = "auto"
    "YUE2_HOST"                 = "127.0.0.1"
    "YUE2_PORT"                 = "8000"
    "YUE2_UPSTREAM"             = "http://127.0.0.1:8000"
    "YUE2_WEB_PORT"             = "8016"
    "HF_HOME"                   = Join-Path $Root ".cache\huggingface"
}

$localModel = Join-Path $Root "models\YuE2-3B"
$localVae = Join-Path $Root "models\YuE2-Vae"
if ((Test-Path $localModel) -and (Test-Path $localVae)) {
    $DefaultEnv["YUE2_MODEL"] = $localModel
    $DefaultEnv["YUE2_VAE"] = $localVae
    $DefaultEnv["YUE2_LOCAL_FILES_ONLY"] = "true"
}

foreach ($entry in $DefaultEnv.GetEnumerator()) {
    if (-not [Environment]::GetEnvironmentVariable($entry.Key, "Process")) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}

$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        $parts = $trimmed -split "=", 2
        if ($parts.Length -eq 2) {
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

$ServeExe = Join-Path $Root ".venv\Scripts\yue2-serve.exe"
$WebExe = Join-Path $Root ".venv\Scripts\yue2-web.exe"
$LogDir = Join-Path $Root "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Đang khởi chạy yue2-serve tại http://$($env:YUE2_HOST):$($env:YUE2_PORT) ..."
$serveOut = Join-Path $LogDir "yue2-serve.stdout.log"
$serveErr = Join-Path $LogDir "yue2-serve.stderr.log"
$serveProc = Start-Process -FilePath $ServeExe -WorkingDirectory $Root -NoNewWindow -PassThru `
    -RedirectStandardOutput $serveOut -RedirectStandardError $serveErr

Write-Host "Đang khởi chạy yue2-web tại http://127.0.0.1:$($env:YUE2_WEB_PORT) ..."
$webOut = Join-Path $LogDir "yue2-web.stdout.log"
$webErr = Join-Path $LogDir "yue2-web.stderr.log"
$webProc = Start-Process -FilePath $WebExe -WorkingDirectory $Root -NoNewWindow -PassThru `
    -RedirectStandardOutput $webOut -RedirectStandardError $webErr

Write-Host ""
Write-Host "Giao diện Studio : http://127.0.0.1:$($env:YUE2_WEB_PORT)"
Write-Host "Tài liệu API     : http://$($env:YUE2_HOST):$($env:YUE2_PORT)/docs"
Write-Host "Thư mục log      : $LogDir"
Write-Host "Nhấn Ctrl+C để dừng cả hai dịch vụ."

try {
    while (-not $serveProc.HasExited -and -not $webProc.HasExited) {
        Start-Sleep -Seconds 2
    }
} finally {
    if ($webProc -and -not $webProc.HasExited) { Stop-Process -Id $webProc.Id -Force -ErrorAction SilentlyContinue }
    if ($serveProc -and -not $serveProc.HasExited) { Stop-Process -Id $serveProc.Id -Force -ErrorAction SilentlyContinue }
}
