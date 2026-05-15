<#
.SYNOPSIS
  Full-stack supervisor for the debris monitoring system on Windows.

.DESCRIPTION
  Manages the four pieces of the system in one place:

    - PostgreSQL  (Windows service "postgresql-x64-16", checked & started)
    - FastAPI     (uvicorn api.main:app  on  port 8502)
    - Streamlit   (streamlit_app/app.py  on  port 8501)
    - Cloudflared (named tunnel, token from .env.tunnel.local)

  Subcommands:
    start    : ensure DB is running, then launch API + UI + tunnel in the
               background, wait for them to come up, and verify the public
               URLs.
    stop     : kill API + UI + tunnel (postgres is left untouched, since it
               is a Windows service that other things may depend on).
    restart  : stop + start.
    status   : show DB service state, listening ports, process roster and
               public URL probe.
    logs     : tail the last 30 lines of every component's log.

  All processes are launched with the venv interpreter:
      venv\Scripts\python.exe
  Logs go under repo\logs\ (api.log, streamlit.log, tunnel_named.log, ...).

.EXAMPLE
  .\scripts\services.ps1 start
  .\scripts\services.ps1 status
  .\scripts\services.ps1 restart
  .\scripts\services.ps1 stop
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "restart", "status", "logs")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$VenvPy   = Join-Path $RepoRoot "venv\Scripts\python.exe"
$CfdExe   = Join-Path $RepoRoot "tools\cloudflared.exe"
$LogDir   = Join-Path $RepoRoot "logs"
$EnvTun   = Join-Path $RepoRoot ".env.tunnel.local"
$DbSvc    = "postgresql-x64-16"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok  ($msg) { Write-Host "  OK   $msg" -ForegroundColor Green }
function Write-Bad ($msg) { Write-Host "  FAIL $msg" -ForegroundColor Red }

function Get-RepoPython {
    Get-WmiObject Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $cmd = $_.CommandLine
            if (-not $cmd) { return $false }
            ($cmd -match [regex]::Escape($RepoRoot)) -or
            ($cmd -match "uvicorn .*api\.main") -or
            ($cmd -match "streamlit run .*streamlit_app")
        }
}

function Get-RepoCloudflared {
    Get-WmiObject Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match [regex]::Escape($CfdExe)) }
}

function Test-Listen($port) {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return ($null -ne $c)
}

function Wait-Port($port, $seconds = 25) {
    for ($i = 0; $i -lt $seconds; $i++) {
        if (Test-Listen $port) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Probe($url, $timeout = 6) {
    try {
        $code = & curl.exe -sS --noproxy "*" --max-time $timeout -o NUL -w "%{http_code}" $url 2>$null
        if ($LASTEXITCODE -ne 0) { return "ERR" }
        return $code
    } catch { return "ERR" }
}

function Get-CfToken {
    if (-not (Test-Path $EnvTun)) { return $null }
    $line = (Get-Content $EnvTun) | Where-Object { $_ -match "^CF_TUNNEL_TOKEN=" } | Select-Object -First 1
    if (-not $line) { return $null }
    return ($line -split "=", 2)[1].Trim()
}

# ----------------------------------------------------------------------------
# start / stop / restart / status / logs
# ----------------------------------------------------------------------------
function Stop-Db { } # we never touch the windows service automatically

function Stop-AppsAndTunnel {
    Write-Step "stopping API + UI + tunnel"
    $py = Get-RepoPython
    foreach ($p in $py) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Ok "killed python pid $($p.ProcessId)" }
        catch { Write-Bad "failed to kill python pid $($p.ProcessId): $_" }
    }
    $cf = Get-RepoCloudflared
    foreach ($p in $cf) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Ok "killed cloudflared pid $($p.ProcessId)" }
        catch { Write-Bad "failed to kill cloudflared pid $($p.ProcessId): $_" }
    }
    Start-Sleep -Seconds 2
}

function Ensure-Db {
    Write-Step "checking PostgreSQL service ($DbSvc)"
    $svc = Get-Service $DbSvc -ErrorAction SilentlyContinue
    if (-not $svc) { Write-Bad "service $DbSvc not found"; return $false }
    if ($svc.Status -ne "Running") {
        Write-Host "  starting $DbSvc ..."
        try { Start-Service $DbSvc -ErrorAction Stop } catch { Write-Bad $_; return $false }
        Start-Sleep -Seconds 2
    }
    $svc = Get-Service $DbSvc
    if ($svc.Status -eq "Running") { Write-Ok "$DbSvc is running"; return $true }
    Write-Bad "$DbSvc final status: $($svc.Status)"
    return $false
}

function Start-Api {
    Write-Step "starting FastAPI on 8502"
    if (Test-Listen 8502) { Write-Ok "8502 already listening"; return }
    $p = Start-Process -FilePath $VenvPy `
        -ArgumentList @("-m","uvicorn","api.main:app","--host","0.0.0.0","--port","8502") `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "api.log") `
        -RedirectStandardError  (Join-Path $LogDir "api.err.log") `
        -PassThru
    if (Wait-Port 8502 25) { Write-Ok "FastAPI pid $($p.Id) listening on 8502" }
    else                   { Write-Bad "FastAPI did not bind 8502 (see logs/api.err.log)" }
}

function Start-Ui {
    Write-Step "starting Streamlit on 8501"
    if (Test-Listen 8501) { Write-Ok "8501 already listening"; return }
    $p = Start-Process -FilePath $VenvPy `
        -ArgumentList @("-m","streamlit","run","streamlit_app/app.py",
                        "--server.port","8501",
                        "--server.address","0.0.0.0",
                        "--server.headless","true",
                        "--browser.gatherUsageStats","false") `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "streamlit.log") `
        -RedirectStandardError  (Join-Path $LogDir "streamlit.err.log") `
        -PassThru
    if (Wait-Port 8501 30) { Write-Ok "Streamlit pid $($p.Id) listening on 8501" }
    else                   { Write-Bad "Streamlit did not bind 8501 (see logs/streamlit.err.log)" }
}

function Start-Tunnel {
    Write-Step "starting cloudflared named tunnel"
    if (-not (Test-Path $CfdExe)) {
        Write-Bad "cloudflared.exe not found at $CfdExe (run scripts/tunnels.ps1 start once to download)"
        return
    }
    $tok = Get-CfToken
    if (-not $tok) {
        Write-Bad "no CF_TUNNEL_TOKEN in $EnvTun (run scripts/setup_named_tunnel.ps1 first)"
        return
    }
    $existing = Get-RepoCloudflared
    if ($existing) { Write-Ok "cloudflared already running (pid $($existing[0].ProcessId))"; return }
    $p = Start-Process -FilePath $CfdExe `
        -ArgumentList @("tunnel","--no-autoupdate","run","--token",$tok) `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "tunnel_named.log") `
        -RedirectStandardError  (Join-Path $LogDir "tunnel_named.err.log") `
        -PassThru
    Write-Ok "cloudflared pid $($p.Id) launched"
    Write-Host "  waiting 8s for QUIC handshake ..."
    Start-Sleep -Seconds 8
}

function Show-Status {
    Write-Step "PostgreSQL"
    $svc = Get-Service $DbSvc -ErrorAction SilentlyContinue
    if ($svc) { Write-Host ("  {0}: {1}" -f $svc.Name, $svc.Status) }
    else      { Write-Bad "$DbSvc not found" }

    Write-Step "process roster"
    $rows = @()
    foreach ($p in (Get-RepoPython)) {
        $hint = "python"
        if     ($p.CommandLine -match "uvicorn")   { $hint = "uvicorn(api)"   }
        elseif ($p.CommandLine -match "streamlit") { $hint = "streamlit(ui)" }
        $rows += [pscustomobject]@{ pid=$p.ProcessId; name="python"; hint=$hint }
    }
    foreach ($p in (Get-RepoCloudflared)) {
        $rows += [pscustomobject]@{ pid=$p.ProcessId; name="cloudflared"; hint="named tunnel" }
    }
    if ($rows.Count -gt 0) { $rows | Format-Table -AutoSize | Out-String | Write-Host }
    else                   { Write-Host "  (no managed processes)" }

    Write-Step "listening ports"
    foreach ($port in 8501, 8502) {
        if (Test-Listen $port) { Write-Ok "$port listening" } else { Write-Bad "$port NOT listening" }
    }

    Write-Step "local probes"
    Write-Host ("  127.0.0.1:8502/docs -> {0}" -f (Probe "http://127.0.0.1:8502/docs"))
    Write-Host ("  127.0.0.1:8501/     -> {0}" -f (Probe "http://127.0.0.1:8501/"))

    Write-Step "public probes"
    Write-Host ("  https://debris-ui.he-ting.com/                              -> {0}" -f (Probe "https://debris-ui.he-ting.com/" 10))
    Write-Host ("  https://debris-api.he-ting.com/docs                         -> {0}" -f (Probe "https://debris-api.he-ting.com/docs" 10))
    Write-Host ("  https://debris-api.he-ting.com/docs/modules/validation      -> {0}" -f (Probe "https://debris-api.he-ting.com/docs/modules/validation" 10))
    Write-Host ("  https://debris-api.he-ting.com/docs/modules/stk_validation  -> {0}" -f (Probe "https://debris-api.he-ting.com/docs/modules/stk_validation" 10))
}

function Show-Logs {
    $files = @(
        "api.log", "api.err.log",
        "streamlit.log", "streamlit.err.log",
        "tunnel_named.log", "tunnel_named.err.log"
    )
    foreach ($f in $files) {
        $p = Join-Path $LogDir $f
        Write-Step $f
        if (Test-Path $p) { Get-Content $p -Tail 30 | ForEach-Object { "  $_" } | Write-Host }
        else              { Write-Host "  (missing)" }
    }
}

function Do-Start {
    if (-not (Test-Path $VenvPy)) { Write-Bad "venv python not found at $VenvPy"; return }
    if (-not (Ensure-Db))         { Write-Bad "DB is not healthy, aborting"; return }
    Start-Api
    Start-Ui
    Start-Tunnel
    Write-Host ""
    Show-Status
    Write-Host ""
    Write-Host "URLs:"  -ForegroundColor Yellow
    Write-Host "  UI : https://debris-ui.he-ting.com/"  -ForegroundColor Yellow
    Write-Host "  API: https://debris-api.he-ting.com/docs"  -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------------
switch ($Action) {
    "start"   { Do-Start }
    "stop"    { Stop-AppsAndTunnel }
    "restart" { Stop-AppsAndTunnel; Start-Sleep -Seconds 1; Do-Start }
    "status"  { Show-Status }
    "logs"    { Show-Logs }
}
