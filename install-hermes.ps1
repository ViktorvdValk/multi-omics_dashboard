<#
.SYNOPSIS
    One-click setup of Hermes Agent (+ Hermes Desktop) on a myDRE Windows VM,
    connected to the workspace's own Ollama GPU server.

.DESCRIPTION
    What this script does, per Windows user (no admin rights needed):
      1. Detects the myDRE web proxy and sets it for this user
         (HTTPS_PROXY / HTTP_PROXY / NO_PROXY, and git).
      2. Checks that GitHub and the Ollama server are reachable.
      3. Installs Hermes Agent + Hermes Desktop from GitHub
         (skipped when Hermes is already installed).
      4. Registers the workspace Ollama server as a named provider and makes it
         the default - no cloud providers. All models on the server appear in
         the Desktop model picker.
      5. Turns off the web toolset, so the agent cannot search or fetch the internet.
      6. Creates the repository's Python environment (.venv) from pyproject.toml
         with uv, so the agent can run and test the loaders.
      7. Puts a 'Hermes - <repo>' shortcut on the desktop that opens Hermes in
         this repository, so AGENTS.md is loaded.

    Keep this script in the repository root. Each user runs it from their own
    clone of the repository. Safe to run again. A log is written to %TEMP%\hermes-install-<timestamp>.log.

.EXAMPLE
    Double-click install-hermes.cmd

.EXAMPLE
    .\install-hermes.ps1 -Model "qwen3.8:27b"
#>
[CmdletBinding()]
param(
    # ---- Workspace settings: edit these defaults once for your workspace ----
    [string]  $OllamaHost       = "CHANGE-ME",     # internal IP of the GPU VM running Ollama
    [int]     $OllamaPort       = 11434,
    [string]  $Model            = "gpt-oss:20b",   # default model, exactly as shown by 'ollama list'
    [string]  $ProviderName     = "mydre-ollama",  # name shown in the Desktop model picker
    [int]     $ContextLength    = 65536,           # Hermes needs at least 64K
    [string[]]$DisabledToolsets = @("web"),        # names: run 'hermes tools list'
    # -------------------------------------------------------------------------
    [switch]  $NoDesktop,                          # command-line only, no desktop app
    [switch]  $Reinstall                           # reinstall even if Hermes is present
)

$ErrorActionPreference = "Stop"
$InstallerUrl = "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1"

function Write-Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    WARN  $msg" -ForegroundColor Yellow }
function Stop-WithError($msg) {
    Write-Host ""
    Write-Host "FAILED: $msg" -ForegroundColor Red
    Write-Host "Send this log file to whoever maintains this setup: $script:LogFile" -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}

$script:LogFile = Join-Path $env:TEMP ("hermes-install-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -Path $script:LogFile | Out-Null

if ($OllamaHost -eq "CHANGE-ME") {
    Stop-WithError "Set `$OllamaHost at the top of install-hermes.ps1 to the internal IP address of the GPU VM."
}

# Older Windows Server builds do not always enable TLS 1.2 by default; GitHub requires it.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$OllamaBase = "http://${OllamaHost}:$OllamaPort"

# ---------------------------------------------------------------------------
# 1. Proxy
# ---------------------------------------------------------------------------
Write-Step "Detecting the workspace proxy"

function Set-UserEnv($name, $value) {
    [Environment]::SetEnvironmentVariable($name, $value, "User")   # permanent, for new programs
    Set-Item -Path "Env:$name" -Value $value                         # this session
}

$probe    = [Uri]"https://github.com"
$proxyUri = [System.Net.WebRequest]::GetSystemWebProxy().GetProxy($probe)
$ProxyUrl = $null
if ($proxyUri -and $proxyUri.Authority -ne $probe.Authority) {
    $ProxyUrl = "http://" + $proxyUri.Authority
    Write-Ok "Proxy found: $ProxyUrl"
} else {
    Write-Warn "No proxy detected; assuming direct internet access."
}

# NO_PROXY: keep existing entries, add localhost and the Ollama server,
# so requests to Ollama never go out through the proxy.
$noProxy  = @()
$existing = [Environment]::GetEnvironmentVariable("NO_PROXY", "User")
if ($existing) { $noProxy += $existing.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ } }
$noProxy += @("localhost", "127.0.0.1", $OllamaHost)
$noProxy  = $noProxy | Select-Object -Unique
Set-UserEnv "NO_PROXY" ($noProxy -join ",")
Write-Ok "NO_PROXY = $env:NO_PROXY"

if ($ProxyUrl) {
    Set-UserEnv "HTTPS_PROXY" $ProxyUrl
    Set-UserEnv "HTTP_PROXY"  $ProxyUrl
    Write-Ok "HTTPS_PROXY and HTTP_PROXY set for this user"
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git config --global http.proxy $ProxyUrl
        Write-Ok "git configured to use the proxy"
    }
}

# ---------------------------------------------------------------------------
# 2. Connectivity checks
# ---------------------------------------------------------------------------
Write-Step "Checking internet access to GitHub"
try {
    Invoke-WebRequest -Uri "https://github.com" -Method Head -UseBasicParsing -TimeoutSec 20 | Out-Null
    Write-Ok "github.com is reachable"
} catch {
    Stop-WithError ("Cannot reach github.com ({0}). Ask myDRE support to allowlist github.com and *.githubusercontent.com for this workspace." -f $_.Exception.Message)
}

Write-Step "Checking the Ollama server at $OllamaBase"
function Get-Direct($url, [int]$timeoutMs = 5000) {
    $req = [System.Net.WebRequest]::Create($url)
    $req.Proxy   = $null          # internal address: never via the proxy
    $req.Timeout = $timeoutMs
    $resp = $req.GetResponse()
    try {
        $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
        return $reader.ReadToEnd()
    } finally {
        $resp.Close()
    }
}
try {
    $tags  = Get-Direct "$OllamaBase/api/tags" | ConvertFrom-Json
    $names = @($tags.models | ForEach-Object { $_.name })
    Write-Ok ("Ollama is reachable. Models on the server: " + ($names -join ", "))
    if ($names -notcontains $Model -and $names -notcontains "${Model}:latest") {
        Write-Warn "Model '$Model' is not on the server yet. On the GPU VM run: ollama pull $Model"
    }
} catch {
    Write-Warn ("Ollama is not reachable ({0}). Is the GPU VM switched on? Continuing anyway." -f $_.Exception.Message)
}

# ---------------------------------------------------------------------------
# 3. Install Hermes (+ Desktop)
# ---------------------------------------------------------------------------
function Find-Hermes {
    # Reload PATH so a fresh install is found without opening a new window
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $cmd = Get-Command hermes -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path $env:LOCALAPPDATA "hermes\bin\hermes.cmd"
    if (Test-Path $fallback) { return $fallback }
    return $null
}

$Hermes        = Find-Hermes
$DesktopFailed = $false

if ($Hermes -and -not $Reinstall) {
    Write-Step "Hermes is already installed ($Hermes) - skipping installation"
} else {
    $what = "Hermes Agent"
    if (-not $NoDesktop) { $what += " + Hermes Desktop" }
    Write-Step "Installing $what from GitHub - this can take 10 to 20 minutes, please wait"

    try {
        $installerText = Invoke-RestMethod -Uri $InstallerUrl -UseBasicParsing -TimeoutSec 60
    } catch {
        Stop-WithError ("Could not download the Hermes installer from GitHub: {0}" -f $_.Exception.Message)
    }
    $installer = Join-Path $env:TEMP "hermes-install.ps1"
    Set-Content -Path $installer -Value $installerText -Encoding UTF8

    # Run the official installer in its own PowerShell process, so it cannot close this window
    $psExe    = (Get-Process -Id $PID).Path
    $baseArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $installer, "-SkipSetup")
    $code     = 1

    if (-not $NoDesktop) {
        & $psExe ($baseArgs + "-IncludeDesktop")
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            $DesktopFailed = $true
            Write-Warn "Installing with the desktop app failed (exit code $code). Retrying without the desktop app..."
        }
    }
    if ($code -ne 0) {
        & $psExe $baseArgs
        $code = $LASTEXITCODE
    }
    if ($code -ne 0) {
        Stop-WithError "The Hermes installer failed (exit code $code). The lines above show which download was blocked."
    }

    $Hermes = Find-Hermes
    if (-not $Hermes) { Stop-WithError "The installer finished, but the 'hermes' command was not found." }
    Write-Ok "Installed: $Hermes"
}

# ---------------------------------------------------------------------------
# 4. Configure Hermes: local Ollama only
# ---------------------------------------------------------------------------
Write-Step "Connecting Hermes to Ollama (no cloud providers)"
function Set-HermesConfig($key, $value, [switch]$Soft) {
    & $Hermes config set $key $value | Out-Null
    if ($LASTEXITCODE -ne 0) {
        if ($Soft) { return $false }
        Stop-WithError "'hermes config set $key' failed."
    }
    Write-Ok "$key = $value"
    return $true
}

# Register the workspace Ollama server as a named provider, so it shows up by
# name in the Desktop model picker. The picker lists the models on the server.
$named = (Set-HermesConfig "providers.$ProviderName.base_url"       "$OllamaBase/v1" -Soft) -and
         (Set-HermesConfig "providers.$ProviderName.context_length" "$ContextLength" -Soft)

if ($named) {
    Set-HermesConfig "model.provider" $ProviderName | Out-Null
} else {
    Write-Warn "Could not register a named provider; using the generic 'custom' provider instead."
    Set-HermesConfig "model.provider" "custom" | Out-Null
}
# Also on the model itself, so the endpoint is set however Hermes resolves the provider
Set-HermesConfig "model.base_url"       "$OllamaBase/v1"  | Out-Null
Set-HermesConfig "model.default"        $Model            | Out-Null
Set-HermesConfig "model.context_length" "$ContextLength"  | Out-Null

# ---------------------------------------------------------------------------
# 5. Turn off toolsets that reach the internet
# ---------------------------------------------------------------------------
if ($DisabledToolsets.Count -gt 0) {
    Write-Step ("Turning off toolsets: " + ($DisabledToolsets -join ", "))
    $list = "[" + ($DisabledToolsets -join ", ") + "]"
    & $Hermes config set agent.disabled_toolsets $list | Out-Null
    $setCode = $LASTEXITCODE
    $check   = ((& $Hermes config get agent.disabled_toolsets) -join " ").Trim()

    # If Hermes stored the value as one literal string instead of a list,
    # remove it again rather than leave a broken setting behind.
    if ($setCode -ne 0 -or $check -eq $list -or -not $check) {
        & $Hermes config unset agent.disabled_toolsets | Out-Null
        Write-Warn "Could not set disabled toolsets automatically. Turn off 'web' by hand: run 'hermes tools'."
    } else {
        Write-Ok "agent.disabled_toolsets = $check"
    }
}

# ---------------------------------------------------------------------------
# 6. Python environment for this repository (packages the loaders need)
# ---------------------------------------------------------------------------
$RepoRoot = $PSScriptRoot
Write-Step "Setting up the project's Python environment in $RepoRoot\.venv"

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    # Hermes ships its own uv
    $hermesRoot = Join-Path $env:LOCALAPPDATA "hermes"
    if (Test-Path $hermesRoot) {
        $found = Get-ChildItem -Path $hermesRoot -Recurse -Depth 4 -Filter uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    # Fallback: download uv from GitHub releases
    $dir = Join-Path $env:LOCALAPPDATA "uv-standalone"
    $found = Get-ChildItem -Path $dir -Recurse -Filter uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.FullName }
    $zip = Join-Path $env:TEMP "uv-windows.zip"
    Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing -TimeoutSec 120
    Expand-Archive -Path $zip -DestinationPath $dir -Force
    $found = Get-ChildItem -Path $dir -Recurse -Filter uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

$VenvOk = $false
if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml")) -and -not (Test-Path (Join-Path $RepoRoot "requirements.txt"))) {
    Write-Warn "No pyproject.toml or requirements.txt next to this script; skipping."
} else {
    try { $Uv = Find-Uv } catch { $Uv = $null }
    if (-not $Uv) {
        Write-Warn "Could not find or download uv; the Python environment was not created."
    } else {
        Write-Ok "uv: $Uv"
        $env:UV_NATIVE_TLS = "1"     # trust the Windows certificate store (the proxy may inspect TLS)
        $env:UV_LINK_MODE  = "copy"  # hardlinks fail on network drives
        $code = 1
        Push-Location $RepoRoot
        try {
            if (Test-Path "pyproject.toml") {
                & $Uv sync              # uses .python-version and uv.lock
            } else {
                & $Uv venv .venv --python 3.12
                if ($LASTEXITCODE -eq 0) { & $Uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt }
            }
            $code = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        $py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
        if ($code -eq 0 -and (Test-Path $py)) {
            $VenvOk = $true
            Write-Ok ("Python environment ready: " + (& $py --version))
        } else {
            Write-Warn "Creating the Python environment failed (exit code $code). The lines above show which download was blocked."
        }
    }
}

# ---------------------------------------------------------------------------
# 7. Shortcut that opens Hermes in this repository (so AGENTS.md is loaded)
# ---------------------------------------------------------------------------
Write-Step "Creating a desktop shortcut"
try {
    $repoName = Split-Path $RepoRoot -Leaf
    $lnkPath  = Join-Path ([Environment]::GetFolderPath("Desktop")) "Hermes - $repoName.lnk"
    $shell    = New-Object -ComObject WScript.Shell
    $lnk      = $shell.CreateShortcut($lnkPath)
    if (-not $NoDesktop -and -not $DesktopFailed) {
        $lnk.TargetPath  = $Hermes
        $lnk.Arguments   = "desktop --cwd `"$RepoRoot`""
        $lnk.WindowStyle = 7    # keep the helper console window minimised
    } else {
        $lnk.TargetPath  = "powershell.exe"
        $lnk.Arguments   = "-NoExit -Command `"Set-Location -LiteralPath '$RepoRoot'; & '$Hermes'`""
    }
    $lnk.WorkingDirectory = $RepoRoot
    $lnk.Description      = "Hermes Agent for $repoName"
    $lnk.Save()
    Write-Ok "Shortcut: $lnkPath"
} catch {
    Write-Warn ("Could not create the shortcut: {0}" -f $_.Exception.Message)
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Step "Done"
Write-Host "    Model : $Model on $OllamaBase/v1 (provider '$ProviderName' in the model picker)"
Write-Host "    Hermes: $Hermes"
if ($VenvOk) { Write-Host "    Python: $RepoRoot\.venv" }
if ($DesktopFailed) { Write-Warn "The desktop app could not be installed; the command-line version is ready." }
Write-Host "    Start : double-click 'Hermes - $(Split-Path $RepoRoot -Leaf)' on the desktop" -ForegroundColor Green
Write-Host "    Log   : $script:LogFile"
try { Stop-Transcript | Out-Null } catch { }
exit 0
