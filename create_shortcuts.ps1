# Creates AirTyper shortcuts (app icon, no console window) in this folder, on the Desktop and in the Start menu.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$script = Join-Path $root "airtyper.py"
$icon = Join-Path $root "assets\airtyper.ico"

$targets = @(
    (Join-Path $root "AirTyper.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "AirTyper.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Programs")) "AirTyper.lnk")
)

$shell = New-Object -ComObject WScript.Shell
foreach ($path in $targets) {
    $lnk = $shell.CreateShortcut($path)
    $lnk.TargetPath = $pythonw
    $lnk.Arguments = "`"$script`""
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = "AirTyper - type in the air and launch apps with hand gestures"
    $lnk.Save()
    Write-Output "Shortcut: $path"
}
