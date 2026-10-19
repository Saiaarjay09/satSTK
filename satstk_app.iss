; Inno Setup recipe: turns the PyInstaller output (dist\satSTK) into a Windows installer.
; Build:  iscc /DAppVersion=2.1.0 satstk_app.iss      -> installer\satSTK-2.1.0-windows-x64-setup.exe
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
[Setup]
AppName=satSTK
AppVersion={#AppVersion}
AppPublisher=satSTK contributors
DefaultDirName={autopf}\satSTK
DefaultGroupName=satSTK
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=installer
OutputBaseFilename=satSTK-{#AppVersion}-windows-x64-setup
Compression=lzma2/fast
SolidCompression=yes
UninstallDisplayName=satSTK
DisableProgramGroupPage=yes
[Files]
Source: "dist\satSTK\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
[Icons]
Name: "{autoprograms}\satSTK"; Filename: "{app}\satSTK.exe"
Name: "{autodesktop}\satSTK"; Filename: "{app}\satSTK.exe"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
[Run]
Filename: "{app}\satSTK.exe"; Description: "Start satSTK"; Flags: nowait postinstall skipifsilent
