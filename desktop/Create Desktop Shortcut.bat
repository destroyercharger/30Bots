@echo off
echo Creating desktop shortcut...

set SCRIPT="%TEMP%\CreateShortcut.vbs"
set SHORTCUT="%USERPROFILE%\Desktop\30Bots Trading.lnk"
set TARGET="%~dp030Bots Trading.bat"
set ICON="%SystemRoot%\System32\shell32.dll,24"

echo Set oWS = WScript.CreateObject("WScript.Shell") > %SCRIPT%
echo sLinkFile = %SHORTCUT% >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oLink.TargetPath = %TARGET% >> %SCRIPT%
echo oLink.WorkingDirectory = "%~dp0" >> %SCRIPT%
echo oLink.Description = "30Bots AI Trading Terminal" >> %SCRIPT%
echo oLink.IconLocation = %ICON% >> %SCRIPT%
echo oLink.Save >> %SCRIPT%

cscript /nologo %SCRIPT%
del %SCRIPT%

echo.
echo Desktop shortcut created!
echo You can now launch "30Bots Trading" from your desktop.
pause
