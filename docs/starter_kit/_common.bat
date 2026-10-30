@echo off
rem Shared by the start_*.bat scripts (Windows, UNTESTED). Finds llama-server.exe.
rem Set LLAMA_SERVER or MODELS before running to override.
if not defined MODELS set "MODELS=%KIT%models"
set "LLAMA="
if defined LLAMA_SERVER set "LLAMA=%LLAMA_SERVER%"
if not defined LLAMA if exist "%KIT%llama.cpp\llama-server.exe" set "LLAMA=%KIT%llama.cpp\llama-server.exe"
if not defined LLAMA for /r "%KIT%llama.cpp" %%F in (llama-server.exe) do if exist "%%F" set "LLAMA=%%F"
if not defined LLAMA for %%F in (llama-server.exe) do if not "%%~$PATH:F"=="" set "LLAMA=%%~$PATH:F"
if not defined LLAMA (
  echo Can't find llama-server.exe.
  echo Unzip a llama.cpp release into the llama.cpp folder next to this script ^(see README.md^).
  pause
  exit /b 1
)
exit /b 0
