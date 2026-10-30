@echo off
setlocal
rem Gemma 4 26B-A4B with vision, on a GPU with 16 GB of memory (UNTESTED on Windows; see start_gemma26b.sh).
set "KIT=%~dp0"
call "%KIT%_common.bat" || exit /b 1
if not defined MODEL (
  if exist "%MODELS%\gemma-4-26B_q4_0-it.gguf" set "MODEL=%MODELS%\gemma-4-26B_q4_0-it.gguf"
  if exist "%MODELS%\gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.i1-Q4_0.gguf" set "MODEL=%MODELS%\gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.i1-Q4_0.gguf"
)
if not defined MODEL (
  echo Missing model file for Gemma 4 26B in: %MODELS%
  echo Download it from: https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf
  echo ^(See README.md for the exact file names.^)
  pause
  exit /b 1
)
if not defined MMPROJ (
  if exist "%MODELS%\gemma-4-26B-it-mmproj.gguf" set "MMPROJ=%MODELS%\gemma-4-26B-it-mmproj.gguf"
  if exist "%MODELS%\gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.mmproj-Q8_0.gguf" set "MMPROJ=%MODELS%\gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.mmproj-Q8_0.gguf"
)
set "VISION="
if defined MMPROJ set VISION=--mmproj "%MMPROJ%"
if not defined MMPROJ (
  echo No vision file ^(mmproj^) found in %MODELS%: starting without image support.
  echo Lena still saves images you attach; set default_vision = false in her settings.txt until you add it.
  echo.
)
echo Starting: "%LLAMA%" -m "%MODEL%" %VISION% -c 20000 --parallel 1 --kv-unified -b 2048 -ub 512 -fa on -ngl 99 --jinja --cache-ram 0 --ctx-checkpoints 0 --host 127.0.0.1 --port 8080
echo.
"%LLAMA%" -m "%MODEL%" %VISION% -c 20000 --parallel 1 --kv-unified -b 2048 -ub 512 -fa on -ngl 99 --jinja --cache-ram 0 --ctx-checkpoints 0 --host 127.0.0.1 --port 8080
echo.
echo llama-server stopped. If that was unexpected, the messages above say why.
pause
