@echo off
setlocal
rem Gemma 4 E4B with vision, runs on almost anything (UNTESTED on Windows; see start_gemma_e4b.sh).
set "KIT=%~dp0"
call "%KIT%_common.bat" || exit /b 1
if not defined MODEL (
  if exist "%MODELS%\gemma-4-E4B-it-Q4_0.gguf" set "MODEL=%MODELS%\gemma-4-E4B-it-Q4_0.gguf"
  if exist "%MODELS%\gemma-4-E4B_q4_0-it.gguf" set "MODEL=%MODELS%\gemma-4-E4B_q4_0-it.gguf"
  if exist "%MODELS%\gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf" set "MODEL=%MODELS%\gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf"
)
if not defined MODEL (
  echo Missing model file for Gemma 4 E4B in: %MODELS%
  echo Download it from: https://huggingface.co/llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF
  echo ^(See README.md for the exact file names.^)
  pause
  exit /b 1
)
if not defined MMPROJ (
  if exist "%MODELS%\mmproj-gemma-4-E4B-it-Q8_0.gguf" set "MMPROJ=%MODELS%\mmproj-gemma-4-E4B-it-Q8_0.gguf"
  if exist "%MODELS%\gemma-4-E4B-it-mmproj.gguf" set "MMPROJ=%MODELS%\gemma-4-E4B-it-mmproj.gguf"
  if exist "%MODELS%\gemma-4-E4B-it-mmproj-BF16.gguf" set "MMPROJ=%MODELS%\gemma-4-E4B-it-mmproj-BF16.gguf"
)
set "VISION="
if defined MMPROJ set VISION=--mmproj "%MMPROJ%"
if not defined MMPROJ (
  echo No vision file ^(mmproj^) found in %MODELS%: starting without image support.
  echo Lena still saves images you attach; set default_vision = false in her settings.txt until you add it.
  echo.
)
echo Starting: "%LLAMA%" -m "%MODEL%" %VISION% -c 16384 --parallel 1 --jinja --host 127.0.0.1 --port 8080
echo.
"%LLAMA%" -m "%MODEL%" %VISION% -c 16384 --parallel 1 --jinja --host 127.0.0.1 --port 8080
echo.
echo llama-server stopped. If that was unexpected, the messages above say why.
pause
