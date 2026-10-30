@echo off
setlocal
rem nomic-embed-text-v1.5 on the CPU, Lena's embedding model (UNTESTED on Windows; see start_embeddings_cpu.sh).
set "KIT=%~dp0"
call "%KIT%_common.bat" || exit /b 1
if not defined MODEL (
  if exist "%MODELS%\nomic-embed-text-v1.5.Q4_K_M.gguf" set "MODEL=%MODELS%\nomic-embed-text-v1.5.Q4_K_M.gguf"
  if exist "%MODELS%\nomic-embed-text-v1.5.Q8_0.gguf" set "MODEL=%MODELS%\nomic-embed-text-v1.5.Q8_0.gguf"
)
if not defined MODEL (
  echo Missing model file for nomic-embed-text v1.5 in: %MODELS%
  echo Download it from: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
  echo ^(See README.md for the exact file names.^)
  pause
  exit /b 1
)
echo Starting: "%LLAMA%" -m "%MODEL%" --embedding --pooling mean -c 2048 -b 2048 -ub 2048 --device none --host 127.0.0.1 --port 8082
echo.
"%LLAMA%" -m "%MODEL%" --embedding --pooling mean -c 2048 -b 2048 -ub 2048 --device none --host 127.0.0.1 --port 8082
echo.
echo llama-server stopped. If that was unexpected, the messages above say why.
pause
