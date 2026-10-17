@echo off
rem Windows entry: run scripts/inject.sh in an MSYS2 UCRT64 login shell.
set MSYSTEM=UCRT64
set CHERE_INVOKING=1
rem Inherit the Windows PATH so Git for Windows works before `mise run deps` installs MSYS2 git.
set MSYS2_PATH_TYPE=inherit
C:\msys64\usr\bin\bash.exe -l scripts/inject.sh
