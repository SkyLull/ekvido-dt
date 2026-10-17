@echo off
rem Windows entry for the mise tasks: run packaging/ekvido/<task>.sh in an MSYS2 UCRT64 login shell.
set MSYSTEM=UCRT64
set CHERE_INVOKING=1
C:\msys64\usr\bin\bash.exe -l packaging/ekvido/%1.sh
