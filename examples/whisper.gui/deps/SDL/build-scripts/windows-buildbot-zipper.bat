@echo off
rem just a helper batch file for collecting up files and zipping them.
rem usage: windows-buildbot-zipper.bat <target> <slndir> <zipfilename>
rem must be run from root of SDL source tree.

IF EXIST %2\%1\Release GOTO okaydir
echo Please run from root of source tree after doing a Release build.
GOTO done

:okaydir
erase /q /f /s zipper
IF EXIST zipper GOTO zippermade
mkdir zipper
:zippermade
mkdir zipper\SDL
mkdir zipper\SDL\include
mkdir zipper\SDL\lib
copy include\*.h include\
copy %2\%1\Release\SDL2.dll zipper\SDL\lib\
copy %2\%1\Release\SDL2.lib zipper\SDL\lib\
copy %2\%1\Release\SDL2main.lib zipper\SDL\lib\
cd zipper
zip -9r ..\%3 SDL
cd ..
erase /q /f /s zipper

:done

