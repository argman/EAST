@echo off

for /f %%i in ('where python') do set python_path=%%i
::set python_path=D:/Python/v3.6.5
if exist %python_path% (
  set python_dir=%python_path:~0,-11%
) else ( 
  @echo error: can't found python.exe, please delete "::" and set python_path manually at line 4
  pause
  exit
)

set vs2017_path="C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
if exist %vs2017_path% (
  call %vs2017_path%
  del adaptor.pyd
  cl adaptor.cpp ./include/clipper/clipper.cpp /I ./include /I %python_dir%/include /LD /Fe:adaptor.pyd /link/LIBPATH:%python_dir%/libs
  del adaptor.exp adaptor.obj clipper.obj adaptor.lib
) else (
  @echo can't found vs2017, please set vs2017_path manually at line 14, example:
  @echo Visual Studio install path is"D:/Visual Studio/Enterprise 2017",set vs2017_path="D:/Visual Studio/Enterprise 2017/VC/Auxiliary/Build/vcvars64.bat"
  pause
  exit
)

if not exist adaptor.pyd (
  @echo build adaptor.pyd failed
  pause
  exit
)