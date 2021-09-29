~~~~shell
> New-Item -Path .\build -ItemType Directory
> Set-Location .\build\
> cmake -DOptiX_INSTALL_DIR='C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.3.0\' ..
> msbuild -p:configuration=release .\optix7courseR.sln
~~~~
