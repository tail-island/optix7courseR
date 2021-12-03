## PowerShellを使用する場合

PowerShell上で、以下を実行してください。

~~~shell
> New-Item -Path .\build -ItemType Directory
> Set-Location .\build\
> cmake -DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0" ..
> msbuild -p:configuration=release -p:platform=x64 ALL_BUILD.vcxproj
~~~

## Visual Studio Codeを使用する場合

.vscodeフォルダにsettings.jsonファイルを作成してください。

~~~json
{
    "cmake.configureSettings": {
        "OptiX_INSTALL_DIR": "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
    }
}
~~~

Visual Studio Codeのステータス・バーをクリックし、以下に状態になるように設定してください（設定するのは、[CMake: [Release]: Ready]の部分と[Visual Studio Community 2019 Release - amd64]の部分です）。

![Visual Studio Code - status bar](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/visual-studio-code-status-bar.png)

CTRL-SHIFT-pで[CMake: Delete Cache and Reconfigure]を実行し、CTRL-SHIFT-pで[CMake: Build]を実行してください。
