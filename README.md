![example11-accumulate-linux](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/example11-accumulate-linux.png)

OptiXのチュートリアルとして有名な[optix7course](https://github.com/ingowald/optix7course)を、C++17（CUDAがまだC++20に対応していなかった……）で書き換えてみました。C++17や、あとCMake3.18が提供する機能の活用と、行列演算を独自ライブラリから[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)に変更したことで、元ネタよりもコードがシンプルになったと思います。

# コース

1. [開発環境を確認する](https://github.com/tail-island/optix7courseR/tree/main/example01-hello-optix)
2. [OptiXを使用するための前準備](https://github.com/tail-island/optix7courseR/tree/main/example02-pipeline-and-raygen)
3. [生成された画像をウィンドウで表示する](https://github.com/tail-island/optix7courseR/tree/main/example03-in-glfw-window)
4. [ポリゴンを表示する](https://github.com/tail-island/optix7courseR/tree/main/example04-first-triangle-mesh)
5. [ポリゴンに陰影をつける](https://github.com/tail-island/optix7courseR/tree/main/example05-first-shading)
6. [オブジェクトとしてポリゴンを構造化する](https://github.com/tail-island/optix7courseR/tree/main/example06-multiple-objects)
7. [既存の3Dモデルを読み込む](https://github.com/tail-island/optix7courseR/tree/main/example07-first-real-model)
8. [テクスチャーの貼り付ける](https://github.com/tail-island/optix7courseR/tree/main/example08-textures)
9. [影を生成する](https://github.com/tail-island/optix7courseR/tree/main/example09-shadow-rays)
10. [リアルな影を生成する](https://github.com/tail-island/optix7courseR/tree/main/example10-soft-shadows)
11. [積算でノイズを除去する](https://github.com/tail-island/optix7courseR/tree/main/example11-accumulate)

# ビルド

## Linuxでシェルを使用する場合

ターミナル上で、以下を実行してください。

~~~shell
$ mkdir build
$ cd build
$ cmake -DOptiX_INSTALL_DIR=/opt/optix -DCMAKE_BUILD_TYPE=Release
$ make
~~~

## WindowsでPowerShellを使用する場合

PowerShell上で、以下を実行してください。

~~~powershell
> New-Item -Path .\\build -ItemType Directory
> Set-Location .\\build\\
> cmake -DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0" ..
> msbuild -p:configuration=release -p:platform=x64 ALL_BUILD.vcxproj
~~~

## Visual Studio Codeを使用する場合

.vscodeフォルダにsettings.jsonファイルを作成してください。

~~~json
{
    "cmake.configureSettings": {
        "OptiX_INSTALL_DIR": "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0",
        "//": "Linuxの場合は\"/opt/optix\""
    }
}
~~~

Visual Studio Codeのステータス・バーをクリックし、以下に状態になるように設定してください（設定するのは、\[CMake: \[Release\]: Ready\]の部分と\[Visual Studio Community 2019 Release - amd64\]の部分です）。

![Visual Studio Code - status bar](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/visual-studio-code-status-bar.png)

CTRL-SHIFT-pで\[CMake: Delete Cache and Reconfigure\]を実行し、CTRL-SHIFT-pで\[CMake: Build\]を実行してください。
