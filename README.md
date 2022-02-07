![example11-accumulate-linux](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/example11-accumulate-linux.png)

[optix7course](https://github.com/ingowald/optix7course)をC++17（CUDAがC++20に対応していなかった……）で書き換えました。C++17や、あとCMake3.18が提供する機能の活用と、行列演算を独自ライブラリから[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)に変更することで、オリジナルよりもコードがシンプルになっています。

# コース

1. [example01-hello-optix](https://github.com/tail-island/optix7courseR/tree/main/example01-hello-optix) - 開発環境が正しく構築されたかの確認
1. [example02-pipeline-and-raygen](https://github.com/tail-island/optix7courseR/tree/main/example02-pipeline-and-raygen) - OptiXを使用するための前準備
1. [example03-in-glfw-window](https://github.com/tail-island/optix7courseR/tree/main/example03-in-glfw-window) - 生成された画像をウィンドウ表示
1. [example04-first-triangle-mesh](https://github.com/tail-island/optix7courseR/tree/main/example04-first-triangle-mesh) - ポリゴンを表示
1. [example05-first-shading](https://github.com/tail-island/optix7courseR/tree/main/example05-first-shading) - ポリゴンに陰影をつける
1. [example06-multiple-objects](https://github.com/tail-island/optix7courseR/tree/main/example06-multiple-objects) - ポリゴンをオブジェクトとして構造化
1. [example07-first-real-model](https://github.com/tail-island/optix7courseR/tree/main/example07-first-real-model) - 既存の3Dモデルの読み込み
1. [example08-textures](https://github.com/tail-island/optix7courseR/tree/main/example08-textures) - テクスチャーの貼り付け
1. [example09-shadow-rays](https://github.com/tail-island/optix7courseR/tree/main/example09-shadow-rays) - 影の生成
1. [example10-soft-shadows](https://github.com/tail-island/optix7courseR/tree/main/example10-soft-shadows) - リアルな影の生成
1. [example11-accumulate](https://github.com/tail-island/optix7courseR/tree/main/example11-accumulate) - 積算によるノイズ除去

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
        "OptiX_INSTALL_DIR": "/opt/optix"  // Windowsの場合は"C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
    }
}
~~~

Visual Studio Codeのステータス・バーをクリックし、以下に状態になるように設定してください（設定するのは、\[CMake: \[Release\]: Ready\]の部分と\[Visual Studio Community 2019 Release - amd64\]の部分です）。

![Visual Studio Code - status bar](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/visual-studio-code-status-bar.png)

CTRL-SHIFT-pで\[CMake: Delete Cache and Reconfigure\]を実行し、CTRL-SHIFT-pで\[CMake: Build\]を実行してください。
