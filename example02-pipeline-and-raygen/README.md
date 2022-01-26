# example02-pipeline-and-raygen

信じられないかもしれませんけど、2つ目であるexample02-pipeline-and-raygenが、本コースの中で一番大変だったりします……。やっていることはOptiXを起動する`optixLaunch()`を呼び出すだけで実はレイ・トレーシングそのものは全くやらないのですけど、`optixLaunch()`できるようにするための設定は膨大な作業量なんです。

で、まずは、以下のOptiXの構造図を見てください。

![OptiXの構造]()

レイ・トレーシングでは、レイ（光）をカメラ側から辿っていきます。もちろん現実のレイはそれとは逆の方向で光源から出ているわけですけど、光源からレイを辿っていくとカメラのセンサーに入らない場合がいっぱい出てきて、それらはすべて計算の無駄となってしまうので、カメラ側から逆に辿るしかないわけ。というわけで、この、カメラからどんなレイを飛ばすのかを`__raygen__xxx()`にプログラミングします。

で、光は物に当たると反射したりしますから、物に衝突した場合の処理も必要です。OptiXでは、この処理を`__closeshit_xxx()`と`___anyhit_xxx()`に書くのですが、本コースでは`___anyhit_xxx()`は使用せずに`___closeshit_xxx()`のみ使用します。

で、あとは、レイを辿ってはみたのだけど何にも衝突しなかった場合の処理も必要です。大気圏の外側までレイを追いかけて、太陽にぶつかるまで頑張れるほどの計算資源は、今どきの高性能なコンピューターであっても持ち合わせてはいませんから。というわけで、`__miss_xxx()`に、レイが衝突しなかった場合の救済処置を書きます。

さて、OptiXはCUDAを使用した高速なレイ・トレーシングが売りなので、これらの処理は全てCUDAとしてプログラミングします。なので、拡張子がcuとなっているDeviceProgram.cuに処理を書きました。

……といっても、example02-pipeline-and-raygenでは、`__faygen__renderFrame()`ではレイを飛ばさずにテスト・パターンの画像を作るだけで、`__closeshit_radiance()`や`___miss_radiance()`では何もしないですけどね。`optixGetLaunchIndex()`で画面上の座標を取得して、座標位置に合わせて適当に色を設定しているだけ。詳しくはDeviceProgram.cuをご参照ください。

このDeviceProgram.cuを見ていただくと、一番最初になんだコレと疑問に感じるのは、最初の方にあるこの行だと思います。

~~~c++
extern "C" {
__constant__ LaunchParams optixLaunchParams;
}
~~~

この部分はCPUで実行するプログラムからGPUで実行するCUDAのプログラムとに渡すパラメーターで、後述するOptixState.hの準備作業の中で型も変数名も自由に設定できます。で、今回使用している方はOptixParams.hで定義している`LaunchParams`構造体で、中身は画像のバッファーです。

~~~c++
struct LaunchParams {
  std::uint32_t *imageBuffer;
};
~~~

ここまでを読んでなんだ別に大変じゃないじゃんと感じた方は、OptixState.hを開いてみてください。ここには400行を超える膨大なコードが書かれているのですけど、この全ては、OptiXを始めるための準備作業でしかないんです。OptiXを初期化して、CUDAのストリームを取得して、OptiXのコンテキストを作成して、OptiXのモジュールを作成して（この際にはCUDAのbin2cで文字列化したバイナリコードを使用したりします）、レイ生成処理と衝突処理と衝突しなかった場合の処理のプログラム・グループを作成して、処理のパイプラインを作成して、最後にシェーダー・バインディング・テーブル（OptiXが処理しやすいように処理をパッケージしたものみたい）を作っていきます。で、それぞれの処理がもー強烈に面倒くさいんですよ……。

まぁ、一度作ってしまえばこの先は小さな修正しかしませんので、こういうモノなんだと考えて写経していただくことにしてここではコードは載せないことにさせてください。設定のパラメーター等は、後でOptiXのリファレンスを参照していただければよいかなと。

というわけで、これで準備が終わりましたから、Render.hでOptiXを実行してみましょう。作業は簡単で、`LaunchParms`構造体をGPUのメモリに転送（この作業を楽にするために、common/DeviceBuffer.hに`DeviceBuffer`というユーティリティ・クラスを作成しました）して、`optixLaunch()`関数を呼び出すだけです。

~~~c++
optixLaunchParamsBuffer_.set(LaunchParams{reinterpret_cast<std::uint32_t *>(imageBuffer_.getData())});

OPTIX_CHECK(optixLaunch(optixState_.getPipeline(), optixState_.getStream(), optixLaunchParamsBuffer_.getData(), optixLaunchParamsBuffer_.getDataSize(), &optixState_.getShaderBindingTable(), width_, height_, 1));
~~~

これで`__raygen__renderFrame()`で作成したテスト・パターンの画像を取得できます。今回は、この画像をstbという超便利なライブラリを使用してosc_example2.pngというファイルに保存して終わりにしました。画像が生成されたら、example02-pipeline-and-raygenは完了です。お疲れさまでした。

![example02-pipeline-and-raygen-linux]()

![example02-pipeline-and-raygen-windows]()

