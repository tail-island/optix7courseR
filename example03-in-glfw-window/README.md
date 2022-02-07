# example03-in-glfw-window

example02みたいに画像ファイルが生成されるだけじゃ物足りない！　画面に表示してレイ・トレーシング結果をすぐに見たい！　そりゃ、そうですよね。というわけで、example03では、画面にウィンドウ表示してみます。

とは言っても、画面表示をイチから自前でやるのは大変なので、GLFWというライブラリを使用して楽をしましょう。GLFWはちょっとレイヤが低いライブラリなので、今回はcommon/Window.hでラップしました。このcommon/Window.hで定義している`osc::common::Window`クラスを継承して、Window.hでexample03用のウィンドウを表現する`osc::Window`クラスを作成します。といってもやっていることは単純で、Renderer.hで定義している`osc::Renderer`との間をつないでいるだけです。

~~~c++
void resize(const Eigen::Vector2i &size) noexcept override {
  renderer_.resize(size.x(), size.y());
}

std::vector<Eigen::Vector3f> render() noexcept override {
  return renderer_.render();
}
~~~

で、楽をした分の罪滅ぼしとして、せっかく画面に表示させるのだから少し動かしてみましょう。

まず、動かすためには何か変わっていくデータが必要なので、OptixParams.hの`osc::LaunchParams`に`int frameId;`を追加します。

~~~c++
struct LaunchParams {
  float3 *imageBuffer;
  int frameId;
};
~~~

追加した`frameId`を使って生成される画像が変わるように、DeviceProgram.cuの`__raygen_renderFrame()`を変更します。

~~~c++
extern "C" __global__ void __raygen__renderFrame() {
  const auto &frameId = optixLaunchParams.frameId;
  const auto &x = optixGetLaunchIndex().x;
  const auto &y = optixGetLaunchIndex().y;

  if (frameId == 0 && x == 0 && y == 0) {
    printf("___raygen__renderFrame() is called.\n");
  }

  // 通常はレイを生成して、トレースして、その結果に基づいて出力を作成するのですが、とりあえず、テスト・パターンを生成してみます。

  const auto r = (x + frameId) % 256;
  const auto g = (y + frameId) % 256;
  const auto b = (x + y + frameId) % 256;

  optixLaunchParams.imageBuffer[x + y * optixGetLaunchDimensions().x] = float3{static_cast<float>(r) / 255, static_cast<float>(g) / 255, static_cast<float>(b) / 255};;
}
~~~

このように`frameId`を加えることで、生成される画像の色が変わるようになりました。あとは、Renderer.hの中に`frameId`を`osc::LaunchParams`に設定すると、`farmeId`をインクリメントする処理を書くだけ。

~~~c++
auto render() noexcept {
  optixLaunchParamsBuffer_.set(LaunchParams{reinterpret_cast<float3 *>(imageBuffer_.getData()), frameId_});

  OPTIX_CHECK(optixLaunch(optixState_.getPipeline(), optixState_.getStream(), optixLaunchParamsBuffer_.getData(), optixLaunchParamsBuffer_.getDataSize(), &optixState_.getShaderBindingTable(), width_, height_, 1));

  frameId_++;

  CUDA_CHECK(cudaDeviceSynchronize());

  return imageBuffer_.get();
}
~~~

以上で、画面に動くテスト・パターンが表示されます。テスト・パターンが画面に表示されたら、example03-in-glfw-windowは完了です。お疲れさまでした。

![example03-in-glfw-window-linux]()

![example03-in-glfw-window-windows]()
