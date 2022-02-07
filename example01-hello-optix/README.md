# example01-hello-optix

まずは、環境が正しく構築されたかを確認しましょう。そのために、OptiXを初期化するAPI（Application Programming Interface）である`optixInit()`を呼び出してみます。

NVIDIAのCUDAやOptiXのAPIでは、古いC言語との互換性のためだと思うのですけど、`throw`と`try/catch`ではなく戻り値でエラーを表現します。API呼び出しのたびに戻り値を調べる`if`文を書くのはあまりに大変なので、common/Util.hにマクロを定義しました。

~~~c++
#define CUDA_CHECK(call)                                                                                                                       \
  {                                                                                                                                            \
    if (call != cudaSuccess) {                                                                                                                 \
      auto error = cudaGetLastError();                                                                                                         \
      std::cerr << "CUDA call (" << #call << ") failed. " << cudaGetErrorName(error) << " (" << cudaGetErrorString(error) << ")" << std::endl; \
      std::exit(2);                                                                                                                            \
    }                                                                                                                                          \
  }

#define CU_CHECK(call)                                                        \
  {                                                                           \
    if (call != CUDA_SUCCESS) {                                               \
      std::cerr << "CU call (" << #call << ") failed. " << call << std::endl; \
      std::exit(2);                                                           \
    }                                                                         \
  }

#define OPTIX_CHECK(call)                                                        \
  {                                                                              \
    if (call != OPTIX_SUCCESS) {                                                 \
      std::cerr << "Optix call (" << #call << ") failed. " << call << std::endl; \
      std::exit(2);                                                              \
    }                                                                            \
  }
~~~

今回は使うAPIがCUDAのランタイムAPIとCUDAのドライバーAPI、OptiXと3種類ありますので、3つのマクロを作成しました。

で、残る作業は、この`OPTIX_CHECK`マクロを使用して`optixInit()`を呼び出すだけ。HelloOptix.hに以下の一行を追加しました。

~~~c++
OPTIX_CHECK(optixInit());
~~~

では、実行してください。エラー・メッセージが表示されなければ、これでexample01-hello-optixは完了です。お疲れさまでした。

![example01-hello-world-linux](https://raw.githubusercontent.com/tail-island/optix7courseR/main/image/example01-hello-world-linux.png)

![example01-hello-world-windows]()
