// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=cuda --iree-cuda-target=sm_89 %s -o /dev/null
// RUN: iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=cuda --iree-cuda-target=rtx4090 %s -o /dev/null

module {
  func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
