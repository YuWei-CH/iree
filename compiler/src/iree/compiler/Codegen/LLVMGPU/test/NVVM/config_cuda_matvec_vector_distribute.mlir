// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=sm_80 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @cuda_vecmat_f32_bf16_f32_with_bias() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6144xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6144x2048xbf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [6144], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6144xf32>> -> tensor<6144xf32>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [6144, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6144x2048xbf16>> -> tensor<6144x2048xbf16>
  %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [2048], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xf32>> -> tensor<2048xf32>
  %7 = tensor.empty() : tensor<2048xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<2048xf32>) -> tensor<2048xf32>
  %9 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d1, d0)>,
      affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%4, %5 : tensor<6144xf32>, tensor<6144x2048xbf16>)
    outs(%8 : tensor<2048xf32>) {
  ^bb0(%in: f32, %in_0: bf16, %out: f32):
    %11 = arith.extf %in_0 : bf16 to f32
    %12 = arith.mulf %in, %11 : f32
    %13 = arith.addf %out, %12 : f32
    linalg.yield %13 : f32
  } -> tensor<2048xf32>
  %10 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%6, %9 : tensor<2048xf32>, tensor<2048xf32>)
    outs(%7 : tensor<2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %11 = arith.addf %in, %in_0 : f32
    linalg.yield %11 : f32
  } -> tensor<2048xf32>
  iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0], sizes = [2048], strides = [1] : tensor<2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf32>>
  return
}

// CHECK-LABEL: func.func @cuda_vecmat_f32_bf16_f32_with_bias
// CHECK-SAME:  #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [96, 1, 1] subgroup_size = 32
// CHECK:       linalg.generic
// CHECK-SAME:  partial_reduction = [0, 768]
// CHECK-SAME:  thread = [0, 8]
// CHECK-SAME:  workgroup = [16, 0]

// -----

#pipeline_layout_bf16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @cuda_vecmat_bf16_bf16_bf16() {
  %cst = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x6144xbf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6144xbf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [2048], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048xbf16>> -> tensor<2048xbf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 6144], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x6144xbf16>> -> tensor<2048x6144xbf16>
  %5 = tensor.empty() : tensor<6144xbf16>
  %6 = linalg.fill ins(%cst : bf16) outs(%5 : tensor<6144xbf16>) -> tensor<6144xbf16>
  %7 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d1, d0)>,
      affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3, %4 : tensor<2048xbf16>, tensor<2048x6144xbf16>)
    outs(%6 : tensor<6144xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %8 = arith.mulf %in, %in_0 : bf16
    %9 = arith.addf %out, %8 : bf16
    linalg.yield %9 : bf16
  } -> tensor<6144xbf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0], sizes = [6144], strides = [1] : tensor<6144xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6144xbf16>>
  return
}

// CHECK-LABEL: func.func @cuda_vecmat_bf16_bf16_bf16
// CHECK-SAME:  #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [32, 1, 1] subgroup_size = 32
// CHECK:       linalg.generic
// CHECK-SAME:  partial_reduction = [0, 256]
// CHECK-SAME:  thread = [0, 8]
// CHECK-SAME:  workgroup = [4, 0]
