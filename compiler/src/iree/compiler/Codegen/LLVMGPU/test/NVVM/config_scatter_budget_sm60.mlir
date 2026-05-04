// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=sm_60 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @scatter_readonly_init_budget_tiled() {
  %c0 = arith.constant 0 : index
  %original = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1x442xf32, strided<[442, 442, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %updates = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<11x1x442xf32, strided<[442, 442, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %indices = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<11xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : memref<32x1x442xf32, strided<[442, 442, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  %original_t = iree_codegen.load_from_buffer %original : memref<32x1x442xf32, strided<[442, 442, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<32x1x442xf32>
  %updates_t = iree_codegen.load_from_buffer %updates : memref<11x1x442xf32, strided<[442, 442, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<11x1x442xf32>
  %indices_t = iree_codegen.load_from_buffer %indices : memref<11xi32, strided<[1], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<11xi32>
  %result = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%updates_t, %indices_t : tensor<11x1x442xf32>, tensor<11xi32>)
    outs(%original_t : tensor<32x1x442xf32>) {
  ^bb0(%update: f32, %original_value: f32):
    iree_linalg_ext.yield %update : f32
  } -> tensor<32x1x442xf32>
  iree_codegen.store_to_buffer %result, %out : tensor<32x1x442xf32> into memref<32x1x442xf32, strided<[442, 442, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
  return
}

// CHECK-LABEL: func.func @scatter_readonly_init_budget_tiled
// CHECK-SAME:  #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<Distribute> workgroup_size = [64, 1, 1] subgroup_size = 32
// CHECK:       iree_linalg_ext.scatter
// CHECK-SAME:  lowering_config = #iree_codegen.lowering_config<tile_sizes = {{\[\[11, 1, 384\]\]}}>
