# Coauthor Sync Bundle

This branch packages local Qwen/IREE sync artifacts without mixing them into
formal IREE PR branches.

LLVM patch from llvm/llvm-project#194203:

```bash
git -C third_party/llvm-project apply \
  ../../coauthor_sync/llvm/0001-mlir-nvgpu-to-nvvm-support-bf16-mma-sync.patch
```

Updated Qwen3.5 IREE demo and decode benchmark:

```bash
cp coauthor_sync/approxMLIR/runtime/examples/example_qwen_iree.py \
  third_party/approxMLIR/runtime/examples/example_qwen_iree.py
cp coauthor_sync/approxMLIR/runtime/examples/benchmark_qwen35_decode.py \
  third_party/approxMLIR/runtime/examples/benchmark_qwen35_decode.py
```
