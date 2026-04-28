#!/usr/bin/env python3
"""Benchmark Qwen3.5-2B decode across Torch CUDA, Torch Inductor, and IREE.

The `all` backend runs each benchmark in a separate subprocess so GPU memory is
released between backends. Timings report the autoregressive decode loop after
prefill, matching `example_qwen_iree.py`'s "Decode: N tokens in T" convention:
the first generated token comes from prefill, so the timed decode loop runs
N - 1 model steps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "runtime/bindings/python").exists() and (
            parent / "third_party/approxMLIR/runtime"
        ).exists():
            return parent
    raise RuntimeError(f"could not find repo root from {here}")


REPO_ROOT = find_repo_root()
BUILD_TMP = REPO_ROOT / "build/tmp"
BUILD_TMP.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TMPDIR", str(BUILD_TMP))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(BUILD_TMP / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(BUILD_TMP / "triton"))
os.environ.setdefault("XDG_CACHE_HOME", str(BUILD_TMP / "xdg-cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["all", "torch-eager", "torch-inductor", "iree"],
        default="all",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=33)
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BUILD_TMP / "qwen35_decode_bench",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--inductor-mode", default="reduce-overhead")
    parser.add_argument("--inductor-fullgraph", action="store_true")
    parser.add_argument(
        "--torch-cast-linear-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iree-token-result", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iree-pretranspose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-iree-memory-types", action="store_true")
    return parser.parse_args()


def result_path(args: argparse.Namespace, backend: str) -> Path:
    return args.output_dir / f"{backend}.json"


def write_result(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def prepend_pythonpath(env: dict[str, str], paths: list[Path]) -> None:
    existing = env.get("PYTHONPATH")
    parts = [str(path) for path in paths]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def run_all(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backends = ["torch-eager", "torch-inductor", "iree"]
    results = []
    for backend in backends:
        out_path = result_path(args, backend)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            backend,
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-seq",
            str(args.max_seq),
            "--output-dir",
            str(args.output_dir),
            "--json-output",
            str(out_path),
            "--inductor-mode",
            args.inductor_mode,
        ]
        if args.inductor_fullgraph:
            cmd.append("--inductor-fullgraph")
        cmd.append(
            "--torch-cast-linear-inputs"
            if args.torch_cast_linear_inputs
            else "--no-torch-cast-linear-inputs"
        )
        cmd.append("--torch-warmup" if args.torch_warmup else "--no-torch-warmup")
        cmd.append("--iree-token-result" if args.iree_token_result else "--no-iree-token-result")
        cmd.append("--iree-pretranspose" if args.iree_pretranspose else "--no-iree-pretranspose")
        if args.print_iree_memory_types:
            cmd.append("--print-iree-memory-types")

        print("\n" + "=" * 80, flush=True)
        print(f"Running {backend}", flush=True)
        print("=" * 80, flush=True)
        env = os.environ.copy()
        env["TMPDIR"] = str(BUILD_TMP)
        proc = subprocess.run(cmd, env=env, text=True)
        if out_path.exists():
            results.append(json.loads(out_path.read_text()))
        else:
            results.append({"backend": backend, "status": "failed", "returncode": proc.returncode})

    print_summary(results, args)
    combined_path = args.output_dir / "summary.json"
    combined_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    write_markdown_summary(args.output_dir / "summary.md", results, args)
    return 0 if all(result.get("status") == "ok" for result in results) else 1


def print_summary(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n" + "=" * 80)
    print("Qwen3.5-2B Decode Benchmark Summary")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Requested generated tokens: {args.max_new_tokens}")
    print("")
    header = (
        f"{'backend':<18} {'status':<8} {'prefill_s':>10} {'decode_s':>10} "
        f"{'tok/s*':>10} {'step/s':>10} {'warmup_s':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        status = result.get("status", "failed")
        prefill = result.get("prefill_s")
        decode = result.get("decode_s")
        tok_s = result.get("reported_tokens_per_s")
        step_s = result.get("decode_steps_per_s")
        warmup = result.get("warmup_s", result.get("compile_warmup_s"))
        print(
            f"{result.get('backend', '?'):<18} {status:<8} "
            f"{prefill if prefill is not None else float('nan'):>10.4f} "
            f"{decode if decode is not None else float('nan'):>10.4f} "
            f"{tok_s if tok_s is not None else float('nan'):>10.2f} "
            f"{step_s if step_s is not None else float('nan'):>10.2f} "
            f"{warmup if warmup is not None else float('nan'):>10.4f}"
        )
        if status != "ok" and result.get("error"):
            print(f"  error: {result['error']}")
    print("")
    print("* tok/s uses the existing IREE demo convention: N reported tokens,")
    print("  while the timed decode loop actually runs N - 1 decode steps.")
    print(f"Wrote JSON results to {args.output_dir}")


def write_markdown_summary(
    path: Path, results: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    lines = [
        "# Qwen3.5-2B Decode Benchmark",
        "",
        f"- Model: `{args.model}`",
        f"- Prompt: `{args.prompt}`",
        f"- Requested generated tokens: `{args.max_new_tokens}`",
        "- Timing convention: decode time excludes prefill and runs `N - 1` model steps.",
        "",
        "| Backend | Status | Prefill (s) | Decode (s) | Reported tok/s | Decode step/s | Warmup (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        def fmt(key: str) -> str:
            value = result.get(key)
            return "" if value is None else f"{value:.4f}"

        warmup = result.get("warmup_s", result.get("compile_warmup_s"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result.get("backend", "?")),
                    str(result.get("status", "failed")),
                    fmt("prefill_s"),
                    fmt("decode_s"),
                    fmt("reported_tokens_per_s"),
                    fmt("decode_steps_per_s"),
                    "" if warmup is None else f"{warmup:.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def load_qwen_text_model(model_name: str, device: str):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class Qwen35TextOnlyCausalLM(torch.nn.Module):
        def __init__(self, conditional_model):
            super().__init__()
            self.config = conditional_model.config.text_config
            self.generation_config = getattr(conditional_model, "generation_config", None)
            self.model = conditional_model.model.language_model
            self.lm_head = conditional_model.lm_head

        def forward(
            self,
            input_ids,
            cache_position=None,
            past_key_values=None,
            return_dict=False,
            use_cache=True,
        ):
            position_ids = None
            if cache_position is not None:
                position_ids = cache_position.reshape(1, -1)
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = self.lm_head(outputs.last_hidden_state)
            return logits, outputs.past_key_values

    conditional_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = Qwen35TextOnlyCausalLM(conditional_model)
    del conditional_model

    def _conv1d_forward_cast_input(self, input):
        return torch.nn.functional.conv1d(
            input.to(self.weight.dtype),
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    for module in model.modules():
        if module.__class__.__name__ == "Qwen3_5GatedDeltaNet" and hasattr(module, "conv1d"):
            module.conv1d = module.conv1d.to(torch.float32)
            module.conv1d.forward = _conv1d_forward_cast_input.__get__(
                module.conv1d, module.conv1d.__class__
            )

    if getattr(load_qwen_text_model, "cast_linear_inputs", True):
        def _linear_forward_cast_input(self, input):
            if input.dtype != self.weight.dtype:
                input = input.to(self.weight.dtype)
            return torch.nn.functional.linear(input, self.weight, self.bias)

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = _linear_forward_cast_input.__get__(
                    module, module.__class__
                )

    model.eval().to(device)
    return tokenizer, model


def tokenize_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    prompt_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(prompt_inputs, "input_ids"):
        return prompt_inputs.input_ids
    if isinstance(prompt_inputs, dict):
        return prompt_inputs["input_ids"]
    return prompt_inputs


def run_torch_backend(args: argparse.Namespace, use_inductor: bool) -> dict[str, Any]:
    import torch
    from transformers import cache_utils

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for torch benchmark")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    backend_name = "torch-inductor" if use_inductor else "torch-eager"
    device = "cuda"
    load_qwen_text_model.cast_linear_inputs = args.torch_cast_linear_inputs
    tokenizer, model = load_qwen_text_model(args.model, device)
    prompt_ids = tokenize_prompt(tokenizer, args.prompt).to(device)
    seq_len = prompt_ids.shape[1]
    if seq_len > args.max_seq:
        raise ValueError(f"prompt length {seq_len} exceeds max_seq {args.max_seq}")
    decode_steps = max(args.max_new_tokens - 1, 0)
    positions = torch.arange(seq_len, seq_len + decode_steps, device=device)

    def new_cache():
        return cache_utils.StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=args.max_seq,
            device=device,
            dtype=torch.bfloat16,
        )

    def cuda_time(fn, *, profile: bool = False):
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        t0 = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return value, time.perf_counter() - t0

    def prefill_once():
        cache = new_cache()
        cache_position = torch.arange(seq_len, device=device)
        logits, kv = model(
            prompt_ids,
            cache_position=cache_position,
            past_key_values=cache,
            return_dict=False,
            use_cache=True,
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token, kv

    def eager_decode_step(input_ids, cache_position, past_key_values):
        logits, kv = model(
            input_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token, kv

    decode_step_fn = eager_decode_step
    compile_warmup_s = None
    if use_inductor:
        decode_step_fn = torch.compile(
            eager_decode_step,
            backend="inductor",
            mode=args.inductor_mode,
            fullgraph=args.inductor_fullgraph,
        )

    def call_decode_step(input_ids, cache_position, past_key_values):
        if use_inductor and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        token, next_kv = decode_step_fn(input_ids, cache_position, past_key_values)
        if use_inductor:
            token = token.clone()
        return token, next_kv

    warmup_s = 0.0
    if args.torch_warmup and decode_steps:
        with torch.inference_mode():
            (warm_token, warm_kv), warm_prefill_s = cuda_time(prefill_once)
            (warm_token, warm_kv), warm_decode_s = cuda_time(
                lambda: call_decode_step(warm_token, positions[:1], warm_kv)
            )
        warmup_s = warm_prefill_s + warm_decode_s
        if use_inductor:
            compile_warmup_s = warm_decode_s
        del warm_token, warm_kv
        torch.cuda.empty_cache()

    with torch.inference_mode():
        (next_token, kv), prefill_s = cuda_time(prefill_once)
        generated = [next_token]

        def decode_loop():
            nonlocal next_token, kv
            for step in range(decode_steps):
                pos = positions[step : step + 1]
                next_token, kv = call_decode_step(next_token, pos, kv)
                generated.append(next_token)
            return next_token, kv

        profile_decode = os.environ.get("PROFILE_CUDA_API", "0") == "1"
        _, decode_s = cuda_time(decode_loop, profile=profile_decode)

    generated_ids = torch.cat(generated, dim=1).detach().cpu().tolist()[0]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    reported_tokens = len(generated_ids)
    result = {
        "backend": backend_name,
        "status": "ok",
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": int(seq_len),
        "reported_tokens": int(reported_tokens),
        "decode_steps": int(decode_steps),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "reported_tokens_per_s": reported_tokens / decode_s if decode_s else None,
        "decode_steps_per_s": decode_steps / decode_s if decode_s else None,
        "warmup_s": warmup_s,
        "compile_warmup_s": compile_warmup_s,
        "torch_cast_linear_inputs": args.torch_cast_linear_inputs,
        "output": text,
    }
    print_backend_result(result)
    return result


def stream_subprocess(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    return proc.wait(), "".join(lines)


def run_iree_backend(args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ.copy()
    env["TMPDIR"] = str(BUILD_TMP)
    env["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    env["MAX_SEQ"] = str(args.max_seq)
    env["PROMPT"] = args.prompt
    env["QWEN_MODEL"] = args.model
    env["SKIP_EAGER_DECODE"] = "1"
    env["PRETRANSPOSE_LINEAR_WEIGHTS"] = "1" if args.iree_pretranspose else "0"
    env["IREE_DECODE_TOKEN_RESULT"] = "1" if args.iree_token_result else "0"
    env["PRINT_STATIC_INPUT_MEMORY_TYPES"] = "1" if args.print_iree_memory_types else "0"
    env["PYTHONUNBUFFERED"] = "1"
    iree_compile = REPO_ROOT / "build/tools/iree-compile"
    if iree_compile.exists():
        env.setdefault("IREE_COMPILE_BIN", str(iree_compile))
    prepend_pythonpath(
        env,
        [
            REPO_ROOT / "build/qwen_iree_run/local_iree_compiler_shim",
            REPO_ROOT / "build/runtime/bindings/python",
            REPO_ROOT / "third_party/approxMLIR/runtime",
        ],
    )

    example = REPO_ROOT / "third_party/approxMLIR/runtime/examples/example_qwen_iree.py"
    cmd = [sys.executable, str(example)]
    returncode, output = stream_subprocess(cmd, env)
    if returncode != 0:
        raise RuntimeError(f"IREE benchmark failed with return code {returncode}")

    prefill_matches = re.findall(r"Prefill\s*:\s*([0-9.]+)s\s+logits\.shape", output)
    decode_matches = re.findall(r"Decode\s*:\s*([0-9]+)\s+tokens in\s+([0-9.]+)s", output)
    upload_match = re.search(r"Static inputs uploaded to device in\s+([0-9.]+)s", output)
    prepack_match = re.search(r"Prepack\s*:\s*([0-9.]+)s", output)
    mem_types = [int(value) for value in re.findall(r"Static input memory: .* type=([0-9]+)", output)]
    if not decode_matches:
        raise RuntimeError("could not parse IREE decode timing")
    reported_tokens, decode_s = decode_matches[-1]
    result = {
        "backend": "iree",
        "status": "ok",
        "model": args.model,
        "prompt": args.prompt,
        "reported_tokens": int(reported_tokens),
        "decode_steps": max(int(reported_tokens) - 1, 0),
        "prefill_s": float(prefill_matches[-1]) if prefill_matches else None,
        "decode_s": float(decode_s),
        "reported_tokens_per_s": int(reported_tokens) / float(decode_s),
        "decode_steps_per_s": max(int(reported_tokens) - 1, 0) / float(decode_s),
        "static_upload_s": float(upload_match.group(1)) if upload_match else None,
        "prepack_s": float(prepack_match.group(1)) if prepack_match else None,
        "iree_token_result": args.iree_token_result,
        "iree_pretranspose": args.iree_pretranspose,
        "printed_static_memory_types": mem_types,
    }
    print_backend_result(result)
    return result


def print_backend_result(result: dict[str, Any]) -> None:
    print("\n--- Benchmark result ---")
    for key in (
        "backend",
        "status",
        "prompt_tokens",
        "reported_tokens",
        "decode_steps",
        "prefill_s",
        "decode_s",
        "reported_tokens_per_s",
        "decode_steps_per_s",
        "warmup_s",
        "compile_warmup_s",
        "static_upload_s",
        "prepack_s",
    ):
        if key in result and result[key] is not None:
            print(f"{key}: {result[key]}")
    if result.get("output"):
        print(f"output: {result['output']}")


def worker_main(args: argparse.Namespace) -> int:
    try:
        if args.backend == "torch-eager":
            result = run_torch_backend(args, use_inductor=False)
        elif args.backend == "torch-inductor":
            result = run_torch_backend(args, use_inductor=True)
        elif args.backend == "iree":
            result = run_iree_backend(args)
        else:
            raise ValueError(f"worker cannot run backend={args.backend}")
    except Exception as e:
        result = {
            "backend": args.backend,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print("\n--- Benchmark failed ---")
        print(result["error"])
        print(result["traceback"])
        write_result(args.json_output, result)
        return 1

    write_result(args.json_output, result)
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "all":
        return run_all(args)
    return worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
