#!/usr/bin/env python3
"""Deploy Qwen2.5-0.5B-Instruct on IREE CUDA via torchax.

Workflow:
1. Load Qwen2.5-0.5B-Instruct from HuggingFace (PyTorch)
2. Convert to JAX via torchax (model.to('jax') + functional_call)
3. Use StaticCache for fixed-shape KV buffers (IREE-compatible)
4. Validate eager-mode correctness (prefill + autoregressive decode)
5. Export to StableHLO MLIR via jax.export
6. Compile to IREE VM flatbuffers targeting CUDA
7. Run full prefill + decode loop via IREE with timing

No approximation is applied (empty config / vanilla compilation).

Usage:
    python example_qwen_iree.py
"""

import os
import shlex
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.export as jax_export
from jax.tree_util import register_pytree_node

import numpy as np
import torch
import torchax
import torchax.interop

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import cache_utils

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BACKEND = os.environ.get("APPROX_BACKEND", "cuda")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "100"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "256"))
PROMPT = os.environ.get("PROMPT", "What is the capital of France?")

print("=" * 70)
print("Qwen IREE Deployment (no approximation)")
print("=" * 70)
print(f"Model         : {MODEL_NAME}")
print(f"IREE backend  : {BACKEND}")
print(f"Max seq len   : {MAX_SEQ}")
print(f"JAX devices   : {jax.devices()}")
print(f"Prompt        : {PROMPT}")

IREE_EXTRA_ARGS = shlex.split(os.environ.get("IREE_EXTRA_ARGS", ""))
if IREE_EXTRA_ARGS:
    print(f"IREE extra args: {IREE_EXTRA_ARGS}")

# ===================================================================
# Step 1: Load model
# ===================================================================
print(f"\n--- Step 1: Loading model ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)
model.eval()

# Untie lm_head / embed_tokens weights so torchax sees all params.
if model.config.tie_word_embeddings:
    model.config.tie_word_embeddings = False
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Loaded {MODEL_NAME} ({n_params:.0f}M params)")

# Collect all stop token IDs from generation config.
eos_ids = model.generation_config.eos_token_id
if isinstance(eos_ids, int):
    eos_ids = [eos_ids]
stop_token_ids = set(eos_ids)
print(f"Stop tokens: {stop_token_ids}")

# ===================================================================
# Step 2: Convert to JAX via torchax + StaticCache
# ===================================================================
print(f"\n--- Step 2: Converting to JAX via torchax ---")

# Register StaticCache as JAX pytree
def _flatten_static_cache(cache):
    return (cache.key_cache, cache.value_cache), (
        getattr(cache, 'max_cache_len', MAX_SEQ),
        getattr(cache, 'max_batch_size', 1),
    )

def _unflatten_static_cache(aux, children):
    cache = object.__new__(cache_utils.StaticCache)
    cache.key_cache = list(children[0])
    cache.value_cache = list(children[1])
    cache.max_cache_len, cache.max_batch_size = aux
    return cache

register_pytree_node(
    cache_utils.StaticCache,
    _flatten_static_cache,
    _unflatten_static_cache,
)

# Move model to torchax ('jax' device)
env = torchax.default_env()
with env:
    model.to('jax')

    # Create StaticCache with fixed shape
    cache = cache_utils.StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=MAX_SEQ,
        device='jax',
        dtype=torch.bfloat16,
    )

    model_weights = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())

print(f"Model on JAX device. StaticCache allocated: "
      f"K[0].shape={cache.key_cache[0].shape}")

# ===================================================================
# Step 3: Define decode_one via torch.func.functional_call
# ===================================================================
# A single forward step: takes token + cache_position, returns logits + updated cache.
# Works for both prefill (seq_len tokens) and decode (1 token) since
# StaticCache has fixed shape — only cache_position changes.

def decode_one(weights, buffers, input_ids, cache_position, past_key_values):
    logits, updated_cache = torch.func.functional_call(
        model,
        (weights, buffers),
        (input_ids,),
        dict(
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True,
        ),
    )
    return logits, updated_cache

# jax_view converts torch callable -> JAX callable (handles torchax tensor conversion)
jax_decode = torchax.interop.jax_view(decode_one)
jitted_decode = torchax.interop.jax_jit(decode_one)

# ===================================================================
# Step 4: Eager-mode validation (CPU via torchax)
# ===================================================================
print(f"\n--- Step 4: Eager-mode validation ---")

messages = [{"role": "user", "content": PROMPT}]
prompt_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
)
SEQ_LEN = prompt_ids.shape[1]
print(f"Prompt tokens: {SEQ_LEN}")

if SEQ_LEN > MAX_SEQ:
    raise ValueError(f"Prompt length {SEQ_LEN} exceeds MAX_SEQ {MAX_SEQ}")

with env:
    prompt_jax = prompt_ids.to('jax')
    cache_position = torch.arange(SEQ_LEN, device='jax')

    # Prefill
    t0 = time.perf_counter()
    logits, kv = jitted_decode(
        model_weights, model_buffers, prompt_jax, cache_position, cache
    )
    t1 = time.perf_counter()
    print(f"Prefill  : {t1-t0:.3f}s  logits.shape={logits.shape}")

    # Greedy autoregressive decode
    next_token_id = int(logits[0, -1].argmax())
    generated = [next_token_id]

    for step in range(MAX_NEW_TOKENS - 1):
        cur_token = torch.tensor([[next_token_id]], device='jax')
        pos = torch.tensor([SEQ_LEN + step], device='jax')
        logits, kv = jitted_decode(
            model_weights, model_buffers, cur_token, pos, kv
        )
        next_token_id = int(logits[0, -1].argmax())
        generated.append(next_token_id)
        if next_token_id in stop_token_ids:
            break

    t2 = time.perf_counter()

text = tokenizer.decode(generated, skip_special_tokens=True)
print(f"Decode   : {len(generated)} tokens in {t2-t1:.3f}s")
print(f"Output   : {text}")

# ===================================================================
# Step 5: Export to StableHLO via jax.export
# ===================================================================
print(f"\n--- Step 5: Exporting to StableHLO ---")

# torchax tensors wrap JAX arrays in ._elem; extract the underlying JAX array.
def _unwrap_torchax(x):
    return x._elem if hasattr(x, '_elem') else x

def _to_jax_shape(x):
    j = _unwrap_torchax(x)
    return jax.ShapeDtypeStruct(j.shape, j.dtype)

weights_shapes = jax.tree.map(_to_jax_shape, model_weights)
buffers_shapes = jax.tree.map(_to_jax_shape, model_buffers)
kv_shapes = jax.tree.map(_to_jax_shape, cache)

jitted_jax_decode = jax.jit(jax_decode)

def export_mlir(name, seq_len, filename):
    """Export a single StableHLO module and write to file."""
    shapes = (
        weights_shapes,
        buffers_shapes,
        jax.ShapeDtypeStruct((1, seq_len), jnp.int32),
        jax.ShapeDtypeStruct((seq_len,), jnp.int32),
        kv_shapes,
    )
    exported = jax_export.export(jitted_jax_decode)(*shapes)
    mlir = str(exported.mlir_module())
    with open(filename, "w") as f:
        f.write(mlir)
    print(f"{name:9s}: {len(mlir):,} chars -> {filename}")
    return mlir

prefill_mlir = export_mlir("Prefill", SEQ_LEN, "qwen_prefill.mlir")
decode_mlir = export_mlir("Decode", 1, "qwen_decode.mlir")

# ===================================================================
# Step 6: Compile to IREE and load
# ===================================================================
import approx_runtime as ar
import iree.runtime as ireert

if BACKEND != "cuda":
    raise ValueError(f"Unsupported backend {BACKEND!r}; this example is CUDA-only")
iree_backend = "cuda"
runtime_backend = "cuda"


def compile_and_load(mlir_text, name, backend, rt_backend):
    """Compile StableHLO -> IREE vmfb and load."""
    t_start = time.perf_counter()
    vmfb = ar.compile_to_iree(
        mlir_text,
        backend=backend,
        input_type="stablehlo",
        extra_args=IREE_EXTRA_ARGS,
    )
    t_compile = time.perf_counter()
    modules, device = ar.load_module(vmfb, backend=rt_backend)
    t_load = time.perf_counter()
    print(f"  {name}: compile {t_compile-t_start:.1f}s, "
          f"load {t_load-t_compile:.1f}s, "
          f"{len(vmfb):,} bytes, device={device} (backend={backend})")
    return modules, device


print(f"\n--- Step 6: Compiling with IREE (backend={iree_backend}) ---")
prefill_mod, _ = compile_and_load(prefill_mlir, "prefill", iree_backend, runtime_backend)
decode_mod, decode_device = compile_and_load(
    decode_mlir, "decode", iree_backend, runtime_backend
)

# Free MLIR strings after compilation
prefill_mlir = decode_mlir = None

# ===================================================================
# Step 7: Run full inference via IREE-compiled modules
# ===================================================================
print(f"\n--- Step 7: IREE inference ---")

iree_prefill = prefill_mod.jit_call_torch["main"]
iree_decode = decode_mod.jit_call_torch["main"]

# Flatten all inputs to numpy arrays in pytree order.
weights_flat, _ = jax.tree.flatten(jax.tree.map(_unwrap_torchax, model_weights))
buffers_flat, _ = jax.tree.flatten(jax.tree.map(_unwrap_torchax, model_buffers))
cache_flat, _ = jax.tree.flatten(jax.tree.map(_unwrap_torchax, cache))

weights_np = [np.asarray(w) for w in weights_flat]
buffers_np = [np.asarray(b) for b in buffers_flat]
cache_np = [np.asarray(c) for c in cache_flat]

input_ids_np = np.asarray(prompt_ids, dtype=np.int32)
cache_pos_np = np.arange(SEQ_LEN, dtype=np.int32)

# -- IREE Prefill (warmup + timed) --
all_prefill_inputs = weights_np + buffers_np + [input_ids_np, cache_pos_np] + cache_np
_ = iree_prefill(*all_prefill_inputs)  # warmup
del _

t0 = time.perf_counter()
prefill_out = iree_prefill(*all_prefill_inputs)
t1 = time.perf_counter()

# Output structure: (logits, kv_cache_leaves...)
iree_logits = np.asarray(prefill_out[0].to_host())
print(f"Prefill  : {t1-t0:.4f}s  logits.shape={iree_logits.shape}")

first_tok = int(np.argmax(iree_logits[0, -1, :]))
print(f"First tok: '{tokenizer.decode([first_tok])}'")

# -- IREE Decode loop --
# StaticCache has fixed shape — only the token and cache_position change.
kv_buffers = [prefill_out[i] for i in range(1, len(prefill_out))]
iree_generated = [first_tok]
static_inputs_np = weights_np + buffers_np  # constant across steps

t_static_upload_start = time.perf_counter()
static_inputs = [
    ireert.asdevicearray(decode_device, x) for x in static_inputs_np
]
t_static_upload_end = time.perf_counter()
print(f"Static inputs uploaded to device in "
      f"{t_static_upload_end-t_static_upload_start:.4f}s")

t_decode_start = time.perf_counter()
for step in range(MAX_NEW_TOKENS - 1):
    next_tok_np = np.array([[iree_generated[-1]]], dtype=np.int32)
    pos_np = np.array([SEQ_LEN + step], dtype=np.int32)

    decode_out = iree_decode(*(static_inputs + [next_tok_np, pos_np] + kv_buffers))

    tok = int(np.argmax(np.asarray(decode_out[0].to_host())[0, -1, :]))
    iree_generated.append(tok)
    kv_buffers = [decode_out[i] for i in range(1, len(decode_out))]

    if tok in stop_token_ids:
        break

t_decode_end = time.perf_counter()

iree_text = tokenizer.decode(iree_generated, skip_special_tokens=True)
print(f"Decode   : {len(iree_generated)} tokens in "
      f"{t_decode_end-t_decode_start:.4f}s")
print(f"Output   : {iree_text}")

# ===================================================================
# Summary
# ===================================================================
print(f"\n--- Summary ---")
print(f"Eager output : {text}")
print(f"IREE  output : {iree_text}")
print("Prefill MLIR : OK")
print("Decode  MLIR : OK")
print("Prefill IREE : OK")
print("Decode  IREE : OK")

print("\n" + "=" * 70)
print("Done!")
