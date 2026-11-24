# inkling-zml

Run Thinking Machines Lab's **Inkling-2-Mini** (Apache 2.0) on the **ZML** inference
stack, re-implementing the architecture in Zig. Specs and decisions live in
[`docs/`](docs/):

- `docs/architecture.md` — full model spec (from llama.cpp `inkling2*.cpp`,
  HF `config.json`, transformers modeling code, parsed GGUF header)
- `docs/gguf-mainline-layout.md` — GGUF `inkling2_loop` layout of the Unsloth
  Mainline_Support files (24 physical blocks + loop control, 48 logical visits)
- `docs/loader-gguf.md` — Zig GGUF loader spec (parser v3, NVFP4→bf16 dequant,
  tensor name mapping)
- `docs/model-struct-zml.md` — ZML model struct spec (48 visits, SWA/global
  attention, MoE, KV cache, sconv)
- `docs/decisions.md` — architecture decisions log
- `docs/roadmap.md` — phases and validation plan

## Dev environment (Nix)

```bash
nix develop          # CPU dev shell: bazelisk (Bazel 9.3.0), zig, python3, uv, gh
nix develop .#cuda   # NVIDIA GPU shell (CUDA toolkit) for --config=cuda
```

ZML is pinned to a master commit via `git_override` in `MODULE.bazel`
(see `zmlCommit` in `flake.nix` — keep both in sync). Bazel fetches the hermetic
Zig 0.17.0 toolchain itself (rules_zig); the nix `zig` is for the editor/ZLS.

## Build / run (once `src/` exists)

```bash
bazel build --config=cpu   //src:inkling-zml        # CPU
bazel build --config=cuda  //src:inkling-zml        # NVIDIA GPU
bazel run   --config=cpu   //src:inkling-zml -- --model=models/Inkling-2-Mini-UD-Q8_K_XL.gguf --prompt="Hello"
```

## Weights

Download the mainline GGUF (rename `.gguf_file` → `.gguf`):

```bash
mkdir -p models
hf download unsloth/Inkling-2-Mini-GGUF --include "Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file" --local-dir models/
mv models/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file models/Inkling-2-Mini-UD-Q8_K_XL.gguf
```

Reference inference for validation: mainline llama.cpp (PR #32694/#32704, build ≥ b12662)
on the same file.
