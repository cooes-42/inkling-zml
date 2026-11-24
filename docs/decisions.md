# Décisions d'architecture — chat projet, 2026-11-24

## D1 — Source de poids : GGUF Mainline_Support UD-Q8_K_XL

On cible `unsloth/Inkling-2-Mini-GGUF:Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`
(renommé `.gguf`, 22,9 Go), layout `inkling2_loop` lu par llama.cpp mainline (PR #32694/#32704).
Les GGUF racine (48 blocs « déroulés » pour Unsloth Studio) sont exclus : non lisibles par
llama.cpp upstream, et de toute façon le parse montre que même Mainline contient 24 blocs
physiques + contrôle de boucle (voir D2). UD-Q8_K_XL = référence quasi pleine précision
(ppl 7.653 vs 7.651 root, KLD 0.0006 selon le README Unsloth).

## D2 — La boucle n'est PAS un simple déroulement séquentiel de 48 couches

Correction de l'hypothèse de départ (« format déroulé en 48 blocs, pas de logique de bouclage ») :
l'en-tête GGUF contient `block_count=24` + `loop.{start=6,end=18,count=3,gate_rank=64}` et des
tenseurs `blk.{6..17}.loop_*` (msg_query + ctrl down/gain/forget rang 64 pour Q/K/V/FFN).
llama.cpp déroule en 48 **visites** qui partagent les poids physiques, avec un mécanisme de
contrôle cross-passe (`mix` + `control`, voir `inkling2-loop.cpp`). Ignorer ce contrôle
produirait des logits faux sur les passes 2 et 3. **Décision** : on implémente les 48 visites
comme 48 couches du graphe ZML partageant les poids des 12 couches du core, AVEC le
mécanisme `mix`/`control` (c'est ~150 lignes, pas un « gros » travail). Le KV cache et l'état
sconv sont alloués pour 48 visites.

## D3 — Déquantification systématique en bf16 au chargement (v1)

Tous les tenseurs quantisés (NVFP4 experts, et Q8_K/IQ des autres quants si on les supporte
plus tard) sont déquantifiés en bf16 au chargement. Les poids quantifiés ne sont PAS gardés en
mémoire en v1. Noyaux FP4/MXFP8 natifs = phase 2 (performance).

## D4 — Texte seul en v1, vision en phase ultérieure

`mmproj-F16.gguf` (639 Mo) exclu du scope v1. L'embedding d'entrée n'est pas normée quand
l'input vient d'images (voir `build_inputs`), à garder en tête pour plus tard.

## D5 — Pattern ZML retenu (à partir de `examples/llm/models/qwen3_5_moe`)

- `Model.init(allocator, store: zml.io.TensorStore.View, config)` avec `store.withPrefix("blk.N.")`
  → le loader GGUF Zig implémente `TensorStore` (mmap + déquant bf16) au lieu de safetensors.
- Structure : `Model { token_embd, token_embd_norm, layers[24] (poids physiques), output, output_norm, ... }`
  et un graphe de 48 visites : `visit[i] = .{ phys = i<24 ? i : 6 + (i-24) % 12, pass }`.
- Attention : `zml.attention.attention(q, k, v, token_index, ...)` pour le softmax causal de base,
  + masque SWA custom (20 couches) + biais relatif `kq_b` ajouté aux logits + sinks.
  Point ouvert : ZML gère-t-il un biais additif par (head, distance) et des sinks ? Sinon
  on calcule les logits nous-mêmes (q@k^T + kq_b) puis softmax — à vérifier dans `zml/attention`.
- MoE : `zml.moe.forwardMoe(x, topk_ids, scores, gate_up, down, backend, mesh(.experts), ...)`
  avec activation SiTU custom (β·tanh(g/β)·sigmoid(g)·β₂·tanh(u/β₂), β=4, β₂=25) et
  `routing_weight_placement` + scale 8.0 à brancher correctement.
- sconv (kernel 4, causal, + taps dynamiques sur les couches globales) : petit état récurrent
  par visite (3 × 6656 canaux), conv1d custom en Zig/ZML — ZML n'a pas de sconv clé en main
  (lfm2 a du linear attention, à voir si réutilisable).
- Tokenizer : `zml.tokenizer` depuis le `tokenizer.json` HF (pre-tokenizer `inkling2` à
  reimplementer si besoin ; le GGUF embarque aussi tokens/merges).

## D6 — Validation

Comparer les logits ZML vs llama.cpp (mainline, build ≥ #32704) sur les mêmes prompts avec
UD-Q8_K_XL, en bf16 déquantifié des deux côtés. Tolérance : écart relatif < 1e-2 sur les top logits
(premier jalon), puis ppl équivalente. Benchmarks tok/s + mémoire ensuite.

## D8 — Env de dev : Nix flake + Bazel bzlmod, ZML pinné par commit

- `flake.nix` : `devShells.default` (bazelisk + wrapper `bazel`, zig, python3, uv, git, gh,
  curl, jq) et `devShells.cuda` (cudatoolkit/cudnn/nccl, `allowUnfree`) pour `--config=cuda`.
  nixpkgs pinné via `flake.lock` (nixos-unstable 2026-11-21).
- Bazel 9.3.0 (`.bazelversion`, == zml), Zig 0.17.0 hermétique téléchargé par rules_zig
  (le `zig` du shell, 0.17.1, ne sert que pour ZLS/édition).
- `MODULE.bazel` : `bazel_dep(name="zml")` + `git_override` au commit master
  `1470ac33f58ad5795076695a6965abf0f38d4272` ; blocs toolchain/overrides copiés de zml
  (seuls les overrides du module racine s'appliquent en bzlmod), patches dans `third_party/`.
- `.bazelrc` : flags communs repris de zml + configs `--config=cpu|cuda|metal`, `release`, `native`.
- Validé ici : `nix flake show`, `nix develop` démarre, `bazel version` = 9.3.0,
  `bazel query @zml//zml:zml` résout (module graph OK). Le premier build complet
  (LLVM/XLA/Zig) est long : à faire en local.

## D7 — Repo

- Git initialisé, remote `cooes-42/inkling-zml` (public), branche `main`.
- La doc vivante est dans `docs/` (ce chat + ces fichiers) ; le code Zig/Bazel arrivera
  dans `src/` quand le dev local démarre.
