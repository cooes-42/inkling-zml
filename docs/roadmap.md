# Roadmap

## Phase 1 — Recherche (fait, 2026-11-24)
- Spec d'architecture : `docs/architecture.md`
- Layout GGUF parsé : `docs/gguf-mainline-layout.md`
- Décisions : `docs/decisions.md`

## Phase 2 — Loader GGUF en Zig (fait, 2026-11-24)
1. Parser binaire GGUF v3 : fait (`src/gguf.zig`), validé sur les vrais fichiers
   (`gguf_dump`, 744 tenseurs mainline).
2. `TensorStore` ZML au-dessus du mmap : fait (`src/gguf_store.zig`, `registryFromGguf` +
   `loadAll` déquant bf16 host).
3. Déquant → bf16 : F32/F16/BF16/Q8_0/Q8_K/NVFP4/MXFP4 faits ; K-quants et IQ* restent
   (pour les builds UD racine, D13).
4. Tokenizer : **non branché** — `zml.tokenizer` exclu du shim `src/zml.zig` (sentencepiece
   cassé en C++ modules). v1 : tokens passés en CLI. À faire : parser `tokenizer.json` HF
   nous-mêmes (pre-tokenizer o200k_base + ignore_merges).

## Phase 3 — Model struct ZML (en cours, 2026-11-24)
Écrit et commité (`src/inkling2.zig` + `depth_mix/loop_ctrl/sconv/moe.zig`) :
1. Model : embeddings, 24 couches, loop_ctrl[12], lm_head, res queries — mapping validé
   (`inkling_dump` : 704 tenseurs résolus sur Mainline UD-Q8_K_XL).
2. Graphe 48 visites (prelude 0..5, 3 passes sur 6..17, coda 18..23), poids partagés,
   KV + sconv par visite — écrit (`forward`/`forwardVisit`).
3. Attention Q/K/V + rel + clips L2 + value norm + τ + sinks + masque SWA/global + kq_b
   additif via `zml.nn.sdpa` — écrit.
4. sconv kernel 4 causal + taps dynamiques — écrit (`sconv.zig`).
5. mix over depth (attn_res/ffn_res/output_res) — écrit (`depth_mix.zig`).
6. Boucle mix bank + control LoRA rank 64 — écrit (`loop_ctrl.zig`).
7. MoE manuel top-8 sigmoid ×8 + SiTU — écrit (`moe.zig`).
Reste : test d'intégration mini `//src:inkling2_test` à rendre vert (debug shapes ZML en
cours), puis inférence réelle (voir blocage RAM ci-dessous).

⚠ Blocage RAM (à traiter en phase 4) : poids déquantifiés = 52,1 Go bf16 > 30 Go RAM.
Stratégie à choisir : chargement couche par couche (48 noyaux `forwardVisit`) ou experts
NVFP4 gardés natifs (mmap 22,9 Go) + déquant dans le graphe.

## Phase 4 — Inférence
- KV cache 48 visites × (K,V) 1024 bf16 ; sconv state 48 × 3 × 6656.
- Prefill + decode, sampling (temp/top-p), chat template jinja du GGUF.

## Phase 5 — Validation
- Logits vs llama.cpp mainline (UD-Q8_K_XL) sur prompts fixes ; puis ppl ; puis bench
  tok/s et mémoire vs llama.cpp (mêmes réglages).

## Phase 6 — Optimisation (plus tard)
- Poids quantifiés en mémoire + noyaux NVFP4/MXFP8 ; pipeline vision (mmproj-F16.gguf) ;
  varloop (loop.varying) hors scope v1.
