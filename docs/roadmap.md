# Roadmap

## Phase 1 — Recherche (fait, 2026-11-24)
- Spec d'architecture : `docs/architecture.md`
- Layout GGUF parsé : `docs/gguf-mainline-layout.md`
- Décisions : `docs/decisions.md`

## Phase 2 — Loader GGUF en Zig (prochaine étape)
1. Parser binaire GGUF v3 : magic/version, KV metadata (tous types 0..12), tensor infos
   (name, n_dims, dims u64, type u32, offset u64), alignement 32 o.
2. `TensorStore` ZML au-dessus du mmap : `createTensor(name, axes)` résout offset/type/dims.
3. Déquant → bf16 : F32/BF16 natifs ; NVFP4 (E4M3 × block-scale E4M3 × expert-scale F32) ;
   Q8_K, Q4_K, IQ* pour les autres quants (v1 : au moins NVFP4 + natifs, le reste ensuite).
4. Tokenizer : `zml.tokenizer` + `tokenizer.json` HF ; pre-tokenizer `inkling2`.

## Phase 3 — Model struct ZML
1. `Inkling2Loop` : embeddings, 24 couches physiques, lm_head, res queries, output.
2. Visite = (couche physique, passe) ; graphe 48 visites, poids partagés, KV + sconv state
   par visite.
3. Attention : Q/K/V + rel (d_rel=16, extent 1024), clips L2 (qk 22.63, rel 8.0), value norm,
   log-scale τ sur globales, sinks, masque SWA 1024 vs global, kq_b additif.
4. sconv kernel 4 causal + taps dynamiques (globales) + residual scalé.
5. attn_res / ffn_res / output_res : `mix` over depth (softmax 1/√H, requêtes apprises).
6. Boucle : `mix` bank + `control` (LoRA rank 64, tanh gain/forget) entre passes.
7. MoE : router F32 + biais, top-8 sigmoid × 8.0, SiTU(4.0, 25.0), experts NVFP4→bf16.

## Phase 4 — Inférence
- KV cache 48 visites × (K,V) 1024 bf16 ; sconv state 48 × 3 × 6656.
- Prefill + decode, sampling (temp/top-p), chat template jinja du GGUF.

## Phase 5 — Validation
- Logits vs llama.cpp mainline (UD-Q8_K_XL) sur prompts fixes ; puis ppl ; puis bench
  tok/s et mémoire vs llama.cpp (mêmes réglages).

## Phase 6 — Optimisation (plus tard)
- Poids quantifiés en mémoire + noyaux NVFP4/MXFP8 ; pipeline vision (mmproj-F16.gguf) ;
  varloop (loop.varying) hors scope v1.
