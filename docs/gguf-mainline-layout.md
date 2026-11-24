# Layout GGUF Mainline_Support (UD-Q8_K_XL) — résultat du parse d'en-tête

Fichier : `unsloth/Inkling-2-Mini-GGUF` → `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`
(22 867 452 896 o, à renommer en `.gguf`). En-tête parsé sur les 64 premiers Mo (script Python
de parse GGUF v3 : magic `GGUF`, version 3, 744 tenseurs, 59 paires KV, data alignée 32 o).

## Metadata clés

```
general.architecture = inkling2_loop
inkling2_loop.block_count = 24          ← 24 blocs PHYSIQUES, pas 48 déroulés
inkling2_loop.loop.start = 6, loop.end = 18, loop.count = 3, loop.gate_rank = 64, loop.varying = false
inkling2_loop.embedding_length = 2304
inkling2_loop.feed_forward_length = 1152          (dense, indicatif)
inkling2_loop.expert_feed_forward_length = 1152
inkling2_loop.expert_count = 128, expert_used_count = 8
inkling2_loop.attention.head_count[24] = 24, head_count_kv[24] = 8
inkling2_loop.attention.key_length = 128, value_length = 128 (= swa aussi)
inkling2_loop.attention.sliding_window = 1024
inkling2_loop.attention.sliding_window_pattern = [T,T,T,T,T,F, ...]  (F sur 5,11,17,23)
inkling2_loop.shortconv.l_cache = 4
inkling2_loop.attn_res.block_size = 6
inkling2_loop.attention.relative_length = 16, relative_extent = 1024
inkling2_loop.attention.qk_clip = 22.627416610717773, relative_clip = 8.0
inkling2_loop.attention.log_scale_floor = 8192, log_scale_alpha = 0.1
inkling2_loop.activation.situ_beta = 4.0, situ_linear_beta = 25.0
inkling2_loop.expert_weights_scale = 8.0
inkling2_loop.final_logit_softcapping = 30.0
inkling2_loop.vocab_size = 201024, vocab_size_unpadded = 200058
inkling2_loop.context_length = 1048576
inkling2_loop.rope.dimension_count = 0          ← pas de RoPE
general.file_type = 32
tokenizer.ggml.model = gpt2, pre = inkling2, tokens/merges/token_type complets embarqués,
  eos/eot = 200006, padding = 199999, add_bos/add_eos = false, chat_template jinja embarqué
```

## Répartition des types de tenseurs (744)

| Type ggml | Nombre | Rôle |
|---|---|---|
| F32 (0) | 374 | normes, convs sconv, clips q/rel, router (`ffn_gate_inp`, `exp_probs_b`), res queries, loop ctrl |
| BF16 (30) | 330 | embeddings, lm_head, projections q/k/v/o/gate/rel, sink_proj |
| NVFP4 (40) | 40 | poids d'experts `ffn_gate_up_exps` / `ffn_down_exps` (+ `.scale` F32 [128] par tenseur) |

NVFP4 = 4 blocs E4M3 avec scale E4M3 (ggml `GGML_TYPE_NVFP4 = 40`). Chaque expert a un
scale F32 global `[128]` (un par expert). Bloc NVFP4 : 16 poids E4M3 + 1 scale E4M3 (block scale),
le `.scale` tensor est un facteur additionnel par expert (Unsloth Dynamic).

## Nommage

`token_embd.weight`, `token_embd_norm.weight`, `output.weight`, `output_norm.weight`,
`output_res_query.weight`, `blk.N.<champ>.weight|.bias`, `blk.N.ffn_*_exps.weight|.scale`,
`blk.N.loop_*` (N ∈ 6..17). Dims ggml = ordre inversé vs PyTorch (dims[0] = dimension contiguë).

## Points d'attention pour le loader Zig

- Le fichier n'est PAS « déroulé en 48 blocs » : 24 blocs + metadata de boucle + tenseurs
  `loop_*` de contrôle. Le déroulement en 48 visites se fait côté inférence (partage des poids).
- Déquantifier NVFP4 → bf16 au chargement (v1) : E4M3 → f32 → bf16, × block scale E4M3, × expert scale.
- `general.file_type = 32` (UD-Q8_K_XL d'Unsloth) ; les tenseurs non-experts BF16/F32 sont natifs.
- Le tokenizer (BPE gpt2, pre-tokenizer `inkling2`) est embarqué dans le GGUF ; le
  `tokenizer.json` HF existe aussi pour ZML (`zml.tokenizer`).
