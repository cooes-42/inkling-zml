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

## Phase 4 — Inférence (FAIT, 2026-11-25)
- CLI `inkling` layer-wise (`src/main.zig`) : 48 visites exécutées une par une, poids d'une
  couche physique (~2,2 Go bf16) chargés à la volée depuis le GGUF mmapé (52 Go > RAM 30 Go).
  7 variantes XLA compilées (is_global x new_block x has_query x has_loop), poids re-bakés
  par visite. KV cache + sconv chaînés entre visites ; msgs de boucle via `msgsFn`.
- Platform CPU 1 device (sinon mesh 4 partitions -> buffers répliqués x4 -> OOM).
- Premier run réel OK : tokens "1 2 3 4" -> logits {s=4, voc=200058} dumpés.

## Phase 5 — Validation (en cours, 2026-11-25)
- Référence : llama.cpp master (commit 29b52d3, arch inkling2_loop) buildé CPU dans
  /home/nico/llama-build, outil /tmp/dump_logits.cpp (prefill ids bruts, dump logits dernier
  token). Le build Unsloth préexistant (883286076) ne connaît que l'arch `inkling2`.
- 1er comparatif : écart énorme (r=0.07) -> bugs trouvés et corrigés :
  1. déquant NVFP4/MXFP4 : vrai layout ggml = bloc 64 elems/36 o (4 scales UE4M3 + 32 o
     nibbles, low nibble -> moitié basse), pas 256/144 (c1839cc) ;
  2. routage MoE : topk sur sigmoid(logits)+biais, poids = sigmoid(logits) non biaisés x8 ;
  3. couches SWA avec messages : normer chaque entrée Q/K/V séparément ;
  4. clip L2 : facteur = threshold/(||x||+1e-6) clampé [0,1].
- Re-comparaison en cours (run7). GGUF patchés loop.count=2 dispo pour bissection
  (Inkling-2-Mini-UD-Q8_K_XL-loop2.gguf).

## Phase 6 — Optimisation (plus tard)
- LoopCD (arXiv 2610.02185) : contrastive decoding depuis la 1re passe de boucle
  (h' = h_R + ω(h_R − h_1) avant coda), + halving des passes (loop_count 3 -> 2, voire 1)
  = -22..48% FLOPs. `--loop-count N` déjà en CLI. Voir scratchpad handoff.
- Poids quantifiés en mémoire + noyaux NVFP4/MXFP8 ; pipeline vision (mmproj-F16.gguf) ;
  varloop (loop.varying) hors scope v1.

