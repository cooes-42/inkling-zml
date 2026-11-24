# Loader GGUF en Zig — spec d'implémentation

Scope v1 : `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf` (arch `inkling2_loop`),
744 tenseurs, header ~12 Mo, fichier 22 867 452 896 o. Types présents : F32, BF16, NVFP4.
Tous les poids sont déquantifiés en **bf16 au chargement** (décision D3).

## 1. Format binaire GGUF v3 (little-endian)

```
header:   "GGUF" | version u32 (=3) | tensor_count u64 | metadata_kv_count u64
KV x N:   key: string | value_type u32 | value
tensor:   name: string | n_dims u32 | dims u64[n_dims] (ordre ggml, ne[0]=contigu) | ggml_type u32 | offset u64
string:   len u64 | bytes UTF-8
value_type: 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32 6=f32 7=bool 8=string 9=array 10=u64 11=i64 12=f64
array:    elem_type u32 | len u64 | elem*
ensuite:  padding jusqu'à alignement `general.alignment` (défaut 32) → data_start
tensor i: données à data_start + offset_i (offsets relatifs à data_start, chaque tenseur aligné)
```

## 2. Parser Zig (module `gguf.zig`)

```zignpub const Gguf = struct {
    file: std.Io.File, mmap: []align(32) const u8,
    data_start: u64,                       // offset absolu du début des données
    kv: std.StringArrayHashMapUnmanaged(KvValue),
    tensors: std.StringArrayHashMapUnmanaged(TensorInfo),
    pub const TensorInfo = struct { dims: [4]u64, n_dims: u3, ggml_type: u32, offset: u64 }; // offset relatif à data_start
    pub const KvValue = union(enum) { u8, i8, ..., f32, bool, string, array: []KvValue };
    pub fn open(io, path) !Gguf            // mmap + parse header + KV + tensor infos
    pub fn tensorBytes(self, name) []const u8;   // slice mmap[data_start+off .. +size]
    pub fn ggmlTypeSize(type, ne0) blockSize;   // taille d'un bloc quantifié
};
```

Points de vigilance : lire le header par `readPositional` (pas de seek sur mmap),
`general.alignment` peut surcharger 32 ; valider `general.architecture == "inkling2_loop"`,
`block_count == 24`, `loop.{start,end,count,gate_rank}` ; les dims ggml sont en ordre
inversé vs PyTorch/ZML (ne[0] = dimension de contraction pour les poids 2D).

## 3. Intégration ZML : construire un `safetensors.TensorRegistry` depuis le GGUF

ZML charge via `TensorRegistry` (`zml/core/safetensors.zig`) :
`Tensor { file_uri, name, shape: Shape, offset }`, puis `TensorStore.fromRegistry(allocator, registry, meshes)`.

- Pour chaque tenseur GGUF **natif** (F32/BF16) : `registry.registerTensor(.{
  .file_uri = gguf_path, .name = name, .shape = Shape.init(dims_zml, .f32/.bf16),
  .offset = data_start + t.offset })`. Les dims ZML = dims gguf inversées.
- Pour les tenseurs **quantifiés** (NVFP4) : enregistrer le poids bf16 comme
  `Binding { transformed = true, sources = [blob_nvfp4, scale] }` et le produire dans
  notre `Loader.loadExecute` custom (mécanisme prévu par ZML, cf. `TransformedTensorNotDelivered`)
  en déquantifiant vers bf16. v1 = déquant systématique ; v2 = garder `.nvfp4` natif
  (ZML a déjà `Quantization.Scheme.nvfp4` : weight `.u8`/`.f4e2m1` + scale `.f8e4m3fn`,
  block 16 — nos noyaux pourront s'y brancher).

## 4. Déquant NVFP4 (ggml type 40) → bf16

Layout vérifié sur le fichier (taille exacte 382 205 952 o pour [2304,2304,128]) :
**bloc de 256 valeurs = 144 o** = `block_scales[16] (E4M3)` + `qs[128]` (nibbles E2M1 packés, 2/o,
nibble basse = élément d'indice pair dans le bloc).

```
w[i] = e2m1(qs_nibble[i]) * f32(e4m3(block_scales[i / 16])) * expert_scale[e]
```
`expert_scale` = tenseur F32 séparé `<name>.scale` [128] (un facteur par expert).

- **E2M1** (4 bits : s|e1e0|m) LUT 16 entrées : `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` pour nibble 0..7,
  négatives pour 8..15 (bit 3 = signe ; pas de NaN/Inf en E2M1).
- **E4M3** (FP8, 1 signe + 4 exp + 3 mant, biais 7, pas d'Inf, NaN = 0x7F/0xFF) :
  `e>0 : (-1)^s * 2^(e-7) * (1 + m/8)` ; `e==0 : (-1)^s * 2^-6 * (m/8)` ; max 448.
- Ordre des éléments dans le bloc = ordre ggml (ne[0] contigu).

## 5. Tous les types de quants à supporter (objectif long terme : les 10 builds UD Unsloth)

Fichiers visés (`unsloth/Inkling-2-Mini-GGUF`) : racine `UD-IQ1_M/IQ1_S/IQ2_XXS/IQ3_XXS/
IQ4_XS/Q2_K_XL/Q3_K_XL/Q4_K_M/Q4_K_XL/Q8_K_XL.gguf` + `Mainline_Support/*.gguf_file` +
`mmproj-F16.gguf` (vision, phase ultérieure) + `dspark/` (drafter MTP, hors scope).

Types observés en parsant les en-têtes (UD-Q4_K_XL racine, UD-Q8_K_XL mainline) :
Q8_0, F32, BF16, NVFP4 — et selon les builds : Q4_K, Q3_K, Q2_K, Q5_K, Q6_K, Q8_K,
IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4.

Table de déquantification (à porter depuis `ggml.c` `dequantize_row_*` en Zig) :

| type ggml | id | bloc (o / elems) | formule |
|---|---|---|---|
| F32 / F16 / BF16 | 0/1/30 | natif | — |
| Q4_0 | 2 | 18 / 32 | `w = (nibble - 8) * d` |
| Q5_0 | 6 | 22 / 32 | `w = (q5 - 16) * d` |
| Q8_0 | 8 | 34 / 32 | `w = q * d` (d en f16) — **présent dans UD-Q4_K_XL** |
| Q2_K | 10 | 84 / 256 | 16 sous-blocs, `w = d*sc*q2 - dmin?` (cf. ggml.c) |
| Q3_K | 11 | 110 / 256 | cf. ggml.c |
| Q4_K | 12 | 144 / 256 | 8 sous-blocs de 32 : `w = d*s_lo*q4 - dmin*s_hi` (scales 6+6 bits dans 12 o) |
| Q5_K | 13 | 176 / 256 | cf. ggml.c |
| Q6_K | 14 | 210 / 256 | cf. ggml.c |
| Q8_K | 15 | 292 / 256 | `{ f32 d; i8 qs[256]; i16 bsums[16] }`, `w = q*d` (bsums ignorable) |
| IQ1_S/IQ1_M | 19/29 | ~34-42 / 256 | LUT `kvalues_iq1s`, cf. ggml.c |
| IQ2_XXS/XS/S | 16/17/22 | 42/46/66? / 256 | LUT `kvalues_iq2xs`, cf. ggml-impl.h |
| IQ3_XXS/S | 18/21 | 46?/110 / 256 | LUT, cf. ggml.c |
| IQ4_NL / IQ4_XS | 20/23 | 18?/144 / 256 | cf. ggml.c |
| MXFP4 | 39 | 136 / 256 | 8 scales E8M0 + 128 o nibbles E2M1 |
| NVFP4 | 40 | 144 / 256 | 16 scales E4M3 + 128 o nibbles E2M1 × scale F32 externe (vérifié) |

Règle générale ggml : la quantification court sur `ne[0]` (dimension contiguë) par blocs
complets ; ici toutes les dims pertinentes sont divisibles par 256 (2304 = 9×256,
1152, 1024, 3072, 384…), donc aucun chevauchement de bloc entre lignes.
Priorité v1 : F32, BF16, Q8_0, NVFP4 (présents dans UD-Q8_K_XL mainline) + Q8_K.
Le reste (K-quants, IQ) vient avec les builds UD intermédiaires.

## 6. Mapping noms GGUF → model struct ZML (dims logiques, ordre ggml inversé)

Globaux : `token_embd.weight` [d=2304, vocab=201024] bf16 · `token_embd_norm.weight` [d] f32 ·
`output.weight` [d, vocab] bf16 · `output_norm.weight` [d] f32 · `output_res_query.weight` [d] f32.

Par couche `blk.N` (N ∈ 0..23), préfixe `store.withPrefix("blk.{N}.")` :

| tenseur GGUF | shape (ZML) | dtype | champ |
|---|---|---|---|
| attn_norm.weight (local) | {d} | f32 | attn_norm |
| attn_q/k/v_inp_norm.weight (global) | {d} | f32 | q/k/v_inp_norm |
| attn_q.weight | {dout=h·hd=3072, d} | bf16 | wq |
| attn_k.weight / attn_v.weight | {dout=hkv·hd=1024, d} | bf16 | wk / wv |
| attn_output.weight | {dout=d, d=h·hd} | bf16 | wo |
| attn_gate.weight | {dout=h·hd, d} | bf16 | wqkv_gate |
| attn_sink_proj.weight | {dout=h=24, d} | bf16 | sink_proj |
| attn_rel.weight | {dout=h·d_rel=384, d} | bf16 | wrel |
| attn_rel_bias.weight | {d_rel=16, extent=1024} | f32 | rel_bias |
| attn_rel_clip.weight | {d_rel, h} | f32 | rel_clip |
| attn_q_clip.weight | {hd=128, h} | f32 | q_clip |
| attn_k_conv.weight / attn_v_conv.weight | {kernel=4, d_kv=1024} | f32 | k_conv / v_conv |
| attn_k_conv_proj.weight / attn_v_conv_proj.weight (global) | {d, dout=kernel·hkv=32} | f32 | k/v_conv_proj |
| attn_post_conv.weight / ffn_post_conv.weight | {kernel=4, d} | f32 | attn/ffn_post_conv |
| attn_res_query.weight | {d} ou {d, q=3} (global) | f32 | attn_res_query |
| ffn_res_query.weight | {d} | f32 | ffn_res_query |
| ffn_norm.weight | {d} | f32 | ffn_norm |
| ffn_gate_inp.weight | {expert=128, d} | f32 | router |
| exp_probs_b.bias | {expert} | f32 | router_bias |
| ffn_gate_up_exps.weight + .scale | {d, dout=2·ff=2304, expert} nvfp4 | bf16 déquant | gate_up_exps |
| ffn_down_exps.weight + .scale | {ff=1152, d, expert} nvfp4 | bf16 déquant | down_exps |
| loop_msg_query.weight (N∈6..17) | {d, ctrl=4} | f32 | loop.msg_query |
| loop_{q,k,v,ffn}_ctrl_down.weight | {dout=2·d=4608, rank=64} | f32 | loop.ctrl_down[c] |
| loop_{q,k,v,ffn}_ctrl_gain/forget.weight | {dout=d, rank=64} | f32 | loop.ctrl_gain/forget[c] |

## 7. Tokenizer

- `tokenizer.ggml.pre = "inkling2"` = **pre-tokenizer GPT-4o (o200k_base) + `ignore_merges=true`**
  (llama.cpp `LLAMA_VOCAB_PRE_TYPE_GPT4O`) — pas de regex custom à écrire.
- Vocab embarquée dans le GGUF (`tokenizer.ggml.tokens/merges/token_type`, 200 058 tokens,
  eos/eot=200006, padding=199999, pas de BOS/EOS auto) ; `tokenizer.json` HF aussi dispo.
- Côté ZML : `zml/tokenizer` (BPE) — à brancher sur le tokenizer.json HF, avec le
  pre-split o200k_base + ignore_merges.

## 8. Empreinte mémoire (bf16 déquantifié)

- Poids totaux ≈ 26,2 B params × 2 o ≈ **52 Go** en buffers bf16 (vs 22,9 Go sur disque).
- KV cache : 48 visites × seq × 2 (K,V) × 1024 × 2 o → **6,4 Go à 32K**, 26 Go à 128K.
- État sconv : 48 visites × 3 × 6656 × 2 o ≈ 1,9 Mo (négligeable, par séquence).

## 9. Plan d'implémentation (ordre proposé)

1. `src/gguf.zig` : parser header/KV/tensor-infos + mmap + tests sur le vrai fichier
   (vérifier 744 tenseurs, data_start, quelques tailles de blocs).
2. `src/gguf_store.zig` : construit le `TensorRegistry` (natif direct, NVFP4 en transformed).
3. Déquant NVFP4 → bf16 (LUT E2M1 + decode E4M3) + test unitaire vs valeurs connues.
4. `src/inkling2.zig` : model struct (voir `docs/architecture.md`), `init(store, config)`.
5. Chargement + dump de quelques tenseurs pour validation vs `llama-cli --dump`.
