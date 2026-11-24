# Model struct ZML — Inkling2Loop (spec)

Ancré sur `examples/llm/models/qwen3_5_moe` (ZML master) : `Model.init(store, config)`,
`Buffers = zml.Bufferized(Model)`, compile prefill/decode, `Session` avec
`runPrefill`/`runDecode`, `KvCache` par couche, `zml.attention.attention`, `zml.moe.forwardMoe`.
Spec mathématique : `docs/architecture.md` (source : `inkling2.cpp` / `inkling2-loop.cpp`).

## 1. Organisation des sources

```
src/
├── gguf.zig            # parser GGUF v3 + mmap (docs/loader-gguf.md)
├── gguf_store.zig      # TensorRegistry ZML depuis GGUF + déquant NVFP4→bf16
├── inkling2.zig        # Config, Model, Layer, forward, KvCache, sconv state
├── sconv.zig           # conv1d causale kernel 4 + taps dynamiques + état récurrent
├── depth_mix.zig       # mix(blocks, cur, query) — attention sur la profondeur
├── loop_ctrl.zig       # control(cur, msg) — LoRA rank 64 cross-passe
├── moe.zig             # router top-8 + SiTU + forwardMoe
└── main.zig            # CLI : prefill/decode, chat template, sampling
```

## 2. Config (lu depuis les KV GGUF, cf. gguf-mainline-layout.md)

```zig
pub const Config = struct {
    d: i64 = 2304, vocab: i64 = 201024, vocab_unpadded: i64 = 200058,
    n_layers_phys: u32 = 24,
    loop: struct { start: u32 = 6, end: u32 = 18, count: u32 = 3, gate_rank: u64 = 64 },
    n_visits: u32 = 48,                       // 6 + 12*3 + 6
    h: i64 = 24, hkv: i64 = 8, hd: i64 = 128,
    d_rel: i64 = 16, rel_extent: i64 = 1024,
    swa: i64 = 1024,                          // couches globales = complement de sliding_window_pattern
    global_layers: [4]u32 = .{ 5, 11, 17, 23 },
    sconv_kernel: i64 = 4, attn_res_block: u32 = 6,
    n_experts: i64 = 128, top_k: u32 = 8, ff_exp: i64 = 1152,
    route_scale: f32 = 8.0, situ_beta: f32 = 4.0, situ_linear_beta: f32 = 25.0,
    qk_clip: f32 = 22.627416610717773, rel_clip: f32 = 8.0,
    log_scale_floor: f32 = 8192.0, log_scale_alpha: f32 = 0.1,
    final_softcap: f32 = 30.0, rms_eps: f32 = 1e-6,
    context_length: u32 = 1048576, eos: u32 = 200006,
};
```

## 3. Mapping visite → couche physique

```zig
fn visitPhys(v: u32) u32 { // v in 0..47
    if (v < 24) return v;
    return 6 + (v - 24) % 12;   // couches 6..17 répétées 3x
}
fn visitPass(v: u32) u32 { return if (v < 24) 0 else 1 + (v - 24) / 12; }
```

Le graphe de forward itère sur 48 visites ; les poids viennent de `layers[visitPhys(v)]`,
le KV cache et l'état sconv sont indexés par `v` (48 slots).

## 4. Model / Layer

```zig
pub const Model = struct {
    token_embd: zml.Tensor,        // {vocab, d} bf16
    token_embd_norm: RmsNorm,      // {d}
    layers: [24]Layer,             // poids physiques
    loop_ctrl: [12]LoopCtrl,       // blk.6..17 : msg_query + ctrl down/gain/forget[4]
    output: zml.Tensor,            // {vocab, d} bf16 (lm_head)
    output_norm: RmsNorm,
    output_res_query: zml.Tensor,  // {d}
    pub fn init(store: zml.io.TensorStore.View, cfg: Config) Model // store.withPrefix("blk.{i}.")...
};

pub const Layer = struct {
    // local (SWA) : attn_norm ; global : attn_{q,k,v}_inp_norm + conv_proj
    is_global: bool,
    attn_norm: ?RmsNorm, q_inp_norm: ?RmsNorm, k_inp_norm: ?RmsNorm, v_inp_norm: ?RmsNorm,
    wq: Linear, wk: Linear, wv: Linear, wo: Linear,      // {dout=h*hd,d} / {dout=hkv*hd,d} / {dout=d,d=h*hd}
    wqkv_gate: Linear, sink_proj: Linear,               // {dout=h*hd,d}, {dout=h,d}
    wrel: Linear,                                       // {dout=h*d_rel, d}
    rel_bias: zml.Tensor,                               // {d_rel, extent} f32
    rel_clip: zml.Tensor, q_clip: zml.Tensor,           // {d_rel,h}, {hd,h} f32
    k_conv: zml.Tensor, v_conv: zml.Tensor,             // {kernel, hkv*hd} f32
    k_conv_proj: ?Linear, v_conv_proj: ?Linear,         // global: {dout=kernel*hkv, d} f32
    attn_post_conv: zml.Tensor, ffn_post_conv: zml.Tensor, // {kernel, d} f32
    attn_res_query: zml.Tensor,                         // {d} ou {d, 3} (global)
    ffn_res_query: zml.Tensor,                          // {d}
    ffn_norm: RmsNorm,
    router: Linear, router_bias: zml.Tensor,            // {expert, d} f32, {expert}
    gate_up_exps: zml.Tensor, down_exps: zml.Tensor,    // {d, 2*ff, expert} / {ff, d, expert} bf16 (déquant)
};
```

## 5. Forward d'une visite `v` (pseudo-ZML, fidèle à `build_layer`)

```zig
fn forwardVisit(m: Model, v: u32, cur: Tensor /*{b,s,d}*/, pos: Tensor /*{s}*/,
                kv: *KvCache, sconv_state: *SconvState,
                blocks: *std.ArrayList(Tensor),  // streams des blocs de 6
                bank: ?Bank, msgs: ?[4]Tensor) Tensor {
    const phys = visitPhys(v);
    const L = m.layers[phys];
    const global = L.is_global;

    // 1. attn_res : mix over depth
    var inputs = [3]Tensor{ cur, cur, cur };
    if (v > 0) {
        const mixed = depth_mix.mix(blocks.items, cur, L.attn_res_query); // softmax(1/√d) · concat
        inputs = if (global) .{ slice(mixed,1), slice(mixed,2), slice(mixed,3) } else .{mixed} ** 3;
    }
    // 2. contrôle de boucle (passes 2/3)
    if (msgs) |ms| for (0..3) |j| inputs[j] = loop_ctrl.control(inputs[j], ms[j], m.loop_ctrl[phys-6], j);
    // 3. norms
    if (!global) { inputs[0] = L.attn_norm.f(inputs[0]); inputs[1]=inputs[0]; inputs[2]=inputs[0]; }
    else { inputs[0]=L.q_inp_norm.f(inputs[0]); inputs[1]=L.k_inp_norm.f(inputs[1]); inputs[2]=L.v_inp_norm.f(inputs[2]); }

    // 4. projections + sconv + clips
    var Q = L.wq.f(inputs[0]);                       // {s, h, hd}
    var K = sconv.conv(L.wk.f(inputs[1]), L.k_conv, global ? L.k_conv_proj.f(inputs[1]) : null, sconv_state.at(v, 0));
    var V = sconv.conv(L.wv.f(inputs[2]), L.v_conv, global ? L.v_conv_proj.f(inputs[2]) : null, sconv_state.at(v, 1024));
    var R = L.wrel.f(inputs[0]);                     // {s, h, d_rel}
    Q = l2clip(Q * L.q_clip, cfg.qk_clip);
    K = l2clip(K, cfg.qk_clip);
    V = rmsnorm(V, cfg.rms_eps);                     // value norm sans gamma
    R = l2clip(R * L.rel_clip, cfg.rel_clip);
    if (global) { const tau = 1 + cfg.log_scale_alpha * log(max(1, (pos+1)/cfg.log_scale_floor));
                  Q = Q * tau; R = R * tau; }

    // 5. biais relatif + sinks → logits
    const table = pad(L.rel_bias, extent+1) @ R;     // {extent+1, h, s}
    const sinks = L.sink_proj.f(inputs[0]);         // {h, s}
    const kq_b = gather(table - sinks, rel_pos(v, pos)); // {h, k, q} additif

    // 6. attention : softmax(QK/√hd + mask + kq_b, sink=0) · V
    const mask = if (global) causalMask(pos) else swaMask(pos, cfg.swa); // {q, k} additif (-inf hors fenêtre)
    const out = attnWithSink(Q, K, V, kv.at(v), mask + kq_b.broad, sinks, 1/√hd);

    // 7. sortie attention : gate sigmoïde + wo + post_conv
    var a = out * sigmoid(L.wqkv_gate.f(inputs[0]));
    a = sconv.conv(L.wo.f(a), L.attn_post_conv, null, sconv_state.at(v, 2048));

    // 8. residual par bloc de 6
    const inpSA = cur;
    if (v % 6 == 0) blocks.append(inpSA) else cur = inpSA + a;

    // 9. FFN MoE
    const ffn_inp = cur;
    cur = depth_mix.mix(blocks.items, cur, L.ffn_res_query);
    if (msgs) |ms| cur = loop_ctrl.control(cur, ms[3], m.loop_ctrl[phys-6], 3);
    cur = L.ffn_norm.f(cur);
    cur = moe.forward(cur, L.router, L.router_bias, L.gate_up_exps, L.down_exps, cfg); // top-8, sigmoid, ×8, SiTU
    cur = sconv.conv(cur, L.ffn_post_conv, null, sconv_state.at(v, 4352));
    return ffn_inp + cur;
}
```

Boucle principale (fidèle à `inkling2-loop.cpp::graph`) :

```zig
var cur = embed + token_embd_norm;
var blocks: ArrayList(Tensor) = .empty;
for (0..6) |v| cur = forwardVisit(v, cur, ...);            // prelude
var prelude = cur; var prelude_blocks = blocks.clone();
for (0..3) |pass| {
    var bank: ?Bank = null;
    if (pass > 0) { bank = .{ blocks.items[prelude_blocks.len+1..], cur }; cur = prelude; blocks = prelude_blocks.clone(); }
    for (6..18) |ip| {
        const v = if (pass == 0) ip else 24 + (pass-1)*12 + (ip-6);
        const msgs = if (pass > 0) depth_mix.mix(bank.blocks, bank.out, loop_ctrl[ip-6].msg_query) else null;
        cur = forwardVisit(v, cur, ..., msgs);
    }
}
for (18..24) |v| cur = forwardVisit(v, cur, ...);
// sortie : mix(blocks, cur, output_res_query) → output_norm → lm_head → softcap 30 → [:vocab_unpadded]
```

## 6. KV cache et état sconv

```zig
pub const KvCache = struct { // 48 visites, comme SelfAttnCache de qwen3_5_moe
    k: Tensor, v: Tensor,     // {b, layer=48, s=max_seq, h=hkv, hd=128} bf16
    layer_index: Tensor,      // scalaire u32, atLayer(v) / scatterSlices update(k, v, token_index)
};
pub const SconvState = struct { // {b, layer=48, kernel-1=3, channels=6656} bf16
    // offsets : K@0, V@1024, attn_post@2048, ffn_post@4352
};
```

## 7. Primitives ZML disponibles vs à écrire

| Besoin | ZML |
|---|---|
| projections, normes | `zml.nn.Linear`, `RmsNorm` |
| attention causale + masque additif | `zml.nn.sdpa(q,k,v,.{.attn_mask})` / `zml.attention.attention` (masque = additif, cf. sdpa: `attn_weights.add(mask)`) |
| masque SWA / causal custom | à construire en `{q,k}` additif (-inf), gather par `token_index` |
| KV cache | pattern `KvCache` de qwen3_5_moe (`scatterSlices`, `atLayer`) |
| MoE top-k | `zml.moe.forwardMoe(x, ids, scores, gate_up, down, backend, mesh, ...)` |
| SiTU | activation custom dans forwardMoe si supporté, sinon wrapper manuel |
| **sinks** | pas de support sdpa → softmax manuel avec colonne sink (logit 0) |
| **sconv kernel 4 causal + taps dynamiques** | à écrire (`sconv.zig`) : matmul 4 taps + état récurrent |
| **mix over depth** | à écrire (`depth_mix.zig`) : rmsnorm + matmul + softmax + matmul |
| **loop control (LoRA rank 64)** | à écrire : 3 matmuls + tanh |
| L2 clip | `x * min(1, t/||x||)` — ops de base |

## 8. Compilation prefill/decode

Pattern `inference.zig` de qwen3_5_moe : deux noyaux compilés (prefill seqlen / decode 1),
`token_index` scalaire/buffer, `active_length`. Session : tokenizer (`zml.tokenizer`,
pre o200k_base + ignore_merges), chat template jinja du GGUF, sampling.

## 9. Validation (todo 5)

1. Dump logits ZML vs `llama-cli` (mainline ≥ #32704) sur UD-Q8_K_XL, mêmes prompts,
   bf16 des deux côtés : top-10 logits, écart max, KLD.
2. ppl sur un corpus fixe vs llama.cpp.
3. Bench tok/s prefill/decode + mémoire (RSS/VRAM) vs llama.cpp mêmes réglages.
