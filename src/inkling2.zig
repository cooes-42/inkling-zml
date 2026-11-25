// Model struct ZML pour Inkling-2-Mini (arch GGUF `inkling2_loop`) — voir
// docs/architecture.md et docs/model-struct-zml.md.
// v1 : mono-device, tout replicated, poids déquantifiés en bf16 au chargement (D3).
// Noms de tenseurs vérifiés sur le GGUF réel (gguf_dump --grep=...).
const std = @import("std");
const zml = @import("zml");
const gguf = @import("gguf.zig");

pub const Config = struct {
    d: i64 = 2304,
    vocab: i64 = 201024,
    vocab_unpadded: i64 = 200058,
    n_layers_phys: u32 = 24,
    loop_start: u32 = 6,
    loop_end: u32 = 18,
    loop_count: u32 = 3,
    gate_rank: u64 = 64,
    n_visits: u32 = 48, // 6 + 12*3 + 6
    h: i64 = 24,
    hkv: i64 = 8,
    hd: i64 = 128,
    d_rel: i64 = 16,
    rel_extent: i64 = 1024,
    swa: i64 = 1024,
    global_layers: [4]u32 = .{ 5, 11, 17, 23 },
    sconv_kernel: i64 = 4,
    attn_res_block: u32 = 6,
    n_experts: i64 = 128,
    top_k: u32 = 8,
    ff_exp: i64 = 1152,
    route_scale: f32 = 8.0,
    situ_beta: f32 = 4.0,
    situ_linear_beta: f32 = 25.0,
    qk_clip: f32 = 22.627416610717773,
    rel_clip: f32 = 8.0,
    log_scale_floor: f32 = 8192.0,
    log_scale_alpha: f32 = 0.1,
    final_softcap: f32 = 30.0,
    rms_eps: f32 = 1e-6,
    context_length: u32 = 1048576,
    eos: u32 = 200006,

    pub fn isGlobal(self: Config, layer: u32) bool {
        for (self.global_layers) |g| if (g == layer) return true;
        return false;
    }

    /// Override depuis les KV du GGUF (clés `inkling2_loop.*`), fallback sur les défauts.
    pub fn fromGguf(g: *const gguf.Gguf, base: Config) Config {
        var c = base;
        if (g.getKvU32("inkling2_loop.block_count")) |v| c.n_layers_phys = v;
        if (g.getKvU32("inkling2_loop.loop.start")) |v| c.loop_start = v;
        if (g.getKvU32("inkling2_loop.loop.end")) |v| c.loop_end = v;
        if (g.getKvU32("inkling2_loop.loop.count")) |v| c.loop_count = v;
        c.n_visits = c.loop_start + (c.loop_end - c.loop_start) * c.loop_count + (c.n_layers_phys - c.loop_end);
        return c;
    }
};

/// Visite v (0..n_visits-1) -> couche physique.
pub fn visitPhys(cfg: Config, v: u32) u32 {
    if (v < cfg.n_layers_phys) return v;
    const core = cfg.loop_end - cfg.loop_start;
    return cfg.loop_start + (v - cfg.n_layers_phys) % core;
}

/// Visite v -> passe de boucle (0 = prelude, 1.. = replays du core).
pub fn visitPass(cfg: Config, v: u32) u32 {
    if (v < cfg.n_layers_phys) return 0;
    const core = cfg.loop_end - cfg.loop_start;
    return 1 + (v - cfg.n_layers_phys) / core;
}

pub const Model = struct {
    config: Config,
    token_embd: zml.Tensor, // {.voc, .d} bf16
    token_embd_norm: zml.Tensor, // {.d} f32
    layers: [24]Layer,
    loop_ctrl: [12]LoopCtrl, // blk.6..17
    output: zml.Tensor, // {.voc, .d} bf16 (lm_head)
    output_norm: zml.Tensor, // {.d} f32
    output_res_query: zml.Tensor, // {.d} f32

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !Model {
        _ = allocator;
        var layers: [24]Layer = undefined;
        for (&layers, 0..) |*l, i| {
            l.* = try Layer.init(store.withPrefix("blk").withLayer(i), cfg, @intCast(i));
        }
        var loop_ctrl: [12]LoopCtrl = undefined;
        for (&loop_ctrl, 0..) |*lc, i| {
            lc.* = LoopCtrl.init(store.withPrefix("blk").withLayer(i + cfg.loop_start));
        }
        return .{
            .config = cfg,
            .token_embd = store.createTensor("token_embd.weight", .{ .voc, .d }, .replicated, .replicated),
            .token_embd_norm = store.createTensor("token_embd_norm.weight", .{.d}, .replicated, .replicated),
            .layers = layers,
            .loop_ctrl = loop_ctrl,
            .output = store.createTensor("output.weight", .{ .voc, .d }, .replicated, .replicated),
            .output_norm = store.createTensor("output_norm.weight", .{.d}, .replicated, .replicated),
            .output_res_query = store.createTensor("output_res_query.weight", .{.d}, .replicated, .replicated),
        };
    }
};

/// Contrôle de boucle cross-passe d'une couche du core (blk.6..17), noms llama.cpp
/// mainline : `loop_msg_query` {ctrl=4, d} + 4 contrôles (q, k, v, ffn), chacun
/// LoRA rank 64 : down {rank, 2d} -> gain/forget {d, rank}.
pub const LoopCtrl = struct {
    pub const names = .{ "q", "k", "v", "ffn" }; // LOOP_CTRL_Q, K, V, FFN
    msg_query: zml.Tensor, // .{.ctrl, .d} f32, ggml [d, 4]
    down: [4]zml.nn.Linear, // weight .{.rank, .d2} f32, contract .d2
    gain: [4]zml.nn.Linear, // weight .{.d, .rank} f32, contract .rank
    forget: [4]zml.nn.Linear, // weight .{.d, .rank} f32, contract .rank

    pub fn init(store: zml.io.TensorStore.View) LoopCtrl {
        var down: [4]zml.nn.Linear = undefined;
        var gain: [4]zml.nn.Linear = undefined;
        var forget: [4]zml.nn.Linear = undefined;
        inline for (names, 0..) |n, i| {
            down[i] = .init(store.createTensor("loop_" ++ n ++ "_ctrl_down.weight", .{ .rank, .d2 }, .replicated, .replicated), null, .d2);
            gain[i] = .init(store.createTensor("loop_" ++ n ++ "_ctrl_gain.weight", .{ .d, .rank }, .replicated, .replicated), null, .rank);
            forget[i] = .init(store.createTensor("loop_" ++ n ++ "_ctrl_forget.weight", .{ .d, .rank }, .replicated, .replicated), null, .rank);
        }
        return .{
            .msg_query = store.createTensor("loop_msg_query.weight", .{ .ctrl, .d }, .replicated, .replicated),
            .down = down,
            .gain = gain,
            .forget = forget,
        };
    }
};

pub const Layer = struct {
    is_global: bool,
    // local (SWA) : attn_norm partagé Q/K/V ; global : attn_{q,k,v}_inp_norm séparées
    attn_norm: ?zml.Tensor, // {.d} f32
    q_inp_norm: ?zml.Tensor,
    k_inp_norm: ?zml.Tensor,
    v_inp_norm: ?zml.Tensor,
    wq: zml.nn.Linear, // attn_q.weight {.dout=h*hd, .d}
    wk: zml.nn.Linear, // attn_k.weight {.dout=hkv*hd, .d}
    wv: zml.nn.Linear, // attn_v.weight {.dout=hkv*hd, .d}
    wo: zml.nn.Linear, // attn_output.weight {.dout=d, .d=h*hd}
    wqkv_gate: zml.nn.Linear, // attn_gate.weight {.dout=h*hd, .d} (gate sigmoïde de sortie)
    sink_proj: zml.nn.Linear, // attn_sink_proj.weight {.dout=h, .d}
    wrel: zml.nn.Linear, // attn_rel.weight {.dout=h*d_rel, .d}
    rel_bias: zml.Tensor, // attn_rel_bias.weight, ggml [d_rel, extent] -> ZML {.extent, .d_rel} f32
    rel_clip: zml.Tensor, // attn_rel_clip.weight, ggml [d_rel, h] -> ZML {.h, .d_rel} f32
    q_clip: zml.Tensor, // attn_q_clip.weight, ggml [hd, h] -> ZML {.h, .hd} f32
    k_conv: zml.Tensor, // attn_k_conv.weight, ggml [kernel, hkv*hd] -> ZML {.d_kv, .kernel} f32
    v_conv: zml.Tensor, // attn_v_conv.weight idem
    k_conv_proj: ?zml.nn.Linear, // global: attn_k_conv_proj.weight, ggml [d, kernel*hkv] -> ZML {.dout, .d} f32
    v_conv_proj: ?zml.nn.Linear, // global: attn_v_conv_proj.weight
    attn_post_conv: zml.Tensor, // attn_post_conv.weight, ggml [kernel, d] -> ZML {.d, .kernel} f32
    ffn_post_conv: zml.Tensor, // ffn_post_conv.weight idem
    attn_res_query: ?zml.Tensor, // null pour blk.0 ; {.d} (local) ou ggml [d, 3] -> ZML {.q, .d} (global) f32
    ffn_res_query: zml.Tensor, // {.d} f32
    ffn_norm: zml.Tensor, // {.d} f32
    router: zml.nn.Linear, // ffn_gate_inp.weight, ggml [d, expert] -> ZML {.expert, .d} f32
    router_bias: zml.Tensor, // exp_probs_b.bias {.expert} f32
    gate_up_exps: zml.nn.Linear, // ffn_gate_up_exps.weight, ggml [d, 2ff, expert] -> ZML {.expert, .dout, .d} bf16 (déquant)
    down_exps: zml.nn.Linear, // ffn_down_exps.weight, ggml [ff, d, expert] -> ZML {.expert, .d, .dout} bf16 (déquant)

    pub fn init(store: zml.io.TensorStore.View, cfg: Config, idx: u32) !Layer {
        const global = cfg.isGlobal(idx);
        return .{
            .is_global = global,
            .attn_norm = if (global) null else store.createTensor("attn_norm.weight", .{.d}, .replicated, .replicated),
            .q_inp_norm = if (global) store.createTensor("attn_q_inp_norm.weight", .{.d}, .replicated, .replicated) else null,
            .k_inp_norm = if (global) store.createTensor("attn_k_inp_norm.weight", .{.d}, .replicated, .replicated) else null,
            .v_inp_norm = if (global) store.createTensor("attn_v_inp_norm.weight", .{.d}, .replicated, .replicated) else null,
            .wq = .init(store.createTensor("attn_q.weight", .{ .dout, .d }, .replicated, .replicated), null, .d),
            .wk = .init(store.createTensor("attn_k.weight", .{ .dout, .d }, .replicated, .replicated), null, .d),
            .wv = .init(store.createTensor("attn_v.weight", .{ .dout, .d }, .replicated, .replicated), null, .d),
            .wo = .init(store.createTensor("attn_output.weight", .{ .dout, .d }, .replicated, .replicated), null, .d),
            .wqkv_gate = .init(store.createTensor("attn_gate.weight", .{ .dout, .d }, .replicated, .replicated), null, .d),
            .sink_proj = .init(store.createTensor("attn_sink_proj.weight", .{ .dout, .d }, .replicated, .replicated), null, .d),
            .wrel = .init(store.createTensor("attn_rel.weight", .{ .dout, .d }, .replicated, .replicated), null, .d),
            .rel_bias = store.createTensor("attn_rel_bias.weight", .{ .extent, .d_rel }, .replicated, .replicated),
            .rel_clip = store.createTensor("attn_rel_clip.weight", .{ .h, .d_rel }, .replicated, .replicated),
            .q_clip = store.createTensor("attn_q_clip.weight", .{ .h, .hd }, .replicated, .replicated),
            .k_conv = store.createTensor("attn_k_conv.weight", .{ .d_kv, .kernel }, .replicated, .replicated),
            .v_conv = store.createTensor("attn_v_conv.weight", .{ .d_kv, .kernel }, .replicated, .replicated),
            .k_conv_proj = if (global) .init(store.createTensor("attn_k_conv_proj.weight", .{ .dout, .d }, .replicated, .replicated), null, .d) else null,
            .v_conv_proj = if (global) .init(store.createTensor("attn_v_conv_proj.weight", .{ .dout, .d }, .replicated, .replicated), null, .d) else null,
            .attn_post_conv = store.createTensor("attn_post_conv.weight", .{ .d, .kernel }, .replicated, .replicated),
            .ffn_post_conv = store.createTensor("ffn_post_conv.weight", .{ .d, .kernel }, .replicated, .replicated),
            .attn_res_query = if (store.hasKey("attn_res_query.weight"))
                if (global)
                    store.createTensor("attn_res_query.weight", .{ .q, .d }, .replicated, .replicated)
                else
                    store.createTensor("attn_res_query.weight", .{.d}, .replicated, .replicated)
            else
                null,
            .ffn_res_query = store.createTensor("ffn_res_query.weight", .{.d}, .replicated, .replicated),
            .ffn_norm = store.createTensor("ffn_norm.weight", .{.d}, .replicated, .replicated),
            .router = .init(store.createTensor("ffn_gate_inp.weight", .{ .expert, .d }, .replicated, .replicated), null, .d),
            .router_bias = store.createTensor("exp_probs_b.bias", .{.expert}, .replicated, .replicated),
            .gate_up_exps = .init(store.createTensor("ffn_gate_up_exps.weight", .{ .expert, .dout, .d }, .replicated, .replicated), null, .d),
            .down_exps = .init(store.createTensor("ffn_down_exps.weight", .{ .expert, .d, .dout }, .replicated, .replicated), null, .dout),
        };
    }
};

/// KV cache pour les 48 visites : k/v {.layer=48, .s=max_seq, .h=hkv, .hd} (mono-batch v1).
pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor, // scalaire u32 = visite courante

    pub fn init(cfg: Config, max_seq: i64, dtype: zml.DataType) KvCache {
        const shape = zml.Shape.init(.{
            .layer = @as(i64, cfg.n_visits),
            .s = max_seq,
            .h = cfg.hkv,
            .hd = cfg.hd,
        }, dtype);
        return .{
            .k = zml.Tensor.fromShape(shape),
            .v = zml.Tensor.fromShape(shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        const zeros_k = try std.heap.smp_allocator.alloc(u8, self.k.shape().byteSize());
        defer std.heap.smp_allocator.free(zeros_k);
        @memset(zeros_k, 0);
        const zeros_v = try std.heap.smp_allocator.alloc(u8, self.v.shape().byteSize());
        defer std.heap.smp_allocator.free(zeros_v);
        @memset(zeros_v, 0);
        return .{
            .k = try zml.Buffer.fromBytes(io, platform, self.k.shape(), zeros_k),
            .v = try zml.Buffer.fromBytes(io, platform, self.v.shape(), zeros_v),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
    }

    pub fn atVisit(self: KvCache, v: usize) KvCache {
        return .{ .k = self.k, .v = self.v, .layer_index = zml.Tensor.scalar(v, .u32) };
    }

    pub fn keys(self: KvCache) zml.Tensor {
        return self.k.slice(.layer, .dynSingle(self.layer_index));
    }

    pub fn values(self: KvCache) zml.Tensor {
        return self.v.slice(.layer, .dynSingle(self.layer_index));
    }

    /// Écrit k/v {.s, .h, .hd} à (.layer_index, token_index) et retourne le cache complet.
    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer); // {.s, .h, .hd}
        return .{
            .k = self.k.scatterSlices(
                .{ .layer = self.layer_index, .s = token_index },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = self.layer_index, .s = token_index },
                new_v.convert(self.v.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.v),
            .layer_index = self.layer_index,
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .layer_index = self.layer_index.reuseBuffer(other.layer_index),
        };
    }
};

/// État récurrent sconv par visite : {.layer=48, .k=kernel-1=3, .mix=6656} (mono-batch v1).
/// Canaux : K@0 (1024), V@1024 (1024), attn_post@2048 (2304), ffn_post@4352 (2304).
pub const SconvState = struct {
    pub const channels: i64 = 6656;
    pub const k_off: i64 = 0;
    pub const v_off: i64 = 1024;
    pub const attn_post_off: i64 = 2048;
    pub const ffn_post_off: i64 = 4352;

    state: zml.Tensor, // {.layer, .k, .mix}
    layer_index: zml.Tensor,

    pub fn init(cfg: Config, dtype: zml.DataType) SconvState {
        return .{
            .state = zml.Tensor.fromShape(zml.Shape.init(.{
                .layer = @as(i64, cfg.n_visits),
                .k = cfg.sconv_kernel - 1,
                .mix = channels,
            }, dtype)),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initBuffer(self: SconvState, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(SconvState) {
        const zeros = try std.heap.smp_allocator.alloc(u8, self.state.shape().byteSize());
        defer std.heap.smp_allocator.free(zeros);
        @memset(zeros, 0);
        return .{
            .state = try zml.Buffer.fromBytes(io, platform, self.state.shape(), zeros),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(SconvState)) void {
        self.state.deinit();
        self.layer_index.deinit();
    }

    pub fn atVisit(self: SconvState, v: usize) SconvState {
        return .{ .state = self.state, .layer_index = zml.Tensor.scalar(v, .u32) };
    }

    /// Slice des canaux [off, off+len) de la visite courante : {.k, .len}
    pub fn sliceChannels(self: SconvState, off: i64, len: i64) zml.Tensor {
        return self.state
            .slice(.layer, .dynSingle(self.layer_index))
            .slice(.mix, .{ .start = off, .end = off + len });
    }

    /// Réinjecte les canaux [off, off+len) : scatterSlices sur {.layer, .mix}, new_vals {.k, .len}.
    pub fn updateChannels(self: SconvState, off: i64, len: i64, new_vals: zml.Tensor) SconvState {
        return .{
            .state = self.state.scatterSlices(
                .{ .layer = self.layer_index, .mix = zml.Tensor.arange(.{ .start = off, .end = off + len }, .u32).withTags(.{.mix}) },
                new_vals,
                .{ .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.state),
            .layer_index = self.layer_index,
        };
    }
};

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

const depth_mix = @import("depth_mix.zig");
const loop_ctrl_mod = @import("loop_ctrl.zig");
const sconv = @import("sconv.zig");
const moe = @import("moe.zig");

/// rmsNorm avec poids (comme build_norm RMS de llama.cpp), calcul en f32.
pub fn rmsNormW(x: zml.Tensor, w: zml.Tensor, eps: f32) zml.Tensor {
    return zml.nn.rmsNorm(x.convert(.f32), .d, eps)
        .mul(w.convert(.f32).broad(x.shape()))
        .convert(x.dtype());
}

/// L2-clip : x * min(1, threshold/||x||) sur l'axe donné (inkling2.cpp graph::clip).
fn l2clip(x: zml.Tensor, threshold: f32, axis: anytype) zml.Tensor {
    const n = x.powByConst(2).sum(axis).sqrt();
    const t = zml.Tensor.scalar(threshold, x.dtype());
    const s = t.div(n.maximum(t));
    return x.mul(s.broad(x.shape()));
}

/// Entrée d'une visite (compilable seule pour le chargement couche par couche).
pub const VisitIn = struct {
    layer: Layer,
    loop: ?LoopCtrl, // présent si passe de boucle > 0 sur une couche du core
    cur: zml.Tensor, // {.s, .d}
    pos: zml.Tensor, // {.s} i64
    token_index: zml.Tensor, // {.s} u32
    kv: KvCache,
    sconv: SconvState,
    blocks: zml.Tensor, // {.nb=nb_max, .s, .d} zéro-paddé
    n_blocks: zml.Tensor, // scalaire u32
    msgs: ?zml.Tensor, // {.q=4, .s, .d} si passe de boucle
    visit: u32,
    cfg: Config,
};

pub const VisitOut = struct {
    cur: zml.Tensor,
    kv: KvCache,
    sconv: SconvState,
    blocks: zml.Tensor,
    n_blocks: zml.Tensor,
};

/// Forward d'une visite v, fidèle à inkling2.cpp build_layer.
pub fn visitForward(inp: VisitIn) VisitOut {
    const cfg = inp.cfg;
    const phys = visitPhys(cfg, inp.visit);
    const L = inp.layer;
    const global = L.is_global;
    const h = cfg.h;
    const hkv = cfg.hkv;
    const hd = cfg.hd;
    const cur = inp.cur;
    const s = cur.dim(.s);
    const n_kv = s; // v1 prefill : tous les tokens sont dans le cache

    const inpSA = cur;

    // 1. attn_res : mix over depth (blk.0 n'a pas de query)
    var inputs = [3]zml.Tensor{ cur, cur, cur };
    if (L.attn_res_query) |q| {
        const mixed = depth_mix.mix(inp.blocks, inp.n_blocks, cur, q).convert(cur.dtype()); // {.q, .s, .d}
        if (global) {
            for (0..3) |j| inputs[j] = mixed.slice(.q, .{ .start = @intCast(j), .end = @intCast(j + 1) }).squeeze(.q);
        } else {
            const m0 = mixed.slice(.q, .{ .start = 0, .end = 1 }).squeeze(.q);
            inputs = .{ m0, m0, m0 };
        }
    }

    // 2. contrôle de boucle (passes 1/2) : messages Q, K, V
    if (inp.msgs) |ms| {
        const lc = inp.loop.?;
        for (0..3) |j| {
            const msg = ms.slice(.q, .{ .start = @intCast(j), .end = @intCast(j + 1) }).squeeze(.q);
            inputs[j] = loop_ctrl_mod.control(inputs[j], msg, lc, j);
        }
    }

    // 3. norms
    if (global) {
        inputs[0] = rmsNormW(inputs[0], L.q_inp_norm.?, cfg.rms_eps);
        inputs[1] = rmsNormW(inputs[1], L.k_inp_norm.?, cfg.rms_eps);
        inputs[2] = rmsNormW(inputs[2], L.v_inp_norm.?, cfg.rms_eps);
    } else {
        const n = rmsNormW(inputs[0], L.attn_norm.?, cfg.rms_eps);
        inputs = .{ n, n, n };
    }

    // 4. projections + sconv
    var Q = L.wq.forward(inputs[0], inputs[0].dtype()).splitAxis(.dout, .{ .h = h, .hd = hd }); // {.s, .h, .hd}
    const Kp = L.wk.forward(inputs[1], inputs[1].dtype()).rename(.{ .dout = .c }); // {.s, .c=hkv*hd}
    const Vp = L.wv.forward(inputs[2], inputs[2].dtype()).rename(.{ .dout = .c });
    var R = L.wrel.forward(inputs[0], inputs[0].dtype()).splitAxis(.dout, .{ .h = h, .d_rel = cfg.d_rel }); // {.s, .h, .d_rel}

    const dyn_k: ?zml.Tensor = if (global) blk: {
        const t = L.k_conv_proj.?.forward(inputs[1], .f32); // {.s, .kernel*hkv}
        break :blk t.reshape(.{ .s = s, .kernel = cfg.sconv_kernel, .h = hkv });
    } else null;
    const dyn_v: ?zml.Tensor = if (global) blk: {
        const t = L.v_conv_proj.?.forward(inputs[2], .f32);
        break :blk t.reshape(.{ .s = s, .kernel = cfg.sconv_kernel, .h = hkv });
    } else null;

    const kc = sconv.convolve(Kp, L.k_conv.rename(.{ .d_kv = .c }), dyn_k, inp.sconv, inp.visit, SconvState.k_off);
    const vc = sconv.convolve(Vp, L.v_conv.rename(.{ .d_kv = .c }), dyn_v, kc.state, inp.visit, SconvState.v_off);
    var K = kc.out.splitAxis(.c, .{ .h = hkv, .hd = hd }); // {.s, .h, .hd}
    var V = vc.out.splitAxis(.c, .{ .h = hkv, .hd = hd });

    // 5. clips + value norm + log-scale (globales)
    Q = l2clip(Q.mul(L.q_clip.convert(Q.dtype()).broad(Q.shape())), cfg.qk_clip, .hd);
    K = l2clip(K, cfg.qk_clip, .hd);
    V = zml.nn.rmsNorm(V.convert(.f32), .hd, cfg.rms_eps).convert(V.dtype());
    R = l2clip(R.mul(L.rel_clip.convert(R.dtype()).broad(R.shape())), cfg.rel_clip, .d_rel);
    if (global) {
        const tau = inp.pos.convert(.f32).addConstant(1).scale(1.0 / cfg.log_scale_floor)
            .maximum(zml.Tensor.scalar(1.0, .f32)).log().scale(cfg.log_scale_alpha).addConstant(1); // {.s}
        Q = Q.mul(tau.convert(Q.dtype()).broad(Q.shape()));
        R = R.mul(tau.convert(R.dtype()).broad(R.shape()));
    }

    // 6. KV cache (K/V après sconv + clip/norm) puis biais relatif + sinks
    var kv = inp.kv.atVisit(inp.visit).update(K, V, inp.token_index);
    const Kc = kv.keys().slice(.s, .{ .start = 0, .end = n_kv }).rename(.{ .s = .k }).transpose(.{ .h, .k, .hd }); // {.h, .k, .hd}
    const Vc = kv.values().slice(.s, .{ .start = 0, .end = n_kv }).rename(.{ .s = .k }).transpose(.{ .h, .k, .hd });

    const sinks = L.sink_proj.forward(inputs[0], .f32).rename(.{ .dout = .h }).transpose(.{ .h, .s }); // {.h, .s}
    const extent: i64 = if (global) cfg.rel_extent else cfg.swa;
    // pad high de 1 sur .extent (ggml_pad(rel_bias, 0, 1, 0, 0)) : entrée d'indice `extent` = 0
    const zeros_row = zml.Tensor.constant(zml.DataType.f32.zero()).broad(zml.Shape.init(.{ .extent = 1, .d_rel = cfg.d_rel }, L.rel_bias.dtype()));
    const table = zml.Tensor.concatenate(&.{ L.rel_bias, zeros_row }, .extent) // {.extent+1, .d_rel}
        .dot(R.transpose(.{ .d_rel, .h, .s }).convert(.f32), .d_rel) // {.extent+1, .h, .s}
        .sub(sinks.broad(zml.Shape.init(.{ .extent = extent + 1, .h = h, .s = s }, .f32))) // {.extent+1, .h, .s}
        .transpose(.{ .h, .extent, .s }); // {.h, .extent+1, .s}

    // positions relatives {k, q} clampées à [0, extent]
    const kp = zml.Tensor.arange(.{ .start = 0, .end = n_kv }, .i64).withTags(.{.k});
    const qp = inp.pos.withTags(.{ .q });
    const kq_shape = zml.Shape.init(.{ .k = n_kv, .q = s }, .i64);
    const rel = qp.broad(kq_shape).sub(kp.broad(kq_shape)); // {.k, .q} = pos_q - pos_k
    const rel_pos = rel.clamp(zml.Tensor.scalar(0, .i64), zml.Tensor.scalar(extent, .i64)).convert(.i32);

    // kq_b = table[h, rel_pos[k,q], q] -> gather multi-axes -> {.k, .q, .h} -> {.h, .q, .k}
    const iota_q = zml.Tensor.iota(zml.Shape.init(.{ .q = s }, .i32), .q);
    const kq_b = table.gather(.{ .extent = rel_pos, .s = iota_q }, .{}).transpose(.{ .h, .q, .k }); // {.h, .q, .k}

    // masque causal (+ SWA) additif {.q, .k}
    var allowed = rel.cmp(.GE, zml.Tensor.scalar(0, .i64)); // pos_k <= pos_q
    if (!global) {
        // fenêtre glissante : pos_q - pos_k < swa
        const in_swa = rel.cmp(.LT, zml.Tensor.scalar(cfg.swa, .i64));
        allowed = allowed.convert(.i32).mul(in_swa.convert(.i32)).cmp(.GT, zml.Tensor.scalar(0, .i32));
    }
    const mask = zml.Tensor.select(
        allowed.transpose(.{ .q, .k }),
        zml.Tensor.constant(zml.DataType.f32.zero()).broad(zml.Shape.init(.{ .q = s, .k = n_kv }, .f32)),
        zml.Tensor.constant(zml.DataType.Value.init(.f32, -std.math.inf(f32))).broad(zml.Shape.init(.{ .q = s, .k = n_kv }, .f32)),
    ); // {.q, .k}

    // sdpa splitte q en {h=hkv, hq} : on replie kq_b (par tête Q) en {h=hkv, hq, q, k}
    const kq_b_h = kq_b.splitAxis(.h, .{ .h = hkv, .hq = @divExact(h, hkv) }); // {.h=hkv, .hq, .q, .k}
    // sdpa ajoute le masque dans le dtype de q (avant de passer en f32)
    const attn_mask = mask.broad(kq_b_h.shape()).add(kq_b_h).convert(Q.dtype()); // {.h=hkv, .hq, .q, .k}

    // 7. attention sdpa : softmax(QK/√hd + mask + kq_b) · V
    const attn = zml.nn.sdpa(
        Q.rename(.{ .s = .q }).transpose(.{ .h, .q, .hd }),
        Kc,
        Vc,
        .{ .attn_mask = attn_mask, .scale = zml.Tensor.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .f32) },
    ); // {.h, .q, .hd}

    // 8. sortie attention : gate sigmoïde + wo + sconv post
    var a = attn.rename(.{ .q = .s }).transpose(.{ .s, .h, .hd }).merge(.{ .dout = .{ .h, .hd } }); // {.s, .dout=h*hd}
    a = a.mul(L.wqkv_gate.forward(inputs[0], a.dtype()).sigmoid());
    const wo_out = L.wo.forward(a.rename(.{ .dout = .d }), a.dtype()).rename(.{ .dout = .c }); // {.s, .c=d}
    const ac = sconv.convolve(wo_out, L.attn_post_conv.rename(.{ .d = .c }), null, vc.state, inp.visit, SconvState.attn_post_off);

    // 9. residual par bloc de 6 : nouveau stream si phys % 6 == 0
    var blocks = inp.blocks;
    var n_blocks = inp.n_blocks;
    var cur2: zml.Tensor = undefined;
    if (phys % cfg.attn_res_block == 0) {
        blocks = inp.blocks.scatterSlices(
            .{ .nb = inp.n_blocks },
            inpSA.reshape(.{ .nb = 1, .s = s, .d = cfg.d }),
            .{ .update_fn = zml.Tensor.ScatterOpts.override },
        );
        n_blocks = inp.n_blocks.addConstant(1);
        cur2 = ac.out.rename(.{ .c = .d });
    } else {
        cur2 = inpSA.add(ac.out.rename(.{ .c = .d }));
    }

    // 10. FFN MoE
    const ffn_inp = cur2;
    var f = depth_mix.mix(blocks, n_blocks, cur2, L.ffn_res_query).slice(.q, .{ .start = 0, .end = 1 }).squeeze(.q).convert(cur2.dtype()); // {.s, .d}
    if (inp.msgs) |ms| {
        const msg = ms.slice(.q, .{ .start = 3, .end = 4 }).squeeze(.q);
        f = loop_ctrl_mod.control(f, msg, inp.loop.?, 3);
    }
    f = rmsNormW(f, L.ffn_norm, cfg.rms_eps);
    f = moe.forward(f, L, cfg);
    const fc = sconv.convolve(f.rename(.{ .d = .c }), L.ffn_post_conv.rename(.{ .d = .c }), null, ac.state, inp.visit, SconvState.ffn_post_off);

    return .{
        .cur = ffn_inp.add(fc.out.rename(.{ .c = .d })),
        .kv = kv,
        .sconv = fc.state,
        .blocks = blocks,
        .n_blocks = n_blocks,
    };
}

pub const OutputIn = struct {
    blocks: zml.Tensor,
    n_blocks: zml.Tensor,
    cur: zml.Tensor,
    output: zml.Tensor, // {.voc, .d}
    output_norm: zml.Tensor,
    output_res_query: zml.Tensor,
    cfg: Config,
};

pub const OutputOut = struct {
    logits: zml.Tensor, // {.s, .voc_unpadded} f32
};

/// Sortie modèle : mix(blocks, cur, output_res_query) -> output_norm -> lm_head -> softcap -> unpadded.
pub fn outputForward(inp: OutputIn) OutputOut {
    const cfg = inp.cfg;
    var out = depth_mix.mix(inp.blocks, inp.n_blocks, inp.cur, inp.output_res_query)
        .slice(.q, .{ .start = 0, .end = 1 }).squeeze(.q).convert(inp.cur.dtype()); // {.s, .d}
    out = rmsNormW(out, inp.output_norm, cfg.rms_eps);
    var logits = inp.output.dot(out, .d); // {.voc, .s}
    logits = logits.scale(1.0 / cfg.final_softcap).tanh().scale(cfg.final_softcap); // softcap 30
    logits = logits.slice(.voc, .{ .start = 0, .end = cfg.vocab_unpadded }).convert(.f32).transpose(.{ .s, .voc }); // {.s, .voc}
    return .{ .logits = logits };
}

pub const ForwardOut = struct {
    logits: zml.Tensor, // {.s, .voc_unpadded} f32
    kv: KvCache,
    sconv: SconvState,
    blocks: zml.Tensor,
    n_blocks: zml.Tensor,
};

/// Forward monolithique (tests) : 48 visites + sortie.
pub fn forward(m: Model, tokens: zml.Tensor, pos: zml.Tensor, token_index: zml.Tensor, kv: KvCache, sconv_state: SconvState) ForwardOut {
    const cfg = m.config;
    const s = tokens.dim(.s);

    var cur = m.token_embd.gather(.{ .voc = tokens }, .{}).convert(.bf16); // {.s, .d}
    cur = rmsNormW(cur, m.token_embd_norm, cfg.rms_eps);

    // tracking host des blocs (le graphe est déroulé à la compilation : host_len est connu)
    var host_blocks: [depth_mix.nb_max]zml.Tensor = undefined;
    var host_len: usize = 0;
    var kv_c = kv;
    var sc = sconv_state;

    const run = struct {
        fn call(
            m_: Model,
            v: u32,
            cur_: zml.Tensor,
            pos_: zml.Tensor,
            idx_: zml.Tensor,
            kv_: KvCache,
            sc_: SconvState,
            host_: []const zml.Tensor,
            loop_: ?LoopCtrl,
            msgs_: ?zml.Tensor,
            cfg_: Config,
        ) VisitOut {
            const s_ = cur_.dim(.s);
            return visitForward(.{
                .layer = m_.layers[visitPhys(cfg_, v)],
                .loop = loop_,
                .cur = cur_,
                .pos = pos_,
                .token_index = idx_,
                .kv = kv_,
                .sconv = sc_,
                .blocks = depth_mix.blocksTensor(host_, s_, cfg_.d, .bf16),
                .n_blocks = zml.Tensor.scalar(@as(u32, @intCast(host_.len)), .u32),
                .msgs = msgs_,
                .visit = v,
                .cfg = cfg_,
            });
        }
    }.call;

    const doVisit = struct {
        fn apply(
            m_: Model,
            v: u32,
            cur_: *zml.Tensor,
            kv_: *KvCache,
            sc_: *SconvState,
            host_: []zml.Tensor,
            host_len_: *usize,
            loop_: ?LoopCtrl,
            msgs_: ?zml.Tensor,
            cfg_: Config,
            pos_: zml.Tensor,
            idx_: zml.Tensor,
        ) void {
            if (visitPhys(cfg_, v) % cfg_.attn_res_block == 0) {
                host_[host_len_.*] = cur_.*;
                host_len_.* += 1;
            }
            const o = run(m_, v, cur_.*, pos_, idx_, kv_.*, sc_.*, host_[0..host_len_.*], loop_, msgs_, cfg_);
            cur_.* = o.cur;
            kv_.* = o.kv;
            sc_.* = o.sconv;
        }
    }.apply;

    // prelude : visites 0..loop_start
    for (0..cfg.loop_start) |v| {
        doVisit(m, @intCast(v), &cur, &kv_c, &sc, &host_blocks, &host_len, null, null, cfg, pos, token_index);
    }
    const prelude = cur;
    const prelude_len = host_len;

    // passes sur le core [loop_start, loop_end)
    const core = cfg.loop_end - cfg.loop_start;
    for (0..cfg.loop_count) |pass_usize| {
        const pass: u32 = @intCast(pass_usize);
        var bank_items: []const zml.Tensor = &.{};
        var bank_out = cur;
        if (pass > 0) {
            // bank = blocs après le premier bloc du core + sortie de la passe précédente
            bank_items = host_blocks[prelude_len + 1 .. host_len];
            bank_out = cur;
            cur = prelude;
            host_len = prelude_len;
        }
        for (0..core) |i_usize| {
            const i: u32 = @intCast(i_usize);
            const ip: u32 = cfg.loop_start + i;
            const v: u32 = if (pass == 0) ip else cfg.n_layers_phys + (pass - 1) * core + i;
            const has_msgs = pass > 0;
            const bank_t = depth_mix.blocksTensor(bank_items, s, cfg.d, .bf16);
            const msgs: ?zml.Tensor = if (has_msgs)
                depth_mix.mix(bank_t, zml.Tensor.scalar(@as(u32, @intCast(bank_items.len)), .u32), bank_out, m.loop_ctrl[ip - cfg.loop_start].msg_query.rename(.{ .ctrl = .q }))
            else
                null;
            const lc: ?LoopCtrl = if (has_msgs) m.loop_ctrl[ip - cfg.loop_start] else null;
            doVisit(m, v, &cur, &kv_c, &sc, &host_blocks, &host_len, lc, msgs, cfg, pos, token_index);
        }
    }

    // coda : visites loop_end..n_layers_phys
    for (cfg.loop_end..cfg.n_layers_phys) |v| {
        doVisit(m, @intCast(v), &cur, &kv_c, &sc, &host_blocks, &host_len, null, null, cfg, pos, token_index);
    }

    const out = outputForward(.{
        .blocks = depth_mix.blocksTensor(host_blocks[0..host_len], s, cfg.d, .bf16),
        .n_blocks = zml.Tensor.scalar(@as(u32, @intCast(host_len)), .u32),
        .cur = cur,
        .output = m.output,
        .output_norm = m.output_norm,
        .output_res_query = m.output_res_query,
        .cfg = cfg,
    });
    return .{ .logits = out.logits, .kv = kv_c, .sconv = sc, .blocks = depth_mix.blocksTensor(host_blocks[0..host_len], s, cfg.d, .bf16), .n_blocks = zml.Tensor.scalar(@as(u32, @intCast(host_len)), .u32) };
}

/// Wrapper à un seul argument struct pour zml.FnExe (qui exige une input struct unique).
pub const ForwardInput = struct {
    // champs bufferisés en préfixe (requis par FnExe.Runner)
    m: Model,
    kv: KvCache,
    sconv: SconvState,
    // entrées runtime
    tokens: zml.Tensor, // {.s} u32
    pos: zml.Tensor, // {.s} i64
    token_index: zml.Tensor, // {.s} u32
};

pub fn forwardFn(inp: ForwardInput) ForwardOut {
    return forward(inp.m, inp.tokens, inp.pos, inp.token_index, inp.kv, inp.sconv);
}
// Constructeur aléatoire (tests d'intégration du graphe sans GGUF).
// ---------------------------------------------------------------------------

fn randT(sh: zml.Shape, stddev: f64) zml.Tensor {
    return zml.Tensor.Rng.normal(sh, .{ .mean = 0, .stddev = stddev });
}

pub fn randomLayer(cfg: Config, idx: u32) Layer {
    const global = cfg.isGlobal(idx);
    const d = cfg.d;
    const extent: i64 = if (global) cfg.rel_extent else cfg.swa;
    return .{
        .is_global = global,
        .attn_norm = if (global) null else randT(.init(.{ .d = d }, .f32), 1),
        .q_inp_norm = if (global) randT(.init(.{ .d = d }, .f32), 1) else null,
        .k_inp_norm = if (global) randT(.init(.{ .d = d }, .f32), 1) else null,
        .v_inp_norm = if (global) randT(.init(.{ .d = d }, .f32), 1) else null,
        .wq = .init(randT(.init(.{ .dout = cfg.h * cfg.hd, .d = d }, .bf16), 0.02), null, .d),
        .wk = .init(randT(.init(.{ .dout = cfg.hkv * cfg.hd, .d = d }, .bf16), 0.02), null, .d),
        .wv = .init(randT(.init(.{ .dout = cfg.hkv * cfg.hd, .d = d }, .bf16), 0.02), null, .d),
        .wo = .init(randT(.init(.{ .dout = d, .d = cfg.h * cfg.hd }, .bf16), 0.02), null, .d),
        .wqkv_gate = .init(randT(.init(.{ .dout = cfg.h * cfg.hd, .d = d }, .bf16), 0.02), null, .d),
        .sink_proj = .init(randT(.init(.{ .dout = cfg.h, .d = d }, .bf16), 0.02), null, .d),
        .wrel = .init(randT(.init(.{ .dout = cfg.h * cfg.d_rel, .d = d }, .bf16), 0.02), null, .d),
        .rel_bias = randT(.init(.{ .extent = extent, .d_rel = cfg.d_rel }, .f32), 0.1),
        .rel_clip = randT(.init(.{ .h = cfg.h, .d_rel = cfg.d_rel }, .f32), 1),
        .q_clip = randT(.init(.{ .h = cfg.h, .hd = cfg.hd }, .f32), 1),
        .k_conv = randT(.init(.{ .d_kv = cfg.hkv * cfg.hd, .kernel = cfg.sconv_kernel }, .f32), 0.1),
        .v_conv = randT(.init(.{ .d_kv = cfg.hkv * cfg.hd, .kernel = cfg.sconv_kernel }, .f32), 0.1),
        .k_conv_proj = if (global) .init(randT(.init(.{ .dout = cfg.sconv_kernel * cfg.hkv, .d = d }, .f32), 0.1), null, .d) else null,
        .v_conv_proj = if (global) .init(randT(.init(.{ .dout = cfg.sconv_kernel * cfg.hkv, .d = d }, .f32), 0.1), null, .d) else null,
        .attn_post_conv = randT(.init(.{ .d = d, .kernel = cfg.sconv_kernel }, .f32), 0.1),
        .ffn_post_conv = randT(.init(.{ .d = d, .kernel = cfg.sconv_kernel }, .f32), 0.1),
        .attn_res_query = if (idx == 0) null else if (global)
            randT(.init(.{ .q = 3, .d = d }, .f32), 0.1)
        else
            randT(.init(.{ .d = d }, .f32), 0.1),
        .ffn_res_query = randT(.init(.{ .d = d }, .f32), 0.1),
        .ffn_norm = randT(.init(.{ .d = d }, .f32), 1),
        .router = .init(randT(.init(.{ .expert = cfg.n_experts, .d = d }, .f32), 0.1), null, .d),
        .router_bias = randT(.init(.{ .expert = cfg.n_experts }, .f32), 0.1),
        .gate_up_exps = .init(randT(.init(.{ .expert = cfg.n_experts, .dout = 2 * cfg.ff_exp, .d = d }, .bf16), 0.02), null, .d),
        .down_exps = .init(randT(.init(.{ .expert = cfg.n_experts, .d = d, .dout = cfg.ff_exp }, .bf16), 0.02), null, .dout),
    };
}

pub fn randomLoopCtrl(cfg: Config) LoopCtrl {
    var down: [4]zml.nn.Linear = undefined;
    var gain: [4]zml.nn.Linear = undefined;
    var forget: [4]zml.nn.Linear = undefined;
    inline for (LoopCtrl.names, 0..) |_, i| {
        down[i] = .init(randT(.init(.{ .rank = cfg.gate_rank, .d2 = 2 * cfg.d }, .f32), 0.1), null, .d2);
        gain[i] = .init(randT(.init(.{ .d = cfg.d, .rank = cfg.gate_rank }, .f32), 0.1), null, .rank);
        forget[i] = .init(randT(.init(.{ .d = cfg.d, .rank = cfg.gate_rank }, .f32), 0.1), null, .rank);
    }
    return .{
        .msg_query = randT(.init(.{ .ctrl = 4, .d = cfg.d }, .f32), 0.1),
        .down = down,
        .gain = gain,
        .forget = forget,
    };
}

pub fn randomModel(cfg: Config) Model {
    var layers: [24]Layer = undefined;
    for (&layers, 0..) |*l, i| l.* = randomLayer(cfg, @intCast(i));
    var loop_ctrl: [12]LoopCtrl = undefined;
    for (&loop_ctrl, 0..) |*lc, i| {
        _ = i;
        lc.* = randomLoopCtrl(cfg);
    }
    return .{
        .config = cfg,
        .token_embd = randT(.init(.{ .voc = cfg.vocab, .d = cfg.d }, .bf16), 0.02),
        .token_embd_norm = randT(.init(.{ .d = cfg.d }, .f32), 1),
        .layers = layers,
        .loop_ctrl = loop_ctrl,
        .output = randT(.init(.{ .voc = cfg.vocab, .d = cfg.d }, .bf16), 0.02),
        .output_norm = randT(.init(.{ .d = cfg.d }, .f32), 1),
        .output_res_query = randT(.init(.{ .d = cfg.d }, .f32), 0.1),
    };
}
