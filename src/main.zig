// inkling <model.gguf> --tokens "id id id ..." [--dump-logits <file>]
// v2 : layer-wise. Les 48 visites sont exécutées une par une ; les poids d'une
// couche physique (~2,2 Go déquantifiés en bf16) ne sont chargés que pendant la
// visite (le modèle complet fait 52 Go > RAM). Un exécutable compilé par variante
// de visite (is_global x new_block x has_query x has_loop, 7 variantes max), les
// poids sont re-bakés à chaque visite. Le tokenizer n'est pas branché (ids en CLI).
const std = @import("std");
const zml = @import("zml");
const gguf_store = @import("gguf_store.zig");
const inkling2 = @import("inkling2.zig");

const VisitExe = zml.FnExe(inkling2.visitForward);
const EmbedExe = zml.FnExe(inkling2.embedFn);
const MsgsExe = zml.FnExe(inkling2.msgsFn);
const OutputExe = zml.FnExe(inkling2.outputForward);

/// Clé de variante compilée : bit0 is_global, bit1 new_block, bit2 has_loop, bit3 !has_query.
fn variantKey(cfg: inkling2.Config, phys: u32, has_loop: bool, has_query: bool) u4 {
    var k: u4 = 0;
    if (cfg.isGlobal(phys)) k |= 1;
    if (phys % cfg.attn_res_block == 0) k |= 2;
    if (has_loop) k |= 4;
    if (!has_query) k |= 8;
    return k;
}

/// Ne free que si le buffer n'a pas été donatéd (et donc déjà libéré par le runner).
fn maybeDeinit(exe: *const zml.Exe, flat: usize, buf: zml.Buffer) void {
    for (exe.donated_input_indices) |d| {
        if (@as(usize, d) == flat) return;
    }
    var b = buf;
    b.deinit();
}

/// Libère les poids baked (préfixe des args) non donatés.
fn freeWeights(exe: *const zml.Exe, wbufs: *zml.Bufferized(inkling2.VisitWeights)) void {
    const n = zml.meta.count(zml.Buffer, wbufs);
    var ptrs: [512]*zml.Buffer = undefined;
    zml.meta.forEachVisit(wbufs, *zml.Buffer, struct {
        fn call(i: usize, b: *zml.Buffer, acc: *[512]*zml.Buffer) void {
            acc[i] = b;
        }
    }.call, .{&ptrs});
    for (ptrs[0..n], 0..) |p, i| maybeDeinit(exe, i, p.*);
}

/// Indices flat des entrées runtime de VisitIn (après le préfixe baked layer+loop).
const Idx = struct {
    cur: usize,
    pos: usize,
    token_index: usize,
    kv_k: usize,
    kv_v: usize,
    kv_li: usize,
    sc_state: usize,
    sc_li: usize,
    blocks: usize,
    n_blocks: usize,
    msgs: usize,
};

fn flatIdx(b: usize) Idx {
    return .{
        .cur = b + 0,
        .pos = b + 1,
        .token_index = b + 2,
        .kv_k = b + 3,
        .kv_v = b + 4,
        .kv_li = b + 5,
        .sc_state = b + 6,
        .sc_li = b + 7,
        .blocks = b + 8,
        .n_blocks = b + 9,
        .msgs = b + 10,
    };
}

const Ctx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    gf: *const gguf_store.GgufFile,
    cfg: inkling2.Config,
    mesh: *const zml.Sharding.Mesh,
    s: i64,
    pos_t: zml.Tensor,
    idx_t: zml.Tensor,
    cur_t: zml.Tensor,
    blocks_t: zml.Tensor,
    nblocks_t: zml.Tensor,
    msgs_t: zml.Tensor,
    visit_exes: *[16]?VisitExe,
};

/// Compile (si besoin) la variante puis exécute une visite avec les poids de `phys` chargés à la volée.
fn runVisit(
    ctx: *const Ctx,
    visit: u32,
    phys: u32,
    has_loop: bool,
    msgs_buf: ?zml.Buffer,
    free_old_cur: bool,
    cur_buf: *zml.Buffer,
    pos_buf: zml.Buffer,
    idx_buf: zml.Buffer,
    kv_buf: *zml.Bufferized(inkling2.KvCache),
    sc_buf: *zml.Bufferized(inkling2.SconvState),
    blocks_buf: *zml.Buffer,
    nblocks_buf: *zml.Buffer,
) !void {
    const cfg = ctx.cfg;
    const blk_view = ctx.store.view().withPrefix("blk").withLayer(phys);
    const has_query = blk_view.hasKey("attn_res_query.weight");
    const key = variantKey(cfg, phys, has_loop, has_query);

    if (ctx.visit_exes[key] == null) {
        std.debug.print("compile visit variant key={d} (phys={d} loop={} query={})...\n", .{ key, phys, has_loop, has_query });
        const sample = inkling2.VisitIn{
            .layer = try inkling2.Layer.init(blk_view, cfg, phys),
            .loop = if (has_loop) inkling2.LoopCtrl.init(blk_view) else null,
            .cur = ctx.cur_t,
            .pos = ctx.pos_t,
            .token_index = ctx.idx_t,
            .kv = inkling2.KvCache.init(cfg, ctx.s, .bf16),
            .sconv = inkling2.SconvState.init(cfg, .bf16),
            .blocks = ctx.blocks_t,
            .n_blocks = ctx.nblocks_t,
            .msgs = if (has_loop) ctx.msgs_t else null,
            .new_block = phys % cfg.attn_res_block == 0,
            .cfg = cfg,
        };
        ctx.visit_exes[key] = try VisitExe.compile(ctx.allocator, ctx.io, ctx.platform, .{ .meshes = &.{ctx.mesh} }, .{sample});
        std.debug.print("  ok\n", .{});
    }
    const exe: *const VisitExe = &ctx.visit_exes[key].?;

    // poids de la couche physique, chargés depuis le GGUF mmapé
    var w = inkling2.VisitWeights{
        .layer = try inkling2.Layer.init(blk_view, cfg, phys),
        .loop = if (has_loop) inkling2.LoopCtrl.init(blk_view) else null,
    };
    var wbufs = try zml.mem.bufferize(ctx.allocator, inkling2.VisitWeights, &w);
    try gguf_store.loadAll(ctx.allocator, ctx.io, ctx.platform, inkling2.VisitWeights, &w, &wbufs, ctx.store, ctx.gf);

    var runner = try VisitExe.Runner(.{ .layer, .loop }).init(exe, ctx.allocator, .{
        .layer = wbufs.layer,
        .loop = wbufs.loop,
    });
    defer runner.deinit(ctx.allocator);

    const li_kv = try zml.Buffer.scalar(ctx.io, ctx.platform, visit, .u32);
    const li_sc = try zml.Buffer.scalar(ctx.io, ctx.platform, visit, .u32);

    const b: usize = zml.meta.count(zml.Buffer, &wbufs);
    const idx = flatIdx(b);
    const old_cur = cur_buf.*;
    const old_kv = kv_buf.*;
    const old_sc = sc_buf.*;
    const old_blocks = blocks_buf.*;
    const old_nb = nblocks_buf.*;

    runner.run(ctx.io, .{
        .inputs = .{
            .cur = cur_buf.*,
            .pos = pos_buf,
            .token_index = idx_buf,
            .kv = .{ .k = kv_buf.k, .v = kv_buf.v, .layer_index = li_kv },
            .sconv = .{ .state = sc_buf.state, .layer_index = li_sc },
            .blocks = blocks_buf.*,
            .n_blocks = nblocks_buf.*,
            .msgs = msgs_buf,
        },
        .outputs = .{
            .cur = cur_buf,
            .kv = kv_buf,
            .sconv = sc_buf,
            .blocks = blocks_buf,
            .n_blocks = nblocks_buf,
        },
    }, .{ .wait = true });

    const raw = &exe.raw;
    if (free_old_cur) maybeDeinit(raw, idx.cur, old_cur);
    maybeDeinit(raw, idx.kv_k, old_kv.k);
    maybeDeinit(raw, idx.kv_v, old_kv.v);
    maybeDeinit(raw, idx.kv_li, old_kv.layer_index);
    maybeDeinit(raw, idx.sc_state, old_sc.state);
    maybeDeinit(raw, idx.sc_li, old_sc.layer_index);
    maybeDeinit(raw, idx.blocks, old_blocks);
    maybeDeinit(raw, idx.n_blocks, old_nb);
    maybeDeinit(raw, idx.kv_li, li_kv);
    maybeDeinit(raw, idx.sc_li, li_sc);
    if (msgs_buf) |mb| maybeDeinit(raw, idx.msgs, mb);
    freeWeights(raw, &wbufs);
}

const MsgQ = struct {
    msg_query: zml.Tensor, // {.ctrl=4, .d} f32
};

fn loadMsgQuery(ctx: *const Ctx, phys: u32) !zml.Buffer {
    const v = ctx.store.view().withPrefix("blk").withLayer(phys);
    var w = MsgQ{ .msg_query = v.createTensor("loop_msg_query.weight", .{ .ctrl, .d }, .replicated, .replicated) };
    var bufs = try zml.mem.bufferize(ctx.allocator, MsgQ, &w);
    try gguf_store.loadAll(ctx.allocator, ctx.io, ctx.platform, MsgQ, &w, &bufs, ctx.store, ctx.gf);
    return bufs.msg_query;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();
    const path = args.next() orelse {
        std.debug.print("usage: inkling <model.gguf> --tokens \"1 2 3\" [--dump-logits f.bin]\n", .{});
        return error.Usage;
    };
    var tokens_str: ?[]const u8 = null;
    var dump_path: ?[]const u8 = null;
    var loop_count: ?u32 = null;
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--tokens")) tokens_str = args.next();
        if (std.mem.eql(u8, a, "--dump-logits")) dump_path = args.next();
        if (std.mem.eql(u8, a, "--loop-count")) {
            if (args.next()) |v| loop_count = try std.fmt.parseInt(u32, v, 10);
        }
    }
    if (tokens_str == null) {
        std.debug.print("--tokens requis\n", .{});
        return error.Usage;
    }

    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(allocator);
    {
        var it = std.mem.splitScalar(u8, tokens_str.?, ' ');
        while (it.next()) |t| {
            if (t.len == 0) continue;
            try tokens.append(allocator, try std.fmt.parseInt(u32, t, 10));
        }
    }
    const seqlen: usize = tokens.items.len;
    const s: i64 = @intCast(seqlen);
    std.debug.print("seqlen={d}\n", .{seqlen});

    // GGUF -> store
    var gf = try gguf_store.GgufFile.open(allocator, init.io, path);
    defer gf.deinit(allocator);
    var reg = try gguf_store.registryFromGguf(allocator, &gf);
    defer reg.deinit();
    var store = zml.io.TensorStore.fromRegistry(allocator, &reg, &.{});

    var cfg = inkling2.Config.fromGguf(&gf.header, .{});
    if (loop_count) |lc| cfg = cfg.withLoopCount(lc);
    std.debug.print("config: layers={d} loop=[{d},{d})x{d} n_visits={d} d={d} voc={d}\n", .{
        cfg.n_layers_phys, cfg.loop_start, cfg.loop_end, cfg.loop_count, cfg.n_visits, cfg.d, cfg.vocab,
    });

    // 1 device CPU : sinon le mesh réplique chaque buffer x4 (OOM)
    const platform = try zml.Platform.init(allocator, init.io, .cpu, .{ .cpu = .{ .device_count = 1 } });
    const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));

    // prototypes de tenseurs runtime
    const tokens_t = zml.Tensor.init(.{ .s = s }, .u32);
    const pos_t = zml.Tensor.init(.{ .s = s }, .i64);
    const idx_t = zml.Tensor.init(.{ .s = s }, .u32);
    const cur_t = zml.Tensor.init(.{ .s = s, .d = cfg.d }, .bf16);
    const blocks_t = zml.Tensor.init(.{ .nb = inkling2.nb_max, .s = s, .d = cfg.d }, .bf16);
    const nblocks_t = zml.Tensor.init(.{}, .u32);
    const msgs_t = zml.Tensor.init(.{ .q = 4, .s = s, .d = cfg.d }, .f32);

    var visit_exes: [16]?VisitExe = @splat(null);
    defer for (&visit_exes) |*e| if (e.*) |*x| x.deinit();

    var ctx = Ctx{
        .allocator = allocator,
        .io = init.io,
        .platform = platform,
        .store = &store,
        .gf = &gf,
        .cfg = cfg,
        .mesh = mesh,
        .s = s,
        .pos_t = pos_t,
        .idx_t = idx_t,
        .cur_t = cur_t,
        .blocks_t = blocks_t,
        .nblocks_t = nblocks_t,
        .msgs_t = msgs_t,
        .visit_exes = &visit_exes,
    };

    // --- embedding (poids baked) ---
    var root = store.view();
    var emb_in = inkling2.EmbedIn{
        .token_embd = root.createTensor("token_embd.weight", .{ .voc, .d }, .replicated, .replicated),
        .token_embd_norm = root.createTensor("token_embd_norm.weight", .{ .d }, .replicated, .replicated),
        .tokens = tokens_t,
        .cfg = cfg,
    };
    var emb_bufs = try zml.mem.bufferize(allocator, inkling2.EmbedIn, &emb_in);
    std.debug.print("loading token_embd...\n", .{});
    try gguf_store.loadAll(allocator, init.io, platform, inkling2.EmbedIn, &emb_in, &emb_bufs, &store, &gf);
    var embed_exe = try EmbedExe.compile(allocator, init.io, platform, .{ .meshes = &.{mesh} }, .{emb_in});
    defer embed_exe.deinit();
    var embed_runner = try EmbedExe.Runner(.{ .token_embd, .token_embd_norm }).init(&embed_exe, allocator, .{
        .token_embd = emb_bufs.token_embd,
        .token_embd_norm = emb_bufs.token_embd_norm,
    });
    defer embed_runner.deinit(allocator);

    // --- sortie (poids baked) ---
    var out_in = inkling2.OutputIn{
        .output = root.createTensor("output.weight", .{ .voc, .d }, .replicated, .replicated),
        .output_norm = root.createTensor("output_norm.weight", .{ .d }, .replicated, .replicated),
        .output_res_query = root.createTensor("output_res_query.weight", .{ .d }, .replicated, .replicated),
        .blocks = blocks_t,
        .n_blocks = nblocks_t,
        .cur = cur_t,
        .cfg = cfg,
    };
    var out_bufs = try zml.mem.bufferize(allocator, inkling2.OutputIn, &out_in);
    std.debug.print("loading output...\n", .{});
    try gguf_store.loadAll(allocator, init.io, platform, inkling2.OutputIn, &out_in, &out_bufs, &store, &gf);
    var out_exe = try OutputExe.compile(allocator, init.io, platform, .{ .meshes = &.{mesh} }, .{out_in});
    defer out_exe.deinit();
    var out_runner = try OutputExe.Runner(.{ .output, .output_norm, .output_res_query }).init(&out_exe, allocator, .{
        .output = out_bufs.output,
        .output_norm = out_bufs.output_norm,
        .output_res_query = out_bufs.output_res_query,
    });
    defer out_runner.deinit(allocator);

    // --- messages de boucle (exe compilé une fois, poids runtime) ---
    const msgs_in = inkling2.MsgsIn{
        .blocks = blocks_t,
        .n_blocks = nblocks_t,
        .cur = cur_t,
        .msg_query = root.withPrefix("blk").withLayer(cfg.loop_start).createTensor("loop_msg_query.weight", .{ .ctrl, .d }, .replicated, .replicated),
        .cfg = cfg,
    };
    var msgs_exe = try MsgsExe.compile(allocator, init.io, platform, .{ .meshes = &.{mesh} }, .{msgs_in});
    defer msgs_exe.deinit();
    var msgs_runner = try MsgsExe.Runner(.{}).init(&msgs_exe, allocator, .{});
    defer msgs_runner.deinit(allocator);

    // --- buffers d'entrée ---
    const tokens_bytes = try allocator.alloc(u8, seqlen * 4);
    defer allocator.free(tokens_bytes);
    @memcpy(tokens_bytes, std.mem.sliceAsBytes(tokens.items));
    const pos_items = try allocator.alloc(i64, seqlen);
    defer allocator.free(pos_items);
    for (pos_items, 0..) |*p, i| p.* = @intCast(i);
    const pos_bytes = std.mem.sliceAsBytes(pos_items);

    var tokens_buf = try zml.Buffer.fromBytes(init.io, platform, tokens_t.shape(), tokens_bytes);
    defer tokens_buf.deinit();
    var pos_buf = try zml.Buffer.fromBytes(init.io, platform, pos_t.shape(), pos_bytes);
    defer pos_buf.deinit();
    var idx_buf = try zml.Buffer.fromBytes(init.io, platform, idx_t.shape(), tokens_bytes);
    defer idx_buf.deinit();

    // --- état persistant : cur, kv, sconv, blocks ---
    var cur_buf: zml.Buffer = undefined;
    embed_runner.run(init.io, .{
        .inputs = .{ .tokens = tokens_buf },
        .outputs = .{ .cur = &cur_buf },
    }, .{ .wait = true });
    std.debug.print("embedded\n", .{});

    const kv = inkling2.KvCache.init(cfg, s, .bf16);
    var kv_buf = try kv.initBuffer(init.io, platform);
    defer inkling2.KvCache.deinitBuffer(&kv_buf);
    const sconv_state = inkling2.SconvState.init(cfg, .bf16);
    var sc_buf = try sconv_state.initBuffer(init.io, platform);
    defer inkling2.SconvState.deinitBuffer(&sc_buf);

    const blocks_zeros = try allocator.alloc(u8, blocks_t.shape().byteSize());
    defer allocator.free(blocks_zeros);
    @memset(blocks_zeros, 0);
    var blocks_buf = try zml.Buffer.fromBytes(init.io, platform, blocks_t.shape(), blocks_zeros);
    defer blocks_buf.deinit();
    var nblocks_buf = try zml.Buffer.scalar(init.io, platform, 0, .u32);
    defer nblocks_buf.deinit();

    // --- 48 visites ---
    const core = cfg.loop_end - cfg.loop_start;
    const prelude_len = inkling2.preludeBlockCount(cfg);

    for (0..cfg.loop_start) |i| {
        const v: u32 = @intCast(i);
        std.debug.print("visit {d}/{d} phys={d}\n", .{ v, cfg.n_visits, inkling2.visitPhys(cfg, v) });
        try runVisit(&ctx, v, inkling2.visitPhys(cfg, v), false, null, true, &cur_buf, pos_buf, idx_buf, &kv_buf, &sc_buf, &blocks_buf, &nblocks_buf);
    }
    const prelude_buf = cur_buf; // handle du prélude, réutilisé au début de chaque passe > 0

    var msgs_list: [64]?zml.Buffer = @splat(null);
    for (0..cfg.loop_count) |p_usize| {
        const p: u32 = @intCast(p_usize);
        if (p > 0) {
            // msgs de la passe = mix(bank, n_bank, cur_fin_passe_précédente, msg_query[ip])
            for (0..core) |i| {
                const ip: u32 = cfg.loop_start + @as(u32, @intCast(i));
                const mq = try loadMsgQuery(&ctx, ip);
                var out: zml.Buffer = undefined;
                msgs_runner.run(init.io, .{
                    .inputs = .{
                        .blocks = blocks_buf,
                        .n_blocks = nblocks_buf,
                        .cur = cur_buf,
                        .msg_query = mq,
                    },
                    .outputs = .{ .msgs = &out },
                }, .{ .wait = true });
                maybeDeinit(&msgs_exe.raw, 3, mq);
                msgs_list[i] = out;
            }
            // reset : cur = prelude, n_blocks = prelude_len (les blocs restent, masqués)
            cur_buf.deinit();
            cur_buf = prelude_buf;
            nblocks_buf.deinit();
            nblocks_buf = try zml.Buffer.scalar(init.io, platform, prelude_len, .u32);
        }
        var first = true;
        for (0..core) |i| {
            const ip: u32 = cfg.loop_start + @as(u32, @intCast(i));
            const v: u32 = if (p == 0) ip else cfg.n_layers_phys + (p - 1) * core + @as(u32, @intCast(i));
            std.debug.print("visit {d}/{d} phys={d} pass={d}\n", .{ v, cfg.n_visits, inkling2.visitPhys(cfg, v), p });
            try runVisit(&ctx, v, inkling2.visitPhys(cfg, v), p > 0, msgs_list[i], !first, &cur_buf, pos_buf, idx_buf, &kv_buf, &sc_buf, &blocks_buf, &nblocks_buf);
            first = false;
        }
    }

    // coda : visites v = loop_end..n_layers_phys-1 (indice de visite = slot KV), exécutées après les passes
    for (cfg.loop_end..cfg.n_layers_phys) |i| {
        const v: u32 = @intCast(i);
        std.debug.print("visit {d}/{d} phys={d} (coda)\n", .{ v, cfg.n_visits, inkling2.visitPhys(cfg, v) });
        try runVisit(&ctx, v, @intCast(i), false, null, true, &cur_buf, pos_buf, idx_buf, &kv_buf, &sc_buf, &blocks_buf, &nblocks_buf);
    }
    std.debug.print("all visits done\n", .{});

    // --- logits ---
    var logits_buf: zml.Buffer = undefined;
    out_runner.run(init.io, .{
        .inputs = .{
            .blocks = blocks_buf,
            .n_blocks = nblocks_buf,
            .cur = cur_buf,
        },
        .outputs = .{ .logits = &logits_buf },
    }, .{ .wait = true });
    std.debug.print("logits shape {f}\n", .{logits_buf.shape()});

    const sl = try logits_buf.toSliceAlloc(allocator, init.io);
    defer sl.free(allocator);
    const vals = sl.items(f32);
    const n = vals.len;
    std.debug.print("logits len={d}\n", .{n});
    const last = vals[(seqlen - 1) * @as(usize, @intCast(cfg.vocab_unpadded)) ..];
    var top: [5]struct { id: usize, v: f32 } = undefined;
    for (&top, 0..) |*t, i| t.* = .{ .id = i, .v = last[i] };
    for (0..@as(usize, @intCast(cfg.vocab_unpadded))) |i| {
        const v = last[i];
        for (&top) |*t| {
            if (v > t.v) {
                t.* = .{ .id = i, .v = v };
                break;
            }
        }
    }
    for (0..5) |i| {
        for (i + 1..5) |j| {
            if (top[j].v > top[i].v) {
                const tmp = top[i];
                top[i] = top[j];
                top[j] = tmp;
            }
        }
    }
    for (top) |t| std.debug.print("  token {d} logit {d:.6}\n", .{ t.id, t.v });

    if (dump_path) |dp| {
        const f = try std.Io.Dir.createFile(.cwd(), init.io, dp, .{});
        defer f.close(init.io);
        var wbuf: [4096]u8 = undefined;
        var w = f.writer(init.io, &wbuf);
        try w.interface.writeAll(std.mem.sliceAsBytes(vals));
        try w.interface.flush();
        std.debug.print("dumped {d} logits to {s}\n", .{ vals.len, dp });
    }
}
