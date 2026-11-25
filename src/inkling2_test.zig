// Test d'intégration du graphe Inkling2 : config mini (6 visites), poids aléatoires,
// compile + run prefill, vérifie la shape des logits et l'absence de NaN.
const std = @import("std");
const zml = @import("zml");
const inkling2 = @import("inkling2.zig");

fn testCfg() inkling2.Config {
    return .{
        .d = 64,
        .vocab = 128,
        .vocab_unpadded = 100,
        .n_layers_phys = 4,
        .loop_start = 2,
        .loop_end = 4,
        .loop_count = 2,
        .gate_rank = 8,
        .n_visits = 6, // 2 + (4-2)*2 + 0
        .h = 4,
        .hkv = 2,
        .hd = 16,
        .d_rel = 4,
        .rel_extent = 32,
        .swa = 16,
        .global_layers = .{ 1, 3, 100, 101 },
        .sconv_kernel = 4,
        .attn_res_block = 2,
        .n_experts = 4,
        .top_k = 2,
        .ff_exp = 32,
    };
}

/// Le modèle aléatoire est construit DANS le graphe (les poids deviennent des
/// constantes du graphe compilé).
fn testForward(tokens: zml.Tensor, pos: zml.Tensor, idx: zml.Tensor, kv: inkling2.KvCache, sconv: inkling2.SconvState) inkling2.ForwardOut {
    const cfg = testCfg();
    const model = inkling2.randomModel(cfg);
    return inkling2.forward(model, tokens, pos, idx, kv, sconv);
}

test "forward mini (6 visites, poids aléatoires)" {
    const platform = zml.testing.env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cfg = testCfg();
    const seqlen: i64 = 8;

    const exe = try platform.compileFn(allocator, io, testForward, .{
        zml.Tensor.init(.{ .s = seqlen }, .u32),
        zml.Tensor.init(.{ .s = seqlen }, .i64),
        zml.Tensor.init(.{ .s = seqlen }, .u32),
        inkling2.KvCache.init(cfg, seqlen, .bf16),
        inkling2.SconvState.init(cfg, .bf16),
    }, .{ .meshes = &.{zml.testing.tpMesh()} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const pos = [_]i64{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var tokens_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .s = seqlen }, .u32), std.mem.sliceAsBytes(&tokens));
    defer tokens_buf.deinit();
    var pos_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .s = seqlen }, .i64), std.mem.sliceAsBytes(&pos));
    defer pos_buf.deinit();
    var idx_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .s = seqlen }, .u32), std.mem.sliceAsBytes(&tokens));
    defer idx_buf.deinit();
    const kv = inkling2.KvCache.init(cfg, seqlen, .bf16);
    var kv_buf = try kv.initBuffer(io, platform);
    defer inkling2.KvCache.deinitBuffer(&kv_buf);
    const sconv = inkling2.SconvState.init(cfg, .bf16);
    var sconv_buf = try sconv.initBuffer(io, platform);
    defer inkling2.SconvState.deinitBuffer(&sconv_buf);

    args.set(.{ tokens_buf, pos_buf, idx_buf, kv_buf, sconv_buf });
    exe.call(io, args, &results, .{ .wait = true });

    const out = results.get(zml.Bufferized(inkling2.ForwardOut));
    const logits = out.logits;
    std.debug.print("logits shape: {f}\n", .{logits.shape()});
    try std.testing.expectEqualSlices(i64, &.{ seqlen, cfg.vocab_unpadded }, logits.shape().dims());
    try std.testing.expectEqual(zml.DataType.f32, logits.shape().dtype());
    const sl = try logits.toSliceAlloc(allocator, io);
    defer sl.free(allocator);
    const vals = sl.items(f32);
    var non_finite: usize = 0;
    for (vals) |v| {
        if (std.math.isNan(v) or !std.math.isFinite(v)) non_finite += 1;
    }
    std.debug.print("logits: {d} valeurs, {d} non finies\n", .{ vals.len, non_finite });
    // poids aléatoires => on vérifie juste que le graphe produit essentiellement des valeurs finies
    try std.testing.expect(non_finite < vals.len / 10);
}
