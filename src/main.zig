// inkling <model.gguf> --tokens "id id id ..." [--dump-logits <file>] [--seqlen N]
// v1 : prefill seul, logits du dernier token dumpés (validation vs llama.cpp, phase 5).
// Le tokenizer n'est pas branché (les ids sont passés en CLI).
const std = @import("std");
const zml = @import("zml");
const gguf_store = @import("gguf_store.zig");
const inkling2 = @import("inkling2.zig");

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
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--tokens")) tokens_str = args.next();
        if (std.mem.eql(u8, a, "--dump-logits")) dump_path = args.next();
    }
    if (tokens_str == null) {
        std.debug.print("--tokens requis\n", .{});
        return error.Usage;
    }

    // parse tokens
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
    std.debug.print("seqlen={d}\n", .{seqlen});

    // GGUF -> store
    var gf = try gguf_store.GgufFile.open(allocator, init.io, path);
    defer gf.deinit(allocator);
    var reg = try gguf_store.registryFromGguf(allocator, &gf);
    defer reg.deinit();
    var store = zml.io.TensorStore.fromRegistry(allocator, &reg, &.{});

    const cfg = inkling2.Config.fromGguf(&gf.header, .{});

    // platform CPU + mesh
    const platform = try zml.Platform.init(allocator, init.io, .cpu, .{});
    const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));

    // modèle + poids
    const model = try inkling2.Model.init(allocator, store.view(), cfg);
    var buffers = try zml.mem.bufferize(allocator, inkling2.Model, &model);
    std.debug.print("loading weights...\n", .{});
    try gguf_store.loadAll(allocator, init.io, platform, inkling2.Model, &model, &buffers, &store, &gf);
    std.debug.print("weights loaded\n", .{});

    // KV cache + sconv state (48 visites)
    const kv = inkling2.KvCache.init(cfg, @intCast(seqlen), .bf16);
    const sconv_state = inkling2.SconvState.init(cfg, .bf16);
    var kv_buf = try kv.initBuffer(init.io, platform);
    defer inkling2.KvCache.deinitBuffer(&kv_buf);
    var sconv_buf = try sconv_state.initBuffer(init.io, platform);
    defer inkling2.SconvState.deinitBuffer(&sconv_buf);

    // compile prefill
    const tokens_t = zml.Tensor.init(.{ .s = @as(i64, @intCast(seqlen)) }, .u32);
    const pos_t = zml.Tensor.init(.{ .s = @as(i64, @intCast(seqlen)) }, .i64);
    std.debug.print("compiling prefill...\n", .{});
    var exe = try zml.FnExe(inkling2.forwardFn).compile(allocator, init.io, platform, .{
        .meshes = &.{mesh},
    }, .{
        .{
            .m = model,
            .tokens = tokens_t,
            .pos = pos_t,
            .token_index = zml.Tensor.init(.{ .s = @as(i64, @intCast(seqlen)) }, .u32),
            .kv = kv,
            .sconv = sconv_state,
        },
    });
    defer exe.deinit();
    std.debug.print("compiled\n", .{});

    var runner = try zml.FnExe(inkling2.forwardFn).Runner(.{ .m, .kv, .sconv }).init(&exe, allocator, .{
        .m = buffers,
        .kv = kv_buf,
        .sconv = sconv_buf,
    });
    defer runner.deinit(allocator);

    // buffers d'entrée
    const tokens_bytes = try allocator.alloc(u8, seqlen * 4);
    defer allocator.free(tokens_bytes);
    @memcpy(tokens_bytes, std.mem.sliceAsBytes(tokens.items));
    const pos_items = try allocator.alloc(i64, seqlen);
    defer allocator.free(pos_items);
    for (pos_items, 0..) |*p, i| p.* = @intCast(i);
    const pos_bytes = std.mem.sliceAsBytes(pos_items);
    const idx_bytes = tokens_bytes; // token_index = pos en prefill depuis 0

    var tokens_buf = try zml.Buffer.fromBytes(init.io, platform, tokens_t.shape(), tokens_bytes);
    defer tokens_buf.deinit();
    var pos_buf = try zml.Buffer.fromBytes(init.io, platform, pos_t.shape(), pos_bytes);
    defer pos_buf.deinit();
    var idx_buf = try zml.Buffer.fromBytes(init.io, platform, tokens_t.shape(), idx_bytes);
    defer idx_buf.deinit();

    var logits_buf: zml.Buffer = undefined;
    var out_kv = kv_buf;
    var out_sconv = sconv_buf;
    runner.run(init.io, .{
        .inputs = .{
            .tokens = tokens_buf,
            .pos = pos_buf,
            .token_index = idx_buf,
        },
        .outputs = .{
            .logits = &logits_buf,
            .kv = &out_kv,
            .sconv = &out_sconv,
        },
    }, .{});
    std.debug.print("ran\n", .{});

    // logits du dernier token : {s, voc} f32 -> top 5
    const logits_shape = logits_buf.shape();
    std.debug.print("logits shape {f}\n", .{logits_shape});
    const n = @divExact(logits_shape.byteSize(), 4);
    const vals = try logits_buf.getValue([]f32, init.io);
    defer allocator.free(vals);
    std.debug.print("logits len={d}\n", .{n});
    const last = vals[(seqlen - 1) * @as(usize, @intCast(cfg.vocab_unpadded)) ..];
    var top: [5]struct { id: usize, v: f32 } = undefined;
    for (&top, 0..) |*t, i| t.* = .{ .id = i, .v = last[i] };
    for (0..@as(usize, @intCast(cfg.vocab_unpadded))) |i| {
        const v = last[i];
        for (&top) |*t| {
            if (v > t.v) {
                const tmp = t.*;
                t.* = .{ .id = i, .v = v };
                // shift down
                _ = tmp;
                break;
            }
        }
    }
    // tri simple du top 5
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
