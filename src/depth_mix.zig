// mix(blocks, n_blocks, cur, query) — attention sur la profondeur (inkling2.cpp graph::mix).
// v2 : blocks est un tenseur {.nb, .s, .d} zéro-paddé, n_blocks scalaire u32 = nb valides.
// cur {.s, .d} est concaténé comme dernier slot (toujours valide).
// query {.d} (SWA, 1 requête) ou {.q, .d} (global, 3). Retourne {.q, .s, .d}.
const std = @import("std");
const zml = @import("zml");

pub const nb_max: i64 = 16; // capacité max du stack de blocs (48/6+1 = 9 suffit)

pub fn mix(blocks: zml.Tensor, n_blocks: zml.Tensor, cur: zml.Tensor, query: zml.Tensor) zml.Tensor {
    const d = cur.dim(.d);
    const s = cur.dim(.s);
    const nb = blocks.dim(.nb); // nb_max

    // states: {.nb+1, .s, .d} = [blocks..., cur]
    const cur_b = cur.reshape(.{ .nb = 1, .s = s, .d = d });
    var states = zml.Tensor.concatenate(&.{ blocks, cur_b }, .nb).convert(.f32);
    states = zml.nn.rmsNorm(states, .d, 1e-6);

    // queries: {.d, .q, .s}
    var q = query.convert(.f32);
    if (q.rank() == 1) {
        q = q.reshape(.{ .d = d, .q = 1 });
    } else {
        q = q.transpose(.{ .d, .q });
    }
    const queries = q.broad(zml.Shape.init(.{ .d = d, .q = q.dim(.q), .s = s }, .f32));

    // scores = states^T @ queries -> {.nb+1, .s, .q}
    var scores = states.dot(queries, .d);

    // masque: slot i valide si i < n_blocks (blocs) ou i == nb (cur, dernier slot)
    const idx = zml.Tensor.arange(.{ .start = 0, .end = nb + 1 }, .u32).withTags(.{.nb});
    const n_b = n_blocks.convert(.u32).broad(zml.Shape.init(.{ .nb = nb + 1 }, .u32));
    const is_block = idx.cmp(.LT, n_b).convert(.i32);
    const is_cur = idx.cmp(.EQ, zml.Tensor.scalar(@as(u32, @intCast(nb)), .u32).broad(zml.Shape.init(.{ .nb = nb + 1 }, .u32))).convert(.i32);
    const allowed = is_block.add(is_cur).cmp(.GT, zml.Tensor.scalar(0, .i32));
    const mask = zml.Tensor.select(
        allowed,
        zml.Tensor.constant(zml.DataType.f32.zero()).broad(zml.Shape.init(.{ .nb = nb + 1 }, .f32)),
        zml.Tensor.constant(zml.DataType.Value.init(.f32, -std.math.inf(f32))).broad(zml.Shape.init(.{ .nb = nb + 1 }, .f32)),
    );
    scores = scores.scale(1.0 / @sqrt(@as(f32, @floatFromInt(d)))).add(mask.broad(scores.shape())).softmax(.nb);

    // out = scores @ states^T -> {.q, .s, .d}
    const states_t = states.transpose(.{ .nb, .s, .d });
    return scores.dot(states_t, .nb).transpose(.{ .q, .s, .d });
}

/// Construit le tenseur de blocs {.nb=nb_max, .s, .d} zéro-paddé depuis une liste host.
pub fn blocksTensor(items: []const zml.Tensor, s: i64, d: i64, dtype: zml.DataType) zml.Tensor {
    var buf: [nb_max]zml.Tensor = undefined;
    const zeros = zml.Tensor.zeroes(zml.Shape.init(.{ .nb = 1, .s = s, .d = d }, dtype));
    for (&buf, 0..) |*b, i| {
        b.* = if (i < items.len) items[i].reshape(.{ .nb = 1, .s = s, .d = d }) else zeros;
    }
    return zml.Tensor.concatenate(&buf, .nb);
}
