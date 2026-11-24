// mix(blocks, cur, query) — attention sur la profondeur (inkling2.cpp graph::mix).
// v1 mono-batch : cur {.s, .d}, query {.d} (SWA, 1 requête) ou {.q, .d} (global, 3).
// Retourne {.q, .s, .d}.
const std = @import("std");
const zml = @import("zml");

pub fn mix(blocks: []const zml.Tensor, cur: zml.Tensor, query: zml.Tensor) zml.Tensor {
    const d = cur.dim(.d);
    const s = cur.dim(.s);

    // states: {.d, .nb, .s}, cur en dernier (llama prépend chaque bloc à cur)
    var states = cur.transpose(.{ .d, .s }).reshape(.{ .d = d, .nb = 1, .s = s });
    for (blocks) |blk| {
        const b = blk.transpose(.{ .d, .s }).reshape(.{ .d = d, .nb = 1, .s = s });
        states = zml.Tensor.concatenate(&.{ b, states }, .nb);
    }
    states = zml.nn.rmsNorm(states.convert(.f32), .d, 1e-6);

    // queries: {.d, .q, .s}
    var q = query.convert(.f32);
    if (q.rank() == 1) {
        q = q.reshape(.{ .d = d, .q = 1 });
    } else {
        q = q.transpose(.{ .d, .q });
    }
    const queries = q.broad(zml.Shape.init(.{ .d = d, .q = q.dim(.q), .s = s }, .f32));

    // scores = states^T @ queries -> {.nb, .s, .q}, softmax sur .nb avec échelle 1/sqrt(d)
    var scores = states.dot(queries, .d);
    scores = scores.scale(1.0 / @sqrt(@as(f32, @floatFromInt(d)))).softmax(.nb);

    // out = scores @ states^T -> {.q, .s, .d}
    const states_t = states.transpose(.{ .nb, .d, .s });
    return scores.dot(states_t, .nb).transpose(.{ .q, .s, .d });
}
