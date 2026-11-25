// sconv : convolution courte causale kernel 4 + taps dynamiques (couches globales),
// fidèle à inkling2.cpp graph::convolve. v1 mono-batch.
//
//   x    = concat(past {.k, .c}, curT {.s, .c}, .t)        -> {.t = k+s, .c}
//   out  = sum_lag x[lag .. lag+s] * weight[:, lag]        -> {.s, .c}
//   dyn  = sum_lag window(x, lag) * tap(lag)               (couches globales)
//   new past = x[s .. s+k]                                 (état récurrent, offset canaux)
//   return (cur + out) / sqrt((dynamic ? 2 : 1)*k + 1)
const std = @import("std");
const zml = @import("zml");
const inkling2 = @import("inkling2.zig");

pub const ConvResult = struct {
    out: zml.Tensor, // {.s, .c}
    state: inkling2.SconvState,
};

/// `cur` {.s, .c}, `weight` {.c, .kernel}, `dynamic` ?{.s, .kernel*.h} (taps dynamiques),
/// état `st` indexé par `st.layer_index` (scalaire u32 runtime), canaux à l'offset `offset`.
pub fn convolve(
    cur: zml.Tensor,
    weight: zml.Tensor,
    dynamic: ?zml.Tensor,
    st: inkling2.SconvState,
    offset: i64,
) ConvResult {
    const kernel = weight.dim(.kernel);
    const c = cur.dim(.c);
    const s = cur.dim(.s);

    var past = st.sliceChannels(offset, c).rename(.{ .mix = .c }); // {.k = kernel-1, .c}
    const cur_t = cur.rename(.{ .s = .t }); // {.t=s, .c}
    past = past.rename(.{ .k = .t }); // {.t = kernel-1, .c}
    const x = zml.Tensor.concatenate(&.{ past, cur_t }, .t); // {.t = kernel-1+s, .c}

    // noyau statique : out[t] = sum_lag x[t+lag] * w[lag]
    var out: zml.Tensor = undefined;
    for (0..@intCast(kernel)) |lag_usize| {
        const lag: i64 = @intCast(lag_usize);
        const win = x.slice(.t, .{ .start = lag, .end = lag + s }).rename(.{ .t = .s }); // {.s, .c}
        const w = weight.slice(.kernel, .{ .start = lag, .end = lag + 1 }).squeeze(.kernel)
            .convert(cur.dtype()); // {.c} (le poids est renommé .c par l'appelant)
        const term = win.mul(w.broad(win.shape()));
        out = if (lag == 0) term else out.add(term);
    }

    // taps dynamiques (couches globales) : dynamic {.s, .kernel * .h}
    if (dynamic) |dyn| {
        const heads = dyn.dim(.h);
        for (0..@intCast(kernel)) |lag_usize| {
            const lag: i64 = @intCast(lag_usize);
            const win = x.slice(.t, .{ .start = kernel - 1 - lag, .end = kernel - 1 - lag + s }).rename(.{ .t = .s }); // {.s, .c}
            const tap = dyn.slice(.kernel, .{ .start = lag, .end = lag + 1 }).squeeze(.kernel).convert(cur.dtype()); // {.s, .h}
            // window {c/heads, heads, s} * tap {s, h}
            const w = win.transpose(.{ .c, .s }).splitAxis(.c, .{ .ch = @divExact(c, heads), .h = heads }); // {.ch, .h, .s}
            const term = w.mul(tap.transpose(.{ .h, .s }).broad(w.shape())).merge(.{ .c = .{ .ch, .h } }); // {.c, .s}
            out = out.add(term.transpose(.{ .s, .c }));
        }
    }

    // nouvel état récurrent : derniers kernel-1 pas de x
    const new_past = x.slice(.t, .{ .start = s, .end = s + kernel - 1 }).rename(.{ .t = .k, .c = .mix }); // {.k, .mix}
    const new_state = st.updateChannels(offset, c, new_past);

    const dyn_factor: i64 = if (dynamic != null) 2 else 1;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dyn_factor * kernel + 1)));
    return .{
        .out = cur.add(out).scale(scale),
        .state = new_state,
    };
}
