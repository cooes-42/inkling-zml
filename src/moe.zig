// MoE Inkling2 : router F32 + biais, top-8, sigmoid × route_scale, experts NVFP4→bf16,
// activation SiTU(g, u) = β·tanh(g/β)·sigmoid(g)·β₂·tanh(u/β₂) (β=4, β₂=25).
// v1 : MoE manuel en stablehlo (pas de SiTU dans zml.moe, décision D10).
const std = @import("std");
const zml = @import("zml");
const inkling2 = @import("inkling2.zig");

/// x {.s, .d} -> {.s, .d}
pub fn forward(
    x: zml.Tensor,
    layer: inkling2.Layer,
    cfg: inkling2.Config,
) zml.Tensor {
    // logits router {.s, .expert} f32
    const logits = layer.router.forward(x, .f32).add(layer.router_bias.broad(zml.Shape.init(.{ .s = x.dim(.s), .expert = cfg.n_experts }, .f32)));
    const routing = logits.topK(.{ .topk = .expert }, cfg.top_k, .{});
    const ids = routing.indices.convert(.i32); // {.s, .topk}
    const scores = routing.values.sigmoid().scale(cfg.route_scale); // {.s, .topk}

    // experts gather: gate_up {.expert, .dout, .d} -> {.s, .topk, .dout, .d}
    const gate_up = layer.gate_up_exps.weight.gather(.{ .expert = ids }, .{});
    const down = layer.down_exps.weight.gather(.{ .expert = ids }, .{}); // {.s, .topk, .d, .dout}

    // h = gate_up(x) : {.s, .topk, .dout}
    const x_in = x.insertAxes(.d, .{.topk}).broad(zml.Shape.init(.{ .s = x.dim(.s), .topk = cfg.top_k, .d = x.dim(.d) }, x.dtype()));
    const h = gate_up.dot(x_in, .d).convert(.f32); // {.s, .topk, .dout}

    // SiTU sur la paire (gate, up) concaténée dans .dout = [gate | up]
    const ff = cfg.ff_exp;
    const g = h.slice(.dout, .{ .start = 0, .end = ff }); // {.s, .topk, .dout=ff}
    const u = h.slice(.dout, .{ .start = ff, .end = 2 * ff });
    const act = g
        .scale(1.0 / cfg.situ_beta).tanh().scale(cfg.situ_beta)
        .mul(g.sigmoid())
        .mul(u.scale(1.0 / cfg.situ_linear_beta).tanh().scale(cfg.situ_linear_beta)); // {.s, .topk, .dout=ff}

    // down : act {.s, .topk, .dout=ff} x down {.s, .topk, .d, .dout} -> {.s, .topk, .d}
    const out_e = down.dot(act, .dout); // {.s, .topk, .d}

    // pondération par les scores et somme sur topk -> {.s, .d}
    const weighted = out_e.mul(scores.broad(out_e.shape()));
    return weighted.sum(.topk).convert(x.dtype());
}
