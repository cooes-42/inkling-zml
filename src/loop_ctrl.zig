// control(cur, msg, loop, ctrl) — injection cross-passe d'un message dans une entrée
// de couche loopée (inkling2.cpp graph::control) :
//   gate = rmsNorm(cur) ++ rmsNorm(msg) ; g = ctrl_down(gate)
//   gain = tanh(ctrl_gain(g)), forget = tanh(ctrl_forget(g))
//   return cur - forget*cur + gain*msg
const std = @import("std");
const zml = @import("zml");
const inkling2 = @import("inkling2.zig");

pub fn control(cur: zml.Tensor, msg: zml.Tensor, loop: inkling2.LoopCtrl, ctrl: usize) zml.Tensor {
    const cur32 = cur.convert(.f32);
    const msg32 = msg.convert(.f32);
    const cn = zml.nn.rmsNorm(cur32, .d, 1e-6);
    const mn = zml.nn.rmsNorm(msg32, .d, 1e-6);
    const gate = zml.Tensor.concatenate(&.{ cn, mn }, .d).rename(.{ .d = .d2 }); // {.s, .d2=2d}
    const g = loop.down[ctrl].forward(gate, .f32); // {.s, .rank}
    const gain = loop.gain[ctrl].forward(g, .f32).tanh(); // {.s, .d}
    const forget = loop.forget[ctrl].forward(g, .f32).tanh(); // {.s, .d}
    return cur32.sub(forget.mul(cur32)).add(gain.mul(msg32)).convert(cur.dtype());
}
