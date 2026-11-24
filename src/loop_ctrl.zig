// control(cur, msg, loop, ctrl) — injection cross-passe d'un message dans une entrée
// de couche loopée (inkling2.cpp graph::control) :
//   gate = rmsNorm(cur) ++ rmsNorm(msg) ; g = ctrl_down(gate)
//   gain = tanh(ctrl_gain(g)), forget = tanh(ctrl_forget(g))
//   return cur - forget*cur + gain*msg
const std = @import("std");
const zml = @import("zml");
const inkling2 = @import("inkling2.zig");

pub fn control(cur: zml.Tensor, msg: zml.Tensor, loop: inkling2.LoopCtrl, ctrl: usize) zml.Tensor {
    const cn = zml.nn.rmsNorm(cur.convert(.f32), .d, 1e-6);
    const mn = zml.nn.rmsNorm(msg.convert(.f32), .d, 1e-6);
    const gate = zml.Tensor.concatenate(&.{ cn, mn }, .d); // {.s, .d2}
    const g = loop.down[ctrl].forward(gate, .f32); // {.s, .rank}
    const gain = loop.gain[ctrl].forward(g, .f32).tanh(); // {.s, .d}
    const forget = loop.forget[ctrl].forward(g, .f32).tanh(); // {.s, .d}
    return cur.sub(forget.mul(cur)).add(gain.mul(msg));
}
