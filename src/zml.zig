// Shim du module "zml" : réexporte les sous-modules de @zml SAUF le tokenizer
// (qui tire sentencepiece, cassé en C++ modules avec notre toolchain — inutile en v1,
// le tokenizer GGUF/HF est géré à part). Évite ainsi de builder sentencepiece.
const core = @import("zml/core");

pub const bazel = @import("bazel");
pub const constants = @import("zml/constants");
pub const meta = @import("zml/meta");
pub const pjrt = @import("pjrt");
pub const stdx = @import("stdx");
pub const tracer = @import("zml/profiling").tracer;

pub const moe = @import("zml/kernels/moe");
pub const nn = @import("zml/nn");
pub const quantization = @import("zml/quantization");
pub const Quantization = quantization.Quantization;

pub const Buffer = core.Buffer;
pub const Compiler = core.Compiler;
pub const dtype = core.dtype;
pub const Data = core.Data;
pub const DataType = core.DataType;
pub const exe = core.exe;
pub const Exe = core.Exe;
pub const FnExe = core.FnExe;
pub const floats = core.floats;
pub const io = core.io;
pub const kernel = core.kernel;
pub const mem = core.mem;
pub const Bufferized = core.Bufferized;
pub const mlir = core.mlir;
pub const ops = core.ops;
pub const pjrtx = core.pjrtx;
pub const platform = core.platform;
pub const Memory = core.Memory;
pub const Platform = core.Platform;
pub const Target = core.Target;
pub const safetensors = core.safetensors;
pub const shape = core.shape;
pub const Shape = core.Shape;
pub const Sharding = core.Sharding;
pub const Mesh = core.Mesh;
pub const slice = core.slice;
pub const Slice = core.Slice;
pub const tensor = core.tensor;
pub const Tensor = core.Tensor;
pub const testing = core.testing;
pub const module = core.module;

pub const KiB = core.KiB;
pub const MiB = core.MiB;
pub const GiB = core.GiB;
pub const TiB = core.TiB;
