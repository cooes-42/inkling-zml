// inkling_dump <file.gguf> — ouvre le GGUF, construit le TensorStore ZML, initialise
// le Model Inkling2 et dump les shapes de tous les tenseurs résolus.
// Validation du mapping noms GGUF -> model struct (docs/loader-gguf.md §6).
const std = @import("std");
const zml = @import("zml");
const gguf_store = @import("gguf_store.zig");
const inkling2 = @import("inkling2.zig");

pub fn main(init: std.process.Init) !void {
    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // nom du programme
    const path = args.next() orelse {
        std.debug.print("usage: inkling_dump <file.gguf>\n", .{});
        return error.Usage;
    };

    var gf = try gguf_store.GgufFile.open(init.gpa, init.io, path);
    defer gf.deinit(init.gpa);

    var reg = try gguf_store.registryFromGguf(init.gpa, &gf);
    defer reg.deinit();
    var store = zml.io.TensorStore.fromRegistry(init.gpa, &reg, &.{});

    const cfg = inkling2.Config.fromGguf(&gf.header, .{});
    std.debug.print("config: layers={d} loop=[{d},{d})x{d} -> n_visits={d}\n", .{
        cfg.n_layers_phys, cfg.loop_start, cfg.loop_end, cfg.loop_count, cfg.n_visits,
    });

    const model = try inkling2.Model.init(init.gpa, store.view(), cfg);

    const Printer = struct {
        fn call(_: usize, t: *const zml.Tensor) void {
            std.debug.print("  {f} {s}\n", .{ t.shape(), @tagName(t.dtype()) });
        }
    };
    zml.meta.forEachVisit(&model, *const zml.Tensor, Printer.call, .{});
    std.debug.print("OK: {d} tensors résolus depuis le GGUF\n", .{zml.meta.count(zml.Tensor, &model)});
}
