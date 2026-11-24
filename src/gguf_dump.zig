// gguf_dump <file.gguf> [tensor-name] — inspecte l'en-tête d'un GGUF.
const std = @import("std");
const gguf = @import("gguf.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;

    var it = std.process.Args.Iterator.init(init.minimal.args);
    defer it.deinit();
    _ = it.next(); // nom du programme
    const path = it.next() orelse {
        std.debug.print("usage: gguf_dump <file.gguf> [tensor-name]\n", .{});
        return error.Usage;
    };
    const maybe_name = it.next();

    const file = try std.Io.Dir.openFile(.cwd(), init.io, path, .{ .mode = .read_only });
    defer file.close(init.io);

    // Lit le header en une fois (64 Mo suffisent : le header fait ~12 Mo à cause du vocab embarqué).
    const buf = try a.alloc(u8, 1 << 26);
    defer a.free(buf);
    const n = try file.readPositional(init.io, &.{buf}, 0);
    var g = try gguf.Gguf.parse(a, buf[0..n]);
    defer g.deinit();
    dump(&g, maybe_name);
}

fn dump(g: *const gguf.Gguf, maybe_name: ?[]const u8) void {
    std.debug.print("version={d} tensors={d} kv={d} alignment={d} data_start={d}\n", .{
        g.version, g.tensor_count, g.metadata_kv_count, g.alignment, g.data_start,
    });
    std.debug.print("arch={s} name={s}\n", .{
        g.getKvString("general.architecture") orelse "?",
        g.getKvString("general.name") orelse "?",
    });
    if (maybe_name) |name| {
        const t = g.tensor(name) orelse {
            std.debug.print("tensor {s} not found\n", .{name});
            return;
        };
        std.debug.print("{s}: dims={any} type={s} offset={d} bytes={d}\n", .{
            name, t.dims, @tagName(t.ggml_type), t.offset, t.byteSize(),
        });
        return;
    }
    var it = g.tensors.iterator();
    var i: u64 = 0;
    while (it.next()) |e| : (i += 1) {
        if (i >= 12) break;
        std.debug.print("  {s} dims={any} type={s} off={d}\n", .{
            e.key_ptr.*, e.value_ptr.dims, @tagName(e.value_ptr.ggml_type), e.value_ptr.offset,
        });
    }
}
