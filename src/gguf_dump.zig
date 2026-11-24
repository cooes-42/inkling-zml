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
        if (std.mem.eql(u8, name, "--stats")) {
            var total_elems: u64 = 0;
            var total_raw: u64 = 0;
            var it2 = g.tensors.iterator();
            while (it2.next()) |e| {
                total_elems += e.value_ptr.elems();
                total_raw += e.value_ptr.byteSize();
            }
            std.debug.print("tensors={d} elems={d} raw={d} ({d:.1} Go) bf16={d} ({d:.1} Go)\n", .{
                g.tensor_count,
                total_elems,
                total_raw,
                @as(f64, @floatFromInt(total_raw)) / 1e9,
                total_elems * 2,
                @as(f64, @floatFromInt(total_elems * 2)) / 1e9,
            });
            return;
        }
        if (std.mem.startsWith(u8, name, "--grep=")) {
            const pat = name["--grep=".len..];
            var it = g.tensors.iterator();
            while (it.next()) |e| {
                if (std.mem.indexOf(u8, e.key_ptr.*, pat)) |_| {
                    std.debug.print("  {s} dims={any} type={s} off={d} bytes={d}\n", .{
                        e.key_ptr.*, e.value_ptr.dims, e.value_ptr.ggml_type.name(), e.value_ptr.offset, e.value_ptr.byteSize(),
                    });
                }
            }
            return;
        }
        const t = g.tensor(name) orelse {
            std.debug.print("tensor {s} not found\n", .{name});
            return;
        };
        std.debug.print("{s}: dims={any} type={s} offset={d} bytes={d}\n", .{
            name, t.dims, t.ggml_type.name(), t.offset, t.byteSize(),
        });
        return;
    }
    var it = g.tensors.iterator();
    var i: u64 = 0;
    while (it.next()) |e| : (i += 1) {
        if (i >= 12) break;
        std.debug.print("  {s} dims={any} type={s} off={d} bytes={d}\n", .{
            e.key_ptr.*, e.value_ptr.dims, e.value_ptr.ggml_type.name(), e.value_ptr.offset, e.value_ptr.byteSize(),
        });
    }
}
