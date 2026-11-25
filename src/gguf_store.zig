// Bridge GGUF -> ZML : mmap du fichier, TensorRegistry, chargement host avec
// déquantification bf16 des tenseurs quantifiés (décision D3, v1).
// Voir docs/loader-gguf.md.
const std = @import("std");
const zml = @import("zml");
const gguf = @import("gguf.zig");

pub const GgufFile = struct {
    io: std.Io,
    file: std.Io.File,
    map: []align(std.heap.pageSize()) const u8,
    header: gguf.Gguf,
    path: []const u8,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !GgufFile {
        const file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
        errdefer file.close(io);

        // header : 64 Mo suffisent (le vocab embarqué fait ~12 Mo)
        const buf = try allocator.alloc(u8, 1 << 26);
        defer allocator.free(buf);
        const n = try file.readPositional(io, &.{buf}, 0);
        var header = try gguf.Gguf.parse(allocator, buf[0..n]);
        errdefer header.deinit();

        const st = try file.stat(io);
        const size: usize = @intCast(st.size);
        const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
        errdefer std.posix.munmap(map);

        return .{
            .io = io,
            .file = file,
            .map = map,
            .header = header,
            .path = try allocator.dupe(u8, path),
        };
    }

    pub fn deinit(self: *GgufFile, allocator: std.mem.Allocator) void {
        std.posix.munmap(self.map);
        self.file.close(self.io);
        self.header.deinit();
        allocator.free(self.path);
    }

    /// Octets bruts d'un tenseur (déjà déquantifiés si natif, blocs sinon).
    pub fn tensorData(self: *const GgufFile, name: []const u8) ![]const u8 {
        const ti = self.header.tensor(name) orelse return error.TensorNotFound;
        const start = self.header.data_start + ti.offset;
        const len = ti.byteSize();
        if (start + len > self.map.len) return error.OutOfBounds;
        return self.map[start .. start + len];
    }
};

/// dtype ZML de sortie pour un type ggml (les quantifiés sont déquantifiés en bf16, D3).
pub fn ggmlOutDtype(t: gguf.GgmlType) zml.DataType {
    return switch (t) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => .bf16,
    };
}

/// Construit le TensorRegistry ZML depuis le GGUF : une entrée par tenseur GGUF,
/// shape = dims ggml inversées (ordre ZML), offset absolu = data_start + offset.
pub fn registryFromGguf(allocator: std.mem.Allocator, g: *const GgufFile) !zml.safetensors.TensorRegistry {
    var reg = zml.safetensors.TensorRegistry.init(allocator);
    errdefer reg.deinit();
    var it = g.header.tensors.iterator();
    while (it.next()) |e| {
        const ti = e.value_ptr;
        var dims = zml.Shape.DimsArray.empty;
        var i: usize = ti.dims.len;
        while (i > 0) : (i -= 1) dims.appendAssumeCapacity(@intCast(ti.dims[i - 1]));
        const shape = zml.Shape.init(dims.slice(), ggmlOutDtype(ti.ggml_type));
        try reg.registerTensor(.{
            .file_uri = g.path,
            .name = ti.name,
            .shape = shape,
            .offset = g.header.data_start + ti.offset,
        });
    }
    return reg;
}

/// Charge tous les tenseurs du modèle dans leurs buffers, depuis le GGUF mmapé.
/// - natifs (f32/f16/bf16) : copie directe des octets ;
/// - quantifiés (nvfp4/q8_0/q8_k/...) : déquantifiés en bf16 côté host (v1).
/// Chaque buffer alloué par `loadBuffers` est remplacé par un `Buffer.fromBytes` (upload host->device).
pub fn loadAll(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    comptime T: type,
    model: *const T,
    buffers: *zml.Bufferized(T),
    store: *const zml.io.TensorStore,
    g: *const GgufFile,
) !void {
    const n = zml.meta.count(zml.Tensor, model);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tensors = try a.alloc(*const zml.Tensor, n);
    const bufs = try a.alloc(*zml.Buffer, n);
    zml.meta.forEachVisit(model, *const zml.Tensor, struct {
        fn call(i: usize, t: *const zml.Tensor, acc: []*const zml.Tensor) void {
            acc[i] = t;
        }
    }.call, .{tensors});
    zml.meta.forEachVisit(buffers, *zml.Buffer, struct {
        fn call(i: usize, b: *zml.Buffer, acc: []*zml.Buffer) void {
            acc[i] = b;
        }
    }.call, .{bufs});

    for (tensors, bufs) |t, b| {
        const sources = store.getSourcesById(t.id) orelse continue;
        if (sources.transformed) continue; // v1 : pas de bindings transformés
        try loadOne(a, io, platform, t, b, g, sources.tensors[0].name);
    }
}

fn loadOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    t: *const zml.Tensor,
    b: *zml.Buffer,
    g: *const GgufFile,
    name: []const u8,
) !void {
    const ti = g.header.tensor(name) orelse return error.TensorNotFound;
    const raw = try g.tensorData(name);
    const shape = t.shape();
    const out_len = shape.byteSize();
    const host = try allocator.alloc(u8, out_len);
    defer allocator.free(host);

    switch (ti.ggml_type) {
        .f32, .f16, .bf16 => {
            if (raw.len < out_len) return error.TensorTooSmall;
            @memcpy(host, raw[0..out_len]);
        },
        else => {
            // quantifié -> bf16 host ; NVFP4 a besoin du tenseur `.scale` (f32 par expert)
            var scale_raw: ?[]const u8 = null;
            if (ti.ggml_type == .nvfp4) {
                // convention GGUF : l'échelle s'appelle `<base>.scale` (sans le suffixe `.weight`)
                const base = if (std.mem.endsWith(u8, name, ".weight")) name[0 .. name.len - ".weight".len] else name;
                const scale_name = try std.fmt.allocPrint(allocator, "{s}.scale", .{base});
                defer allocator.free(scale_name);
                scale_raw = try g.tensorData(scale_name);
            }
            try gguf.dequantToBf16Host(ti.ggml_type, ti.elems(), ti.dims, raw, scale_raw, host);
        },
    }

    const nb = try zml.Buffer.fromBytes(io, platform, shape, host);
    b.deinit();
    b.* = nb;
}
