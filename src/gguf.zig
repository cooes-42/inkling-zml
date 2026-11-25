// GGUF v3 parser + déquantification (bf16/f32) — voir docs/loader-gguf.md.
// v1 couvre : F32, F16, BF16, Q8_0, Q8_K, NVFP4, MXFP4.
// Les autres quants (Q4_K, Q2_K..Q6_K, IQ*) : à porter depuis ggml.c (dequantize_row_*).
const std = @import("std");

pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
    _,

    /// Nombre d'éléments par bloc pour les types quantifiés, null si non quantifié.
    pub fn blockElems(self: GgmlType) ?u32 {
        return switch (self) {
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => 256,
            .mxfp4 => 32, // block_mxfp4 : 1 scale E8M0 + 16 octets nibbles
            .nvfp4 => 64, // block_nvfp4 : 4 scales UE4M3 + 32 octets nibbles
            else => null,
        };
    }

    /// Taille en octets d'un bloc quantifié (cf. ggml-impl.h; IQ à vérifier).
    pub fn blockBytes(self: GgmlType) ?u32 {
        return switch (self) {
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .mxfp4 => 17, // 1 scale E8M0 + 16 octets nibbles E2M1 (QK_MXFP4=32)
            .nvfp4 => 36, // 4 scales UE4M3 + 32 octets nibbles E2M1 (QK_NVFP4=64)
            else => null, // TODO: tailles IQ depuis ggml-impl.h
        };
    }

    /// Nom affichable sans panique sur les valeurs inconnues (wildcard `_`).
    pub fn name(t: GgmlType) []const u8 {
        return std.enums.tagName(GgmlType, t) orelse "unknown";
    }

    pub fn elemBytes(self: GgmlType) u32 {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .f64, .i64 => 8,
            .i8 => 1,
            else => 0, // quantifié : voir blockBytes
        };
    }
};

pub const KvValue = union(enum) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    u64: u64,
    i64: i64,
    f32: f32,
    f64: f64,
    bool: bool,
    string: []const u8,
    array: []KvValue,
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: []const u64, // ordre ggml : dims[0] = ne[0] = dimension contiguë
    ggml_type: GgmlType,
    offset: u64, // relatif à data_start

    pub fn ne0(self: TensorInfo) u64 {
        return self.dims[0];
    }

    pub fn elems(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *= d;
        return n;
    }

    /// Taille des données du tenseur en octets (avec blocs de quant)
    pub fn byteSize(self: TensorInfo) u64 {
        const e = self.elems();
        const n0 = self.ne0();
        if (self.ggml_type.blockElems()) |be| {
            const bb = self.ggml_type.blockBytes() orelse return 0; // IQ*: taille à définir
            const n_blocks = (n0 + be - 1) / be;
            return n_blocks * bb * (e / n0);
        }
        return e * self.ggml_type.elemBytes();
    }
};

pub const Gguf = struct {
    allocator: std.mem.Allocator,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: u32,
    data_start: u64, // offset absolu du début des données (header aligné)
    kv: std.StringArrayHashMapUnmanaged(KvValue),
    tensors: std.StringArrayHashMapUnmanaged(TensorInfo),

    pub fn deinit(self: *Gguf) void {
        var it = self.kv.iterator();
        while (it.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            freeKvValue(self.allocator, e.value_ptr.*);
        }
        self.kv.deinit(self.allocator);
        var it2 = self.tensors.iterator();
        while (it2.next()) |e| {
            // value.name == key (même allocation) : on ne free que la clé et dims
            self.allocator.free(e.key_ptr.*);
            self.allocator.free(e.value_ptr.dims);
        }
        self.tensors.deinit(self.allocator);
    }

    pub fn getKv(self: *const Gguf, key: []const u8) ?KvValue {
        return self.kv.get(key);
    }

    pub fn getKvString(self: *const Gguf, key: []const u8) ?[]const u8 {
        const v = self.kv.get(key) orelse return null;
        return switch (v) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn getKvU32(self: *const Gguf, key: []const u8) ?u32 {
        return switch (self.kv.get(key) orelse return null) {
            .u32 => |v| v,
            .u8 => |v| v,
            else => null,
        };
    }

    pub fn tensor(self: *const Gguf, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    /// Parse l'en-tête GGUF depuis le début du fichier (lire >= data_start octets).
    pub fn parse(allocator: std.mem.Allocator, buf: []const u8) !Gguf {
        var r = Reader{ .buf = buf };
        const magic = try r.bytes(4);
        if (!std.mem.eql(u8, magic, "GGUF")) return error.InvalidMagic;
        const version = try r.rU32();
        const tensor_count = try r.rU64();
        const kv_count = try r.rU64();

        var g = Gguf{
            .allocator = allocator,
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = kv_count,
            .alignment = 32,
            .data_start = 0,
            .kv = .{},
            .tensors = .{},
        };
        errdefer g.deinit();

        var i: u64 = 0;
        while (i < kv_count) : (i += 1) {
            const key = try r.string(allocator);
            errdefer allocator.free(key);
            const v = try r.kvValue(allocator);
            try g.kv.put(allocator, key, v);
            if (std.mem.eql(u8, key, "general.alignment")) {
                g.alignment = switch (v) {
                    .u32 => |x| x,
                    .u8 => |x| x,
                    else => 32,
                };
            }
        }

        i = 0;
        while (i < tensor_count) : (i += 1) {
            const name = try r.string(allocator);
            errdefer allocator.free(name);
            const n_dims = try r.rU32();
            if (n_dims > 8) return error.TooManyDims;
            const dims = try allocator.alloc(u64, n_dims);
            errdefer allocator.free(dims);
            for (dims) |*d| d.* = try r.rU64();
            const t = try r.rU32();
            const offset = try r.rU64();
            try g.tensors.put(allocator, name, .{
                .name = name,
                .dims = dims,
                .ggml_type = @enumFromInt(t),
                .offset = offset,
            });
        }

        const pos = r.pos;
        g.data_start = std.mem.alignForward(u64, pos, g.alignment);
        return g;
    }
};

fn freeKvValue(a: std.mem.Allocator, v: KvValue) void {
    switch (v) {
        .string => |s| a.free(s),
        .array => |arr| {
            for (arr) |x| freeKvValue(a, x);
            a.free(arr);
        },
        else => {},
    }
}

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) ![]const u8 {
        if (self.pos + n > self.buf.len) return error.OutOfBounds;
        const s = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }
    fn bytes(self: *Reader, n: usize) ![]const u8 {
        return self.take(n);
    }
    fn rU8(self: *Reader) !u8 {
        return (try self.take(1))[0];
    }
    fn rU32(self: *Reader) !u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }
    fn rU64(self: *Reader) !u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }
    fn rI32(self: *Reader) !i32 {
        return @bitCast(try self.rU32());
    }
    fn rF32(self: *Reader) !f32 {
        return @bitCast(try self.readIntLe(u32));
    }
    fn rF64(self: *Reader) !f64 {
        return @bitCast(try self.readIntLe(u64));
    }
    fn string(self: *Reader, allocator: std.mem.Allocator) ![]u8 {
        const n = try self.rU64();
        const s = try self.take(@intCast(n));
        return allocator.dupe(u8, s);
    }
    fn kvValue(self: *Reader, allocator: std.mem.Allocator) !KvValue {
        const t = try self.rU32();
        return switch (t) {
            0 => .{ .u8 = try self.rU8() },
            1 => .{ .i8 = @bitCast(try self.rU8()) },
            2 => .{ .u16 = @bitCast(try self.readIntLe(u16)) },
            3 => .{ .i16 = @bitCast(try self.readIntLe(i16)) },
            4 => .{ .u32 = try self.rU32() },
            5 => .{ .i32 = try self.rI32() },
            6 => .{ .f32 = try self.rF32() },
            7 => .{ .bool = (try self.rU8()) != 0 },
            8 => .{ .string = try self.string(allocator) },
            10 => .{ .u64 = try self.rU64() },
            11 => .{ .i64 = @bitCast(try self.rU64()) },
            12 => .{ .f64 = try self.rF64() },
            9 => blk: {
                const et = try self.rU32();
                const n = try self.rU64();
                const arr = try allocator.alloc(KvValue, n);
                errdefer allocator.free(arr);
                // les éléments d'un array partagent le même type ; on triche en
                // re-lisant le type à chaque fois (conforme au format : type única fois)
                var j: u64 = 0;
                while (j < n) : (j += 1) {
                    // relit le type de l'élément (le format stocke elem_type une fois, mais
                    // l'API lit chaque valeur avec son propre tag — on garde elem_type)
                    arr[j] = try self.kvValueTyped(allocator, et);
                }
                break :blk .{ .array = arr };
            },
            else => return error.UnknownKvType,
        };
    }
    fn readIntLe(self: *Reader, comptime T: type) !T {
        const n = @sizeOf(T);
        return std.mem.readInt(T, (try self.take(n))[0..n], .little);
    }
    fn kvValueTyped(self: *Reader, allocator: std.mem.Allocator, t: u32) !KvValue {
        return switch (t) {
            0 => .{ .u8 = try self.rU8() },
            1 => .{ .i8 = @bitCast(try self.rU8()) },
            2 => .{ .u16 = try self.readIntLe(u16) },
            3 => .{ .i16 = try self.readIntLe(i16) },
            4 => .{ .u32 = try self.rU32() },
            5 => .{ .i32 = try self.rI32() },
            6 => .{ .f32 = try self.rF32() },
            7 => .{ .bool = (try self.rU8()) != 0 },
            8 => .{ .string = try self.string(allocator) },
            10 => .{ .u64 = try self.rU64() },
            11 => .{ .i64 = @bitCast(try self.rU64()) },
            12 => .{ .f64 = try self.rF64() },
            else => return error.UnknownKvElemType,
        };
    }
};

// ---------------------------------------------------------------------------
// Conversions de précision
// ---------------------------------------------------------------------------

pub fn f32ToBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    const rounding_bias: u32 = 0x7FFF + ((bits >> 16) & 1);
    return @intCast((bits +% rounding_bias) >> 16);
}

pub fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn f16ToF32(h: u16) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
}

// ---------------------------------------------------------------------------
// Formats FP4/FP8
// ---------------------------------------------------------------------------

const e2m1_lut = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };

pub fn e2m1ToF32(nibble: u4) f32 {
    return e2m1_lut[nibble];
}

/// FP8 E4M3 (biais 7, pas d'Inf, NaN = 0x7F/0xFF), max 448.
pub fn e4m3ToF32(byte: u8) f32 {
    const sign = byte >> 7;
    const e: u4 = @intCast((byte >> 3) & 0xF);
    const m: u3 = @intCast(byte & 0x7);
    const mag: f32 = if (e == 0)
        @as(f32, @floatFromInt(m)) / 8.0 * 0x1p-6
    else
        (1.0 + @as(f32, @floatFromInt(m)) / 8.0) * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(e)) - 7.0);
    return if (sign == 1) -mag else mag;
}

pub fn e8m0ToF32(byte: u8) f32 {
    return std.math.pow(f32, 2.0, @as(f32, @floatFromInt(byte)) - 127.0);
}

/// UE4M3 non signé de ggml (ggml_ue4m3_to_fp32) : 0x00/0x7F -> 0, bit de signe ignoré, facteur 0.5.
pub fn ue4m3ToF32(x: u8) f32 {
    if (x == 0 or x == 0x7F) return 0.0;
    const exp: u4 = @intCast((x >> 3) & 0xF);
    const man: u3 = @intCast(x & 0x7);
    const raw: f32 = if (exp == 0)
        @as(f32, @floatFromInt(man)) * 0x1p-9
    else
        (1.0 + @as(f32, @floatFromInt(man)) / 8.0) * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(exp)) - 7.0);
    return raw * 0.5;
}

// ---------------------------------------------------------------------------
// Déquantification par bloc. `block` = octets du bloc, `out` = elems du bloc.
// ---------------------------------------------------------------------------

pub fn dequantBlock(ggml_type: GgmlType, block: []const u8, out: []f32, global_scale: ?f32) !void {
    switch (ggml_type) {
        .f32 => {
            for (out, 0..) |*o, i| o.* = @as(f32, @bitCast(std.mem.readInt(u32, block[i * 4 ..][0..4], .little)));
        },
        .f16 => {
            for (out, 0..) |*o, i| o.* = f16ToF32(std.mem.readInt(u16, block[i * 2 ..][0..2], .little));
        },
        .bf16 => {
            for (out, 0..) |*o, i| o.* = bf16ToF32(std.mem.readInt(u16, block[i * 2 ..][0..2], .little));
        },
        .q8_0 => {
            const d = f16ToF32(std.mem.readInt(u16, block[0..2], .little));
            for (out, 0..) |*o, i| o.* = @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + i])))) * d;
        },
        .q8_k => {
            const d: f32 = @bitCast(std.mem.readInt(u32, block[0..4], .little));
            for (out, 0..) |*o, i| o.* = @as(f32, @floatFromInt(@as(i8, @bitCast(block[4 + i])))) * d;
        },
        .nvfp4 => try dequantFp4(block, out, .e4m3, global_scale),
        .mxfp4 => try dequantFp4(block, out, .e8m0, global_scale),
        else => return error.UnsupportedQuant,
    }
}

/// NVFP4 (block_nvfp4) : 64 elems = 4 sous-blocs de 16, 4 scales UE4M3 (octets 0..3) + 32 octets
/// de nibbles E2M1. MXFP4 (block_mxfp4) : 32 elems, 1 scale E8M0 + 16 octets de nibbles.
/// Sous-bloc s, octet j (0..7) : nibble bas -> y[s*16+j], nibble haut -> y[s*16+8+j].
/// `global_scale` = facteur F32 externe (tenseur `.scale` par expert pour NVFP4 Unsloth).
fn dequantFp4(block: []const u8, out: []f32, scale_kind: enum { e4m3, e8m0 }, global_scale: ?f32) !void {
    const gs = global_scale orelse 1.0;
    switch (scale_kind) {
        .e4m3 => { // nvfp4 : 64 elems, 36 octets
            if (block.len < 36) return error.BlockTooSmall;
            if (out.len < 64) return error.OutTooSmall;
            for (0..4) |s| {
                const d = ue4m3ToF32(block[s]) * gs;
                const base = s * 16;
                for (0..8) |j| {
                    const byte = block[4 + s * 8 + j];
                    out[base + j] = e2m1ToF32(@intCast(byte & 0xF)) * d;
                    out[base + 8 + j] = e2m1ToF32(@intCast(byte >> 4)) * d;
                }
            }
        },
        .e8m0 => { // mxfp4 : 32 elems, 17 octets
            if (block.len < 17) return error.BlockTooSmall;
            if (out.len < 32) return error.OutTooSmall;
            const d = e8m0ToF32(block[0]) * gs;
            for (0..16) |j| {
                const byte = block[1 + j];
                out[j] = e2m1ToF32(@intCast(byte & 0xF)) * d;
                out[16 + j] = e2m1ToF32(@intCast(byte >> 4)) * d;
            }
        },
    }
}

/// Déquantifie un tenseur entier en bf16 (little-endian), côté host.
/// `raw` = octets du tenseur (tous blocs), `scale_raw` = octets f32 du tenseur `.scale`
/// (NVFP4 Unsloth : un f32 par expert, indexé par bloc / blocs_par_expert), `out` = elems*2 octets.
pub fn dequantToBf16Host(ggml_type: GgmlType, elems: u64, dims: []const u64, raw: []const u8, scale_raw: ?[]const u8, out: []u8) !void {
    if (out.len < elems * 2) return error.OutTooSmall;
    switch (ggml_type) {
        .bf16 => {
            @memcpy(out[0..elems * 2], raw[0 .. elems * 2]);
        },
        .f16 => {
            const out16 = std.mem.bytesAsSlice(u16, out[0 .. elems * 2]);
            for (out16, 0..) |*o, i| {
                const h = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                o.* = f32ToBf16(f16ToF32(h));
            }
        },
        .q8_0 => {
            const bb = ggml_type.blockBytes().?;
            var pos: usize = 0;
            var off: usize = 0;
            while (off < elems) {
                const block = raw[pos .. pos + bb];
                var tmp: [32]f32 = undefined;
                try dequantBlock(.q8_0, block, &tmp, null);
                for (tmp, 0..) |w, k| {
                    if (off + k >= elems) break;
                    std.mem.writeInt(u16, out[(off + k) * 2 ..][0..2], f32ToBf16(w), .little);
                }
                pos += bb;
                off += 32;
            }
        },
        .q8_k => {
            const bb = ggml_type.blockBytes().?;
            var pos: usize = 0;
            var off: usize = 0;
            while (off < elems) {
                const block = raw[pos .. pos + bb];
                var tmp: [256]f32 = undefined;
                try dequantBlock(.q8_k, block, &tmp, null);
                for (tmp, 0..) |w, k| {
                    if (off + k >= elems) break;
                    std.mem.writeInt(u16, out[(off + k) * 2 ..][0..2], f32ToBf16(w), .little);
                }
                pos += bb;
                off += 256;
            }
        },
        .nvfp4, .mxfp4 => {
            const be: u64 = ggml_type.blockElems().?;
            const bb = ggml_type.blockBytes().?;
            const n_blocks = elems / be;
            const per_expert_elems: u64 = if (dims.len >= 3) dims[0] * dims[1] else elems;
            const blocks_per_expert = per_expert_elems / be;
            const scales: []const f32 = if (scale_raw) |sr|
                @alignCast(std.mem.bytesAsSlice(f32, sr))
            else
                &[_]f32{1.0};
            for (0..n_blocks) |blk| {
                const gs: f32 = if (blk / blocks_per_expert < scales.len) scales[blk / blocks_per_expert] else 1.0;
                const block = raw[blk * bb .. (blk + 1) * bb];
                var tmp: [256]f32 = undefined;
                try dequantBlock(ggml_type, block, &tmp, gs);
                for (tmp[0..be], 0..) |w, k| {
                    std.mem.writeInt(u16, out[(blk * be + k) * 2 ..][0..2], f32ToBf16(w), .little);
                }
            }
        },
        else => return error.UnsupportedQuant,
    }
}

/// Déquantifie un tenseur entier (tous ses blocs) en f32. `data` = octets du tenseur.
pub fn dequantTensor(ggml_type: GgmlType, data: []const u8, out: []f32, global_scale: ?f32) !void {
    const be = ggml_type.blockElems() orelse {
        try dequantBlock(ggml_type, data, out, null);
        return;
    };
    const bb = ggml_type.blockBytes().?;
    var pos: usize = 0;
    var off: usize = 0;
    while (off < out.len) {
        const n = @min(be, out.len - off);
        if (pos + bb > data.len) return error.TensorTooSmall;
        try dequantBlock(ggml_type, data[pos .. pos + bb], out[off .. off + n], global_scale);
        pos += bb;
        off += n;
    }
}
