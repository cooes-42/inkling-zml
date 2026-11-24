const std = @import("std");
const gguf = @import("gguf.zig");

const testing = std.testing;

// Helpers d'écriture little-endian dans un ArrayList (API 0.17 : plus de .writer(allocator)).
fn wAll(buf: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    try buf.appendSlice(a, s);
}
fn wInt(buf: *std.ArrayList(u8), a: std.mem.Allocator, comptime T: type, v: T) !void {
    var tmp: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &tmp, v, .little);
    try buf.appendSlice(a, &tmp);
}

// Construit un mini GGUF en mémoire : 3 KV + 2 tenseurs (f32[4], q8_0[32]).
fn buildMiniGguf(allocator: std.mem.Allocator) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try wAll(&buf, allocator, "GGUF");
    try wInt(&buf, allocator, u32, 3); // version
    try wInt(&buf, allocator, u64, 2); // tensor_count
    try wInt(&buf, allocator, u64, 3); // kv_count

    // KV 1: general.architecture = "inkling2_loop" (string)
    try wInt(&buf, allocator, u64, "general.architecture".len);
    try wAll(&buf, allocator, "general.architecture");
    try wInt(&buf, allocator, u32, 8); // type string
    try wInt(&buf, allocator, u64, "inkling2_loop".len);
    try wAll(&buf, allocator, "inkling2_loop");
    // KV 2: general.alignment = 32 (u32)
    try wInt(&buf, allocator, u64, "general.alignment".len);
    try wAll(&buf, allocator, "general.alignment");
    try wInt(&buf, allocator, u32, 4); // type u32
    try wInt(&buf, allocator, u32, 32);
    // KV 3: test.arr = [1,2,3] (array of u32)
    try wInt(&buf, allocator, u64, "test.arr".len);
    try wAll(&buf, allocator, "test.arr");
    try wInt(&buf, allocator, u32, 9); // type array
    try wInt(&buf, allocator, u32, 4); // elem type u32
    try wInt(&buf, allocator, u64, 3);
    for ([_]u32{ 1, 2, 3 }) |v| try wInt(&buf, allocator, u32, v);

    // tenseur 1: a.f32 [4] offset 0 ; tenseur 2: b.q8_0 [32] offset 32
    try writeTensorInfo(&buf, allocator, "a.f32", &[_]u64{4}, 0, 0);
    try writeTensorInfo(&buf, allocator, "b.q8_0", &[_]u64{32}, 8, 32);

    // data_start aligné sur 32
    const header_end = buf.items.len;
    const data_start = std.mem.alignForward(usize, header_end, 32);
    try buf.appendNTimes(allocator, 0, data_start - header_end);

    // données : a = [1.0, -2.5, 3.0, 0.0] f32 (16 o), pad à 32, puis bloc q8_0 (34 o)
    for ([_]f32{ 1.0, -2.5, 3.0, 0.0 }) |v| try wInt(&buf, allocator, u32, @bitCast(v));
    try buf.appendNTimes(allocator, 0, 32 - 16);
    // q8_0 block: d = f16(2.0) = 0x4000, qs = 32 x i8(3) → w = 6.0
    try wInt(&buf, allocator, u16, 0x4000);
    try buf.appendNTimes(allocator, 3, 32);

    return buf.toOwnedSlice(allocator);
}

fn writeTensorInfo(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8, dims: []const u64, ggml_type: u32, offset: u64) !void {
    try wInt(buf, allocator, u64, name.len);
    try wAll(buf, allocator, name);
    try wInt(buf, allocator, u32, @intCast(dims.len));
    for (dims) |d| try wInt(buf, allocator, u64, d);
    try wInt(buf, allocator, u32, ggml_type);
    try wInt(buf, allocator, u64, offset);
}

test "parse mini gguf header" {
    const allocator = testing.allocator;
    const bytes = try buildMiniGguf(allocator);
    defer allocator.free(bytes);

    var g = try gguf.Gguf.parse(allocator, bytes);
    defer g.deinit();

    try testing.expectEqual(@as(u32, 3), g.version);
    try testing.expectEqual(@as(u64, 2), g.tensor_count);
    try testing.expectEqual(@as(u32, 32), g.alignment);
    try testing.expectEqualStrings("inkling2_loop", g.getKvString("general.architecture").?);
    try testing.expectEqual(@as(u32, 32), g.getKvU32("general.alignment").?);
    const arr = g.getKv("test.arr").?.array;
    try testing.expectEqual(@as(usize, 3), arr.len);
    try testing.expectEqual(@as(u32, 2), arr[1].u32);

    const a = g.tensor("a.f32").?;
    try testing.expectEqual(@as(u64, 4), a.ne0());
    try testing.expectEqual(@as(u64, 16), a.byteSize());
    try testing.expectEqual(gguf.GgmlType.f32, a.ggml_type);

    const b = g.tensor("b.q8_0").?;
    try testing.expectEqual(@as(u64, 32), b.ne0());
    try testing.expectEqual(@as(u64, 34), b.byteSize()); // 1 bloc de 34 o
    try testing.expectEqual(@as(u64, 32), b.offset);

    try testing.expect(g.data_start % 32 == 0);
    try testing.expect(g.data_start <= bytes.len);
}

test "dequant q8_0 block" {
    var block: [34]u8 = undefined;
    std.mem.writeInt(u16, block[0..2], 0x4000, .little); // d = 2.0
    @memset(block[2..], 3); // qs = 3
    var out: [32]f32 = undefined;
    try gguf.dequantBlock(.q8_0, &block, &out, null);
    for (out) |v| try testing.expectEqual(@as(f32, 6.0), v);
}

test "dequant nvfp4 block" {
    // 16 scales E4M3 = 0x38 (1.0) + 128 o de nibbles 0x6 (E2M1 = 4.0)
    var block: [144]u8 = undefined;
    @memset(block[0..16], 0x38);
    @memset(block[16..], 0x66);
    var out: [256]f32 = undefined;
    try gguf.dequantBlock(.nvfp4, &block, &out, null);
    for (out) |v| try testing.expectEqual(@as(f32, 4.0), v);

    // avec scale globale F32 (tenseur .scale Unsloth) = 2.0
    var out2: [256]f32 = undefined;
    try gguf.dequantTensor(.nvfp4, &block, &out2, 2.0);
    for (out2) |v| try testing.expectEqual(@as(f32, 8.0), v);
}

test "fp conversions" {
    try testing.expectEqual(@as(u16, 0x3F80), gguf.f32ToBf16(1.0));
    try testing.expectEqual(@as(f32, 1.0), gguf.bf16ToF32(0x3F80));
    try testing.expectEqual(@as(f32, 4.0), gguf.e2m1ToF32(6));
    try testing.expectEqual(@as(f32, -1.5), gguf.e2m1ToF32(11));
    try testing.expectEqual(@as(f32, 1.0), gguf.e4m3ToF32(0x38));
    try testing.expectEqual(@as(f32, 448.0), gguf.e4m3ToF32(0x7E));
    try testing.expectEqual(@as(f32, 2.0), gguf.e8m0ToF32(128));
    try testing.expectEqual(@as(f32, 2.0), gguf.f16ToF32(0x4000));
}

test "dequantToBf16Host nvfp4 with per-expert scale" {
    // 2 experts x 256 elems : bloc 0 expert 0, bloc 1 expert 1
    var raw: [2 * 144]u8 = undefined;
    @memset(raw[0..16], 0x38); // expert 0: scales 1.0
    @memset(raw[16..144], 0x66); // nibbles 6 -> E2M1 4.0
    @memset(raw[144 .. 144 + 16], 0x40); // expert 1: E4M3 0x40 = e=8 -> 2^(8-7)*(1+0)=2.0
    @memset(raw[144 + 16 ..], 0x66);
    var scale: [2]f32 = .{ 2.0, 3.0 };
    var out: [512 * 2]u8 = undefined;
    try gguf.dequantToBf16Host(.nvfp4, 512, &.{ 256, 1, 2 }, &raw, std.mem.asBytes(&scale), &out);
    const o = std.mem.bytesAsSlice(u16, &out);
    // expert 0: 4.0 * 1.0 * 2.0 = 8.0 -> bf16 0x4100
    try testing.expectEqual(@as(u16, 0x4100), o[0]);
    try testing.expectEqual(@as(u16, 0x4100), o[255]);
    // expert 1: 4.0 * 2.0 * 3.0 = 24.0 -> bf16 0x41C0
    try testing.expectEqual(@as(u16, 0x41C0), o[256]);
    try testing.expectEqual(@as(u16, 0x41C0), o[511]);
}

test "block sizes" {
    try testing.expectEqual(@as(?u32, 32), gguf.GgmlType.q8_0.blockElems());
    try testing.expectEqual(@as(?u32, 34), gguf.GgmlType.q8_0.blockBytes());
    try testing.expectEqual(@as(?u32, 256), gguf.GgmlType.q8_k.blockElems());
    try testing.expectEqual(@as(?u32, 292), gguf.GgmlType.q8_k.blockBytes());
    try testing.expectEqual(@as(?u32, 144), gguf.GgmlType.nvfp4.blockBytes());
    try testing.expectEqual(@as(?u32, null), gguf.GgmlType.f32.blockElems());
}
