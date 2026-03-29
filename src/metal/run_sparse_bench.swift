// run_sparse_bench.swift
// Compiles sparse_ops.metal at runtime and benchmarks all 4 kernels.
// Run: swift run_sparse_bench.swift
import Metal
import Foundation

// ── Config ─────────────────────────────────────────────────────
let B: Int = 64; let N_H: Int = 256; let K: Int = 100
let D: Int = 4;  let N_IN: Int = 25088; let RUNS: Int = 50

print("B=\(B) N_H=\(N_H) K=\(K) D=\(D) N_IN=\(N_IN)")

// ── Device + shader source ──────────────────────────────────────
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
print("GPU: \(device.name)")

let shaderURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("sparse_ops.metal")
let src = try! String(contentsOf: shaderURL)

// Compile at runtime (equivalent to nvcc for CUDA)
var options = MTLCompileOptions()
let library = try! device.makeLibrary(source: src, options: options)
let queue   = device.makeCommandQueue()!

// ── Buffers ────────────────────────────────────────────────────
func makeFloatBuf(_ n: Int, val: Float = 0) -> MTLBuffer {
    let buf = device.makeBuffer(length: n * MemoryLayout<Float>.size,
                                options: .storageModeShared)!
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: n)
    for i in 0..<n { ptr[i] = val == 0 ? Float.random(in: -1...1) : val }
    return buf
}
func makeUIntBuf(_ n: Int, maxVal: Int) -> MTLBuffer {
    let buf = device.makeBuffer(length: n * MemoryLayout<UInt32>.size,
                                options: .storageModeShared)!
    let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: n)
    for i in 0..<n { ptr[i] = UInt32.random(in: 0..<UInt32(maxVal)) }
    return buf
}

let bufA    = makeFloatBuf(B * N_IN * D)          // [B, N_in, D]
let bufConn = makeUIntBuf(N_H * K, maxVal: N_IN)  // [N_h, K]
let bufZ    = makeFloatBuf(B * N_H * D, val: 0)   // [B, N_h, D] output

// ── Function constants ──────────────────────────────────────────
func makeConstants() -> MTLFunctionConstantValues {
    let fc = MTLFunctionConstantValues()
    var b_ = UInt32(B); fc.setConstantValue(&b_, type: .uint, index: 0)
    var nh = UInt32(N_H); fc.setConstantValue(&nh, type: .uint, index: 1)
    var k_ = UInt32(K); fc.setConstantValue(&k_, type: .uint, index: 2)
    var d_ = UInt32(D); fc.setConstantValue(&d_, type: .uint, index: 3)
    var ni = UInt32(N_IN); fc.setConstantValue(&ni, type: .uint, index: 4)
    return fc
}

// ── Timing helper ───────────────────────────────────────────────
func bench(_ name: String, _ pipelineFn: () -> MTLComputePipelineState?,
           gridFn: (MTLComputeCommandEncoder, MTLComputePipelineState) -> Void) {
    guard let pso = pipelineFn() else { print("  \(name): pipeline failed"); return }

    // Warmup
    for _ in 0..<5 {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufConn, offset: 0, index: 1)
        enc.setBuffer(bufZ, offset: 0, index: 2)
        gridFn(enc, pso)
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    let t0 = Date()
    for _ in 0..<RUNS {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufConn, offset: 0, index: 1)
        enc.setBuffer(bufZ, offset: 0, index: 2)
        gridFn(enc, pso)
        enc.endEncoding(); cb.commit()
    }
    queue.makeCommandBuffer()!.commit()
    // Wait for all
    let lastCb = queue.makeCommandBuffer()!; lastCb.commit(); lastCb.waitUntilCompleted()

    let ms = -t0.timeIntervalSinceNow * 1000 / Double(RUNS)
    print("  \(name.padding(toLength: 30, withPad: " ", startingAt: 0))  \(String(format: "%.2f", ms)) ms")
}

print("\n=== Metal Kernel Benchmarks ===")

// Kernel 1: naive_fanin — grid [N_H, D, B]
bench("1. naive_fanin") {
    let fn = try? library.makeFunction(name: "naive_fanin", constantValues: makeConstants())
    return fn.flatMap { try? device.makeComputePipelineState(function: $0) }
} gridFn: { enc, pso in
    let tpg = MTLSize(width: pso.maxTotalThreadsPerThreadgroup, height: 1, depth: 1)
    let grid = MTLSize(width: N_H, height: D, depth: B)
    enc.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
}

// Kernel 2: simd_fanin — grid [N_H*32, D, B]
bench("2. simd_fanin") {
    let fn = try? library.makeFunction(name: "simd_fanin", constantValues: makeConstants())
    return fn.flatMap { try? device.makeComputePipelineState(function: $0) }
} gridFn: { enc, pso in
    let grid = MTLSize(width: N_H * 32, height: D, depth: B)
    let tpg  = MTLSize(width: 32, height: 1, depth: 1)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
}

// Kernel 3: tiled_block_fanin — threadgroup [64, D, 1]
bench("3. tiled_block_fanin") {
    let fn = try? library.makeFunction(name: "tiled_block_fanin", constantValues: makeConstants())
    return fn.flatMap { try? device.makeComputePipelineState(function: $0) }
} gridFn: { enc, pso in
    let tileK = 64
    let grid  = MTLSize(width: N_H * tileK, height: D, depth: B)
    let tpg   = MTLSize(width: tileK, height: 1, depth: 1)
    let smem  = tileK * D * MemoryLayout<Float>.size
    enc.setThreadgroupMemoryLength(smem, index: 0)
    enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
}

print("\nNote: PyTorch dense einsum baseline ≈ 3.9ms (from Python benchmark)")
print("block_sparse_seed kernel requires bool C_blk buffer — skipped here (needs arch change)")
