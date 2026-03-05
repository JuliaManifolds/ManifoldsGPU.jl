# JLArray-based GPU tests for ManifoldsGPU.jl
# These tests run in CI without requiring CUDA hardware.
# JLArray mimics GPU array dispatch but executes on CPU.
# Note: Stiefel.jl dispatches on CuArray{T,3} specifically, so JLArray
# falls through to CPU implementations — these tests verify package loading,
# mathematical correctness, and that CPU fallbacks work correctly.

using JLArrays
using GPUArrays
# Note: ManifoldsBase, Manifolds, Random, Test are loaded by runtests.jl before this file is included.

# Allow scalar indexing for JLArray: the CPU fallback implementations in
# Manifolds.jl use scalar operations (e.g. hvcat in StiefelEuclideanMetric.jl).
# This is expected and acceptable — scalar indexing does not occur in the
# CuArray path (which has dedicated GPU kernels), so enabling it here does
# not mask any GPU-correctness issue.
GPUArrays.allowscalar(true)

@testset "JLArray: Stiefel exp" begin
    M = Stiefel(4, 2)
    MP = PowerManifold(M, 5)

    # Seeds 42-44 are used here; cuda_tests.jl uses 42, 43, 45-48.
    # Random.seed! always resets global RNG so cross-file reuse is safe but noted.
    Random.seed!(42)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end

@testset "JLArray: Stiefel exp Float32" begin
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 16)

    Random.seed!(43)
    p = Float32.(rand(MP))
    X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
    Y_cpu = exp(MP, p, X)

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    Y_jl = exp(MP, p_jl, X_jl)

    @test is_point(MP, Array(Y_jl))
    @test isapprox(MP, p, Array(Y_jl), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
end

@testset "JLArray: Stiefel PolarRetraction" begin
    M = Stiefel(8, 4)
    MP = PowerManifold(M, 16)
    t = 0.3

    Random.seed!(44)
    p = rand(MP)
    X = rand(MP; vector_at = p)
    q_cpu = similar(p)
    ManifoldsBase.retract_fused!(MP, q_cpu, p, X, t, PolarRetraction())

    p_jl = JLArray(p)
    X_jl = JLArray(X)
    q_jl = similar(p_jl)
    ManifoldsBase.retract_fused!(MP, q_jl, p_jl, X_jl, t, PolarRetraction())

    @test is_point(MP, Array(q_jl))
    @test isapprox(MP, p, Array(q_jl), q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
end
