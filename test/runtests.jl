using ManifoldsGPU
using Test
using Random

using ManifoldsBase, Manifolds
using CUDA

@testset "ManifoldsGPU.jl" begin
    # Write your tests here.

    @testset "Stiefel" begin
        M = Stiefel(4, 2)
        MP = PowerManifold(M, 5)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)
        @test isapprox(MP, p, Array(Y_cu), Y; atol = 1.0e-10)
    end

    @testset "Stiefel batched stress" begin
        Random.seed!(42)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test isapprox(MP, p, Y_cu_h, Y; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    @testset "Stiefel batched stress Float32" begin
        Random.seed!(43)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = Float32.(rand(MP))
            X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
            Y = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test isapprox(MP, p, Y_cu_h, Y; atol = 2.0f-5, rtol = 2.0f-5)
        end
    end

    @testset "Stiefel retract_polar_fused batched" begin
        Random.seed!(46)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 64)
        t = 0.3

        p = rand(MP)
        X = rand(MP; vector_at = p)

        q = similar(p)
        ManifoldsBase.retract_fused!(MP, q, p, X, t, PolarRetraction())

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        q_cu = similar(p_cu)
        ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test isapprox(MP, p, q_cu_h, q; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "Stiefel retract_polar_fused batched Float32" begin
        Random.seed!(47)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 64)
        t = Float32(0.3)

        p = Float32.(rand(MP))
        X = Float32.(rand(MP; vector_at = p))

        q = similar(p)
        ManifoldsBase.retract_fused!(MP, q, p, X, t, PolarRetraction())

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        q_cu = similar(p_cu)
        ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test isapprox(MP, p, q_cu_h, q; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "Stiefel retract_polar_fused fallback stress" begin
        Random.seed!(48)

        M = Stiefel(48, 16)
        MP = PowerManifold(M, 8)
        t = 0.2

        for _ in 1:4
            p = rand(MP)
            X = 0.2 * rand(MP; vector_at = p)

            q = similar(p)
            ManifoldsBase.retract_fused!(MP, q, p, X, t, PolarRetraction())

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            q_cu = similar(p_cu)
            ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())
            q_cu_h = Array(q_cu)

            @test is_point(MP, q_cu_h)
            @test isapprox(MP, p, q_cu_h, q; atol = 2.0e-12, rtol = 2.0e-12)
        end
    end
end
