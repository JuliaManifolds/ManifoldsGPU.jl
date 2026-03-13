using GPUArrays

# JLArray tests for plain (non-PowerManifold) Stiefel.
# These test that generic Manifolds.jl code works with GPU arrays without scalar indexing.
# CuArray-specific PowerManifold overrides are tested in test/cuda/ only.

@testset "Stiefel JLArray" begin
    GPUArrays.allowscalar(false)

    # Stiefel retract! (Polar, Float64)
    @testset "Stiefel retract Polar Float64" begin
        Random.seed!(42)

        M = Stiefel(8, 4)

        for _ in 1:3
            p = rand(M)
            X = 0.25 * rand(M; vector_at = p)
            q_cpu = retract(M, p, X, PolarRetraction())

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            q_jl = retract(M, p_jl, X_jl, PolarRetraction())
            q_jl_h = Array(q_jl)

            @test is_point(M, q_jl_h)
            @test isapprox(q_jl_h, q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    # Stiefel retract! (Polar, Float32)
    @testset "Stiefel retract Polar Float32" begin
        Random.seed!(43)

        M = Stiefel(8, 4)

        for _ in 1:3
            p = Float32.(rand(M))
            X = Float32(0.25) .* Float32.(rand(M; vector_at = p))
            q_cpu = retract(M, p, X, PolarRetraction())

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            q_jl = retract(M, p_jl, X_jl, PolarRetraction())
            q_jl_h = Array(q_jl)

            @test is_point(M, q_jl_h)
            @test isapprox(q_jl_h, q_cpu; atol = 2.0f-5, rtol = 2.0f-5)
        end
    end

    # Stiefel project! tangent (Float64)
    @testset "Stiefel project tangent Float64" begin
        Random.seed!(44)

        M = Stiefel(8, 4)

        for _ in 1:3
            p = rand(M)
            X = randn(8, 4)
            Y_cpu = similar(X)
            project!(M, Y_cpu, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            Y_jl = similar(X_jl)
            project!(M, Y_jl, p_jl, X_jl)
            Y_jl_h = Array(Y_jl)

            @test is_vector(M, p, Y_jl_h; atol = 2.0e-14)
            @test isapprox(Y_jl_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end
end
