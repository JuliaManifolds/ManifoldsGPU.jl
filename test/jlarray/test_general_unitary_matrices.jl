using GPUArrays

# JLArray tests for plain (non-PowerManifold) GeneralUnitaryMatrices.
# These test that generic Manifolds.jl code works with GPU arrays without scalar indexing.
# CuArray-specific PowerManifold overrides are tested in test/cuda/ only.

@testset "GeneralUnitaryMatrices JLArray" begin
    GPUArrays.allowscalar(false)

    # Rotations retract! (Polar, Float64)
    @testset "Rotations retract Polar Float64" begin
        Random.seed!(42)

        M = Rotations(4)

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

    # Rotations retract! (Polar, Float32)
    @testset "Rotations retract Polar Float32" begin
        Random.seed!(43)

        M = Rotations(4)

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

    # Grassmann retract! (Polar, Float64)
    @testset "Grassmann retract Polar Float64" begin
        Random.seed!(44)

        M = Grassmann(6, 3)

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

    # Grassmann project! tangent (Float64)
    @testset "Grassmann project tangent Float64" begin
        Random.seed!(45)

        M = Grassmann(6, 3)

        for _ in 1:3
            p = rand(M)
            X = randn(6, 3)
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

    # Grassmann project! point (Float64)
    @testset "Grassmann project point Float64" begin
        Random.seed!(46)

        M = Grassmann(6, 3)

        for _ in 1:3
            p = rand(M)
            p_noisy = p .+ 0.01 .* randn(size(p)...)
            q_cpu = similar(p)
            project!(M, q_cpu, p_noisy)

            p_noisy_jl = JLArray(p_noisy)
            q_jl = similar(p_noisy_jl)
            project!(M, q_jl, p_noisy_jl)
            q_jl_h = Array(q_jl)

            @test is_point(M, q_jl_h)
            @test isapprox(q_jl_h, q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end
end
