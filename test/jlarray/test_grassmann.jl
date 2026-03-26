using GPUArrays

# JLArray tests for Grassmann (CPU code path on GPU arrays).
# allowscalar(true) needed because CPU exp! uses Array(qr(z).Q).

@testset "Grassmann JLArray" begin
    GPUArrays.allowscalar(true)

    # Grassmann exp (Float64)
    @testset "Grassmann exp Float64" begin
        Random.seed!(50)

        M = Grassmann(6, 3)

        for _ in 1:3
            p = rand(M)
            X = 0.25 * rand(M; vector_at = p)
            q_cpu = exp(M, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            q_jl = exp(M, p_jl, X_jl)
            q_jl_h = Array(q_jl)

            @test is_point(M, q_jl_h)
            @test isapprox(q_jl_h, q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    # Grassmann exp (Float32)
    @testset "Grassmann exp Float32" begin
        Random.seed!(51)

        M = Grassmann(6, 3)

        for _ in 1:3
            p = Float32.(rand(M))
            X = Float32(0.25) .* Float32.(rand(M; vector_at = p))
            q_cpu = exp(M, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            q_jl = exp(M, p_jl, X_jl)
            q_jl_h = Array(q_jl)

            @test is_point(M, q_jl_h)
            @test isapprox(q_jl_h, q_cpu; atol = 2.0f-5, rtol = 2.0f-5)
        end
    end
end
