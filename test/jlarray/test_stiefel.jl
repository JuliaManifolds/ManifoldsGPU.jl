using GPUArrays

@testset "Stiefel JLArray" begin
    # JLArray tests verify numerical correctness without GPU hardware.
    # Scalar indexing is allowed because the CuArray-specific overrides
    # (CUBLAS, CUSOLVER) do not dispatch on JLArray; the default
    # ManifoldsBase PowerManifold path loops per element.
    GPUArrays.allowscalar(true)

    @testset "exp! basic" begin
        M = Stiefel(4, 2)
        MP = PowerManifold(M, 5)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y = exp(MP, p, X)

        p_jl = JLArray(p)
        X_jl = JLArray(X)
        Y_jl = exp(MP, p_jl, X_jl)
        @test isapprox(MP, p, Array(Y_jl), Y; atol = 1.0e-10)
    end

    @testset "exp! batched stress" begin
        Random.seed!(42)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 16)

        for _ in 1:3
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y = exp(MP, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            Y_jl = exp(MP, p_jl, X_jl)
            Y_jl_h = Array(Y_jl)

            @test is_point(MP, Y_jl_h)
            @test isapprox(MP, p, Y_jl_h, Y; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    @testset "exp! batched stress Float32" begin
        Random.seed!(43)

        M = Stiefel(8, 4)
        MP = PowerManifold(M, 16)

        for _ in 1:3
            p = Float32.(rand(MP))
            X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
            Y = exp(MP, p, X)

            p_jl = JLArray(p)
            X_jl = JLArray(X)
            Y_jl = exp(MP, p_jl, X_jl)
            Y_jl_h = Array(Y_jl)

            @test is_point(MP, Y_jl_h)
            @test isapprox(MP, p, Y_jl_h, Y; atol = 2.0f-5, rtol = 2.0f-5)
        end
    end

    GPUArrays.allowscalar(false)
end
