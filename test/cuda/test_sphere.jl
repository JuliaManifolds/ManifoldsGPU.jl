using Manifolds, ManifoldsGPU, Test, Random, CUDA

@testset "Sphere CUDA" begin
    @testset "inner and norm" begin
        Random.seed!(70)

        M = Sphere(7)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y = rand(MP; vector_at = p)

        i_cpu = inner(MP, p, X, Y)
        n_cpu = norm(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)

        i_gpu = inner(MP, p_cu, X_cu, Y_cu)
        n_gpu = norm(MP, p_cu, X_cu)

        @test isapprox(i_gpu, i_cpu; atol = 1.0e-10, rtol = 1.0e-10)
        @test isapprox(n_gpu, n_cpu; atol = 1.0e-10, rtol = 1.0e-10)
    end

    @testset "inner and norm Float32" begin
        Random.seed!(71)

        M = Sphere(7)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        X = Float32.(rand(MP; vector_at = p))
        Y = Float32.(rand(MP; vector_at = p))

        i_cpu = inner(MP, p, X, Y)
        n_cpu = norm(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)

        i_gpu = inner(MP, p_cu, X_cu, Y_cu)
        n_gpu = norm(MP, p_cu, X_cu)

        @test isapprox(i_gpu, i_cpu; atol = 1.0f-4, rtol = 1.0f-4)
        @test isapprox(n_gpu, n_cpu; atol = 1.0f-4, rtol = 1.0f-4)
    end

    @testset "project!" begin
        for T in [Float32, Float64]
            Random.seed!(72)

            M = Sphere(7)
            MP = PowerManifold(M, 32)

            p = T.(rand(MP))
            q_cpu = similar(p)
            project!(MP, q_cpu, p)

            p_cu = CuArray(p)
            q_cu = similar(p_cu)
            project!(MP, q_cu, p_cu)

            if T === Float32
                @test isapprox(Array(q_cu), q_cpu; atol = 1.0f-5, rtol = 1.0f-5)
            else
                @test isapprox(Array(q_cu), q_cpu; atol = 1.0e-12, rtol = 1.0e-12)
            end
        end
    end

    @testset "project! tangent" begin
        for (seed, T, atol, rtol) in (
                (721, Float32, 1.0f-5, 1.0f-5),
                (722, Float64, 1.0e-12, 1.0e-12),
            )
            Random.seed!(seed)

            M = Sphere(7)
            MP = PowerManifold(M, 32)

            p = T.(rand(MP))
            X = T.(randn(size(p)))

            Y_cpu = similar(p)
            project!(MP, Y_cpu, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = similar(p_cu)
            project!(MP, Y_cu, p_cu, X_cu)

            @test isapprox(Array(Y_cu), Y_cpu; atol = atol, rtol = rtol)
        end
    end

    @testset "exp!" begin
        for (seed, T, atol, rtol) in (
                (73, Float32, 1.0f-4, 1.0f-4),
                (74, Float64, 1.0e-10, 1.0e-10),
            )
            Random.seed!(seed)

            M = Sphere(7)
            MP = PowerManifold(M, 32)

            p = T.(rand(MP))
            X = T.(rand(MP; vector_at = p))

            q_cpu = similar(p)
            exp!(MP, q_cpu, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            q_cu = similar(p_cu)
            exp!(MP, q_cu, p_cu, X_cu)

            @test isapprox(Array(q_cu), q_cpu; atol = atol, rtol = rtol)
        end
    end

    @testset "log!" begin
        for (seed, T, atol, rtol) in (
                (75, Float32, 5.0f-4, 5.0f-4),
                (76, Float64, 1.0e-9, 1.0e-9),
            )
            Random.seed!(seed)

            M = Sphere(7)
            MP = PowerManifold(M, 32)

            p = T.(rand(MP))
            q = T.(rand(MP))

            X_cpu = similar(p)
            log!(MP, X_cpu, p, q)

            p_cu = CuArray(p)
            q_cu = CuArray(q)
            X_cu = similar(p_cu)
            log!(MP, X_cu, p_cu, q_cu)

            @test isapprox(Array(X_cu), X_cpu; atol = atol, rtol = rtol)
        end
    end
end
