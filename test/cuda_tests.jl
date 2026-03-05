using CUDA

@testset "ManifoldsGPU.jl" begin
    # Write your tests here.

    @testset "Stiefel" begin
        Random.seed!(45)

        M = Stiefel(4, 2)
        MP = PowerManifold(M, 5)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)
        @test is_point(MP, Array(Y_cu))
        @test isapprox(MP, p, Array(Y_cu), Y; atol = 2.0e-14, rtol = 2.0e-14)
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
            # CPU fallback via serial svd! (used when gesvdj! fails for large matrices)
            # accumulates slightly more rounding error than the batched GPU path
            @test isapprox(MP, p, q_cu_h, q; atol = 2.0e-12, rtol = 2.0e-12)
        end
    end

    @testset "Grassmann exp Float64" begin
        Random.seed!(50)
        M = Grassmann(6, 3)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        X = 0.25 .* rand(MP; vector_at = p)
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test is_point(MP, Array(Y_cu))
        # Grassmann points are subspaces (equivalence classes); use 3-arg isapprox
        # (manifold-aware, checks geodesic distance) rather than element-wise comparison.
        # CPU and GPU may produce different n×k representatives of the same subspace
        # when singular values are nearly degenerate.
        @test isapprox(MP, Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "Grassmann exp Float32" begin
        Random.seed!(51)
        M = Grassmann(6, 3)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test is_point(MP, Array(Y_cu))
        # Grassmann: 3-arg isapprox for subspace comparison (see Float64 testset comment)
        @test isapprox(MP, Array(Y_cu), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "Grassmann PolarRetraction" begin
        Random.seed!(52)
        M = Grassmann(6, 3)
        MP = PowerManifold(M, 32)
        t = 0.3

        p = rand(MP)
        X = rand(MP; vector_at = p)
        q_cpu = similar(p)
        ManifoldsBase.retract_fused!(MP, q_cpu, p, X, t, PolarRetraction())

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        q_cu = similar(p_cu)
        ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())

        @test is_point(MP, Array(q_cu))
        @test isapprox(MP, p, Array(q_cu), q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "Grassmann exp stress Float64" begin
        Random.seed!(53)
        M = Grassmann(8, 4)
        MP = PowerManifold(M, 128)

        for _ in 1:4
            p = rand(MP)
            X = 0.25 .* rand(MP; vector_at = p)
            Y_cpu = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)

            @test is_point(MP, Array(Y_cu))
            # Grassmann: 3-arg isapprox for subspace comparison (see exp Float64 comment)
            @test isapprox(MP, Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    @testset "Sphere exp Float64" begin
        Random.seed!(60)
        M = Sphere(3)
        MP = PowerManifold(M, 256)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test is_point(MP, Array(Y_cu))
        @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "Sphere exp Float32" begin
        Random.seed!(61)
        M = Sphere(3)
        MP = PowerManifold(M, 256)

        p = Float32.(rand(MP))
        X = Float32(0.5) .* Float32.(rand(MP; vector_at = p))
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test is_point(MP, Array(Y_cu))
        @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "Sphere log Float64" begin
        Random.seed!(62)
        M = Sphere(3)
        MP = PowerManifold(M, 256)

        p = rand(MP)
        q = rand(MP)
        X_cpu = log(MP, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = log(MP, p_cu, q_cu)

        @test isapprox(Array(X_cu), X_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "Euclidean exp Float64" begin
        Random.seed!(70)
        MP = PowerManifold(Euclidean(4), 1024)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test isapprox(Array(Y_cu), Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "Euclidean log Float64" begin
        Random.seed!(71)
        MP = PowerManifold(Euclidean(4), 1024)

        p = rand(MP)
        q = rand(MP)
        X_cpu = log(MP, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = log(MP, p_cu, q_cu)

        @test isapprox(Array(X_cu), X_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "UnitaryMatrices exp ComplexF64" begin
        Random.seed!(80)
        M = UnitaryMatrices(3)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test is_point(MP, Array(Y_cu))
        @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0e-12, rtol = 2.0e-12)
    end

    @testset "UnitaryMatrices exp ComplexF32" begin
        Random.seed!(81)
        M = UnitaryMatrices(3)
        MP = PowerManifold(M, 32)

        p = ComplexF32.(rand(MP))
        X = ComplexF32.(rand(MP; vector_at = p))
        Y_cpu = exp(MP, p, X)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = exp(MP, p_cu, X_cu)

        @test is_point(MP, Array(Y_cu))
        @test isapprox(MP, p, Array(Y_cu), Y_cpu; atol = 2.0f-4, rtol = 2.0f-4)
    end
end
