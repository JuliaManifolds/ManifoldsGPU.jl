@testset "GeneralUnitaryMatrices CUDA" begin

    # 1. Rotations exp! batched
    @testset "Rotations exp! batched" begin
        Random.seed!(42)

        M = Rotations(8)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y_cpu = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test isapprox(Y_cu_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    # 2. Rotations exp! Float32
    @testset "Rotations exp! Float32" begin
        Random.seed!(43)

        M = Rotations(8)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = Float32.(rand(MP))
            X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
            Y_cpu = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test isapprox(Y_cu_h, Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
        end
    end

    # 3. UnitaryMatrices exp! batched
    @testset "UnitaryMatrices exp! batched" begin
        Random.seed!(44)

        M = UnitaryMatrices(8)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y_cpu = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test isapprox(Y_cu_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
        end
    end

    # 4. Rotations retract_polar_fused batched
    @testset "Rotations retract_polar_fused batched" begin
        Random.seed!(46)

        M = Rotations(8)
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
        @test isapprox(q_cu_h, q; atol = 2.0e-14, rtol = 2.0e-14)
    end

    # 5. Rotations retract_polar_fused Float32
    @testset "Rotations retract_polar_fused Float32" begin
        Random.seed!(47)

        M = Rotations(8)
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
        @test isapprox(q_cu_h, q; atol = 2.0f-5, rtol = 2.0f-5)
    end

    # 6. UnitaryMatrices retract_polar_fused batched
    # NOTE: CPU retract_polar_fused! has upstream bug for UnitaryMatrices (check_det kwarg).
    # Only test GPU path via is_point, do NOT compare with CPU.
    @testset "UnitaryMatrices retract_polar_fused batched" begin
        Random.seed!(48)

        M = UnitaryMatrices(8)
        MP = PowerManifold(M, 64)
        t = 0.3

        p = rand(MP)
        X = 0.25 .* rand(MP; vector_at = p)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        q_cu = similar(p_cu)
        ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, PolarRetraction())
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
    end

    # 7. Rotations project! tangent
    @testset "Rotations project! tangent" begin
        Random.seed!(50)

        M = Rotations(8)
        MP = PowerManifold(M, 64)

        p = rand(MP)
        X = randn(size(p)...)

        Y_cpu = similar(X)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(Y_cpu, :, :, i), view(p, :, :, i), view(X, :, :, i)
            )
        end

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = similar(X_cu)
        ManifoldsBase.project!(MP, Y_cu, p_cu, X_cu)
        Y_cu_h = Array(Y_cu)

        @test isapprox(Y_cu_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    # 8. UnitaryMatrices project! tangent
    @testset "UnitaryMatrices project! tangent" begin
        Random.seed!(51)

        M = UnitaryMatrices(8)
        MP = PowerManifold(M, 64)

        p = rand(MP)
        X = randn(ComplexF64, size(p)...)

        Y_cpu = similar(X)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(Y_cpu, :, :, i), view(p, :, :, i), view(X, :, :, i)
            )
        end

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = similar(X_cu)
        ManifoldsBase.project!(MP, Y_cu, p_cu, X_cu)
        Y_cu_h = Array(Y_cu)

        @test isapprox(Y_cu_h, Y_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    # 9. OrthogonalMatrices project! point
    @testset "OrthogonalMatrices project! point" begin
        Random.seed!(52)

        M = OrthogonalMatrices(8)
        MP = PowerManifold(M, 64)

        p = rand(MP)
        p_noisy = p .+ 0.01 .* randn(size(p)...)

        q_cpu = similar(p)
        for i in 1:size(p, 3)
            s = svd(p_noisy[:, :, i])
            q_cpu[:, :, i] .= s.U * s.Vt
        end

        p_noisy_cu = CuArray(p_noisy)
        q_cu = similar(p_noisy_cu)
        ManifoldsBase.project!(MP, q_cu, p_noisy_cu)
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test isapprox(q_cu_h, q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    # 11. Rotations log! batched
    @testset "Rotations log! batched" begin
        Random.seed!(60)

        M = Rotations(8)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = rand(MP)
            q = rand(MP)
            X_cpu = log(MP, p, q)

            p_cu = CuArray(p)
            q_cu = CuArray(q)
            X_cu = log(MP, p_cu, q_cu)
            X_cu_h = Array(X_cu)

            @test is_vector(MP, p, X_cu_h; atol = 2.0e-6)
            @test isapprox(X_cu_h, X_cpu; atol = 2.0e-6, rtol = 2.0e-6)
        end
    end

    # 12. Rotations log! Float32
    @testset "Rotations log! Float32" begin
        Random.seed!(61)

        M = Rotations(8)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = Float32.(rand(MP))
            q = Float32.(rand(MP))
            X_cpu = log(MP, p, q)

            p_cu = CuArray(p)
            q_cu = CuArray(q)
            X_cu = log(MP, p_cu, q_cu)
            X_cu_h = Array(X_cu)

            @test is_vector(MP, p, X_cu_h; atol = 1.0f-3)
            @test isapprox(X_cu_h, X_cpu; atol = 2.0f-3, rtol = 2.0f-3)
        end
    end

    # 13. UnitaryMatrices log! batched
    @testset "UnitaryMatrices log! batched" begin
        Random.seed!(62)

        M = UnitaryMatrices(8)
        MP = PowerManifold(M, 64)

        for _ in 1:6
            p = rand(MP)
            q = rand(MP)
            X_cpu = log(MP, p, q)

            p_cu = CuArray(p)
            q_cu = CuArray(q)
            X_cu = log(MP, p_cu, q_cu)
            X_cu_h = Array(X_cu)

            @test is_vector(MP, p, X_cu_h; atol = 1.0e-4)
            @test isapprox(X_cu_h, X_cpu; atol = 1.0e-4, rtol = 1.0e-4)
        end
    end

    # 14. retract_polar_fused fallback stress (large matrices trigger CPU fallback)
    @testset "retract_polar_fused fallback stress" begin
        Random.seed!(53)

        M = Rotations(48)
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
            @test isapprox(q_cu_h, q; atol = 2.0e-12, rtol = 2.0e-12)
        end
    end
end
