@testset "Grassmann CUDA" begin
    @testset "inner and norm" begin
        Random.seed!(72)

        M = Grassmann(8, 4)
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
        Random.seed!(73)

        M = Grassmann(8, 4)
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

    # GPU exp! uses polar (not QR) — same subspace, different matrix. Compare via distance.

    @testset "exp! batched" begin
        Random.seed!(80)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        for _ in 1:6
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test distance(MP, Y_cu_h, Y) < 2.0e-14
        end
    end

    @testset "exp! batched Float32" begin
        Random.seed!(81)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        for _ in 1:6
            p = Float32.(rand(MP))
            X = Float32(0.25) .* Float32.(rand(MP; vector_at = p))
            Y = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test distance(MP, Y_cu_h, Y) < 2.0f-5
        end
    end

    @testset "exp! fallback large matrices" begin
        Random.seed!(82)

        # Exceeds 32×32 gesvdj! limit, exercises gesvda! fallback
        M = Grassmann(64, 32)
        MP = PowerManifold(M, 4)

        for _ in 1:3
            p = rand(MP)
            X = 0.25 * rand(MP; vector_at = p)
            Y = exp(MP, p, X)

            p_cu = CuArray(p)
            X_cu = CuArray(X)
            Y_cu = exp(MP, p_cu, X_cu)
            Y_cu_h = Array(Y_cu)

            @test is_point(MP, Y_cu_h)
            @test distance(MP, Y_cu_h, Y) < 2.0e-12
        end
    end

    @testset "project! tangent Float64" begin
        Random.seed!(83)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

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

    @testset "project! tangent Float32" begin
        Random.seed!(84)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        X = Float32.(randn(size(p)...))

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

        @test isapprox(Y_cu_h, Y_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "project! point Float64" begin
        Random.seed!(85)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        p_noisy = p .+ 0.01 .* randn(size(p)...)

        q_cpu = similar(p)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(q_cpu, :, :, i), view(p_noisy, :, :, i)
            )
        end

        p_noisy_cu = CuArray(p_noisy)
        q_cu = similar(p_noisy_cu)
        ManifoldsBase.project!(MP, q_cu, p_noisy_cu)
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test isapprox(q_cu_h, q_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "project! point Float32" begin
        Random.seed!(86)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        p_noisy = p .+ Float32(0.01) .* Float32.(randn(size(p)...))

        q_cpu = similar(p)
        for i in 1:size(p, 3)
            ManifoldsBase.project!(
                M, view(q_cpu, :, :, i), view(p_noisy, :, :, i)
            )
        end

        p_noisy_cu = CuArray(p_noisy)
        q_cu = similar(p_noisy_cu)
        ManifoldsBase.project!(MP, q_cu, p_noisy_cu)
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test isapprox(q_cu_h, q_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "retract_polar_fused Float64" begin
        Random.seed!(87)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)
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

    @testset "retract_polar_fused Float32" begin
        Random.seed!(88)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)
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

    # GPU uses Cholesky-QR, CPU uses Householder QR — same subspace, different matrix.
    # Compare via distance (same pattern as exp! tests above).

    @testset "retract_qr_fused Float64" begin
        Random.seed!(89)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)
        t = 0.3

        p = rand(MP)
        X = rand(MP; vector_at = p)

        q = similar(p)
        ManifoldsBase.retract_fused!(MP, q, p, X, t, QRRetraction())

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        q_cu = similar(p_cu)
        ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, QRRetraction())
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test distance(MP, q_cu_h, q) < 2.0e-14
    end

    @testset "retract_qr_fused Float32" begin
        Random.seed!(891)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)
        t = Float32(0.3)

        p = Float32.(rand(MP))
        X = Float32.(rand(MP; vector_at = p))

        q = similar(p)
        ManifoldsBase.retract_fused!(MP, q, p, X, t, QRRetraction())

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        q_cu = similar(p_cu)
        ManifoldsBase.retract_fused!(MP, q_cu, p_cu, X_cu, t, QRRetraction())
        q_cu_h = Array(q_cu)

        @test is_point(MP, q_cu_h)
        @test distance(MP, q_cu_h, q) < 2.0f-5
    end

    @testset "inverse_retract_polar Float64" begin
        Random.seed!(90)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        q = rand(MP)

        X_cpu = similar(p)
        ManifoldsBase.inverse_retract!(MP, X_cpu, p, q, PolarInverseRetraction())

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = similar(p_cu)
        ManifoldsBase.inverse_retract!(MP, X_cu, p_cu, q_cu, PolarInverseRetraction())
        X_cu_h = Array(X_cu)

        @test isapprox(X_cu_h, X_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "inverse_retract_polar Float32" begin
        Random.seed!(91)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        q = Float32.(rand(MP))

        X_cpu = similar(p)
        ManifoldsBase.inverse_retract!(MP, X_cpu, p, q, PolarInverseRetraction())

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = similar(p_cu)
        ManifoldsBase.inverse_retract!(MP, X_cu, p_cu, q_cu, PolarInverseRetraction())
        X_cu_h = Array(X_cu)

        @test isapprox(X_cu_h, X_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "log! Float64" begin
        Random.seed!(92)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        q = rand(MP)

        X_cpu = similar(p)
        log!(MP, X_cpu, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = similar(p_cu)
        log!(MP, X_cu, p_cu, q_cu)
        X_cu_h = Array(X_cu)

        @test isapprox(X_cu_h, X_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "log! Float32" begin
        Random.seed!(93)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        q = Float32.(rand(MP))

        X_cpu = similar(p)
        log!(MP, X_cpu, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = similar(p_cu)
        log!(MP, X_cu, p_cu, q_cu)
        X_cu_h = Array(X_cu)

        @test isapprox(X_cu_h, X_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "log! fallback large matrices Float32" begin
        Random.seed!(931)

        # Exceeds 32×32 gesvdj! limit, exercises gesvda! fallback
        M = Grassmann(64, 32)
        MP = PowerManifold(M, 4)

        p = Float32.(rand(MP))
        q = Float32.(rand(MP))

        X_cpu = similar(p)
        log!(MP, X_cpu, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        X_cu = similar(p_cu)
        log!(MP, X_cu, p_cu, q_cu)
        X_cu_h = Array(X_cu)

        @test isapprox(X_cu_h, X_cpu; atol = 2.0f-4, rtol = 2.0f-4)
    end

    @testset "parallel_transport_direction! Float64" begin
        Random.seed!(94)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        X = rand(MP; vector_at = p)
        Y = 0.25 * rand(MP; vector_at = p)

        Z_cpu = similar(X)
        parallel_transport_direction!(MP, Z_cpu, p, X, Y)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)
        Z_cu = similar(X_cu)
        parallel_transport_direction!(MP, Z_cu, p_cu, X_cu, Y_cu)
        Z_cu_h = Array(Z_cu)

        @test isapprox(Z_cu_h, Z_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "parallel_transport_direction! Float32" begin
        Random.seed!(95)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        X = Float32.(rand(MP; vector_at = p))
        Y = Float32(0.25) .* Float32.(rand(MP; vector_at = p))

        Z_cpu = similar(X)
        parallel_transport_direction!(MP, Z_cpu, p, X, Y)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)
        Z_cu = similar(X_cu)
        parallel_transport_direction!(MP, Z_cu, p_cu, X_cu, Y_cu)
        Z_cu_h = Array(Z_cu)

        @test isapprox(Z_cu_h, Z_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "parallel_transport_direction! fallback large matrices Float32" begin
        Random.seed!(951)

        # Exceeds 32×32 gesvdj! limit, exercises gesvda! fallback
        M = Grassmann(64, 32)
        MP = PowerManifold(M, 4)

        p = Float32.(rand(MP))
        X = Float32.(rand(MP; vector_at = p))
        Y = Float32(0.25) .* Float32.(rand(MP; vector_at = p))

        Z_cpu = similar(X)
        parallel_transport_direction!(MP, Z_cpu, p, X, Y)

        p_cu = CuArray(p)
        X_cu = CuArray(X)
        Y_cu = CuArray(Y)
        Z_cu = similar(X_cu)
        parallel_transport_direction!(MP, Z_cu, p_cu, X_cu, Y_cu)
        Z_cu_h = Array(Z_cu)

        @test isapprox(Z_cu_h, Z_cpu; atol = 2.0f-4, rtol = 2.0f-4)
    end

    @testset "distance Float64" begin
        Random.seed!(96)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = rand(MP)
        q = rand(MP)

        d_cpu = distance(MP, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        d_gpu = distance(MP, p_cu, q_cu)

        @test isapprox(d_gpu, d_cpu; atol = 2.0e-14, rtol = 2.0e-14)
    end

    @testset "distance Float32" begin
        Random.seed!(97)

        M = Grassmann(8, 4)
        MP = PowerManifold(M, 32)

        p = Float32.(rand(MP))
        q = Float32.(rand(MP))

        d_cpu = distance(MP, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        d_gpu = distance(MP, p_cu, q_cu)

        @test isapprox(d_gpu, d_cpu; atol = 2.0f-5, rtol = 2.0f-5)
    end

    @testset "distance fallback large matrices Float32" begin
        Random.seed!(971)

        # Exceeds 32×32 gesvdj! limit, exercises gesvda! fallback
        M = Grassmann(64, 32)
        MP = PowerManifold(M, 4)

        p = Float32.(rand(MP))
        q = Float32.(rand(MP))

        d_cpu = distance(MP, p, q)

        p_cu = CuArray(p)
        q_cu = CuArray(q)
        d_gpu = distance(MP, p_cu, q_cu)

        @test isapprox(d_gpu, d_cpu; atol = 2.0f-4, rtol = 2.0f-4)
    end
end
