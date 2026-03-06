using GPUArrays

@testset "Euclidean JLArray" begin
    # JLArray tests verify numerical correctness without GPU hardware.
    # Scalar indexing is allowed because the CuArray-specific overrides
    # do not dispatch on JLArray; the default ManifoldsBase PowerManifold
    # path loops per element. CUDA tests verify no scalar indexing.
    GPUArrays.allowscalar(true)

    @testset "exp! and log!" begin
        Random.seed!(50)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        X = randn(8, 4, 64)

        Y_cpu = exp(MP, p, X)

        p_jl = JLArray(p)
        X_jl = JLArray(X)
        Y_jl = exp(MP, p_jl, X_jl)
        @test is_point(MP, Array(Y_jl))
        @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0e-14)

        V_cpu = log(MP, p, Y_cpu)

        V_jl = log(MP, p_jl, JLArray(Y_cpu))
        @test isapprox(Array(V_jl), V_cpu; atol = 2.0e-14)
    end

    @testset "exp! and log! Float32" begin
        Random.seed!(51)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = Float32.(randn(8, 4, 64))
        X = Float32.(randn(8, 4, 64))

        Y_cpu = exp(MP, p, X)

        p_jl = JLArray(p)
        X_jl = JLArray(X)
        Y_jl = exp(MP, p_jl, X_jl)
        @test is_point(MP, Array(Y_jl))
        @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0f-5)

        V_cpu = log(MP, p, Y_cpu)

        V_jl = log(MP, p_jl, JLArray(Y_cpu))
        @test isapprox(Array(V_jl), V_cpu; atol = 2.0f-5)
    end

    @testset "distance, inner, norm" begin
        Random.seed!(52)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        q = randn(8, 4, 64)
        X = randn(8, 4, 64)
        Y = randn(8, 4, 64)

        p_jl = JLArray(p)
        q_jl = JLArray(q)
        X_jl = JLArray(X)
        Y_jl = JLArray(Y)

        d_cpu = distance(MP, p, q)
        d_jl = distance(MP, p_jl, q_jl)
        @test isapprox(d_jl, d_cpu; atol = 1.0e-12)

        v_cpu = inner(MP, p, X, Y)
        v_jl = inner(MP, p_jl, X_jl, Y_jl)
        @test isapprox(v_jl, v_cpu; atol = 1.0e-12)

        n_cpu = norm(MP, p, X)
        n_jl = norm(MP, p_jl, X_jl)
        @test isapprox(n_jl, n_cpu; atol = 1.0e-12)
    end

    @testset "parallel_transport_to!" begin
        Random.seed!(53)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        q = randn(8, 4, 64)
        X = randn(8, 4, 64)

        Z_cpu = similar(X)
        parallel_transport_to!(MP, Z_cpu, p, X, q)

        p_jl = JLArray(p)
        q_jl = JLArray(q)
        X_jl = JLArray(X)
        Z_jl = similar(X_jl)
        parallel_transport_to!(MP, Z_jl, p_jl, X_jl, q_jl)

        @test isapprox(Array(Z_jl), Z_cpu; atol = 2.0e-14)
    end

    @testset "project!" begin
        Random.seed!(54)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        X = randn(8, 4, 64)

        q_cpu = similar(p)
        project!(MP, q_cpu, p)

        p_jl = JLArray(p)
        q_jl = similar(p_jl)
        project!(MP, q_jl, p_jl)
        @test isapprox(Array(q_jl), q_cpu; atol = 2.0e-14)

        Y_cpu = similar(X)
        project!(MP, Y_cpu, p, X)

        X_jl = JLArray(X)
        Y_jl = similar(X_jl)
        project!(MP, Y_jl, p_jl, X_jl)
        @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0e-14)
    end

    @testset "zero_vector!" begin
        Random.seed!(55)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        p_jl = JLArray(p)

        X_cpu = similar(p)
        zero_vector!(MP, X_cpu, p)

        X_jl = similar(p_jl)
        zero_vector!(MP, X_jl, p_jl)
        @test isapprox(Array(X_jl), X_cpu; atol = 2.0e-14)
        @test all(Array(X_jl) .== 0)
    end

    @testset "mid_point!" begin
        Random.seed!(56)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p1 = randn(8, 4, 64)
        p2 = randn(8, 4, 64)

        q_cpu = similar(p1)
        mid_point!(MP, q_cpu, p1, p2)

        p1_jl = JLArray(p1)
        p2_jl = JLArray(p2)
        q_jl = similar(p1_jl)
        mid_point!(MP, q_jl, p1_jl, p2_jl)
        @test isapprox(Array(q_jl), q_cpu; atol = 2.0e-14)
    end

    @testset "vector_transport_to!" begin
        Random.seed!(57)

        M = Euclidean(8, 4)
        MP = PowerManifold(M, 64)

        p = randn(8, 4, 64)
        q = randn(8, 4, 64)
        X = randn(8, 4, 64)

        Y_cpu = similar(X)
        vector_transport_to!(MP, Y_cpu, p, X, q, ParallelTransport())

        p_jl = JLArray(p)
        q_jl = JLArray(q)
        X_jl = JLArray(X)
        Y_jl = similar(X_jl)
        vector_transport_to!(MP, Y_jl, p_jl, X_jl, q_jl, ParallelTransport())
        @test isapprox(Array(Y_jl), Y_cpu; atol = 2.0e-14)
    end

    GPUArrays.allowscalar(false)
end
