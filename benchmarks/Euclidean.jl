include("common.jl")

using LinearAlgebra

function _setup_euclidean_data(; n::Int, k::Int, batch::Int, seed::Int)
    Random.seed!(seed)

    M = Euclidean(n, k)
    MP = PowerManifold(M, batch)

    p_cpu = Float32.(randn(n, k, batch))
    X_cpu = Float32.(randn(n, k, batch))
    q_cpu = Float32.(randn(n, k, batch))
    Y_cpu = Float32.(randn(n, k, batch))

    p_gpu = CuArray(p_cpu)
    X_gpu = CuArray(X_cpu)
    q_gpu = CuArray(q_cpu)
    Y_gpu = CuArray(Y_cpu)

    return (; MP, p_cpu, X_cpu, q_cpu, Y_cpu, p_gpu, X_gpu, q_gpu, Y_gpu)
end

function benchmark_euclidean_exp(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> exp(MP, data.p_cpu, data.X_cpu),
        () -> CUDA.@sync exp(MP, data.p_gpu, data.X_gpu);
        samples = samples,
    )

    Y_cpu = exp(MP, data.p_cpu, data.X_cpu)
    Y_gpu = Array(CUDA.@sync exp(MP, data.p_gpu, data.X_gpu))
    relerr = norm(Y_cpu .- Y_gpu) / max(norm(Y_cpu), eps(Float32))

    return _print_results(;
        name = "exp",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Ycpu - Ygpu||/||Ycpu||",
    )
end

function benchmark_euclidean_log(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> log(MP, data.p_cpu, data.q_cpu),
        () -> CUDA.@sync log(MP, data.p_gpu, data.q_gpu);
        samples = samples,
    )

    V_cpu = log(MP, data.p_cpu, data.q_cpu)
    V_gpu = Array(CUDA.@sync log(MP, data.p_gpu, data.q_gpu))
    relerr = norm(V_cpu .- V_gpu) / max(norm(V_cpu), eps(Float32))

    return _print_results(;
        name = "log",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Vcpu - Vgpu||/||Vcpu||",
    )
end

function benchmark_euclidean_distance(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> distance(MP, data.p_cpu, data.q_cpu),
        () -> CUDA.@sync distance(MP, data.p_gpu, data.q_gpu);
        samples = samples,
    )

    d_cpu = distance(MP, data.p_cpu, data.q_cpu)
    d_gpu = CUDA.@sync distance(MP, data.p_gpu, data.q_gpu)
    relerr = abs(d_cpu - d_gpu) / max(abs(d_cpu), eps(Float32))

    return _print_results(;
        name = "distance",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "|dcpu - dgpu|/|dcpu|",
    )
end

function benchmark_euclidean_inner(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> inner(MP, data.p_cpu, data.X_cpu, data.Y_cpu),
        () -> CUDA.@sync inner(MP, data.p_gpu, data.X_gpu, data.Y_gpu);
        samples = samples,
    )

    v_cpu = inner(MP, data.p_cpu, data.X_cpu, data.Y_cpu)
    v_gpu = CUDA.@sync inner(MP, data.p_gpu, data.X_gpu, data.Y_gpu)
    relerr = abs(v_cpu - v_gpu) / max(abs(v_cpu), eps(Float32))

    return _print_results(;
        name = "inner",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "|vcpu - vgpu|/|vcpu|",
    )
end

function benchmark_euclidean_norm(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> norm(MP, data.p_cpu, data.X_cpu),
        () -> CUDA.@sync norm(MP, data.p_gpu, data.X_gpu);
        samples = samples,
    )

    n_cpu = norm(MP, data.p_cpu, data.X_cpu)
    n_gpu = CUDA.@sync norm(MP, data.p_gpu, data.X_gpu)
    relerr = abs(n_cpu - n_gpu) / max(abs(n_cpu), eps(Float32))

    return _print_results(;
        name = "norm",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "|ncpu - ngpu|/|ncpu|",
    )
end

function benchmark_euclidean_parallel_transport(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    Z_cpu = similar(data.X_cpu)
    Z_gpu = similar(data.X_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> parallel_transport_to!(MP, Z_cpu, data.p_cpu, data.X_cpu, data.q_cpu),
        () -> CUDA.@sync parallel_transport_to!(MP, Z_gpu, data.p_gpu, data.X_gpu, data.q_gpu);
        samples = samples,
    )

    parallel_transport_to!(MP, Z_cpu, data.p_cpu, data.X_cpu, data.q_cpu)
    CUDA.@sync parallel_transport_to!(MP, Z_gpu, data.p_gpu, data.X_gpu, data.q_gpu)
    Z_gpu_h = Array(Z_gpu)
    relerr = norm(Z_cpu .- Z_gpu_h) / max(norm(Z_cpu), eps(Float32))

    return _print_results(;
        name = "parallel_transport_to!",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Zcpu - Zgpu||/||Zcpu||",
    )
end

function benchmark_euclidean_project_point(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    q_cpu = similar(data.p_cpu)
    q_gpu = similar(data.p_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> project!(MP, q_cpu, data.p_cpu),
        () -> CUDA.@sync project!(MP, q_gpu, data.p_gpu);
        samples = samples,
    )

    project!(MP, q_cpu, data.p_cpu)
    CUDA.@sync project!(MP, q_gpu, data.p_gpu)
    q_gpu_h = Array(q_gpu)
    relerr = norm(q_cpu .- q_gpu_h) / max(norm(q_cpu), eps(Float32))

    return _print_results(;
        name = "project! (point)",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Qcpu - Qgpu||/||Qcpu||",
    )
end

function benchmark_euclidean_project_vector(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    Y_cpu = similar(data.X_cpu)
    Y_gpu = similar(data.X_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> project!(MP, Y_cpu, data.p_cpu, data.X_cpu),
        () -> CUDA.@sync project!(MP, Y_gpu, data.p_gpu, data.X_gpu);
        samples = samples,
    )

    project!(MP, Y_cpu, data.p_cpu, data.X_cpu)
    CUDA.@sync project!(MP, Y_gpu, data.p_gpu, data.X_gpu)
    Y_gpu_h = Array(Y_gpu)
    relerr = norm(Y_cpu .- Y_gpu_h) / max(norm(Y_cpu), eps(Float32))

    return _print_results(;
        name = "project! (vector)",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Ycpu - Ygpu||/||Ycpu||",
    )
end

function benchmark_euclidean_zero_vector(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    Z_cpu = similar(data.p_cpu)
    Z_gpu = similar(data.p_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> zero_vector!(MP, Z_cpu, data.p_cpu),
        () -> CUDA.@sync zero_vector!(MP, Z_gpu, data.p_gpu);
        samples = samples,
    )

    zero_vector!(MP, Z_cpu, data.p_cpu)
    CUDA.@sync zero_vector!(MP, Z_gpu, data.p_gpu)
    relerr = norm(Array(Z_gpu))

    return _print_results(;
        name = "zero_vector!",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Zgpu|| (should be 0)",
    )
end

function benchmark_euclidean_mid_point(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    q_cpu = similar(data.p_cpu)
    q_gpu = similar(data.p_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> mid_point!(MP, q_cpu, data.p_cpu, data.q_cpu),
        () -> CUDA.@sync mid_point!(MP, q_gpu, data.p_gpu, data.q_gpu);
        samples = samples,
    )

    mid_point!(MP, q_cpu, data.p_cpu, data.q_cpu)
    CUDA.@sync mid_point!(MP, q_gpu, data.p_gpu, data.q_gpu)
    q_gpu_h = Array(q_gpu)
    relerr = norm(q_cpu .- q_gpu_h) / max(norm(q_cpu), eps(Float32))

    return _print_results(;
        name = "mid_point!",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Qcpu - Qgpu||/||Qcpu||",
    )
end

function benchmark_euclidean_vector_transport(; n::Int = 32, k::Int = 16, batch::Int = 2048, samples::Int = 6, seed::Int = 1234)
    data = _setup_euclidean_data(; n = n, k = k, batch = batch, seed = seed)
    MP = data.MP

    Y_cpu = similar(data.X_cpu)
    Y_gpu = similar(data.X_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> vector_transport_to!(MP, Y_cpu, data.p_cpu, data.X_cpu, data.q_cpu, ParallelTransport()),
        () -> CUDA.@sync vector_transport_to!(MP, Y_gpu, data.p_gpu, data.X_gpu, data.q_gpu, ParallelTransport());
        samples = samples,
    )

    vector_transport_to!(MP, Y_cpu, data.p_cpu, data.X_cpu, data.q_cpu, ParallelTransport())
    CUDA.@sync vector_transport_to!(MP, Y_gpu, data.p_gpu, data.X_gpu, data.q_gpu, ParallelTransport())
    Y_gpu_h = Array(Y_gpu)
    relerr = norm(Y_cpu .- Y_gpu_h) / max(norm(Y_cpu), eps(Float32))

    return _print_results(;
        name = "vector_transport_to!",
        manifold_label = "PowerManifold(Euclidean($n, $k), $batch)",
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Ycpu - Ygpu||/||Ycpu||",
    )
end

function main()
    n = _parse_arg(1, 32)
    k = _parse_arg(2, 16)
    batch = _parse_arg(3, 2048)
    samples = _parse_arg(4, 6)

    println("Running Euclidean benchmarks with n=$n, k=$k, batch=$batch, samples=$samples")

    for bench_fn in [
            benchmark_euclidean_exp,
            benchmark_euclidean_log,
            benchmark_euclidean_distance,
            benchmark_euclidean_inner,
            benchmark_euclidean_norm,
            benchmark_euclidean_parallel_transport,
            benchmark_euclidean_project_point,
            benchmark_euclidean_project_vector,
            benchmark_euclidean_zero_vector,
            benchmark_euclidean_mid_point,
            benchmark_euclidean_vector_transport,
        ]
        println()
        bench_fn(; n = n, k = k, batch = batch, samples = samples)
    end
    return
end

main()
