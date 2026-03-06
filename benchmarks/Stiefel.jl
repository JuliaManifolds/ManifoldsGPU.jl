include("common.jl")

function _setup_stiefel_data(; n::Int, k::Int, batch::Int, scale::Float32, seed::Int)
    Random.seed!(seed)

    M = Stiefel(n, k)
    MP = PowerManifold(M, batch)

    p_cpu = Float32.(rand(MP))
    X_cpu = scale .* Float32.(rand(MP; vector_at = p_cpu))

    p_gpu = CuArray(p_cpu)
    X_gpu = CuArray(X_cpu)

    return (; MP, p_cpu, X_cpu, p_gpu, X_gpu)
end

function _method_label(method::AbstractRetractionMethod)
    return string(nameof(typeof(method)))
end

function benchmark_stiefel_retraction(method::AbstractRetractionMethod; n::Int = 32, k::Int = 16, batch::Int = 2048, scale::Float32 = 0.2f0, t::Float32 = 0.3f0, samples::Int = 6, seed::Int = 1234)
    data = _setup_stiefel_data(; n = n, k = k, batch = batch, scale = scale, seed = seed)
    MP = data.MP
    p_cpu = data.p_cpu
    X_cpu = data.X_cpu
    p_gpu = data.p_gpu
    X_gpu = data.X_gpu

    method_name = _method_label(method)
    manifold_label = "PowerManifold(Stiefel($n, $k), $batch)"

    if method isa ExponentialRetraction
        cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
            () -> exp(MP, p_cpu, X_cpu),
            () -> CUDA.@sync exp(MP, p_gpu, X_gpu);
            samples = samples,
        )

        relerr = begin
            Y_cpu = exp(MP, p_cpu, X_cpu)
            Y_gpu = Array(CUDA.@sync exp(MP, p_gpu, X_gpu))
            norm(Y_cpu .- Y_gpu) / max(norm(Y_cpu), eps(Float32))
        end

        return _print_results(
            name = method_name,
            manifold_label = manifold_label,
            samples = samples,
            cpu_all = cpu_all,
            gpu_all = gpu_all,
            cpu_ms = cpu_ms,
            gpu_ms = gpu_ms,
            relerr = relerr,
            relerr_label = "||Ycpu - Ygpu||/||Ycpu||",
        )
    end

    q_cpu = similar(p_cpu)
    q_gpu = similar(p_gpu)

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> ManifoldsBase.retract_fused!(MP, q_cpu, p_cpu, X_cpu, t, method),
        () -> CUDA.@sync ManifoldsBase.retract_fused!(MP, q_gpu, p_gpu, X_gpu, t, method);
        samples = samples,
    )

    relerr = begin
        ManifoldsBase.retract_fused!(MP, q_cpu, p_cpu, X_cpu, t, method)
        CUDA.@sync ManifoldsBase.retract_fused!(MP, q_gpu, p_gpu, X_gpu, t, method)
        q_gpu_h = Array(q_gpu)
        norm(q_cpu .- q_gpu_h) / max(norm(q_cpu), eps(Float32))
    end

    return _print_results(
        name = method_name,
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Qcpu - Qgpu||/||Qcpu||",
        extra_lines = ["Retraction scalar t: $t", "Retraction method: $method_name"],
    )
end

function main()
    n = _parse_arg(1, 32)
    k = _parse_arg(2, 16)
    batch = _parse_arg(3, 2048)
    samples = _parse_arg(4, 6)

    println("Running with n=$n, k=$k, batch=$batch, samples=$samples")
    println()
    benchmark_stiefel_retraction(ExponentialRetraction(); n = n, k = k, batch = batch, samples = samples)
    println()
    return benchmark_stiefel_retraction(PolarRetraction(); n = n, k = k, batch = batch, samples = samples)
end

main()
