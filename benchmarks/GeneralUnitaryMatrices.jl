using Random
using Statistics
using LinearAlgebra

using ManifoldsGPU
using Manifolds
using ManifoldsBase
using CUDA

function _time_median(f; samples::Int = 6)
    timings = Vector{Float64}(undef, samples)
    for i in 1:samples
        GC.gc()
        t0 = time_ns()
        f()
        timings[i] = (time_ns() - t0) / 1.0e6
    end
    return median(timings), timings
end

function _benchmark_cpu_gpu(cpu_f, gpu_f; samples::Int)
    cpu_f()
    gpu_f()

    cpu_ms, cpu_all = _time_median(cpu_f; samples = samples)
    gpu_ms, gpu_all = _time_median(gpu_f; samples = samples)

    return cpu_ms, cpu_all, gpu_ms, gpu_all
end

function _print_results(;
        name::String,
        manifold_label::String,
        samples::Int,
        cpu_all,
        gpu_all,
        cpu_ms::Float64,
        gpu_ms::Float64,
        relerr,
        relerr_label::String,
        extra_lines::Vector{String} = String[],
    )
    speedup = cpu_ms / gpu_ms

    println("=== ManifoldsGPU benchmark: $name on $manifold_label ===")
    println("Element type: Float32")
    for line in extra_lines
        println(line)
    end
    println("Samples: $samples")
    println("CPU times [ms]: ", round.(cpu_all; digits = 2))
    println("GPU times [ms]: ", round.(gpu_all; digits = 2))
    println("Median CPU [ms]: ", round(cpu_ms; digits = 2))
    println("Median GPU [ms]: ", round(gpu_ms; digits = 2))
    println("Speedup (CPU/GPU): ", round(speedup; digits = 2), "x")
    return println("Relative error $relerr_label: ", relerr)
end

function _parse_arg(i::Int, default)
    return length(ARGS) >= i ? parse(typeof(default), ARGS[i]) : default
end

function _setup_rotations_data(; n::Int, batch::Int, scale::Float32, seed::Int)
    Random.seed!(seed)

    M = Rotations(n)
    MP = PowerManifold(M, batch)

    p_cpu = Float32.(rand(MP))
    X_cpu = scale .* Float32.(rand(MP; vector_at = p_cpu))
    q_cpu = Float32.(rand(MP))

    p_gpu = CuArray(p_cpu)
    X_gpu = CuArray(X_cpu)
    q_gpu = CuArray(q_cpu)

    return (; MP, p_cpu, X_cpu, q_cpu, p_gpu, X_gpu, q_gpu)
end

function _setup_unitary_data(; n::Int, batch::Int, scale::Float32, seed::Int)
    Random.seed!(seed)

    M = UnitaryMatrices(n)
    MP = PowerManifold(M, batch)

    p_cpu = ComplexF32.(rand(MP))
    X_cpu = scale .* ComplexF32.(rand(MP; vector_at = p_cpu))
    q_cpu = ComplexF32.(rand(MP))

    p_gpu = CuArray(p_cpu)
    X_gpu = CuArray(X_cpu)
    q_gpu = CuArray(q_cpu)

    return (; MP, p_cpu, X_cpu, q_cpu, p_gpu, X_gpu, q_gpu)
end

function benchmark_rotations(;
        n::Int = 16,
        batch::Int = 2048,
        scale::Float32 = 0.2f0,
        t::Float32 = 0.3f0,
        samples::Int = 6,
        seed::Int = 1234,
    )
    data = _setup_rotations_data(; n = n, batch = batch, scale = scale, seed = seed)
    MP = data.MP
    p_cpu = data.p_cpu
    X_cpu = data.X_cpu
    q_cpu = data.q_cpu
    p_gpu = data.p_gpu
    X_gpu = data.X_gpu
    q_gpu = data.q_gpu

    manifold_label = "PowerManifold(Rotations($n), $batch)"

    # exp!
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
    _print_results(;
        name = "ExponentialRetraction",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Ycpu - Ygpu||/||Ycpu||",
    )

    println()

    # log!
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> log(MP, p_cpu, q_cpu),
        () -> CUDA.@sync log(MP, p_gpu, q_gpu);
        samples = samples,
    )
    relerr = begin
        X_log_cpu = log(MP, p_cpu, q_cpu)
        X_log_gpu = Array(CUDA.@sync log(MP, p_gpu, q_gpu))
        norm(X_log_cpu .- X_log_gpu) / max(norm(X_log_cpu), eps(Float32))
    end
    _print_results(;
        name = "LogarithmicMap",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Xcpu - Xgpu||/||Xcpu||",
    )

    println()

    # retract polar
    q_cpu = similar(p_cpu)
    q_gpu = similar(p_gpu)
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> ManifoldsBase.retract_fused!(MP, q_cpu, p_cpu, X_cpu, t, PolarRetraction()),
        () -> CUDA.@sync ManifoldsBase.retract_fused!(
            MP, q_gpu, p_gpu, X_gpu, t, PolarRetraction(),
        );
        samples = samples,
    )
    relerr = begin
        ManifoldsBase.retract_fused!(MP, q_cpu, p_cpu, X_cpu, t, PolarRetraction())
        CUDA.@sync ManifoldsBase.retract_fused!(
            MP, q_gpu, p_gpu, X_gpu, t, PolarRetraction(),
        )
        q_gpu_h = Array(q_gpu)
        norm(q_cpu .- q_gpu_h) / max(norm(q_cpu), eps(Float32))
    end
    return _print_results(;
        name = "PolarRetraction",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Qcpu - Qgpu||/||Qcpu||",
        extra_lines = ["Retraction scalar t: $t"],
    )
end

function benchmark_unitary(;
        n::Int = 16,
        batch::Int = 2048,
        scale::Float32 = 0.2f0,
        samples::Int = 6,
        seed::Int = 1234,
    )
    data = _setup_unitary_data(; n = n, batch = batch, scale = scale, seed = seed)
    MP = data.MP
    p_cpu = data.p_cpu
    X_cpu = data.X_cpu
    q_cpu = data.q_cpu
    p_gpu = data.p_gpu
    X_gpu = data.X_gpu
    q_gpu = data.q_gpu

    manifold_label = "PowerManifold(UnitaryMatrices($n), $batch)"

    # exp!
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
    _print_results(;
        name = "ExponentialRetraction",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Ycpu - Ygpu||/||Ycpu||",
    )

    println()

    # log!
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> log(MP, p_cpu, q_cpu),
        () -> CUDA.@sync log(MP, p_gpu, q_gpu);
        samples = samples,
    )
    relerr = begin
        X_log_cpu = log(MP, p_cpu, q_cpu)
        X_log_gpu = Array(CUDA.@sync log(MP, p_gpu, q_gpu))
        norm(X_log_cpu .- X_log_gpu) / max(norm(X_log_cpu), eps(Float32))
    end
    return _print_results(;
        name = "LogarithmicMap",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Xcpu - Xgpu||/||Xcpu||",
    )
end

function main()
    n = _parse_arg(1, 16)
    batch = _parse_arg(2, 2048)
    samples = _parse_arg(3, 6)

    println("=== Rotations benchmarks ===")
    println("Running with n=$n, batch=$batch, samples=$samples")
    println()
    benchmark_rotations(; n = n, batch = batch, samples = samples)

    println()
    println("=== UnitaryMatrices benchmarks ===")
    println("Running with n=$n, batch=$batch, samples=$samples")
    println()
    return benchmark_unitary(; n = n, batch = batch, samples = samples)
end

main()
