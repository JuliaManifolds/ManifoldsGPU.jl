using Random
using Statistics

using ManifoldsGPU
using Manifolds
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

function benchmark_stiefel_exp(; n::Int = 32, k::Int = 16, batch::Int = 2048, scale::Float32 = 0.2f0, samples::Int = 6, seed::Int = 1234)
    Random.seed!(seed)

    M = Stiefel(n, k)
    MP = PowerManifold(M, batch)

    p_cpu = Float32.(rand(MP))
    X_cpu = scale .* Float32.(rand(MP; vector_at = p_cpu))

    p_gpu = CuArray(p_cpu)
    X_gpu = CuArray(X_cpu)

    exp(MP, p_cpu, X_cpu)
    CUDA.@sync exp(MP, p_gpu, X_gpu)

    cpu_ms, cpu_all = _time_median(; samples = samples) do
        exp(MP, p_cpu, X_cpu)
    end

    gpu_ms, gpu_all = _time_median(; samples = samples) do
        CUDA.@sync exp(MP, p_gpu, X_gpu)
    end

    speedup = cpu_ms / gpu_ms
    relerr = begin
        Y_cpu = exp(MP, p_cpu, X_cpu)
        Y_gpu = Array(CUDA.@sync exp(MP, p_gpu, X_gpu))
        norm(Y_cpu .- Y_gpu) / max(norm(Y_cpu), eps(Float32))
    end

    println("=== ManifoldsGPU benchmark: exp on PowerManifold(Stiefel($n, $k), $batch) ===")
    println("Element type: Float32")
    println("Samples: $samples")
    println("CPU times [ms]: ", round.(cpu_all; digits = 2))
    println("GPU times [ms]: ", round.(gpu_all; digits = 2))
    println("Median CPU [ms]: ", round(cpu_ms; digits = 2))
    println("Median GPU [ms]: ", round(gpu_ms; digits = 2))
    println("Speedup (CPU/GPU): ", round(speedup; digits = 2), "x")
    return println("Relative error ||Ycpu - Ygpu||/||Ycpu||: ", relerr)
end

function _parse_arg(i::Int, default)
    return length(ARGS) >= i ? parse(typeof(default), ARGS[i]) : default
end

function main()
    n = _parse_arg(1, 32)
    k = _parse_arg(2, 16)
    batch = _parse_arg(3, 2048)
    samples = _parse_arg(4, 6)

    println("Running with n=$n, k=$k, batch=$batch, samples=$samples")

    return benchmark_stiefel_exp(; n = n, k = k, batch = batch, samples = samples)
end

main()
