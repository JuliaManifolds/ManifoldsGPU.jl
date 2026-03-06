using Random
using Statistics

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
