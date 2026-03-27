using Random
using Statistics
using LinearAlgebra

using ManifoldsGPU
using Manifolds
using ManifoldsBase
using CUDA

# --- Timing ---

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

# --- Printing ---

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

function _benchmark_result(; manifold_label::String, operation::String, samples::Int, cpu_ms::Float64, gpu_ms::Float64, relerr)
    speedup = gpu_ms == 0.0 ? Inf : cpu_ms / gpu_ms
    return (
        manifold = manifold_label,
        operation = operation,
        samples = samples,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        speedup = speedup,
        relerr = relerr,
    )
end

function generate_markdown_summary_table(results)
    lines = String[
        "| Manifold | Operation | CPU median [ms] | GPU median [ms] | Speedup CPU/GPU | Relative error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for r in results
        cpu_s = string(round(r.cpu_ms; digits = 2))
        gpu_s = string(round(r.gpu_ms; digits = 2))
        speedup_s = string(round(r.speedup; digits = 2))
        relerr_s = string(round(Float64(r.relerr); sigdigits = 4))
        push!(
            lines,
            "| $(r.manifold) | $(r.operation) | $cpu_s | $gpu_s | $speedup_s | $relerr_s |",
        )
    end

    return join(lines, "\n")
end

# --- Error metrics ---

function _relative_error(cpu_res, gpu_res)
    if cpu_res isa Number
        return abs(cpu_res - gpu_res) / max(abs(cpu_res), eps(Float32))
    end
    return norm(cpu_res .- gpu_res) / max(norm(cpu_res), eps(Float32))
end

# Distance-based error for manifolds where GPU and CPU give different
# matrix representatives of the same point (e.g. Grassmann: polar vs QR).
function _subspace_error(MP, cpu_res, gpu_res)
    return distance(MP, cpu_res, gpu_res)
end

# --- Helpers ---

function _method_label(method::AbstractRetractionMethod)
    return string(nameof(typeof(method)))
end

function _parse_arg(i::Int, default)
    return length(ARGS) >= i ? parse(typeof(default), ARGS[i]) : default
end

# --- Data setup ---

function _setup_data(
        M;
        batch::Int,
        scale::Float32,
        seed::Int,
        point_type = Float32,
        use_power_manifold::Bool = true,
    )
    Random.seed!(seed)

    MB = use_power_manifold ? PowerManifold(M, batch) : M

    p_cpu = point_type.(rand(MB))
    q_cpu = point_type.(rand(MB))
    X_cpu = scale .* point_type.(rand(MB; vector_at = p_cpu))
    Y_cpu = scale .* point_type.(rand(MB; vector_at = p_cpu))
    Z_cpu = scale .* rand(point_type, size(p_cpu)...)

    p_gpu = CuArray(p_cpu)
    q_gpu = CuArray(q_cpu)
    X_gpu = CuArray(X_cpu)
    Y_gpu = CuArray(Y_cpu)
    Z_gpu = CuArray(Z_cpu)

    return (; MB, p_cpu, q_cpu, X_cpu, Y_cpu, Z_cpu, p_gpu, q_gpu, X_gpu, Y_gpu, Z_gpu)
end

# --- Generic operation benchmarks ---

function _benchmark_exp(; MP, p_cpu, X_cpu, p_gpu, X_gpu, samples::Int, manifold_label::String, error_fn = nothing)
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> exp(MP, p_cpu, X_cpu),
        () -> CUDA.@sync exp(MP, p_gpu, X_gpu);
        samples = samples,
    )

    cpu_res = exp(MP, p_cpu, X_cpu)
    gpu_res = Array(CUDA.@sync exp(MP, p_gpu, X_gpu))
    relerr = isnothing(error_fn) ? _relative_error(cpu_res, gpu_res) : error_fn(MP, cpu_res, gpu_res)
    relerr_label = isnothing(error_fn) ? "||Ycpu - Ygpu||/||Ycpu||" : "distance(Ycpu, Ygpu)"

    _print_results(
        name = "exp",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = relerr_label,
    )
    return _benchmark_result(
        manifold_label = manifold_label,
        operation = "exp",
        samples = samples,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
    )
end

function _benchmark_log!(; MP, p_cpu, q_cpu, p_gpu, q_gpu, X_cpu, X_gpu, samples::Int, manifold_label::String)
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> log!(MP, X_cpu, p_cpu, q_cpu),
        () -> CUDA.@sync log!(MP, X_gpu, p_gpu, q_gpu);
        samples = samples,
    )

    cpu_res = log!(MP, X_cpu, p_cpu, q_cpu)
    gpu_res = Array(CUDA.@sync log!(MP, X_gpu, p_gpu, q_gpu))
    relerr = _relative_error(cpu_res, gpu_res)

    _print_results(
        name = "log!",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Xcpu - Xgpu||/||Xcpu||",
    )
    return _benchmark_result(
        manifold_label = manifold_label,
        operation = "log!",
        samples = samples,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
    )
end

function _benchmark_inner(; MP, p_cpu, X_cpu, Y_cpu, p_gpu, X_gpu, Y_gpu, samples::Int, manifold_label::String)
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> inner(MP, p_cpu, X_cpu, Y_cpu),
        () -> CUDA.@sync inner(MP, p_gpu, X_gpu, Y_gpu);
        samples = samples,
    )

    cpu_res = inner(MP, p_cpu, X_cpu, Y_cpu)
    gpu_res = CUDA.@sync inner(MP, p_gpu, X_gpu, Y_gpu)
    relerr = _relative_error(cpu_res, gpu_res)

    _print_results(
        name = "inner",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "|icpu - igpu|/|icpu|",
    )
    return _benchmark_result(
        manifold_label = manifold_label,
        operation = "inner",
        samples = samples,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
    )
end

function _benchmark_norm(; MP, p_cpu, X_cpu, p_gpu, X_gpu, samples::Int, manifold_label::String)
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> norm(MP, p_cpu, X_cpu),
        () -> CUDA.@sync norm(MP, p_gpu, X_gpu);
        samples = samples,
    )

    cpu_res = norm(MP, p_cpu, X_cpu)
    gpu_res = CUDA.@sync norm(MP, p_gpu, X_gpu)
    relerr = _relative_error(cpu_res, gpu_res)

    _print_results(
        name = "norm",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "|ncpu - ngpu|/|ncpu|",
    )
    return _benchmark_result(
        manifold_label = manifold_label,
        operation = "norm",
        samples = samples,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
    )
end

function _benchmark_project!(; MP, p_cpu, Z_cpu, p_gpu, Z_gpu, X_cpu, X_gpu, samples::Int, manifold_label::String)
    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> project!(MP, X_cpu, p_cpu, Z_cpu),
        () -> CUDA.@sync project!(MP, X_gpu, p_gpu, Z_gpu);
        samples = samples,
    )

    cpu_res = project!(MP, X_cpu, p_cpu, Z_cpu)
    gpu_res = Array(CUDA.@sync project!(MP, X_gpu, p_gpu, Z_gpu))
    relerr = _relative_error(cpu_res, gpu_res)

    _print_results(
        name = "project!",
        manifold_label = manifold_label,
        samples = samples,
        cpu_all = cpu_all,
        gpu_all = gpu_all,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
        relerr_label = "||Xcpu - Xgpu||/||Xcpu||",
    )
    return _benchmark_result(
        manifold_label = manifold_label,
        operation = "project!",
        samples = samples,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
    )
end

function _benchmark_retraction(
        method::AbstractRetractionMethod;
        MP,
        p_cpu,
        X_cpu,
        p_gpu,
        X_gpu,
        t::Float32,
        samples::Int,
        manifold_label::String,
    )
    q_cpu = similar(p_cpu)
    q_gpu = similar(p_gpu)
    method_name = _method_label(method)

    if method isa ExponentialRetraction
        cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
            () -> exp(MP, p_cpu, X_cpu),
            () -> CUDA.@sync exp(MP, p_gpu, X_gpu);
            samples = samples,
        )

        cpu_res = exp(MP, p_cpu, X_cpu)
        gpu_res = Array(CUDA.@sync exp(MP, p_gpu, X_gpu))
        relerr = _relative_error(cpu_res, gpu_res)

        _print_results(
            name = method_name,
            manifold_label = manifold_label,
            samples = samples,
            cpu_all = cpu_all,
            gpu_all = gpu_all,
            cpu_ms = cpu_ms,
            gpu_ms = gpu_ms,
            relerr = relerr,
            relerr_label = "||Ycpu - Ygpu||/||Ycpu||",
            extra_lines = ["Retraction method: $method_name"],
        )

        return _benchmark_result(
            manifold_label = manifold_label,
            operation = "exp($method_name)",
            samples = samples,
            cpu_ms = cpu_ms,
            gpu_ms = gpu_ms,
            relerr = relerr,
        )
    end

    cpu_ms, cpu_all, gpu_ms, gpu_all = _benchmark_cpu_gpu(
        () -> ManifoldsBase.retract_fused!(MP, q_cpu, p_cpu, X_cpu, t, method),
        () -> CUDA.@sync ManifoldsBase.retract_fused!(MP, q_gpu, p_gpu, X_gpu, t, method);
        samples = samples,
    )

    cpu_res = ManifoldsBase.retract_fused!(MP, q_cpu, p_cpu, X_cpu, t, method)
    gpu_res = Array(CUDA.@sync ManifoldsBase.retract_fused!(MP, q_gpu, p_gpu, X_gpu, t, method))
    relerr = _relative_error(cpu_res, gpu_res)

    _print_results(
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

    return _benchmark_result(
        manifold_label = manifold_label,
        operation = "retract_fused!($method_name)",
        samples = samples,
        cpu_ms = cpu_ms,
        gpu_ms = gpu_ms,
        relerr = relerr,
    )
end

# --- Generic manifold benchmark ---

function benchmark_manifold(
        name::String,
        M;
        batch::Int,
        scale::Float32,
        samples::Int,
        seed::Int,
        point_type,
        use_power_manifold::Bool = true,
        exp_error_fn = nothing,
    )
    data = _setup_data(
        M;
        batch = batch,
        scale = scale,
        seed = seed,
        point_type = point_type,
        use_power_manifold = use_power_manifold,
    )

    MB = data.MB
    manifold_label = use_power_manifold ? "PowerManifold($name, $batch)" : name
    results = NamedTuple[]
    println("=== $name benchmarks ===")
    println("Point element type: $(eltype(data.p_cpu))")
    println()

    push!(results, _benchmark_exp(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label, error_fn = exp_error_fn))
    println()

    push!(results, _benchmark_log!(; MP = MB, p_cpu = data.p_cpu, q_cpu = data.q_cpu, p_gpu = data.p_gpu, q_gpu = data.q_gpu, X_cpu = data.X_cpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_inner(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, Y_cpu = data.Y_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, Y_gpu = data.Y_gpu, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_norm(; MP = MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    push!(results, _benchmark_project!(; MP = MB, p_cpu = data.p_cpu, Z_cpu = data.Z_cpu, p_gpu = data.p_gpu, Z_gpu = data.Z_gpu, X_cpu = data.X_cpu, X_gpu = data.X_gpu, samples = samples, manifold_label = manifold_label))
    println()

    return results
end
