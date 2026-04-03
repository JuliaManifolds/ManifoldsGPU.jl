# Grassmann-specific benchmarks.
# Run standalone: julia --project benchmarks/Grassmann.jl [n] [k] [batch] [samples]
# Or included by main.jl for the combined benchmark suite.

if !@isdefined(_benchmark_exp)
    include(joinpath(@__DIR__, "utils.jl"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = _parse_arg(1, 32)
    k = _parse_arg(2, 16)
    batch = _parse_arg(3, 2048)
    samples = _parse_arg(4, 6)
    scale = 0.2f0
    t = 0.3f0

    println("Running Grassmann benchmarks: n=$n, k=$k, batch=$batch, samples=$samples")
    println()

    results = benchmark_manifold("Grassmann($n, $k)", Grassmann(n, k); batch = batch, scale = scale, samples = samples, seed = 1234, point_type = Float32, exp_error_fn = _subspace_error)

    data = _setup_data(Grassmann(n, k); batch = batch, scale = scale, seed = 1234, point_type = Float32)
    manifold_label = "PowerManifold(Grassmann($n, $k), $batch)"
    push!(results, _benchmark_retraction(PolarRetraction(); MP = data.MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, t = t, samples = samples, manifold_label = manifold_label))
    println()

    println(generate_markdown_summary_table(results))
end
