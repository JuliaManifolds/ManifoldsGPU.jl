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

    println("Running Grassmann benchmarks: n=$n, k=$k, batch=$batch, samples=$samples")
    println()

    results = benchmark_manifold("Grassmann($n, $k)", Grassmann(n, k); batch = batch, scale = 0.2f0, samples = samples, seed = 1234, point_type = Float32, exp_error_fn = _subspace_error)

    println(generate_markdown_summary_table(results))
end
