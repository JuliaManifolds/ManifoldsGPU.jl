# Combined benchmark suite for all manifolds.
# Run: julia --project benchmarks/main.jl [n] [k] [batch] [samples]
#
# Per-manifold benchmarks can also be run standalone:
#   julia --project benchmarks/Grassmann.jl [n] [k] [batch] [samples]
#   julia --project benchmarks/GeneralUnitaryMatrices.jl [n] [batch] [samples]

include(joinpath(@__DIR__, "utils.jl"))

function _benchmark_extra_retractions(name::String, M; batch::Int, scale::Float32, t::Float32, samples::Int, seed::Int, point_type, methods, error_fn = nothing)
    data = _setup_data(M; batch = batch, scale = scale, seed = seed, point_type = point_type, use_power_manifold = true)
    manifold_label = "PowerManifold($name, $batch)"
    results = NamedTuple[]

    println("=== $name retraction benchmarks ===")
    println("Point element type: $(eltype(data.p_cpu))")
    println()

    for method in methods
        push!(results, _benchmark_retraction(method; MP = data.MB, p_cpu = data.p_cpu, X_cpu = data.X_cpu, p_gpu = data.p_gpu, X_gpu = data.X_gpu, t = t, samples = samples, manifold_label = manifold_label, error_fn = error_fn))
        println()
    end

    return results
end

function main()
    n = _parse_arg(1, 32)
    k = _parse_arg(2, 16)
    batch = _parse_arg(3, 2048)
    samples = _parse_arg(4, 6)
    scale = 0.2f0
    t = 0.3f0
    seed = 1234

    println("Running with n=$n, k=$k, batch=$batch, samples=$samples")
    println()

    all_results = NamedTuple[]

    append!(all_results, benchmark_manifold("Euclidean($n, $k, $batch)", Euclidean(n, k, batch); batch = batch, scale = scale, samples = samples, seed = seed, point_type = Float32, use_power_manifold = false))

    append!(all_results, benchmark_manifold("Sphere($(n - 1))", Sphere(n - 1); batch = batch, scale = scale, samples = samples, seed = seed + 1, point_type = Float32))

    append!(all_results, benchmark_manifold("Rotations($n)", Rotations(n); batch = batch, scale = scale, samples = samples, seed = seed + 2, point_type = Float32))

    append!(all_results, _benchmark_extra_retractions("Rotations($n)", Rotations(n); batch = batch, scale = scale, t = t, samples = samples, seed = seed + 2, point_type = Float32, methods = [PolarRetraction(), QRRetraction()]))

    append!(all_results, benchmark_manifold("UnitaryMatrices($n)", UnitaryMatrices(n); batch = batch, scale = scale, samples = samples, seed = seed + 3, point_type = ComplexF32))

    append!(all_results, benchmark_manifold("Grassmann($n, $k)", Grassmann(n, k); batch = batch, scale = scale, samples = samples, seed = seed + 4, point_type = Float32, exp_error_fn = _subspace_error))

    append!(all_results, _benchmark_extra_retractions("Grassmann($n, $k)", Grassmann(n, k); batch = batch, scale = scale, t = t, samples = samples, seed = seed + 4, point_type = Float32, methods = [PolarRetraction(), QRRetraction()], error_fn = _subspace_error))

    append!(all_results, _benchmark_extra_retractions("Stiefel($n, $k)", Stiefel(n, k); batch = batch, scale = scale, t = t, samples = samples, seed = seed + 5, point_type = Float32, methods = [ExponentialRetraction(), PolarRetraction(), QRRetraction()]))

    markdown_table = generate_markdown_summary_table(all_results)
    println("=== Markdown summary table ===")
    println("Device: ", CUDA.name(CUDA.device()), ", eltype: Float32/ComplexF32")
    println(markdown_table)

    return markdown_table
end

main()
