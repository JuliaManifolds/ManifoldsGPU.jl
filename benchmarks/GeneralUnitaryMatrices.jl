# GeneralUnitaryMatrices-specific benchmarks (Rotations + UnitaryMatrices).
# Run standalone: julia --project benchmarks/GeneralUnitaryMatrices.jl [n] [batch] [samples]
# Or included by main.jl for the combined benchmark suite.

if !@isdefined(_benchmark_exp)
    include(joinpath(@__DIR__, "utils.jl"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = _parse_arg(1, 16)
    batch = _parse_arg(2, 2048)
    samples = _parse_arg(3, 6)

    println("Running GeneralUnitaryMatrices benchmarks: n=$n, batch=$batch, samples=$samples")
    println()

    all_results = NamedTuple[]
    append!(all_results, benchmark_manifold("Rotations($n)", Rotations(n); batch = batch, scale = 0.2f0, samples = samples, seed = 1234, point_type = Float32))
    append!(all_results, benchmark_manifold("UnitaryMatrices($n)", UnitaryMatrices(n); batch = batch, scale = 0.2f0, samples = samples, seed = 1235, point_type = ComplexF32))

    println(generate_markdown_summary_table(all_results))
end
