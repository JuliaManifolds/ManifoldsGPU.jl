using ManifoldsGPU
using Test
using Random
using LinearAlgebra

using ManifoldsBase, Manifolds

@testset "ManifoldsGPU.jl" begin
    # JLArray tests (CI-safe, no GPU hardware required)
    @testset "JLArray" begin
        using JLArrays
        include(joinpath(@__DIR__, "jlarray", "test_stiefel.jl"))
        include(joinpath(@__DIR__, "jlarray", "test_euclidean.jl"))
    end

    # CUDA tests (requires GPU hardware)
    using CUDA
    if CUDA.functional()
        @testset "CUDA" begin
            include(joinpath(@__DIR__, "cuda", "test_stiefel.jl"))
            include(joinpath(@__DIR__, "cuda", "test_euclidean.jl"))
        end
    else
        @info "CUDA not available, skipping CUDA tests"
    end
end
