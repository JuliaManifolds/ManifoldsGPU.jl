using ManifoldsGPU
using Test
using Random
using LinearAlgebra

using ManifoldsBase, Manifolds
using CUDA
using GPUArrays

@testset "ManifoldsGPU.jl" begin
    # JLArray tests (CI-safe, no GPU hardware required)
    @testset "JLArray" begin
        using JLArrays
        include(joinpath(@__DIR__, "jlarray", "test_stiefel.jl"))
        include(joinpath(@__DIR__, "jlarray", "test_general_unitary_matrices.jl"))
    end

    # CUDA tests (requires GPU hardware)
    if CUDA.functional()
        @testset "CUDA" begin
            include(joinpath(@__DIR__, "cuda", "test_stiefel.jl"))
            include(joinpath(@__DIR__, "cuda", "test_general_unitary_matrices.jl"))
        end
    else
        @info "CUDA not available, skipping CUDA tests"
    end
end
