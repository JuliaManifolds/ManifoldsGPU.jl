module ManifoldsGPUCUDAExt

using Manifolds
using ManifoldsBase

using CUDACore
using cuBLAS
using cuSOLVER
using LinearAlgebra: norm, dot

import ManifoldsGPU: _matrix_log_gpu, _matrix_exp_gpu

import ManifoldsBase: distance, inverse_retract!, inverse_retract_polar!, log!
import ManifoldsBase: parallel_transport_direction!, vector_transport_direction!

using Manifolds: PolarInverseRetraction, ParallelTransport

include("helpers.jl")

include("GeneralUnitaryMatrices.jl")
include("Grassmann.jl")
include("Sphere.jl")
include("Stiefel.jl")

end
