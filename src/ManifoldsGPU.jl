"""
    ManifoldsGPU

A package providing GPU support for the JuliaManifolds ecosystem.
"""
module ManifoldsGPU

using ManifoldsBase, Manifolds, ManifoldDiff

"""
    _matrix_exp_gpu(A::AbstractArray{T, 2})
    _matrix_exp_gpu(A::AbstractArray{T, 3})

GPU matrix exponentials helper function. For a 3-index array `A` it computes exponentials
of `A[:, :, i]` for each `i` separately.
"""
function _matrix_exp_gpu(A::AbstractArray) end

"""
    _matrix_log_gpu(A::AbstractArray{T, 2})
    _matrix_log_gpu(A::AbstractArray{T, 3})

GPU matrix logarithm helper function. For a 3-index array `A` it computes logarithms
of `A[:, :, i]` for each `i` separately.
"""
function _matrix_log_gpu(A::AbstractArray) end

end
