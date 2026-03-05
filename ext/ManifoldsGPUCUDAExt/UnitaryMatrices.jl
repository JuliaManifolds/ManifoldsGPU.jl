"""
    exp!(M::PowerManifold{ℂ, <:UnitaryMatrices{ℂ}, ...}, q, p, X)

GPU-native geodesic exponential on batched UnitaryMatrices.

Strategy: for skew-Hermitian tangent X, the geodesic is q = p · expm(X).
Uses the complex Scaling & Squaring Taylor series matrix exponential
(_matrix_exp_gpu extended to CuArray{Complex{T},3}).

expm of a skew-Hermitian matrix is unitary, so q inherits p's unitarity.
"""
function ManifoldsBase.exp!(
    ::PowerManifold{ℂ, <:UnitaryMatrices{ℂ}, <:Tuple, ArrayPowerRepresentation},
    q::CuArray{Complex{T}, 3},
    p::CuArray{Complex{T}, 3},
    X::CuArray{Complex{T}, 3},
) where {T <: Real}
    # X is skew-Hermitian: expm(X) is unitary
    # exp_p(X) = p · expm(X)  (geodesic on unitary group)
    expm_X = _matrix_exp_gpu(X)
    q .= CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, expm_X)
    return q
end
