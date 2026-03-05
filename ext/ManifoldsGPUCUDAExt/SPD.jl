using LinearAlgebra

"""
    exp!(M::PowerManifold{ℝ, <:SymmetricPositiveDefinite, ...}, q, p, X)

GPU-assisted geodesic exponential on batched SPD matrices.

Strategy: serial loop over batch — each (n,n) SPD exp uses CPU eigendecomposition
via eigen(Symmetric(...)). Slices are transferred CPU↔GPU per batch element.

TODO: Replace with true GPU-batched implementation when CUDA.CUSOLVER exposes
syevjBatched (batched symmetric eigendecomposition).

Formula: exp_p(X) = p^{1/2} · expm(p^{-1/2} · X · p^{-1/2}) · p^{1/2}
"""
function ManifoldsBase.exp!(
    ::PowerManifold{ℝ, <:SymmetricPositiveDefinite, <:Tuple, ArrayPowerRepresentation},
    q::CuArray{T, 3},
    p::CuArray{T, 3},
    X::CuArray{T, 3},
) where {T <: Real}
    batch = size(q, 3)
    for i in 1:batch
        # Transfer to CPU for eigendecomposition (no batched GPU alternative yet)
        p_i = Array(@view p[:, :, i])
        X_i = Array(@view X[:, :, i])

        # Eigendecomposition of p_i (symmetric PD): p = V * diag(λ) * V'
        λ, V = eigen(Symmetric(p_i))
        sqrt_λ    = sqrt.(λ)
        invsqrt_λ = inv.(sqrt_λ)
        p_sqrt    = V * Diagonal(sqrt_λ)    * V'
        p_invsqrt = V * Diagonal(invsqrt_λ) * V'

        # Symmetric inner matrix: S = p^{-1/2} · X · p^{-1/2}
        S = Symmetric(p_invsqrt * X_i * p_invsqrt)

        # Matrix exponential of symmetric S via eigendecomposition
        μ, W  = eigen(S)
        expS  = W * Diagonal(exp.(μ)) * W'

        # Geodesic: q = p^{1/2} · exp(S) · p^{1/2}
        # Symmetrize to cancel floating-point rounding: product of symmetric
        # matrices is only approximately symmetric in finite precision.
        result = p_sqrt * expS * p_sqrt
        @view(q[:, :, i]) .= CuArray(T.(0.5 .* (result .+ result')))
    end

    return q
end
