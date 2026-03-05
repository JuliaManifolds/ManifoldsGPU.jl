using LinearAlgebra

"""
    exp!(M::PowerManifold{ℝ, <:Grassmann{ℝ}, ...}, q, p, X)

GPU-native geodesic exponential on the batched Grassmann manifold.

Strategy: thin SVD of tangent X via cuSOLVER gesvdj! (batched), then
  exp_p(X) = (p·V·cos(S) + U·sin(S))·V'
where X = U·diag(S)·V' is the thin SVD.

gesvdj! returns V directly (not V'); the 'T' flag in the final
gemm_strided_batched call transposes it to obtain V'.
"""
function ManifoldsBase.exp!(
    M::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
    q::CuArray{T, 3},
    p::CuArray{T, 3},
    X::CuArray{T, 3},
) where {T <: Real}
    _, k, batch = size(q)
    X_svd = copy(X)  # gesvdj! overwrites input in place
    U_full, S, V = CUDA.CUSOLVER.gesvdj!('V', X_svd)
    # gesvdj! returns full U: (n, n, batch); take thin columns
    # S: (k, batch), V: (k, k, batch)
    U = @view U_full[:, 1:k, :]  # thin U: (n, k, batch)

    cos_S = reshape(cos.(S), 1, k, batch)  # broadcast over spatial dim
    sin_S = reshape(sin.(S), 1, k, batch)

    # exp_p(X) = (p·V·cos(S) + U·sin(S)) · V'
    pV     = CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, V)  # (n, k, batch)
    pV_cos = pV .* cos_S                                        # scale cols by cos(S)
    U_sin  = U  .* sin_S                                        # scale cols by sin(S)

    # (pV_cos + U_sin) · V'  — 'T' transposes V to get V'
    q .= CUDA.CUBLAS.gemm_strided_batched('N', 'T', pV_cos .+ U_sin, V)

    return q
end

"""
    retract_polar_fused!(M::PowerManifold{ℝ, <:Grassmann{ℝ}, ...}, q, p, X, t)

GPU-native polar retraction on the batched Grassmann manifold.

Strategy: project p + t·X onto Stiefel via batched SVD (gesvdj!).
Same formula as Stiefel polar retraction — the resulting orthonormal
columns represent the same subspace under the Grassmann quotient.
"""
function ManifoldsBase.retract_polar_fused!(
    ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
    q::CuArray{T, 3},
    p::CuArray{T, 3},
    X::CuArray{T, 3},
    t::Number,
) where {T <: Real}
    q .= p .+ t .* X

    # NOTE: This fallback block is intentionally non-differentiable.
    # It must NOT be called inside a Zygote differentiation path.
    try
        U, _, V = CUDA.CUSOLVER.gesvdj!('V', q)
        k_thin = min(size(U, 2), size(V, 1))
        U_thin = @view U[:, 1:k_thin, :]
        q .= CUDA.CUBLAS.gemm_strided_batched('N', 'T', U_thin, V)
    catch e
        if e isa ArgumentError
            # gesvdj! size limit exceeded: fall back to per-slice svd!.
            # copy(@view q[:,:,i]) preserves the GPU array type (CuArray),
            # so svd! here dispatches to cuSOLVER (not LAPACK) — no CPU transfer.
            for i in 1:size(q, 3)
                q_i = copy(@view q[:, :, i])
                s = svd!(q_i)
                @view(q[:, :, i]) .= s.U * s.Vt
            end
        else
            rethrow()
        end
    end

    return q
end

function ManifoldsBase.retract_fused!(
    M::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
    q::CuArray{T, 3},
    p::CuArray{T, 3},
    X::CuArray{T, 3},
    t::Number,
    ::PolarRetraction,
) where {T <: Real}
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, t)
end
