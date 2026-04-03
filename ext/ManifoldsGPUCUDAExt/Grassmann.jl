function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        Y::CuArray{T, 3},
    ) where {T <: Real}
    return dot(X, Y)
end

function ManifoldsBase.norm(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    return sqrt(dot(X, X))
end

# GPU Grassmann exp! via SVD + polar orthogonalization (replaces CPU's qr(z).Q).
# X = U*Σ*V' → z = (p*V*cos(Σ) + U*sin(Σ))*V' → q = polar(z)
function ManifoldsBase.exp!(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    n, k, batch = size(X)

    U, S, V = _batched_svd_gpu(X)
    U_thin = @view U[:, 1:k, :]

    S_col = reshape(S, 1, k, batch)
    V_cos = V .* cos.(S_col)
    U_sin = U_thin .* sin.(S_col)

    term1 = CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, V_cos)
    z_pre = term1 .+ U_sin
    q .= CUDA.CUBLAS.gemm_strided_batched('N', 'C', z_pre, V)

    _polar_project_gpu!(q)

    return q
end

function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    A = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, X)    # p' * X
    Y .= X .- CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, A)  # X - p * (p' * X)
    return Y
end

function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
    ) where {T <: Real}
    q .= p
    return _polar_project_gpu!(q)
end

function ManifoldsBase.retract_polar_fused!(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
    ) where {T <: Real}
    q .= p .+ t .* X
    return _polar_project_gpu!(q)
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
