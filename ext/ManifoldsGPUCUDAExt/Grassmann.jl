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

function ManifoldsBase.retract_qr_fused!(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
    ) where {T <: Real}
    q .= p .+ t .* X
    return _cholesky_qr_gpu!(q)
end

function ManifoldsBase.retract_fused!(
        M::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
        ::QRRetraction,
    ) where {T <: Real}
    return ManifoldsBase.retract_qr_fused!(M, q, p, X, t)
end

function ManifoldsBase.inverse_retract_polar!(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, 3},
        p::CuArray{T, 3},
        q::CuArray{T, 3},
    ) where {T <: Real}
    A = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, q)    # p'q, k×k×batch
    Ainv = _batched_inv_gpu(A)                                # inv(p'q)
    X .= .-p
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), q, Ainv, one(T), X)
    return X
end

function ManifoldsBase.inverse_retract!(
        M::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, 3},
        p::CuArray{T, 3},
        q::CuArray{T, 3},
        ::PolarInverseRetraction,
    ) where {T <: Real}
    return ManifoldsBase.inverse_retract_polar!(M, X, p, q)
end

function ManifoldsBase.log!(
        M::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, 3},
        p::CuArray{T, 3},
        q::CuArray{T, 3},
    ) where {T <: Real}
    n, k, batch = size(p)

    # Step 1: polar inverse retraction
    ManifoldsBase.inverse_retract_polar!(M, X, p, q)

    # Step 2: SVD of X
    U, S, V = _batched_svd_gpu(X)
    U_thin = @view U[:, 1:k, :]

    # Step 3: X = U * diag(atan(S)) * V'
    S_atan = reshape(atan.(S), 1, k, batch)
    U_atan = U_thin .* S_atan
    CUDA.CUBLAS.gemm_strided_batched!('N', 'C', one(T), U_atan, V, zero(T), X)

    return X
end

# GPU parallel transport of X along geodesic in direction Y at point p.
# Formula: Z = (-p*V*sin(S) + U*cos(S)) * U'X + X - U*(U'X)
# where Y = U*S*V' is the SVD decomposition.
# The (I - U*U')*X term is expanded to X - U*(U'X) to avoid an n×n matrix.
function ManifoldsBase.parallel_transport_direction!(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        Z::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        Y::CuArray{T, 3},
    ) where {T <: Real}
    n, k, batch = size(Y)

    U, S, V = _batched_svd_gpu(Y)
    U_thin = CuArray(U[:, 1:k, :])   # materialize: view stride ≠ batch stride for gemm
    S_col = reshape(S, 1, k, batch)

    # U'X  — reused in two terms
    UtX = CUDA.CUBLAS.gemm_strided_batched('C', 'N', U_thin, X)    # k×k×batch

    # (-p*V*diag(sin S) + U*diag(cos S)) * U'X
    pV = CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, V)          # n×k×batch
    core = U_thin .* cos.(S_col) .- pV .* sin.(S_col)              # n×k×batch

    # Z = core * (U'X) + X - U*(U'X)
    Z .= X
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), core, UtX, one(T), Z)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', -one(T), U_thin, UtX, one(T), Z)

    return Z
end

function ManifoldsBase.vector_transport_direction!(
        M::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        Z::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        Y::CuArray{T, 3},
        ::ParallelTransport,
    ) where {T <: Real}
    return ManifoldsBase.parallel_transport_direction!(M, Z, p, X, Y)
end

function ManifoldsBase.distance(
        ::PowerManifold{ℝ, <:Grassmann{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        q::CuArray{T, 3},
    ) where {T <: Real}
    # Polar inverse retraction: diff = q * inv(p'q) - p
    A = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, q)
    Ainv = _batched_inv_gpu(A)
    diff = CUDA.CUBLAS.gemm_strided_batched('N', 'N', q, Ainv) .- p

    # SVD → singular values
    _, S, _ = _batched_svd_gpu(diff)

    # norm(atan.(S)) over (k, batch) gives correct PowerManifold r=2 distance
    return norm(atan.(S))
end
