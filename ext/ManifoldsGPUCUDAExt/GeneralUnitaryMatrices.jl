using LinearAlgebra

function ManifoldsBase.exp!(
        ::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Number}
    E = _matrix_exp_gpu(X)
    q .= CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, E)
    return q
end

function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        Y::CuArray{T, 3},
    ) where {T <: Number}
    return dot(X, Y)
end

function ManifoldsBase.norm(
        ::PowerManifold{ℝ, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Number}
    return sqrt(dot(X, X))
end

function ManifoldsBase.retract_polar_fused!(
        ::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
    ) where {T <: Number}
    q .= p .+ CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, T(t) .* X)

    # NOTE: This fallback block is intentionally non-differentiable.
    # retract! functions are not differentiated through directly.
    return _polar_project_gpu!(q)
end

function ManifoldsBase.retract_fused!(
        M::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
        ::PolarRetraction,
    ) where {T <: Number}
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, t)
end

function ManifoldsBase.retract_qr_fused!(
        ::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
    ) where {T <: Number}
    q .= p
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', T(t), p, X, one(T), q)
    return _cholesky_qr_gpu!(q)
end

function ManifoldsBase.retract_fused!(
        M::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
        ::QRRetraction,
    ) where {T <: Number}
    return ManifoldsBase.retract_qr_fused!(M, q, p, X, t)
end

function ManifoldsBase.project!(
        ::PowerManifold{
            <:Any,
            <:Manifolds.GeneralUnitaryMatrices{
                <:Any,
                <:Any,
                Manifolds.AbsoluteDeterminantOneMatrixType,
            },
            <:Tuple,
            ArrayPowerRepresentation,
        },
        q::CuArray{T, 3},
        p::CuArray{T, 3},
    ) where {T <: Number}
    q .= p

    # NOTE: This fallback block is intentionally non-differentiable.
    # project! for points is not differentiated through directly.
    return _polar_project_gpu!(q)
end

function ManifoldsBase.project!(
        ::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Number}
    A = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, X)    # p' * X
    B = CUDA.CUBLAS.gemm_strided_batched('C', 'N', X, p)    # X' * p
    Y .= (A .- B) ./ 2
    return Y
end

function ManifoldsBase.log!(
        ::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, 3},
        p::CuArray{T, 3},
        q::CuArray{T, 3},
    ) where {T <: Real}
    U = CUDA.CUBLAS.gemm_strided_batched('T', 'N', p, q)
    X .= _matrix_log_gpu(U)
    X .= (X .- permutedims(X, (2, 1, 3))) ./ T(2)
    return X
end

function ManifoldsBase.log!(
        ::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, 3},
        p::CuArray{T, 3},
        q::CuArray{T, 3},
    ) where {T <: Complex}
    U = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, q)
    X .= _matrix_log_gpu(U)
    X .= (X .- conj.(permutedims(X, (2, 1, 3)))) ./ T(2)
    return X
end
