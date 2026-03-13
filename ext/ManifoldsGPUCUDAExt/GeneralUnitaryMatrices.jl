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
    try
        U, _, V = CUDA.CUSOLVER.gesvdj!('V', q)
        q .= CUDA.CUBLAS.gemm_strided_batched('N', 'C', U, V)
    catch e
        if e isa ArgumentError
            # CPU fallback: gesvdj! fails for matrices larger than supported size
            batch = size(q, 3)
            for i in 1:batch
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
        M::PowerManifold{<:Any, <:Manifolds.GeneralUnitaryMatrices, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
        ::PolarRetraction,
    ) where {T <: Number}
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, t)
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
    try
        U, _, V = CUDA.CUSOLVER.gesvdj!('V', q)
        q .= CUDA.CUBLAS.gemm_strided_batched('N', 'C', U, V)
    catch e
        if e isa ArgumentError
            # CPU fallback: gesvdj! fails for matrices larger than supported size
            batch = size(q, 3)
            for i in 1:batch
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
