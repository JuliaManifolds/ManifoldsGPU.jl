using LinearAlgebra


function ManifoldsBase.exp!(
        M::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    _, k = ManifoldsBase.get_parameter(M.manifold.size)
    batch = size(q, 3)
    A = CUDA.CUBLAS.gemm_strided_batched('T', 'N', p, X)
    XtX = CUDA.CUBLAS.gemm_strided_batched('T', 'N', X, X)

    B_arg = similar(A, 2 * k, 2 * k, batch)
    B_arg[1:k, 1:k, :] .= A
    B_arg[1:k, (k + 1):(2 * k), :] .= -XtX
    B_arg[(k + 1):(2 * k), (k + 1):(2 * k), :] .= A
    B_arg[(k + 1):(2 * k), 1:k, :] .= reshape(CuArray(Matrix{T}(I, k, k)), k, k, 1)

    B = _matrix_exp_gpu(B_arg)
    B11 = view(B, 1:k, 1:k, :)
    B21 = view(B, (k + 1):(2 * k), 1:k, :)

    r = CUDA.CUBLAS.gemm_strided_batched('N', 'N', p, B11)
    r .+= CUDA.CUBLAS.gemm_strided_batched('N', 'N', X, B21)

    E = _matrix_exp_gpu(-A)
    q .= CUDA.CUBLAS.gemm_strided_batched('N', 'N', r, E)

    return q
end

function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        Y::CuArray{T, 3},
    ) where {T <: Real}
    return dot(X, Y)
end

function ManifoldsBase.norm(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    return sqrt(dot(X, X))
end

function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
    ) where {T <: Real}
    q .= p
    return _polar_project_gpu!(q)
end

function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
    ) where {T <: Real}
    A = CUDA.CUBLAS.gemm_strided_batched('C', 'N', p, X)    # p'X, k×k×batch
    sym = A .+ permutedims(A, (2, 1, 3))                     # A + A'
    Y .= X
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', T(-0.5), p, sym, one(T), Y)
    return Y
end

function ManifoldsBase.retract_polar_fused!(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
    ) where {T <: Real}
    q .= p .+ t .* X
    return _polar_project_gpu!(q)
end

function ManifoldsBase.retract_fused!(
        M::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
        ::PolarRetraction,
    ) where {T <: Real}
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, t)
end
