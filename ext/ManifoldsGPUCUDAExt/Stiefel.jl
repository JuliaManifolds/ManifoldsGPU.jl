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

function ManifoldsBase.retract_polar_fused!(
        ::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
    ) where {T <: Real}
    q .= p .+ t .* X

    try
        U, _, V = CUDA.CUSOLVER.gesvdj!('V', q)
        k = min(size(U, 2), size(V, 1))
        U_thin = @view U[:, 1:k, :]
        q .= CUDA.CUBLAS.gemm_strided_batched('N', 'T', U_thin, V)
    catch e
        if e isa ArgumentError
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
        M::PowerManifold{ℝ, <:Stiefel{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, 3},
        p::CuArray{T, 3},
        X::CuArray{T, 3},
        t::Number,
        ::PolarRetraction,
    ) where {T <: Real}
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, t)
end
