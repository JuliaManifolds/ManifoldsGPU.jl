function _matrix_exp_gpu(A::CuArray{T, 2}) where {T <: Real}
    E = _matrix_exp_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(E, size(A))
end

function _matrix_exp_gpu(A::CuArray{T, 3}) where {T <: Real}
    n, m, batch = size(A)
    n == m || throw(DimensionMismatch("matrix exponential requires square matrices, got ($n, $m, $batch)"))

    I_n = reshape(CuArray(Matrix{T}(I, n, n)), n, n, 1)
    I_batch = similar(A)
    I_batch .= I_n

    maxabs = maximum(abs, A)
    theta = T <: Float32 ? T(1) : T(1 // 2)
    s = max(0, ceil(Int, log2(float(maxabs * n / theta))))
    As = A ./ T(2)^s

    order = T <: Float32 ? 18 : 30
    E = copy(I_batch)
    term = copy(I_batch)
    for j in 1:order
        term = CUDA.CUBLAS.gemm_strided_batched('N', 'N', term, As)
        term ./= T(j)
        E .+= term
    end

    for _ in 1:s
        E = CUDA.CUBLAS.gemm_strided_batched('N', 'N', E, E)
    end

    return E
end
