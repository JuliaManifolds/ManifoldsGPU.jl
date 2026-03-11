function _matrix_exp_gpu(A::CuArray{T, 2}) where {T <: Real}
    E = _matrix_exp_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(E, size(A))
end

function _matrix_exp_gpu(A::CuArray{T, 2}) where {T <: Complex}
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

function _matrix_exp_gpu(A::CuArray{T, 3}) where {T <: Complex}
    n, m, batch = size(A)
    n == m ||
        throw(DimensionMismatch("matrix exponential requires square matrices, got ($n, $m, $batch)"))

    RT = real(T)
    I_n = reshape(CuArray(Matrix{T}(I, n, n)), n, n, 1)
    I_batch = similar(A)
    I_batch .= I_n

    maxabs = maximum(abs, A)
    theta = RT <: Float32 ? RT(1) : RT(1 // 2)
    s = max(0, ceil(Int, log2(float(real(maxabs) * n / theta))))
    As = A ./ T(2)^s

    order = RT <: Float32 ? 18 : 30
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

"""
    _matrix_log_gpu(A::CuArray{T, 2})
    _matrix_log_gpu(A::CuArray{T, 3})

GPU matrix logarithm via Inverse Scaling & Squaring with Denman-Beavers iteration.

Uses the identity `log(A) = 2^s * log(A^{1/2^s})`, where repeated matrix square
roots bring `A` close to `I`, then a Taylor series computes `log(I + X)`. Matrix
square roots are computed via the Denman-Beavers iteration using batched LU-based
inversion (`getrf_strided_batched!` + `getri_strided_batched!`).

Real inputs are promoted to complex (square roots of unitary matrices may have
complex entries), then the real part is taken.

Parameters (tuned for `Rotations(n)` with `n ≤ 32`):
- `sqrtm_count=4`: number of repeated square roots (scaling factor `2^s`)
- `db_iters=10`: Denman-Beavers iterations per square root
- `taylor_order=16`: terms in `log(I+X)` Taylor series
"""
function _matrix_log_gpu(A::CuArray{T, 2}) where {T <: Real}
    L = _matrix_log_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(L, size(A))
end

function _matrix_log_gpu(A::CuArray{T, 2}) where {T <: Complex}
    L = _matrix_log_gpu(reshape(A, size(A, 1), size(A, 2), 1))
    return reshape(L, size(A))
end

# Batched matrix inverse via LU factorization (used by Denman-Beavers)
function _batched_inv_gpu(A::CuArray{T, 3}) where {T}
    A_lu = copy(A)
    pivot = CUDA.zeros(Int32, size(A, 1), size(A, 3))
    CUDA.CUBLAS.getrf_strided_batched!(A_lu, pivot)
    C = similar(A)
    CUDA.CUBLAS.getri_strided_batched!(A_lu, C, pivot)
    return C
end

# Denman-Beavers iteration for batched matrix square root
function _batched_sqrtm_gpu(A::CuArray{T, 3}; iters::Int = 10) where {T}
    nn = size(A, 1)
    I_n = reshape(CuArray(Matrix{T}(I, nn, nn)), nn, nn, 1)
    Y = copy(A)
    Z = similar(A)
    Z .= I_n
    for _ in 1:iters
        Zinv = _batched_inv_gpu(Z)
        Yinv = _batched_inv_gpu(Y)
        Y = (Y .+ Zinv) ./ T(2)
        Z = (Z .+ Yinv) ./ T(2)
    end
    return Y
end

function _matrix_log_gpu(
        A::CuArray{T, 3}; sqrtm_count::Int = 4, db_iters::Int = 10, taylor_order::Int = 16,
    ) where {T <: Complex}
    n, m, batch = size(A)
    n == m ||
        throw(DimensionMismatch("matrix logarithm requires square matrices, got ($n, $m, $batch)"))

    # Inverse Scaling: take s repeated square roots to bring A close to I
    B = copy(A)
    for _ in 1:sqrtm_count
        B = _batched_sqrtm_gpu(B; iters = db_iters)
    end

    # Taylor series: log(I + X) = X - X²/2 + X³/3 - ...
    I_n = reshape(CuArray(Matrix{T}(I, n, n)), n, n, 1)
    X_mat = B .- I_n
    L = copy(X_mat)
    term = copy(X_mat)
    for j in 2:taylor_order
        term = CUDA.CUBLAS.gemm_strided_batched('N', 'N', term, X_mat)
        sign_j = iseven(j) ? T(-1) : T(1)
        L .+= sign_j .* term ./ T(j)
    end

    # Squaring: undo the scaling
    L .*= T(2)^sqrtm_count
    return L
end

function _matrix_log_gpu(A::CuArray{T, 3}) where {T <: Real}
    CT = complex(T)
    Ac = CuArray{CT}(A)
    logAc = _matrix_log_gpu(Ac)
    return real.(logAc)
end
