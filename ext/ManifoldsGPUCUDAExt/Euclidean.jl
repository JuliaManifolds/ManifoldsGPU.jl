using LinearAlgebra

"""
    exp!(M::PowerManifold{ℝ, <:Euclidean, ...}, q, p, X)

GPU-native exponential on batched Euclidean space.
Strategy: flat manifold — geodesic is q = p + X (pure broadcasting).
"""
function ManifoldsBase.exp!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, N},
        p::CuArray{T, N},
        X::CuArray{T, N},
    ) where {T <: Real, N}
    q .= p .+ X
    return q
end

"""
    log!(M::PowerManifold{ℝ, <:Euclidean, ...}, X, p, q)

GPU-native logarithm on batched Euclidean space.
Strategy: flat manifold — log is X = q - p (pure broadcasting).
"""
function ManifoldsBase.log!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, N},
        p::CuArray{T, N},
        q::CuArray{T, N},
    ) where {T <: Real, N}
    X .= q .- p
    return X
end

"""
    distance(M::PowerManifold{ℝ, <:Euclidean, ...}, p, q)

GPU-native distance on batched Euclidean space.
Strategy: fused broadcast avoids temp array from default log + norm chain.
"""
function ManifoldsBase.distance(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T, N},
        q::CuArray{T, N},
    ) where {T <: Real, N}
    return sqrt(sum((p .- q) .^ 2))
end

"""
    inner(M::PowerManifold{ℝ, <:Euclidean, ...}, p, X, Y)

GPU-native inner product on batched Euclidean space.
Strategy: fused broadcast dot product, summed over all dimensions.
"""
function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        ::CuArray{T, N},
        X::CuArray{T, N},
        Y::CuArray{T, N},
    ) where {T <: Real, N}
    return sum(X .* Y)
end

"""
    norm(M::PowerManifold{ℝ, <:Euclidean, ...}, p, X)

GPU-native norm on batched Euclidean space.
Strategy: fused broadcast, avoids intermediate from default sqrt(inner(...)).
"""
function LinearAlgebra.norm(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        ::CuArray{T, N},
        X::CuArray{T, N},
    ) where {T <: Real, N}
    return sqrt(sum(X .^ 2))
end

"""
    parallel_transport_to!(M::PowerManifold{ℝ, <:Euclidean, ...}, Y, p, X, q)

GPU-native parallel transport on batched Euclidean space.
Strategy: identity transport (Y .= X), avoids PowerManifold per-element loop.
"""
function ManifoldsBase.parallel_transport_to!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, N},
        ::CuArray{T, N},
        X::CuArray{T, N},
        ::CuArray{T, N},
    ) where {T <: Real, N}
    Y .= X
    return Y
end

"""
    project!(M::PowerManifold{ℝ, <:Euclidean, ...}, q, p)

GPU-native point projection on batched Euclidean space.
Strategy: identity (q .= p), avoids PowerManifold per-element loop.
"""
function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, N},
        p::CuArray{T, N},
    ) where {T <: Real, N}
    q .= p
    return q
end

"""
    project!(M::PowerManifold{ℝ, <:Euclidean, ...}, Y, p, X)

GPU-native tangent vector projection on batched Euclidean space.
Strategy: identity (Y .= X), avoids PowerManifold per-element loop.
"""
function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, N},
        ::CuArray{T, N},
        X::CuArray{T, N},
    ) where {T <: Real, N}
    Y .= X
    return Y
end

"""
    zero_vector!(M::PowerManifold{ℝ, <:Euclidean, ...}, X, p)

GPU-native zero tangent vector on batched Euclidean space.
Strategy: fill with zeros, avoids PowerManifold per-element loop.
"""
function ManifoldsBase.zero_vector!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T, N},
        ::CuArray{T, N},
    ) where {T <: Real, N}
    X .= zero(T)
    return X
end

"""
    mid_point!(M::PowerManifold{ℝ, <:Euclidean, ...}, q, p1, p2)

GPU-native geodesic midpoint on batched Euclidean space.
Strategy: q = (p1 + p2) / 2, avoids PowerManifold per-element loop.
"""
function ManifoldsBase.mid_point!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T, N},
        p1::CuArray{T, N},
        p2::CuArray{T, N},
    ) where {T <: Real, N}
    q .= (p1 .+ p2) ./ 2
    return q
end

"""
    vector_transport_to!(M::PowerManifold{ℝ, <:Euclidean, ...}, Y, p, X, q, ...)

GPU-native vector transport on batched Euclidean space.
Strategy: identity (Y .= X), avoids PowerManifold per-element loop.
"""
function ManifoldsBase.vector_transport_to!(
        ::PowerManifold{ℝ, <:Euclidean, <:Tuple, ArrayPowerRepresentation},
        Y::CuArray{T, N},
        ::CuArray{T, N},
        X::CuArray{T, N},
        ::CuArray{T, N},
        ::AbstractVectorTransportMethod,
    ) where {T <: Real, N}
    Y .= X
    return Y
end
