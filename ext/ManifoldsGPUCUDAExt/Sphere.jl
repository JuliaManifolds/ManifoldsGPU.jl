function ManifoldsBase.inner(
        ::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T},
        X::CuArray{T},
        Y::CuArray{T},
    ) where {T <: Real}
    return dot(X, Y)
end

function ManifoldsBase.norm(
        ::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
        p::CuArray{T},
        X::CuArray{T},
    ) where {T <: Real}
    return sqrt(dot(X, X))
end

function ManifoldsBase.exp!(
        ::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T},
        p::CuArray{T},
        X::CuArray{T},
    ) where {T <: Real}
    θ = sqrt.(sum(abs2, X; dims = 1))
    q .= cos.(θ) .* p .+ Manifolds.usinc.(θ) .* X
    return q
end

function ManifoldsBase.log!(
        M::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
        X::CuArray{T},
        p::CuArray{T},
        q::CuArray{T},
    ) where {T <: Real}
    cosθ = clamp.(sum(p .* q; dims = 1), -one(T), one(T))
    θ = acos.(cosθ)

    X_regular = (q .- cosθ .* p) ./ Manifolds.usinc.(θ)

    antipodal = abs.(cosθ .+ one(T)) .<= sqrt(eps(T))
    basis = CUDA.zeros(T, size(p))
    basis[1, :] .= one(T)
    if size(p, 1) > 1
        p1_is_one = abs.(p[1:1, :] .- one(T)) .<= sqrt(eps(T))
        basis[1, :] .-= T.(p1_is_one[1, :])
        basis[2, :] .= T.(p1_is_one[1, :])
    end

    X_antipodal = basis .- p .* sum(p .* basis; dims = 1)
    X_antipodal .*= T(π) ./ sqrt.(sum(abs2, X_antipodal; dims = 1))

    X .= ifelse.(antipodal, X_antipodal, X_regular)
    return project!(M, X, p, X)
end

function ManifoldsBase.project!(
        ::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
    Y::CuArray{T},
    p::CuArray{T},
    X::CuArray{T},
    ) where {T <: Real}
    Y .= X .- p .* sum(p .* X; dims = 1)
    return Y
end

function ManifoldsBase.project!(
    ::PowerManifold{ℝ, <:Sphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
        q::CuArray{T},
        p::CuArray{T},
    ) where {T <: Real}
    norms_p = sqrt.(sum(abs2, p; dims = 1))
    q .= p ./ norms_p
    return q
end
