"""
    exp!(M::PowerManifold{ℝ, <:AbstractSphere{ℝ}, ...}, q, p, X)

GPU-native geodesic exponential on the batched sphere.

Strategy: pure broadcasting — no LAPACK required.
  exp_p(X) = cos(θ)·p + sin(θ)/θ·X  where θ = ||X||
Edge case: θ ≈ 0 → sin(θ)/θ → 1 (l'Hôpital), so exp_p(0) = p.
"""
function ManifoldsBase.exp!(
    ::PowerManifold{ℝ, <:AbstractSphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
    q::CuArray{T, 2},
    p::CuArray{T, 2},
    X::CuArray{T, 2},
) where {T <: Real}
    # θ = ||X|| per sample: sum over spatial dimension (dim=1), then sqrt
    θ = sqrt.(sum(X .^ 2; dims = 1))                           # (1, batch)
    sinc_θ = ifelse.(θ .< eps(T), one(T), sin.(θ) ./ θ)       # sin(θ)/θ, handles θ≈0
    q .= cos.(θ) .* p .+ sinc_θ .* X
    return q
end

"""
    log!(M::PowerManifold{ℝ, <:AbstractSphere{ℝ}, ...}, X, p, q)

GPU-native geodesic logarithm on the batched sphere.

Strategy: pure broadcasting.
  log_p(q) = d · (q - cos(d)·p) / sin(d)  where d = arccos(⟨p,q⟩)
Edge case: d ≈ 0 → d/sin(d) → 1, so log_p(p) = 0.
"""
function ManifoldsBase.log!(
    ::PowerManifold{ℝ, <:AbstractSphere{ℝ}, <:Tuple, ArrayPowerRepresentation},
    X::CuArray{T, 2},
    p::CuArray{T, 2},
    q::CuArray{T, 2},
) where {T <: Real}
    cos_d = clamp.(sum(p .* q; dims = 1), -one(T), one(T))    # (1, batch), clamp for acos
    d = acos.(cos_d)
    sinc_d = ifelse.(d .< eps(T), one(T), d ./ sin.(d))        # d/sin(d), handles d≈0
    X .= sinc_d .* (q .- cos_d .* p)
    return X
end
