"""
    mean_cov(backend, f, q) -> (m, V)
    mean_cov(prepared, f) -> (m, V)

The mean and covariance of the pushforward of `q` through `f`: `m = E[f(x)]` and
`V = E[(f(x) - m)(f(x) - m)ᵀ]`, a variance for a scalar `f`, for `x ~ q`. Unlike
`BayesBase.mean_cov(q)`, the moments of a distribution, these are the moments of a function of
it, computed by `backend` or by an expectation prepared once for many functions.
"""
function mean_cov(prepared, f::F) where {F}
    m = expectation(prepared, f)
    V = expectation(prepared, x -> outer(f(x) - m))
    return m, V
end

mean_cov(backend, f::F, q) where {F} = mean_cov(prepare_expectation(backend, q), f)

"""
    mean_cov_cross(backend, f, q) -> (m, V, C)
    mean_cov_cross(prepared, f) -> (m, V, C)

[`mean_cov`](@ref) and the cross-covariance `C = E[(x - E[x])(f(x) - m)ᵀ]` between the input and
the output: a scalar, a vector for a vector input and a scalar output, a row for a scalar input
and a vector output, a matrix otherwise.
"""
function mean_cov_cross(prepared, f::F) where {F}
    mx = expectation(prepared, identity)
    m, V = mean_cov(prepared, f)
    C = expectation(prepared, x -> (x - mx) * (f(x) - m)')
    return m, V, C
end

mean_cov_cross(backend, f::F, q) where {F} = mean_cov_cross(prepare_expectation(backend, q), f)

"""
    tilted_mean_cov(backend, logf, q) -> (logZ, m, V)

The moments of the tilted distribution `q(x) exp(logf(x)) / Z`, the step of moment matching in
expectation propagation: the log-normaliser `logZ = log E_q[exp(logf)]`, the mean and the
covariance. Computed in log space, shifted by the largest log-density at a point, so extreme
log-densities neither overflow nor underflow. For a product of distributions of mixed kinds the
mean and covariance are tuples, one entry per component.

The backend must prepare a point set for `q`: every quadrature, sampling and exact-sum backend
does, and a router uses the one it chooses for `q`.
"""
function tilted_mean_cov(backend, logf::L, q) where {L}
    chosen = which_expectation_backend(backend, Any, typeof(q))
    isnothing(chosen) && throw(UnsupportedExpectationError(backend, logf, q))
    return tilted_mean_cov(prepare_expectation(chosen, q), logf)
end

function tilted_mean_cov(prepared::Union{ExpectationPointSet, ExpectationProductPointSet}, logf::L) where {L}
    # `logf` once per point: it is the costly part, a projection's whole tilted density
    points = collect(eachpoint(prepared))
    logs = map(((x, _),) -> logf(x), points)
    shift = maximum(logs)
    tilted = map(((x, w), l) -> (x, w * exp(l - shift)), points, logs)
    total = weightedsum(last, tilted)
    m = weightedsum(((x, t),) -> scaled(t / total, x), tilted)
    V = weightedsum(((x, t),) -> scaled(t / total, outer(minus(x, m))), tilted)
    return shift + log(total), m, V
end

# Points and weights of a point set, and of the product rule over several
eachpoint(set::ExpectationPointSet) = zip(set.points, set.weights)

function eachpoint(product::ExpectationProductPointSet)
    sets = product.sets
    indices = CartesianIndices(map(set -> Base.OneTo(length(set)), sets))
    return (
        (map((set, i) -> set.points[i], sets, Tuple(index)), prod(map((set, i) -> set.weights[i], sets, Tuple(index))))
            for index in indices
    )
end

# A sum that starts from its first term, so its type follows the summands (dual numbers,
# vectors, tuples)
weightedsum(g, points) = foldl(add, Iterators.map(g, points))

# Arithmetic on points that may be tuples, for products: componentwise, and with marginal second
# moments
add(a, b) = a + b
add(a::Tuple, b::Tuple) = map(add, a, b)
scaled(w, x) = w * x
scaled(w, x::Tuple) = map(xi -> scaled(w, xi), x)
minus(x, m) = x - m
minus(x::Tuple, m::Tuple) = map(minus, x, m)
outer(d::Number) = d * d'
outer(d::AbstractVector) = d * d'
outer(d::Tuple) = map(outer, d)
