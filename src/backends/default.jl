"""
    EADefault(n = 16)

The default backend: a router that sends each distribution to the rule made for its family, with
`n` points per dimension. It is

```julia
BayesBase.ExpectationFirstOf(ExactSum(), GaussHermite(n), GaussLaguerre(n), GaussJacobi(n), QuantileLegendre(n))
```

so the first backend that supports a distribution computes its expectations:

| Distribution | Backend |
|:-------------|:--------|
| finite discrete (`Categorical`, `Bernoulli`, `Binomial`, …) | [`ExactSum`](@ref), exact |
| `Normal`, `MvNormal`, `LogNormal`, `LogitNormal`, [`TransformedGaussian`](@ref) | [`GaussHermite`](@ref) |
| `Gamma`, `Exponential` | [`GaussLaguerre`](@ref) |
| `Beta` | [`GaussJacobi`](@ref) |
| any other continuous univariate distribution | [`QuantileLegendre`](@ref) |

The choice is made from the types alone, so the compiler resolves it. A product of distributions,
a `BayesBase.FactorizedJoint`, is routed component by component. A distribution none of them
covers, such as a `Poisson`, is not supported, and `BayesBase.expectation` throws
`BayesBase.UnsupportedExpectationError`.

```jldoctest
julia> BayesBase.expectation(EA.EADefault(), x -> x^2, Gamma(2.0, 1.0)) ≈ 6.0
true

julia> BayesBase.which_expectation_backend(EA.EADefault(), log, Beta(2.0, 3.0)) isa EA.GaussJacobi
true

julia> BayesBase.which_expectation_backend(EA.EADefault(), log, Poisson(3.0)) === nothing
true
```
"""
function EADefault(n::Integer = 16)
    return ExpectationFirstOf(ExactSum(), GaussHermite(n), GaussLaguerre(n), GaussJacobi(n), QuantileLegendre(n))
end
