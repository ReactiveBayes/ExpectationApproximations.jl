"""
    EADefault(n = 16)

The default backend: a router that sends each distribution to the rule made for its family, with
`n` points per dimension for a Gaussian and for the quantile rule. The rule in unconstrained
coordinates keeps its own default, 8 points per panel. It is

```julia
BayesBase.ExpectationFirstOf(ExactSum(), GaussHermite(n), UnconstrainedLegendre(), QuantileLegendre(n))
```

so the first backend that supports a distribution computes its expectations:

| Distribution | Backend |
|:-------------|:--------|
| finite discrete (`Categorical`, `Bernoulli`, `Binomial`, …) | [`ExactSum`](@ref), exact |
| `Normal`, `MvNormal`, `LogNormal`, `LogitNormal`, [`TransformedGaussian`](@ref) | [`GaussHermite`](@ref), with at most 10 000 points in many dimensions |
| `Gamma`, `Exponential`, `Chisq`, `Erlang`, `InverseGamma`, `Weibull`, `Beta` | [`UnconstrainedLegendre`](@ref), in `log x` or `logit x` |
| any other continuous univariate distribution with a quantile function | [`QuantileLegendre`](@ref) |

Gauss–Laguerre and Gauss–Jacobi are exact for polynomials in `x`, but integrands such as `log x`
are singular at the boundary, where they converge slowly: with 16 points, `E[log x]` under
`Gamma(0.1, 1)` is off by 6.1. [`GaussLaguerre`](@ref) and [`GaussJacobi`](@ref) remain available
to choose explicitly.

The choice is made from the types alone, so the compiler resolves it. A product of distributions,
a `BayesBase.FactorizedJoint`, is routed component by component. A distribution none of them
covers, such as a `Poisson`, is not supported, and `BayesBase.expectation` throws
`BayesBase.UnsupportedExpectationError`.

```jldoctest
julia> BayesBase.expectation(EA.EADefault(), x -> x^2, Gamma(2.0, 1.0)) ≈ 6.0
true

julia> BayesBase.which_expectation_backend(EA.EADefault(), log, Beta(2.0, 3.0)) isa EA.UnconstrainedLegendre
true

julia> BayesBase.which_expectation_backend(EA.EADefault(), log, Poisson(3.0)) === nothing
true
```
"""
function EADefault(n::Integer = 16)
    return ExpectationFirstOf(ExactSum(), GaussHermite(n), UnconstrainedLegendre(), QuantileLegendre(n))
end
