```@meta
CurrentModule = ExpectationApproximations.MPRACompat
```

# Migrating

## From ExpectationApproximations 0.x

Version 1.0 is a rewrite. The previous API is removed: `approximate` and its kernels, CVI, CVMP,
sampling with AdvancedHMC and the constrained unscented transform. Most dependencies are removed
with it. The replacements are:

| Before | Now |
|:-------|:----|
| `approximate(method, f, q)` for an expectation | `BayesBase.expectation(backend, f, q)` with a backend from [Choosing a backend](@ref) |
| Gauss–Hermite and other kernels | [`GaussHermite`](@ref ExpectationApproximations.GaussHermite), [`GaussLaguerre`](@ref ExpectationApproximations.GaussLaguerre), …, chosen by [`EADefault`](@ref ExpectationApproximations.EADefault) |
| CVI and Laplace projections | [`gaussian_variational`](@ref ExpectationApproximations.gaussian_variational) and [`laplace`](@ref ExpectationApproximations.laplace), or RxInfer's `fallback = NaturalGradientPropagation()` and `LocalLaplace()` |
| constrained transforms | the domains of [Constrained variables](@ref) |

## From MessagePassingRulesApproximations 1.0

`ExpectationApproximations.MPRACompat` contains the API of MessagePassingRulesApproximations 1.0,
and MessagePassingRulesApproximations 1.1 re-exports it. Rule packages built on that API keep
working while they move to the expectation protocol. The results are those of 1.0, except that
the Gauss–Hermite weights come from the more accurate rule of this package.

| MessagePassingRulesApproximations 1.0 | The protocol |
|:--------------------------------------|:-------------|
| `approximate(Unscented(), f, (m,), (V,))` | [`mean_cov`](@ref ExpectationApproximations.mean_cov)`(EA.Unscented(), f, MvNormal(m, V))` |
| `approximate(Linearization(), f, (x,))` | [`linearize`](@ref ExpectationApproximations.linearize)`(f, x)` |
| `approximate_meancov(GaussHermiteCubature(n), g, m, V)` | [`tilted_mean_cov`](@ref ExpectationApproximations.tilted_mean_cov)`(EA.GaussHermite(n), log ∘ g, MvNormal(m, V))`, which also returns the log-normaliser |

```@docs
MPRACompat
AbstractApproximationMethod
approximation_name
approximation_short_name
Unscented
UT
UnscentedTransform
sigma_point_parameters
approximate
unscented_statistics
sigma_points_weights
Linearization
local_linearization
GaussHermiteCubature
ghcubature
getweights
getpoints
approximate_meancov
smoothRTS
```
