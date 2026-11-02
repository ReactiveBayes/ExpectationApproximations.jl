# Projections

The operations that approximate inference builds on: the Laplace approximation of a density and, computed through any backend, the moments
of a function of a random variable, its cross-covariance with the input, and the moments of a
tilted distribution, the step of moment matching.

```@docs
ExpectationApproximations.laplace
ExpectationApproximations.mean_cov
ExpectationApproximations.mean_cov_cross
ExpectationApproximations.tilted_mean_cov
ExpectationApproximations.linearize
```
