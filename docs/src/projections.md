# Projections

The operations that approximate inference builds on: the Laplace approximation of a density, its variational Gaussian (the Gaussian that minimises `KL(q ‖ p)`), its moment-matched Gaussian and, computed through any backend, the moments
of a function of a random variable, its cross-covariance with the input, and the moments of a
tilted distribution, the step of moment matching.

```@docs
ExpectationApproximations.laplace
ExpectationApproximations.gaussian_variational
ExpectationApproximations.moment_matching
ExpectationApproximations.mean_cov
ExpectationApproximations.mean_cov_cross
ExpectationApproximations.tilted_mean_cov
ExpectationApproximations.linearize
```
