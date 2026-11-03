# Backends

Each backend computes expectations through the protocol of BayesBase:
`BayesBase.expectation(backend, f, q)`, or `BayesBase.prepare_expectation(backend, q)` once and
`BayesBase.expectation(prepared, f)` for many integrands. Every backend here prepares a
`BayesBase.ExpectationPointSet`, so its expectations are weighted sums that allocate nothing and
pass dual numbers through, and products of distributions combine them component by component.

```@docs
ExpectationApproximations.EADefault
ExpectationApproximations.register_backends!
ExpectationApproximations.GaussHermite
ExpectationApproximations.GaussLaguerre
ExpectationApproximations.GaussJacobi
ExpectationApproximations.QuantileLegendre
ExpectationApproximations.ExactSum
ExpectationApproximations.MonteCarlo
ExpectationApproximations.SNIS
ExpectationApproximations.effective_sample_size
```

## Sigma points

Sigma points are few, deterministic and cheap: they match the first moments of a distribution
rather than integrate it, so they suit smooth functions of Gaussians, as in nonlinear filtering.

```@docs
ExpectationApproximations.Unscented
ExpectationApproximations.SigmaPointSet
ExpectationApproximations.GenUnscented
ExpectationApproximations.SphericalRadialCubature
```
