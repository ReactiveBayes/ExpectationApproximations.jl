# Backends

Each backend computes expectations through the protocol of BayesBase:
`BayesBase.expectation(backend, f, q)`, or `BayesBase.prepare_expectation(backend, q)` once and
`BayesBase.expectation(prepared, f)` for many integrands. Every backend here prepares a
`BayesBase.PointSet`, so its expectations are weighted sums that allocate nothing and pass dual
numbers through, and products of distributions combine them component by component.

```@docs
ExpectationApproximations.EADefault
ExpectationApproximations.GaussHermite
ExpectationApproximations.GaussLaguerre
ExpectationApproximations.GaussJacobi
ExpectationApproximations.QuantileLegendre
ExpectationApproximations.ExactSum
ExpectationApproximations.MonteCarlo
```
