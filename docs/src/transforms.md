# Domain transforms

A variable that must be positive, a probability, a probability vector or a block of such values
is approximated by a Gaussian in unconstrained coordinates, pushed forward onto its domain. The
domain is read from the support of a distribution, usually the variable's prior, so a model keeps
its natural priors (a Gamma, a Beta, a Dirichlet) and the transform lives in the approximation.

```@docs
ExpectationApproximations.Domain
ExpectationApproximations.RealLine
ExpectationApproximations.Positive
ExpectationApproximations.UnitInterval
ExpectationApproximations.LowerBounded
ExpectationApproximations.UpperBounded
ExpectationApproximations.Bounded
ExpectationApproximations.Simplex
ExpectationApproximations.Componentwise
ExpectationApproximations.domain
ExpectationApproximations.forward
ExpectationApproximations.inverse
ExpectationApproximations.logjacobian
ExpectationApproximations.pushforward
ExpectationApproximations.TransformedGaussian
```
