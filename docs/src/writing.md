```@meta
CurrentModule = ExpectationApproximations
```

# Writing a backend

A backend is any value for which two methods of BayesBase are defined:

- `BayesBase.supports_expectation(backend, ::Type{F}, ::Type{Q})`, which declares that the
  backend handles integrands of type `F` under distributions of type `Q`;
- either `BayesBase.prepare_expectation(backend, q)`, which returns something
  `BayesBase.expectation(prepared, f)` can evaluate, or `BayesBase.expectation(backend, f, q)`
  directly, for a closed form.

The simplest way is to return a `BayesBase.ExpectationPointSet(points, weights)` from
`prepare_expectation`. Products of distributions, routers and the projections of this package then
work with the new backend. For example, a two-point rule for Gaussians, exact up to degree 3:

```@example writing
using ExpectationApproximations, Distributions
import BayesBase
const EA = ExpectationApproximations

struct TwoPoint end
BayesBase.supports_expectation(::TwoPoint, ::Type, ::Type{<:Normal}) = true
BayesBase.prepare_expectation(::TwoPoint, q::Normal) =
    BayesBase.ExpectationPointSet(mean(q) .+ std(q) .* [-1.0, 1.0], [0.5, 0.5])

BayesBase.expectation(TwoPoint(), x -> x^2, Normal(1.0, 2.0))
```

## The conformance suite

[`ExpectationApproximations.Testing`](@ref) checks a backend against expectations with known
values. The backends of this package pass the same checks:

- the backend supports each case and computes it correctly;
- a backend labelled exact is exact;
- the prepared expectation equals the direct one, or is reproducible for a stochastic backend;
- the result is inferred, and evaluating a prepared expectation allocates nothing;
- dual numbers pass through;
- a product of two cases is supported and correct.

[`Testing.conformance`](@ref) returns the checks, and [`Testing.test_backend`](@ref) records them
as tests in the test suite of the backend's package.

```@example writing
using ExpectationApproximations.Testing: Case, conformance
cases = [Case(Normal(0.5, 1.5), x -> x^2, 0.5^2 + 1.5^2), Case(Normal(0.0, 1.0), x -> x^3, 0.0)]
checks = conformance(TwoPoint(), cases)
unique(check.name for check in checks), all(check -> check.passed, checks)
```

A case that the rule cannot compute exactly, ``\mathbb{E}[x^4] = 3`` under a standard Gaussian,
fails the "correct" check:

```@example writing
checks = conformance(TwoPoint(), [Case(Normal(0.0, 1.0), x -> x^4, 3.0)])
[check.name for check in checks if !check.passed]
```

```@docs
ExpectationApproximations.Testing
ExpectationApproximations.Testing.test_backend
ExpectationApproximations.Testing.conformance
ExpectationApproximations.Testing.Case
ExpectationApproximations.Testing.Check
```
