```@meta
CurrentModule = ExpectationApproximations.MPRACompat
```

# MessagePassingRulesApproximations 1.0

The API of MessagePassingRulesApproximations 1.0 lives on in `ExpectationApproximations.MPRACompat`,
which its 1.1 re-exports, so that the rule packages built on it keep working while they move to
the expectation protocol.

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
