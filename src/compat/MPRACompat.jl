"""
    ExpectationApproximations.MPRACompat

The API of MessagePassingRulesApproximations 1.0, ported here so that its version 1.1 can
re-export it. It propagates Gaussian moments through a function and computes expectations under a
normal, taking and returning means and covariances.

- `Unscented`: the unscented transform, the mean and covariance of `f(x)` from `2d + 1` sigma
  points, through `approximate` and `unscented_statistics`;
- `Linearization`: the first-order expansion of `f` at the inputs' means, through `approximate`
  and `local_linearization`;
- `GaussHermiteCubature`: expectations under a normal, and the moments of a normal reweighted by
  a function, through `approximate_meancov`;
- `smoothRTS`: the Rauch–Tung–Striebel correction of an input's marginal.

The results equal those of MessagePassingRulesApproximations 1.0, which the tests check, with two
exceptions. The Gauss–Hermite weights come from this package's Golub–Welsch rule, which is more
accurate. And a covariance that is not positive definite throws a `PosDefException` from
`cholesky`, where 1.0 factorised a nearby positive definite matrix and returned numbers. New code should use the expectation protocol of BayesBase and the backends of this
package.

```jldoctest; setup = :(using ExpectationApproximations.MPRACompat)
julia> m, V = approximate(Unscented(), x -> 2x + 1, (1.0,), (0.5,));

julia> m ≈ 3.0 && V ≈ 2.0
true

julia> approximate(Linearization(), x -> x^2, (3.0,))
(6.0, -9.0)
```
"""
module MPRACompat

using ..ExpectationApproximations: gausshermite
using Compat: @compat
using LinearAlgebra
import ForwardDiff

export AbstractApproximationMethod, approximation_name, approximation_short_name
export Unscented, UT, UnscentedTransform
export approximate, unscented_statistics, sigma_points_weights, smoothRTS
export Linearization, local_linearization
export GaussHermiteCubature, ghcubature, getweights, getpoints, approximate_meancov

@compat public sigma_point_parameters

# The matrix square root and inverse that MessagePassingRulesApproximations took from FastCholesky
cholsqrt(A::AbstractMatrix) = cholesky(Symmetric(A)).L
cholinv(x::Real) = inv(x)
cholinv(A::AbstractMatrix) = inv(cholesky(Symmetric(A)))

include("approximations.jl")
include("shared.jl")
include("unscented.jl")
include("linearization.jl")
include("gausshermite.jl")
include("rts.jl")

end
