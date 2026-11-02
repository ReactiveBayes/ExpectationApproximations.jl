"""
    ProtoBase

A prototype of the expectation protocol planned for BayesBase (milestone U0.2 of PLAN.md).
Nothing is exported: every name is public and used qualified, `ProtoBase.expectation`, so the
protocol never clashes with names exported by other packages (`Stochastic` in ReactiveMP,
`expectation` in MessagePassingRulesApproximations).
"""
module ProtoBase

using BayesBase: BayesBase, FactorizedJoint, components, logpdf
using Compat: @compat

@compat public expectation, prepare_expectation, Prepared, NoBackendError, supports, which_backend
@compat public ExpectationLabel, Exact, Stochastic, haslabel, isexact, isstochastic, combine_label
@compat public withrng, getrng, Logpdf, PointSet, ProductPointSet, FirstOf
@compat public DynamicRouter, BackendRegistry, register_backend!

include("protocol.jl")
include("labels.jl")
include("rng.jl")
include("logpdf.jl")
include("pointset.jl")
include("products.jl")
include("firstof.jl")
include("dynamic.jl")

end
