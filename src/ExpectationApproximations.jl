module ExpectationApproximations

using ExponentialFamily, BayesBase, FastCholesky

include("helpers/helpers.jl")
include("helpers/weighted_sum.jl")
include("transforms.jl")
include("log.jl")
include("approximate.jl")
include("cache.jl")

include("kernels/raw_moment.jl")
include("kernels/central_moment.jl")
include("kernels/cross_covariance.jl")
include("kernels/moment_tensors.jl")

include("functions/functions.jl")

include("methods/deterministic/quadrature/classical/classical_quadrature.jl")
include("methods/stochastic/importance/importance_sampling.jl")
include("methods/deterministic/series/linearization.jl")
include("methods/deterministic/series/laplace.jl")
include("methods/deterministic/quadrature/classical/unscented.jl")
include("methods/deterministic/quadrature/classical/gen_unscented.jl")
include("methods/deterministic/quadrature/classical/gen_constrained_unscented.jl")
include("methods/hybrid/gradients.jl")
include("methods/hybrid/common.jl")
include("methods/hybrid/cvmp/cvmp.jl")
include("methods/hybrid/cvi/cvi.jl")
include("methods/stochastic/mcmc/advanced_hmc.jl")

end
