module ExpectationApproximations

using ExponentialFamily

include("transforms.jl")
include("log.jl")
include("approximate.jl")
include("helpers/helpers.jl")
include("helpers/algebra/cholesky.jl")
    

include("kernels/raw_moment.jl")
include("kernels/central_moment.jl")

include("methods/deterministic/quadrature/classical/classical_quadrature.jl")
include("methods/stochastic/importance/importance_sampling.jl")
include("methods/deterministic/series/linearization.jl")
include("methods/deterministic/series/laplace.jl")
include("methods/deterministic/quadrature/classical/unscented.jl")
include("methods/hybrid/cvi/cvmp.jl")

end
