module ExpectationApproximations

    include("transforms.jl")
    include("log.jl")
    include("approximate.jl")
    include("helpers.jl")

    include("kernels/raw_moment.jl")
    include("kernels/central_moment.jl")

    include("methods/deterministic/quadrature/classical/classical_quadrature.jl")
    include("methods/stochastic/importance/importance_sampling.jl")
    include("methods/deterministic/series/linearization.jl")
    include("methods/deterministic/series/laplace.jl")
end
