using ExpectationApproximations
using Test

@testset "ExpectationApproximations.jl" begin
    
    # include("test_transforms.jl")
    # include("test_log.jl")

    # include("methods/deterministic/quadrature/classical/test_classical_quadrature.jl")
    # include("methods/stochastic/importance/test_importance_sampling.jl")
    include("methods/deterministic/series/test_linearization.jl")
end
