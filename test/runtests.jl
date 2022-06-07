using ExpectationApproximations
using Test

@testset "ExpectationApproximations.jl" begin
    
    include("test_transforms.jl")
    include("test_log.jl")

    include("methods/unscented/test_unscented.jl")

end
