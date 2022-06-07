using ExpectationApproximations
using Test

@testset "ExpectationApproximations.jl" begin
    
    include("test_common.jl")

    include("methods/unscented/test_unscented.jl")

end
