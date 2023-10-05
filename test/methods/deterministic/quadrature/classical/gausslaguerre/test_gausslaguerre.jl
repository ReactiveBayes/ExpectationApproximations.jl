module TestsClassicalQuadratureGaussLaguerre

using Test
using ExpectationApproximations

@testset "GaussLaguerre creation" begin
    import ExpectationApproximations: getrule, getpoints, getweights, getlogweights

    for npoints in (10, 20, 100)
        gl = GaussLaguerre(npoints)

        @test getrule(ClassicalQuadrature(gl)) === gl

        @test length(gl) === npoints
        @test length(getpoints(gl)) === npoints
        @test length(getweights(gl)) === npoints
        @test length(getlogweights(gl)) === npoints
    end
end

include("test_gausslaguerre_gamma.jl")
include("test_gausslaguerre_exponential.jl")

end
