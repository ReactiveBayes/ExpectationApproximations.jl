module TestsClassicalQuadratureGaussHermite

    using Test
    using ExpectationApproximations

    @testset "GaussHermite creation" begin 
        import ExpectationApproximations: getrule, getpoints, getweights, getlogweights

        for npoints in (10, 20, 100)
            gl = GaussHermite(npoints)

            @test getrule(ClassicalQuadrature(gl)) === gl

            @test length(gl) === npoints
            @test length(getpoints(gl)) === npoints
            @test length(getweights(gl)) === npoints
            @test length(getlogweights(gl)) === npoints
        end
    end

    include("test_gausshermite_normal.jl")

end