
@testitem "Cross-covariance method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import ExpectationApproximations: CrossCovariance, adjust_integrand

    @testset "Cross-covariance (univariate-univariate)" begin
        mg = @inferred CrossCovariance(1.0, 3.0)
        f = (x) -> x^3
        adjustedf = @inferred adjust_integrand(mg, f, Val(false))

        @test @inferred(adjustedf(0.0)) == 3
        @test @inferred(adjustedf(1.0)) == 0
        @test @inferred(adjustedf(2.0)) == -7
        
    end

    @testset "Cross-covariance construction (univariate-multivariate)" begin
        mg = @inferred CrossCovariance(Univariate, Multivariate, 2.0, [3.0, 1.0])
        f = (x) -> tr(x*x')
        adjustedf = @inferred adjust_integrand(mg, f, Val(false))
        @test @inferred(adjustedf([0.0, 0.0])) == [6.0, 2.0]
    end

    @testset "2d Cross-covariance with multiple argument function with univariate inputs" begin
        mg = @inferred CrossCovariance(Multivariate, Multivariate, [0, 0], [2, 3])
        f = (x, y) -> [x, y ]
        adjustedf = @inferred adjust_integrand(mg, f)

        @test @inferred(adjustedf(0.0,0.0)) == [0.0 0.0; 0.0 0.0]   
        @test @inferred(adjustedf(1.0,0.0)) == [-1.0 0.0; -3.0 0.0]  
        @test @inferred(adjustedf(0.0,1.0)) == [0.0 -2.0; 0.0 -2.0] 
    end

end