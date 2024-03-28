@testitem "Moment tensors" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import ExpectationApproximations: MomentGenerator, DiagonalMomentGenerator, adjust_integrand

    @testset "rank 1 moment construction" begin
        mg = @inferred MomentGenerator(Univariate, Val(2), 3.0)
        f = (x) -> x^3
        adjustedf = @inferred adjust_integrand(mg, f)

        @test adjustedf(0.0) == 9.0
        @test adjustedf(1.0) == 4.0
        @test adjustedf(2.0) == 25.0
        
    end

    @testset "rank 2 moment construction" begin
        mg = @inferred MomentGenerator(Multivariate, Val(2), [3.0, 1.0])
        f = (x) -> x.^3
        adjustedf = @inferred adjust_integrand(mg, f)
        @test @inferred(adjustedf([0.0, 0.0])) == [3.0, 1.0]*[3.0, 1.0]'
        @test @inferred(adjustedf([1.0, 0.0])) == [2.0, 1.0]*[2.0, 1.0]'
        @test @inferred(adjustedf([0.0, 1.0])) == [3.0, 0.0]*[3.0, 0.0]'
    end

    @testset "rank 3 moment construction" begin
        mg = @inferred MomentGenerator(Multivariate, Val(3), [3.0, 1.0])
        f = (x) -> x.^3
        adjustedf = @inferred adjust_integrand(mg, f)
        @show adjustedf([0.0, 0.0])
        # @test @inferred(adjustedf([0.0, 0.0])) == [-27.0, -9.0; -9.0, -3.0 ;; -9.0, -3.0; -3.0 -1.0]
        # @test @inferred(adjustedf([1.0, 0.0])) == [2.0, 1.0]*[2.0, 1.0]'
        # @test @inferred(adjustedf([0.0, 1.0])) == [3.0, 0.0]*[3.0, 0.0]'
    end




end