@testitem "Moment tensors" tags = [:kernels] begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import ExpectationApproximations: MomentGenerator, DiagonalMomentGenerator, adjust_integrand

    @testset "Univariate variance moment construction" begin
        mg = @inferred MomentGenerator(Univariate, Val(2), 3.0)
        f = (x) -> x^3
        adjustedf = @inferred adjust_integrand(mg, f, Val(false))

        @test @inferred(adjustedf(0.0)) == 9.0
        @test @inferred(adjustedf(1.0)) == 4.0
        @test @inferred(adjustedf(2.0)) == 25.0
    end

    @testset "2d covariance moment construction with multiple argument function with univariate input" begin
        mg = @inferred MomentGenerator(Multivariate, Val(2), [2, 3])
        f = (x, y) -> [x, y]
        adjustedf = @inferred adjust_integrand(mg, f)

        @test @inferred(adjustedf(0.0, 0.0)) == [4.0 6.0; 6.0 9.0]
        @test @inferred(adjustedf(1.0, 0.0)) == [1, 3] * [1, 3]'
    end

    @testset "2d covariance moment construction with multiple argument function with univariate inputs" begin
        mg = @inferred MomentGenerator(Multivariate, Val(2), [2, 3])
        f = (x, y) -> [x, y]
        adjustedf = @inferred adjust_integrand(mg, f)

        @test @inferred(adjustedf(0.0, 0.0)) == [4.0 6.0; 6.0 9.0]
        @test @inferred(adjustedf(1.0, 0.0)) == [1, 3] * [1, 3]'
    end

    @testset "2d covariance moment construction with multiple argument function with multivariate-univariate input" begin
        mg = @inferred MomentGenerator(Multivariate, Val(2), [2, 4, 3])
        f = (x, y) -> [x; y]
        adjustedf = @inferred adjust_integrand(mg, f)
        @test @inferred(adjustedf([0.0, 0.0], 0.0)) == [4.0 8.0 6.0; 8.0 16.0 12.0; 6.0 12.0 9.0]
    end

    @testset "2d covariance moment construction" begin
        mg = @inferred MomentGenerator(Multivariate, Val(2), [3.0, 1.0])
        f = (x) -> x .^ 3
        adjustedf = @inferred adjust_integrand(mg, f, Val(false))
        @test @inferred(adjustedf([0.0, 0.0])) == [3.0, 1.0] * [3.0, 1.0]'
        @test @inferred(adjustedf([1.0, 0.0])) == [2.0, 1.0] * [2.0, 1.0]'
        @test @inferred(adjustedf([0.0, 1.0])) == [3.0, 0.0] * [3.0, 0.0]'
    end

    @testset "2d rank 3 skewness tensor construction" begin
        mg = @inferred MomentGenerator(Multivariate, Val(3), [3.0, 1.0])
        f = (x) -> x .^ 3
        adjustedf = @inferred adjust_integrand(mg, f, Val(false))
        @test @inferred(adjustedf([0.0, 0.0])) == [-27.0 -9.0; -9.0 -3.0;;; -9.0 -3.0; -3.0 -1.0]
        @test @inferred(adjustedf([1.0, 0.0])) == [-8.0 -4.0; -4.0 -2.0;;; -4.0 -2.0; -2.0 -1.0]
        @test @inferred(adjustedf([0.0, 1.0])) == [-27.0 0.0; 0.0 0.0;;; 0.0 0.0; 0.0 0.0]
    end

    @testset "2d Diagonal variance generator" begin
        mg = @inferred DiagonalMomentGenerator(Val(2), [3.0, 1.0])
        f = (x) -> x .^ 3
        adjustedf = @inferred adjust_integrand(mg, f, Val(false))
        @test @inferred(adjustedf([0.0, 0.0])) == [3.0, 1.0] .^ 2
        @test @inferred(adjustedf([1.0, 0.0])) == [2.0, 1.0] .^ 2
        @test @inferred(adjustedf([0.0, 1.0])) == [3.0, 0.0] .^ 2
    end
end
