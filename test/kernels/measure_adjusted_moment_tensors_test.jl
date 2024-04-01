@testitem "MeasureAdjustedMomentGenerator" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import ExpectationApproximations: MeasureAdjustedMomentGenerator, adjust_integrand

    @testset "Univariate MeasureAdjustedMomentGenerator Val(1)" begin
        
        mamg = @inferred(MeasureAdjustedMomentGenerator(1.0, 0.0))
        measure = (x) -> pdf(NormalMeanVariance(0,1), x)

        adjusted_integrand = @inferred(adjust_integrand(mamg, measure))
        @test adjusted_integrand(0.0) == 0.0
        @test adjusted_integrand(1.0) == measure(1.0)
        @test adjusted_integrand(2.0) == measure(2.0)*2.0

    end

    @testset "Univariate MeasureAdjustedMomentGenerator Val(2)" begin
        
        mamg = @inferred(MeasureAdjustedMomentGenerator(Val(2), 1.0, 0.0))
        measure = (x) -> pdf(NormalMeanVariance(0,1), x)

        adjusted_integrand = @inferred(adjust_integrand(mamg, measure))
        @test adjusted_integrand(0.0) == 0.0
        @test adjusted_integrand(1.0) == measure(1.0)
        @test adjusted_integrand(2.0) == measure(2.0)*4
    end

end