module TestsImportancecSampling

    using Test
    using ExpectationApproximations, Distributions, StableRNGs

    import ExpectationApproximations: ImportanceSampling
    import Distributions: rand!
    @testset "ImportanceSampling: Basic integrations/approximationsl" begin
    
        irng            = StableRNG(123)
        importance_dist = Normal(10.0, 1.0)
        npoints         = [10000]
        brtol           = 0.1
        
        for n in npoints
        
            method = ImportanceSampling(irng, n,  importance_dist; resampling = true )
            @test isapprox(approximate(method, (x) -> 1), 1.0, rtol = brtol)
            @test isapprox(approximate(method, (x) -> x), mean(importance_dist), rtol = brtol)
            @test isapprox(approximate(method, (x) -> (x - mean(importance_dist))^2), var(importance_dist ), rtol = brtol)

        end
    end


end