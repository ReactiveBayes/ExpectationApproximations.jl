module TestsUnscentedGaussLaguerreExponential

    using Test
    using ExpectationApproximations, Distributions, StableRNGs

    import ExpectationApproximations: UnscentedStrategyAutomatic

    @testset "GaussLaguerre: Basic integrations/approximations when q is Exponential" begin
    
        irng = StableRNG(123)
    
        btol(::UseLogWeights, n) = 1e-10 / log(n) # LogWeights should have better accuracy overall
        btol(::UseWeights, n) = 1e-10
        btol(::UnscentedStrategyAutomatic, n) = 1e-10
        
        strategies = (UseLogWeights(), UseWeights(), UnscentedStrategyAutomatic())
        npoints = (10, 20, 100, 200)
        qs = (Exponential(20rand(irng)) for _ in 1:50)
        
        for strategy in strategies, n in npoints, q in qs
        
            method = Unscented(GaussLaguerre(n), strategy)
            
            # base rel tolerance 
            brtol = btol(strategy, n)
            
            @test isapprox(approximate(method, q, (x) -> 1), 1.0, rtol = brtol)
            @test isapprox(approximate(method, q, (x) -> -1), -1.0, rtol = brtol)
            @test isapprox(approximate(method, q, LogIntegrand((x) -> 0.0)), 1.0, rtol = brtol)
            
            @test isapprox(approximate(method, q, (x) -> x), mean(q), rtol = brtol)
            @test isapprox(approximate(method, q, (x) -> -x), -mean(q), rtol = brtol)
            @test isapprox(approximate(method, q, LogIntegrand((x) -> log(x))), mean(q), rtol = brtol)
            
            @test isapprox(approximate(method, q, (x) -> (x - mean(q))^2), var(q), rtol = brtol)
            @test isapprox(approximate(method, q, (x) -> -(x - mean(q))^2), -var(q), rtol = brtol)
            @test isapprox(approximate(method, q, LogIntegrand((x) -> 2log(abs(x - mean(q))))), var(q), rtol = brtol)
        end
    end

end