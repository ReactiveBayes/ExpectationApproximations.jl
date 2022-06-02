module TestsUnscentedGaussLaguerreGamma

    using Test
    using ExpectationApproximations, Distributions, StableRNGs

    @testset "GaussLaguerre: Basic integrations/approximations when q is Gamma" begin
    
        irng = StableRNG(123)
    
        btol(::UseLogWeights, n) = 1e-2 / log(n) # LogWeights should have better accuracy overall
        btol(::UseWeights, n) = 1e-2
        
        strategies = (UseLogWeights(), UseWeights())
        npoints = (10, 20, 100, 200)
        qs = (Gamma(20rand(irng) + 1, 20rand(irng) + 1) for _ in 1:50)
        
        for strategy in strategies, n in npoints, q in qs
        
            method = Unscented(strategy, GaussLaguerre(n))
            
            # base rel tolerance 
            brtol = btol(strategy, n)
            
            @test isapprox(approximate(method, q, (x) -> 1), 1.0, rtol = brtol)
            @test isapprox(approximate(method, q, (x) -> -1), -1.0, rtol = brtol)
            @test isapprox(approximate(method, q, LogFn((x) -> 0.0)), 1.0, rtol = brtol)
            
            @test isapprox(approximate(method, q, (x) -> x), mean(q), rtol = brtol)
            @test isapprox(approximate(method, q, (x) -> -x), -mean(q), rtol = brtol)
            @test isapprox(approximate(method, q, LogFn((x) -> log(x))), mean(q), rtol = brtol)
            
            @test isapprox(approximate(method, q, (x) -> (x - mean(q))^2), var(q), rtol = brtol)
            @test isapprox(approximate(method, q, (x) -> -(x - mean(q))^2), -var(q), rtol = brtol)
            @test isapprox(approximate(method, q, LogFn((x) -> 2log(abs(x - mean(q))))), var(q), rtol = brtol)
        end
    end

    # @testset "GaussLaguerre" begin

    #     import ExpectationApproximations: mean_var

    #     for α1 in (1.3, 2.1, 10.1, ), 
    #         α2 in (1.3, 2.1, 10.1, ),
    #         β1 in (1.3, 2.1, 10.1, ),
    #         β2 in (1.3, 2.1, 10.1, ),
    #         n in (61, 121, 1121, 5121)

    #         d1 = Gamma(α1, β1)
    #         d2 = Gamma(α2, β2)

    #         pα = α1 + α2 - 1
    #         pβ = (β1 * β2) / (β1 + β2)
    #         p  = Gamma(pα, pβ)

    #         m, v = mean_var(Unscented(GaussLaguerre(n)), d1, (x) -> pdf(d2, x))
            
    #         @test isapprox(m, mean(p), rtol = 1e-1 / log(n))
    #         @test isapprox(v, var(p), rtol = 1e-1 / log(n))

    #         # ForceLog version
    #         m, v = mean_var(Unscented(GaussLaguerre(n, Val(true))), d1, (x) -> pdf(d2, x))
            
    #         @test isapprox(m, mean(p), rtol = 1e-1 / log(n))
    #         @test isapprox(v, var(p), rtol = 1e-1 / log(n))

    #         # Log-Version 
    #         m, v = mean_var(Unscented(GaussLaguerre(n)), d1, LogFn((x) -> logpdf(d2, x)))
                
    #         @test isapprox(m, mean(p), rtol = 1e-1 / log(n))
    #         @test isapprox(v, var(p), rtol = 1e-1 / log(n))

    #         # ForceLog Log-Version 
    #         m, v = mean_var(Unscented(GaussLaguerre(n, Val(true))), d1, LogFn((x) -> logpdf(d2, x)))
                
    #         @test isapprox(m, mean(p), rtol = 1e-1 / log(n))
    #         @test isapprox(v, var(p), rtol = 1e-1 / log(n))

    #     end

    # end

end