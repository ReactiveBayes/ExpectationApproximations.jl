module TestsImportancecSampling

    using Test
    using ExpectationApproximations, Distributions, StableRNGs

    import ExpectationApproximations: ImportanceSampling
    import Distributions: rand!
    import SpecialFunctions: gamma, digamma
    @testset "ImportanceSampling: Basic integrations/approximations with Gaussian p" begin
    
        irng            = StableRNG(123)
        # importance_dist = Normal(10.0, 0.001)
        true_dist       = Normal(5.0, 6.0)
        npoints         = [10000]
        sigmas          = [5,6,7,8,9,10,11,16]
        means           = [3,4,7,8 ]
        brtol           = 0.1
        resampling      = [true, false]
        
        for sigma in sigmas, n in npoints, m in means,resample in resampling
            importance_dist = Normal(m, sigma)
            method = ImportanceSampling(irng, n,  importance_dist; resampling = resample)
            @test isapprox(approximate(method, true_dist, (x) -> 1), 1.0, rtol = brtol)
            tmp = approximate(method, true_dist, (x) -> x)
            @test isapprox(tmp, mean(true_dist), rtol = brtol)
            @test isapprox(approximate(method, true_dist, (x) -> (x - tmp)^2), var(true_dist), rtol = brtol)

        end
    end

    @testset "ImportanceSampling: Basic integration with unnormalized distribution: exp(-x^2)"  begin 
        irng            = StableRNG(123)
        true_dist       = x -> exp(-0.5(x-5)^2/36)
        npoints         = [10000]
        sigmas          = [5,6,7,8,9,10,11,16]
        means           = [3,4,7,8 ]
        brtol           = 0.1
        resampling      = [true, false]
        for sigma in sigmas, n in npoints, m in means,resample in resampling
            importance_dist = Normal(m, sigma)
            method = ImportanceSampling(irng, n,  importance_dist; resampling = resample)
            @test isapprox(approximate(method, true_dist, (x) -> 1),1 , rtol = brtol)
            tmp = approximate(method, true_dist, (x) -> x)
            @test isapprox(tmp, 5, rtol = brtol)
            @test isapprox(approximate(method, true_dist, (x) -> (x - tmp)^2), 36, rtol = brtol)

        end
    end


    @testset "ImportanceSampling: Basic integration with unnormalized distribution: Gamma"  begin 
        irng            = StableRNG(123)
        alpha           = 5
        beta            = 4
        true_dist       = Gamma(alpha,1/beta)
        npoints         = [10000]
        alphas          = [3,4,5,6,7]
        betas           = [3,4,5,6,7]
        brtol           = 1.0
        resampling      = [true, false]
        for a in alphas, b in betas, n in npoints, resample in resampling
            importance_dist = Gamma(a, 1/b)
            method = ImportanceSampling(irng, n,  importance_dist; resampling = resample)
            @test isapprox(approximate(method, true_dist, (x) -> 1),1 , rtol = brtol)
            tmp = approximate(method, true_dist, (x) -> x)
            @test isapprox(tmp, alpha/beta, rtol = brtol)
            @test isapprox(approximate(method, true_dist, (x) -> log(x)), digamma(alpha)-log(beta), rtol = brtol)

        end
    end



end