@testitem "Generalized Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily
    import Distributions: skewness, kurtosis,cov,var
    import ExpectationApproximations: GenUnscented, unscented_statistics, approximate_expectation
    skewness(d::ExponentialFamily.UnivariateGaussianDistributionsFamily) = skewness(convert(Normal, d))
    kurtosis(d::ExponentialFamily.UnivariateGaussianDistributionsFamily) = kurtosis(convert(Normal, d))
    Distributions.cov(d::UnivariateGaussianDistributionsFamily) where T = var(d)
    
    @testset "GenUnscented: Basic integrations/approximations with Gaussian p" begin
        true_dist = NormalMeanVariance(5.0, 6.0)
        sigmas    = [5, 6, 7, 8, 9, 10, 11, 16]
        means     = [3, 4, 7, 8]
        brtol     = 0.1

        for sigma in sigmas, m in means
            importance_dist = Normal(m, sigma)
            method = GenUnscented()
            @test isapprox(approximate_expectation(method, true_dist, (x) -> 1), 1.0, rtol = brtol)
            tmp = approximate_expectation(method, true_dist, (x) -> x)
            @test isapprox(tmp, mean(true_dist), rtol = brtol)
            @test isapprox(approximate_expectation(method, true_dist, (x) -> (x - tmp)^2), var(true_dist), rtol = brtol)
        end
    end

    @testset "GenUnscented: Basic integrations with Beta and Gamma" begin
        method = GenUnscented()
        dist = (Gamma(8,4),Beta(3,4), Gamma(20,1))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = skewness.(dist)
        kdist = kurtosis.(dist, false)

        g(x,y,z) = [x*y; z]
        estimate = ExpectationApproximations.unscented_statistics(method,g,mdist,vdist,sdist,kdist)
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand.(dist) for i=1:10000];
        mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_mc , mean(mapped_samples), rtol = 0.00001)
    end

    @testset "GenUnscented: Basic integrations with Beta, Gamma, Normal" begin
        method = GenUnscented()
        dist = (Gamma(8,4),Beta(3,4), NormalMeanVariance(20,1))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = skewness.(dist)
        kdist = kurtosis.(dist, false)

        g(x,y,z) = [x*y; sin(z)*x*y^3; y/x]
        estimate = ExpectationApproximations.unscented_statistics(method,g,mdist,vdist,sdist,kdist)
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand.(dist) for i=1:10000];
        mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_mc , mean(mapped_samples), rtol = 0.00001)
    end


end