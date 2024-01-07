@testitem "Generalized Constrained Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import Distributions: skewness, kurtosis,cov,var
    import ExpectationApproximations: GenConstrainedUnscented, unscented_statistics, approximate_expectation,sigma_points_weights
    skewness(d::ExponentialFamily.UnivariateGaussianDistributionsFamily) = skewness(convert(Normal, d))
    kurtosis(d::ExponentialFamily.UnivariateGaussianDistributionsFamily) = kurtosis(convert(Normal, d))
    Distributions.cov(d::UnivariateDistribution)          = var(d)
    
    @testset "GenConstrainedUnscented: Basic integrations/approximations with Gaussian p" begin
        method = GenConstrainedUnscented((0.9,0.1),(5,0.9))
        dist = [Gamma(1,2), Beta(2,3)]
        m = mean.(dist)
        v = Matrix(Diagonal(cov.(dist)))
        s = skewness.(dist)
        k = kurtosis.(dist, false)
        points, weights = sigma_points_weights(method, m,v,s,k)
        @show points 
        # for sigma in sigmas, m in means
        #     importance_dist = Normal(m, sigma)
            
        #     @test isapprox(approximate_expectation(method, true_dist, (x) -> 1), 1.0, rtol = brtol)
        #     tmp = approximate_expectation(method, true_dist, (x) -> x)
        #     @test isapprox(tmp, mean(true_dist), rtol = brtol)
        #     @test isapprox(approximate_expectation(method, true_dist, (x) -> (x - tmp)^2), var(true_dist), rtol = brtol)
        # end
    end

    # @testset "GenUnscented: Basic integrations with Beta and Gamma" begin
    #     method = GenUnscented()
    #     dist = (Gamma(8,4),Beta(3,4), Gamma(20,1))
    #     mdist = mean.(dist)
    #     vdist = var.(dist)
    #     sdist = skewness.(dist)
    #     kdist = kurtosis.(dist, false)
    #     brtol = 1e-6
    #     g(x,y,z) = [x*y; z]
    #     estimate = ExpectationApproximations.unscented_statistics(method,g,mdist,vdist,sdist,kdist)
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand.(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_mc , mean(mapped_samples), rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with Beta, Gamma, Normal" begin
    #     method = GenUnscented()
    #     dist = (Gamma(8,4),Beta(3,4), NormalMeanVariance(20,1))
    #     mdist = mean.(dist)
    #     vdist = var.(dist)
    #     sdist = skewness.(dist)
    #     kdist = kurtosis.(dist, false)
    #     brtol = 1e-6
    #     g(x,y,z) = [x*y; sin(z)*x*y^3; y/x]
    #     estimate = ExpectationApproximations.unscented_statistics(method,g,mdist,vdist,sdist,kdist)
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand.(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_mc , mean(mapped_samples), rtol = brtol)
    # end


end