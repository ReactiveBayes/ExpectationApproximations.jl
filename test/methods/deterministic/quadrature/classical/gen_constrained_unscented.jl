@testitem "Generalized Constrained Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import Distributions: skewness, kurtosis,cov,var
    import ExpectationApproximations: GenConstrainedUnscented, unscented_statistics, approximate_expectation,sigma_points_weights
    skewness(d::ExponentialFamily.UnivariateGaussianDistributionsFamily) = skewness(convert(Normal, d))
    kurtosis(d::ExponentialFamily.UnivariateGaussianDistributionsFamily) = kurtosis(convert(Normal, d))
    Distributions.cov(d::UnivariateDistribution)          = var(d)
    
    @testset "GenConstrainedUnscented: Basic integrations/approximations with single Gamma" begin
      
        dist = Gamma(1,2)
        m = mean(dist)
        method = GenConstrainedUnscented((0.9,),(m+0.3,))
        v = var(dist)
        s = skewness(dist)
        k = kurtosis(dist, false)
        points, weights = sigma_points_weights(method, m,v,s,k)
        g(x) = [3x+2x^2]
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)

  
    end

    @testset "GenConstrainedUnscented: Basic integrations/approximations with single Gamma" begin
        method = GenConstrainedUnscented((0.9,1 ),(3,2.3))
        dist = (Gamma(1,2), Normal(2,1))
        m = vcat(mean.(dist)...)
   
        v = Matrix(Diagonal(vcat(var.(dist)...)))
        s = vcat(skewness.(dist)...)
        k = vcat(kurtosis.(dist, false)...)
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



end