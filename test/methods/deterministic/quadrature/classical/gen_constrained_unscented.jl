@testitem "Generalized Constrained Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import Distributions: skewness, kurtosis,cov,var
    import ExpectationApproximations: GenConstrainedUnscented, unscented_statistics, approximate_expectation,sigma_points_weights

    
    @testset "GenConstrainedUnscented: Basic integrations/approximations with single Gamma" begin
      
        dist = Gamma(1,2)
        m = mean(dist)
        lb = 0.9
        ub = m + 10
        tdist = truncated(dist, lb, ub)
        method = GenConstrainedUnscented((lb,),(ub,),0.9)
        brtol = 1e-2
        g(x) = 3x+2x^2
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)

        samples     = [rand(tdist) for i=1:100000]
        estimate_mc = mean(g.(samples));
        
        
        @test g(lb) < estimate_genut < g(ub)
        @test g(lb) < estimate_mc < g(ub)
        # @test isapprox(estimate_genut, estimate_mc,rtol=brtol)
        
    end

    @testset "GenConstrainedUnscented: Basic integrations/approximations with single Gamma" begin
        method = GenConstrainedUnscented((0.9,1 ),(3,2.3))
        dist = (Gamma(1,2), Normal(2,1))
        m = vcat(mean.(dist)...)
   
        v = Matrix(Diagonal(vcat(var.(dist)...)))
        s = vcat(skewness.(dist)...)
        k = vcat(kurtosis.(dist, false)...)
        points, weights = sigma_points_weights(method, m,v,s,k)

    end



end