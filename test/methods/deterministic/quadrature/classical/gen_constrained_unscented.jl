@testitem "Generalized Constrained Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import Distributions: skewness, kurtosis,cov,var
    import ExpectationApproximations: GenConstrainedUnscented, unscented_statistics, approximate_expectation,sigma_points_weights!, getcache

    @testset "sigma_points_weights!: check sigma points are in the bounds and weights are normalized" begin
        lb = ([2, 1], 3, 4)
        ub = ([3,5],6, 6)
        @inferred GenConstrainedUnscented{Multivariate, Multivariate}(lb, ub, 4, 4)
        method = GenConstrainedUnscented{Multivariate, Multivariate}(lb, ub, 4, 4)
        m = [2.2, 3.1, 3.5, 4.2]
        V = diagm(rand(4))
        S = zeros(4)
        K = rand(4)

        @inferred sigma_points_weights!(method, m, V ,S, K)
        points = getcache(method.cache, (:points, (4, 9)))
        weights = getcache(method.cache, (:weights, 9))
        lowerbound = reduce(vcat, lb)
        upperbound = reduce(vcat, ub)

        @test all(map((x) -> all(lowerbound .< x .< upperbound), eachcol(points)))
        @test sum(weights) ≈ 1

    end
    
    @testset "sigma_points_weights!: check sigma points are in the bounds and weights are normalized" begin
        dist = Gamma(1,2)
        m = mean(dist)
        v = var(dist)
        s = skewness(dist)
        k = kurtosis(dist, false)
        lb = 0.9
        ub = m + 10
        @inferred GenConstrainedUnscented(lb,ub)
        method = GenConstrainedUnscented(lb,ub)
        @inferred sigma_points_weights!(method, m, v ,s, k)
        points = getcache(method.cache, (:points, (1, 3)))
        weights = getcache(method.cache, (:weights, 3))
        
        @test all(map((x) -> all(lb .< x .< ub), eachcol(points)))
        @test sum(weights) ≈ 1
    end

    # @testset "GenConstrainedUnscented: Basic integrations/approximations with single Gamma" begin
    #     method = GenConstrainedUnscented((0.9,1 ),(3,2.3))
    #     dist = (Gamma(1,2), Normal(2,1))
    #     m = vcat(mean.(dist)...)
   
    #     v = Matrix(Diagonal(vcat(var.(dist)...)))
    #     s = vcat(skewness.(dist)...)
    #     k = vcat(kurtosis.(dist, false)...)
    #     points, weights = sigma_points_weights(method, m,v,s,k)

    # end



end