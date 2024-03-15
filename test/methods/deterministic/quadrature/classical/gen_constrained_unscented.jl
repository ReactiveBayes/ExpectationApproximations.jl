@testitem "Generalized Constrained Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import ExpectationApproximations: GenConstrainedUnscented, unscented_statistics, approximate_expectation,sigma_points_weights!, getcache, _cov, _kurtosis, _skewness, FromTo

    @testset "sigma_points_weights!: check sigma points are in the bounds and weights are normalized" begin
        lb = ([2, 1], 3, 4)
        ub = ([3,5],6, 6)
        @inferred GenConstrainedUnscented{FromTo(Multivariate, Multivariate)}(lb, ub, 4, 4)

        method = GenConstrainedUnscented{FromTo(Multivariate, Multivariate)}(lb, ub, 4, 4)
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
        s = _skewness(dist)
        k = _kurtosis(dist)
        lb = 0.9
        ub = m + 10
        @inferred GenConstrainedUnscented(lb,ub)
        method = GenConstrainedUnscented(lb,ub)
        @inferred sigma_points_weights!(method, m, v ,s, k)
        points = getcache(method.cache, (:points, (method.inputdim, 2*method.inputdim+1)))
        weights = getcache(method.cache, (:weights, 3))
        
        @test all(map((x) -> all(lb .< x .< ub), eachcol(points)))
        @test sum(weights) ≈ 1
    end

    @testset "sigma_points_weights!: Poisson sigma points" begin
        m = [1.5, 1]
        P = [1.5 0; 0 1]
        S = [1.5, 1]
        K = [8.25, 4]
        ub =(Inf,Inf)
        lb = (0, 0)

        method = GenConstrainedUnscented{FromTo(Multivariate, Multivariate)}(lb, ub, 2, 2, 0.9)
        @inferred sigma_points_weights!(method, m, P ,S, K)
        points = getcache(method.cache, (:points, (2, 5)))
        weights = getcache(method.cache, (:weights, 5))
        result = [[1.5,1], [0.15, 1], [1.5, 0.1], [3.85, 1], [1.5, 2.9]]
        weights_results = [-0.0576, 0.3003, 0.3968, 0.1725, 0.188 ] 
        @test all(map((x,y) ->  all(x ≈ y), eachcol(points), result))
        @test weights ≈ weights_results atol = 1e-4


    end


    @testset "GenConstrainedUnscented: Check kurtosis and skewness of Normal with constrained sigma points" begin
        lb = 3.0
        ub = 20.0
        dist = NormalMeanVariance(5.0,2.0)
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-4
        method = GenConstrainedUnscented{FromTo(Univariate, Univariate)}(lb,ub,1,1, 0.99)
        sigma_points_weights!(method, mdist, vdist ,sdist, kdist)
        points = getcache(method.cache, (:points, (1, 3)))
        weights = getcache(method.cache, (:weights, 3))
        estimate_genut = ExpectationApproximations.unscented_statistics(method,dist,x -> x)
        @test estimate_genut[1] ≈ mdist
        @test estimate_genut[2] ≈ vdist
    end


end