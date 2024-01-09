@testitem "Generalized Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import Distributions: skewness, kurtosis, cov, var
    import ExpectationApproximations: GenUnscented, unscented_statistics, approximate_expectation
    skewness(dist::MvNormalMeanCovariance) = zeros(length(dist.μ))
    kurtosis(dist::MvNormalMeanCovariance,correction::Bool=false) = 3var(dist)
   
    @testset "GenUnscented: Basic integrations with Beta and Gamma" begin
        method = GenUnscented()
        dist = (Gamma(8,4),Beta(3,4), Gamma(20,1))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = skewness.(dist)
        kdist = kurtosis.(dist, false)
        brtol = 1e-2
        g(x,y,z) = [x*y; z]
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand.(dist) for i=1:1e5];
        mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end

    @testset "GenUnscented: Basic integrations with Beta, Gamma, Normal" begin
        method = GenUnscented()
        dist = (Gamma(8,4),Beta(30,4), NormalMeanVariance(2,1))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = skewness.(dist)
        kdist = kurtosis.(dist, false)
        brtol = 1e-1
        g(x,y,z) = [x*y; sin(z)*x; y/x]
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand.(dist) for i=1:1e5];
        mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end


    @testset "GenUnscented: Basic integrations with Poisson, Rayleigh" begin
        method = GenUnscented()
        dist = (Poisson(0.1),Rayleigh(1.0))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = skewness.(dist)
        kdist = kurtosis.(dist, false)
        brtol = 1e-1
        g(x,y) = [sin(x*y), cos(x*y)]
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand.(dist) for i=1:10000];
        mapped_samples = map(d -> g(d[1],d[2]),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance,Poisson, Rayleigh" begin
        method = GenUnscented()
        dist = (MvNormalMeanCovariance([2.0,5.3],[3.0 4.0;4.0 10.0]), Poisson(0.1), Rayleigh(1.0))
        mdist = mean.(dist)
        vdist = cov.(dist)
        sdist = skewness.(dist)
        kdist = kurtosis.(dist, false)
        brtol = 1e-1
        g(x,y,z) = [sin(z*y)*x[1], cos(z*y)*x[2]]
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand.(dist) for i=1:10000];
        mapped_samples = map(d -> g(first(d),d[2],d[3]),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance" begin
        method = GenUnscented()
        dist = MvNormalMeanCovariance([2.0, 5.3],[3.0 4.0;4.0 10.0])
        mdist = mean(dist)
        vdist = cov(dist)
        sdist = skewness(dist)
        kdist = kurtosis(dist, false)
        brtol = 1e-2
        g(x) = x*x'
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand(dist) for i=1:10000];
        mapped_samples = map(d -> g(d),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance" begin
        method = GenUnscented()
        dist = MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2],Matrix(Diagonal([0.1,1.0,3.0,5.0])))
        mdist = mean(dist)
        vdist = cov(dist)
        sdist = skewness(dist)
        kdist = kurtosis(dist, false)
        brtol = 1e-2
        g(x) = tr(x*x') 
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand(dist) for i=1:10000];
        mapped_samples = map(d -> g(d),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance,Chisq" begin
        method = GenUnscented()
        dist = (MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2],Matrix(Diagonal([0.1,1.0,3.0,5.0]))), Chisq(2.0))
        mdist = mean.(dist)
        vdist = cov.(dist)
        sdist = skewness.(dist)
        kdist = kurtosis.(dist, false)
        brtol = 1e-1
        g(x,ν) = ν*x*x'
        estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
        samples = [rand.(dist) for i=1:10000];
        mapped_samples = map(d -> g(d[1],d[2]),samples);
        estimate_mc = mean(mapped_samples);
        @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end


end