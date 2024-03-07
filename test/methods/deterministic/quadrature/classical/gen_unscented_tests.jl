@testitem "Generalized Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra
    import Distributions: skewness, kurtosis, cov, var
    import ExpectationApproximations: GenUnscented, unscented_statistics, approximate_expectation,_cov
    skewness(dist::MvNormalMeanCovariance)   = zeros(length(dist.μ))
    kurtosis(dist::MvNormalMeanCovariance, correction::Bool=false)   = 3*var(dist).^2
    skewness(dist::MvNormalMeanPrecision)   = zeros(length(dist.μ))
    kurtosis(dist::MvNormalMeanPrecision, correction::Bool=false)   = 3*var(dist).^2
    skewness(dist::MvNormalWeightedMeanPrecision)   = zeros(length(dist.xi))
    kurtosis(dist::MvNormalWeightedMeanPrecision, correction::Bool=false)   = 3*var(dist).^2
    skewness(dist::UnivariateGaussianDistributionsFamily) = 0.0
    kurtosis(dist::UnivariateGaussianDistributionsFamily, correction::Bool=false) = 3*var(dist)^2
   
    # @testset "GenUnscented: Check kurtosis and skewness of Normal" begin

    #     dist = Normal(10randn(),5rand())
    #     mdist = mean(dist)
    #     vdist = _cov(dist)
    #     sdist = skewness(dist)
    #     kdist = kurtosis(dist, false)
    #     brtol = 1e-4
    #     method = GenUnscented(1,1)
    #     g(x) = (x-mdist)^3
    #     f(x) = (x-mdist)^4
    #     estimate_skew_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     estimate_kurtosis_genut = ExpectationApproximations.approximate_expectation(method,dist,f)
        
    #     @test isapprox(estimate_skew_genut , sdist, rtol = brtol)
    #     @test isapprox(estimate_kurtosis_genut ,kdist, rtol = brtol)
    # end

    # @testset "GenUnscented: Check kurtosis and skewness of Gamma" begin
    #     dist = Gamma(10rand(), 10rand())
    #     mdist = mean(dist)
    #     vdist = _cov(dist)
    #     sdist = skewness(dist)
    #     kdist = kurtosis(dist, false)
    #     brtol = 1e-4
    #     method = GenUnscented(1,1)
    #     g(x) = (x-mdist)^3
    #     f(x) = (x-mdist)^4
    #     estimate_skew_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     estimate_kurtosis_genut = ExpectationApproximations.approximate_expectation(method,dist,f)
        
    #     @test isapprox(estimate_skew_genut , sdist, rtol = brtol)
    #     @test isapprox(estimate_kurtosis_genut , kdist, rtol = brtol)
    # end

    
    # @testset "GenUnscented: Basic integrations with Beta and Gamma with non-linearity [x*y; z^3] " begin
    #     method = GenUnscented(3, 2)
    #     dist = (Gamma(8,4),Beta(3,4), Gamma(20,1))
    #     mdist = mean.(dist)
    #     vdist = var.(dist)
    #     sdist = skewness.(dist)
    #     kdist = kurtosis.(dist, false)
    #     brtol = 1e-2
    #     g(x,y,z) = [x*y; z^3]
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand.(dist) for i=1:1e5];
    #     mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with Beta, Gamma, Normal" begin
    #     method = GenUnscented(3, 3)
    #     dist = (Gamma(8,4),Beta(30,4), NormalMeanVariance(2,1))
    #     mdist = mean.(dist)
    #     vdist = var.(dist)
    #     sdist = skewness.(dist)
    #     kdist = kurtosis.(dist, false)
    #     brtol = 1e-2
    #     g(x,y,z) = [x*y; sin(z)*x; y/x]
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand.(dist) for i=1:1e5];
    #     mapped_samples = map(d -> g(d[1],d[2],d[3]),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end


    # @testset "GenUnscented: Basic integrations with Poisson, Rayleigh over sin(x*y) and cos(x*y)" begin
    #     method = GenUnscented(2,2)
    #     dist = (Poisson(0.1),Rayleigh(1.0))
    #     mdist = mean.(dist)
    #     vdist = var.(dist)
    #     sdist = skewness.(dist)
    #     kdist = kurtosis.(dist, false)
    #     brtol = 1e-1
    #     g(x,y) = [sin(x*y), cos(x*y)]
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand.(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(d[1],d[2]),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance,Poisson, Rayleigh with sin and multiplication non-linearities" begin
    #     method = GenUnscented(4,2)
    #     dist = (MvNormalMeanCovariance([2.0,5.3],[3.0 4.0;4.0 10.0]), Poisson(0.1), Rayleigh(1.0))
    #     mdist = mean.(dist)
    #     vdist = _cov.(dist)
    #     sdist = skewness.(dist)
    #     kdist = kurtosis.(dist, false)
    #     brtol = 1e-1
    #     g(x,y,z) = [sin(z*y)*x[1], cos(z*y)*x[2]]
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand.(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(first(d),d[2],d[3]),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with outer product" begin
    #     method = GenUnscented(2, 4)
    #     dist = MvNormalMeanCovariance([2.0, 5.3],[3.0 4.0;4.0 10.0])
    #     mdist = mean(dist)
    #     vdist = _cov(dist)
    #     sdist = skewness(dist)
    #     kdist = kurtosis(dist, false)
    #     brtol = 1e-2
    #     g(x) = vec(x*x')
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand(dist) for i=1:50000];
    #     mapped_samples = map(d -> g(d),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with trace over outer product" begin
    #     method = GenUnscented(4, 1)
    #     dist = MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2],Matrix(Diagonal([0.1,1.0,3.0,5.0])))
    #     mdist = mean(dist)
    #     vdist = _cov(dist)
    #     sdist = skewness(dist)
    #     kdist = kurtosis(dist, false)
    #     brtol = 1e-2
    #     g(x) = tr(x*x') 
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(d),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance,Chisq over outer vectorized outer product" begin
    #     method = GenUnscented(5,16)
    #     dist = (MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2],Matrix(Diagonal([0.1,1.0,3.0,5.0]))), Chisq(2.0))
    #     mdist = mean.(dist)
    #     vdist = _cov.(dist)
    #     sdist = skewness.(dist)
    #     kdist = kurtosis.(dist, false)
    #     brtol = 1e-1
    #     g(x,ν) = vec(ν*x*x')
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand.(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(d[1],d[2]),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with abs" begin
    #     method = GenUnscented(4,4)
    #     dist = MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2],Matrix(Diagonal([1e-10,1.0,3.0,1e-8])))
    #     mdist = mean(dist)
    #     vdist = _cov(dist)
    #     sdist = skewness(dist)
    #     kdist = kurtosis(dist, false)
    #     brtol = 1e-2
    #     g(x) = abs.(x) 
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(d),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    # @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with abs" begin

    #     dist = MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2],Matrix(Diagonal([1e-10,1.0,3.0,1e-8])))
       
    #     mdist = mean(dist)
    #     vdist = _cov(dist)
    #     sdist = skewness(dist)
    #     kdist = kurtosis(dist, false)
    #     brtol = 1e-2
    #     method = GenUnscented(length(mdist),1)
    #     g(x) = abs(x'*x) + abs.(x)'*abs.(x)
    #     estimate_genut = ExpectationApproximations.approximate_expectation(method,dist,g)
    #     samples = [rand(dist) for i=1:10000];
    #     mapped_samples = map(d -> g(d),samples);
    #     estimate_mc = mean(mapped_samples);
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    # end

    @testset "Unscented_approximate: " begin
   
        dist = NormalMeanVariance(randn(),0.1rand())
       
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = skewness(dist)
        kdist = kurtosis(dist, false)
        brtol = 1e-2
        method = GenUnscented(length(mdist),1)
        g(x) = abs(x'*x) + abs.(x)'*abs.(x)
        estimate_genut = ExpectationApproximations.Unscented_approximate(method,(dist, ),g)

        samples = [rand(dist) for i=1:100000];
        mapped_samples = map(d -> g(d),samples);
        mean_estimate_mc = mean(mapped_samples);
        var_estimate_mc = var(mapped_samples);
        skewness_estimate_mc = mean(map(d -> (d - mean_estimate_mc)^3, mapped_samples))
        kurtosis_estimate_mc = mean(map(d -> (d - mean_estimate_mc)^4, mapped_samples))

        @show mean_estimate_mc, var_estimate_mc, skewness_estimate_mc, kurtosis_estimate_mc
    #     @test isapprox( estimate_genut , estimate_mc, rtol = brtol)
    end


    

end