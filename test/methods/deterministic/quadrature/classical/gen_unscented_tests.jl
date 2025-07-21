@testitem "Generalized Unscented approximation method" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra, StableRNGs
    import ExpectationApproximations: GenUnscented, unscented_statistics, approximate_expectation, _cov, getcache, _skewness, _kurtosis, FromTo, getoutputvariate, getinputvariate

    @testset "FromTo" begin
        T = FromTo(Univariate, Univariate)
        @inferred FromTo(Univariate, Univariate)
        Tout = getoutputvariate(T)
        Tin  = getinputvariate(T)

        @test Tout == Univariate
        @test Tin == Univariate

        T2 = FromTo(Multivariate, Univariate)
        T2out = getoutputvariate(T2)
        T2in = getinputvariate(T2)

        @test T2out == Univariate
        @test T2in == Multivariate
    end

    @testset "GenUnscented: Check kurtosis and skewness of Normal" begin
        dist = NormalMeanVariance(10.0, 3.0)
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-4
        method = @inferred GenUnscented{FromTo(Univariate, Univariate)}(1, (1,))
        f(x) = (x - mdist)^4
        estimate_kurtosis_genut = @inferred ExpectationApproximations.unscented_statistics(method, dist, f)
        _, _, _, kestimate = @inferred ExpectationApproximations.unscented_statistics(method, dist, identity)

        estimate_kurtosis_genut_2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, f)
        _, _, _, kestimate2 = @inferred ExpectationApproximations.unscented_statistics(method, mdist, vdist, sdist, kdist, identity, Val(false), Val(false))

        @test isapprox(estimate_kurtosis_genut[1], estimate_kurtosis_genut_2, rtol = brtol)
        @test isapprox(estimate_kurtosis_genut[1], kdist, rtol = brtol)
        @test isapprox(estimate_kurtosis_genut[1], kestimate, rtol = brtol)
        @test isapprox(kestimate, kestimate2, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 3))) ≈ 1
    end

    @testset "GenUnscented: Check kurtosis and skewness of Gamma" begin
        dist = Gamma(10rand(), 10rand())
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-4
        method = GenUnscented{FromTo(Univariate, Univariate)}(1, (1,))
        g(x) = (x - mdist)^3
        f(x) = (x - mdist)^4
        estimate_skew_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_kurtosis_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, f)
        estimate_skew_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        estimate_kurtosis_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, f)

        @test isapprox(estimate_skew_genut, estimate_skew_genut2, rtol = 1e-7)
        @test isapprox(estimate_kurtosis_genut, estimate_kurtosis_genut2, rtol = 1e-7)
        @test isapprox(estimate_skew_genut, sdist, rtol = brtol)
        @test isapprox(estimate_kurtosis_genut, kdist, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 3))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with Beta and Gamma with non-linearity [x*y; z^3] " begin
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(3, (2,))
        dist = (Gamma(8, 4), Beta(3, 4), Gamma(20, 1))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        brtol = 1e-2
        g(x, y, z) = [x * y; z^3]
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand.(dist) for i in 1:1e5]
        mapped_samples = map(d -> g(d[1], d[2], d[3]), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 7))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with Beta, Gamma, Normal" begin
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(3, (3,))
        dist = (Gamma(8, 4), Beta(30, 4), NormalMeanVariance(2, 1))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        brtol = 1e-2
        g(x, y, z) = [x * y; sin(z) * x; y / x]
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand.(dist) for i in 1:1e5]
        mapped_samples = map(d -> g(d[1], d[2], d[3]), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 7))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with Poisson, Rayleigh over sin(x*y) and cos(x*y)" begin
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(2, (2,))
        dist = (Poisson(0.1), Rayleigh(1.0))
        mdist = mean.(dist)
        vdist = var.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        brtol = 1e-1
        g(x, y) = [sin(x * y), cos(x * y)]
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand.(dist) for i in 1:10000]
        mapped_samples = map(d -> g(d[1], d[2]), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 5))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance,Poisson, Rayleigh with sin and multiplication non-linearities" begin
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(4, (2,))
        dist = (MvNormalMeanCovariance([2.0, 5.3], [3.0 4.0; 4.0 10.0]), Poisson(0.1), Rayleigh(1.0))
        mdist = mean.(dist)
        vdist = _cov.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        brtol = 1e-1
        g(x, y, z) = [sin(z * y) * x[1], cos(z * y) * x[2]]
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand.(dist) for i in 1:10000]
        mapped_samples = map(d -> g(first(d), d[2], d[3]), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
        @test sum(getcache(method.cache, (:weights, 9))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with outer product" begin
        method = @inferred GenUnscented{FromTo{Multivariate, Multivariate}}(2, (4,))
        dist = MvNormalMeanCovariance([2.0, 5.3], [3.0 4.0; 4.0 10.0])
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-2
        g(x) = vec(x * x')
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand(dist) for i in 1:50000]
        mapped_samples = map(d -> g(d), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 5))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with trace over outer product" begin
        method = @inferred GenUnscented{FromTo(Multivariate, Univariate)}(4, (1,))
        dist = MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2], Matrix(Diagonal([0.1, 1.0, 3.0, 5.0])))
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-2
        g(x) = tr(x * x')
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand(dist) for i in 1:10000]
        mapped_samples = map(d -> g(d), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
        @test sum(getcache(method.cache, (:weights, 9))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance,Chisq over outer vectorized outer product" begin
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(5, (16, 1))
        dist = (MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2], Matrix(Diagonal([0.1, 1.0, 3.0, 5.0]))), Chisq(2.0))
        mdist = mean.(dist)
        vdist = _cov.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        brtol = 1e-1
        g(x, ν) = vec(ν * x * x')
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand.(dist) for i in 1:10000]
        mapped_samples = map(d -> g(d[1], d[2]), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 11))) ≈ 1
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with abs" begin
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(4, (4,))
        dist = MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2], Matrix(Diagonal([1e-10, 1.0, 3.0, 1e-8])))
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-2
        g(x) = abs.(x)
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand(dist) for i in 1:10000]
        mapped_samples = map(d -> g(d), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
        @test sum(getcache(method.cache, (:weights, 9))) ≈ 1
    end

    @testset "GenUnscented: Basic integrations with MvNormalMeanCovariance with abs" begin
        dist = MvNormalMeanCovariance([2.0, 5.3, 4.3, 10.2], Matrix(Diagonal([1e-10, 1.0, 3.0, 1e-8])))

        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-2
        method = @inferred GenUnscented{FromTo(Multivariate, Univariate)}(length(mdist), (1,))
        g(x) = abs(x' * x) + abs.(x)' * abs.(x)
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, mdist, vdist, sdist, kdist, g)
        samples = [rand(dist) for i in 1:10000]
        mapped_samples = map(d -> g(d), samples)
        estimate_mc = mean(mapped_samples)
        @test isapprox(estimate_genut, estimate_mc, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 9))) ≈ 1
        @test isapprox(estimate_genut, estimate_genut2, rtol = 1e-7)
    end

    @testset "Unscented_approximate: Normal with cascaded abs " begin
        dist = NormalMeanVariance(10randn(), 0.1rand())
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        brtol = 1e-2
        method = @inferred GenUnscented{FromTo(Univariate, Univariate)}(length(mdist), (1,))
        g(x) = abs(x' * x) + abs.(x)' * abs.(x)
        estimate_genut = @inferred ExpectationApproximations.unscented_statistics(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.unscented_statistics(method, mdist, vdist, sdist, kdist, g, Val(false), Val(false))
        samples = [rand(dist) for i in 1:100000]
        mapped_samples = map(d -> g(d), samples)
        mean_estimate_mc = mean(mapped_samples)
        var_estimate_mc = var(mapped_samples)
        @test isapprox(estimate_genut2[1], estimate_genut[1], rtol = 1e-7)
        @test isapprox(estimate_genut2[2], estimate_genut[2], rtol = 1e-7)
        @test isapprox(estimate_genut[2], var_estimate_mc, rtol = brtol)
        @test isapprox(estimate_genut[1], mean_estimate_mc, rtol = brtol)
        @test sum(getcache(method.cache, (:weights, 3))) ≈ 1
    end

    @testset "Unscented_approximate: MvNormal with abs " begin
        dist = MvNormalMeanCovariance(randn(3), Matrix(Diagonal([1e-3, 1e-3, 1e-3])))
        brtol = 1e-2
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(3, (3,))
        g(x) = abs.(x)
        estimate_genut = @inferred ExpectationApproximations.unscented_statistics(method, dist, g)
        mdist = mean(dist)
        vdist = _cov(dist)
        sdist = _skewness(dist)
        kdist = _kurtosis(dist)
        estimate_genut2 = @inferred ExpectationApproximations.unscented_statistics(method, mdist, vdist, sdist, kdist, g, Val(false), Val(false))
        samples = [rand(dist) for i in 1:10000]
        mapped_samples = map(d -> g(d), samples)
        mean_estimate_mc = mean(mapped_samples)
        var_estimate_mc = cov(mapped_samples)

        @test isapprox(estimate_genut2[1], estimate_genut[1], rtol = 1e-7)
        @test isapprox(estimate_genut2[2], estimate_genut[2], rtol = 1e-7)
        @test isapprox(estimate_genut[1], mean_estimate_mc, atol = brtol)
        @test isapprox(estimate_genut[2], var_estimate_mc, atol = brtol)
        @test sum(getcache(method.cache, (:weights, 7))) ≈ 1
    end

    @testset "Unscented_approximate: Multiple MvNormal sum and reduce dimensionality" begin
        dist = (MvNormalMeanCovariance(randn(3), rand(InverseWishart(5, diagm(ones(3))))), MvNormalMeanCovariance(randn(5), diagm(rand(5))))
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(8, (4,))
        g(x, y) = [x + y[1:3]; y[5]]
        mdist = mean.(dist)
        vdist = _cov.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        estimate_genut = @inferred ExpectationApproximations.unscented_statistics(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.unscented_statistics(method, mdist, vdist, sdist, kdist, g, Val(false), Val(true))
        samples = rand.(dist, 100000)
        mapped_samples = map((x, y) -> g(x, y), eachcol(first(samples)), eachcol(last(samples)))
        mean_estimate_mc = mean(mapped_samples)
        cov_estimate_mc = cov(mapped_samples)

        @test isapprox(estimate_genut2[1], estimate_genut[1], rtol = 1e-7)
        @test isapprox(estimate_genut2[2], estimate_genut[2], rtol = 1e-7)
        @test isapprox(estimate_genut[1], mean_estimate_mc, rtol = 1e-2)
        @test isapprox(estimate_genut[2], cov_estimate_mc, rtol = 1e-2)
        @test sum(getcache(method.cache, (:weights, 17))) ≈ 1
    end

    @testset "Unscented_approximate: Multiple MvNormal,Gamma, Beta trace scaled by inputs , square and multiplication" begin
        dist = (MvNormalMeanCovariance(randn(3), rand(InverseWishart(5, diagm(ones(3))))), Gamma(3, 4), Beta(0.1))
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(5, (3,))
        g(x, y, z) = [tr(y * x * x'); z^2; z * y]
        mdist = mean.(dist)
        vdist = _cov.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        estimate_genut = @inferred ExpectationApproximations.unscented_statistics(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.unscented_statistics(method, mdist, vdist, sdist, kdist, g, Val(false), Val(true))
        samples = rand.(dist, 1000000)
        mapped_samples = map((x, y, z) -> g(x, y, z), eachcol(first(samples)), getindex(samples, 2), last(samples))
        mean_estimate_mc = mean(mapped_samples)

        @test isapprox(estimate_genut2[1], estimate_genut[1], rtol = 1e-7)
        @test isapprox(estimate_genut2[2], estimate_genut[2], rtol = 1e-7)
        @test isapprox(estimate_genut[1], mean_estimate_mc, rtol = 1e-2)
        @test sum(getcache(method.cache, (:weights, 11))) ≈ 1
    end

    @testset "Unscented_approximate: Multiple MvNormal,Gamma, Beta trace scaled by inputs , square and multiplication" begin
        dist = (Poisson(3.0), Gamma(0.1, 4), Beta(2, 4))
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(3, (3,))
        g(x, y, z) = [x * y * z; z^2; z * y]
        mdist = mean.(dist)
        vdist = _cov.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        estimate_genut = @inferred ExpectationApproximations.unscented_statistics(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.unscented_statistics(method, mdist, vdist, sdist, kdist, g, Val(false), Val(true))
        samples = rand.(dist, 1000000)
        mapped_samples = map((x, y, z) -> g(x, y, z), first(samples), getindex(samples, 2), last(samples))
        mean_estimate_mc = mean(mapped_samples)

        @test isapprox(estimate_genut2[1], estimate_genut[1], rtol = 1e-7)
        @test isapprox(estimate_genut2[2], estimate_genut[2], rtol = 1e-7)
        @test isapprox(estimate_genut[1], mean_estimate_mc, rtol = 1e-2)
        @test sum(getcache(method.cache, (:weights, 7))) ≈ 1
    end

    @testset "approximate: Multiple MvNormal approximate the resulting distribution " begin
        dist = (MvNormalMeanCovariance([1, 2, 3], rand(InverseWishart(5, diagm(ones(3))))), MvNormalMeanCovariance([10.0, 0.9, 0.1, 1.0, 0.75], diagm(rand(5))))
        method = @inferred GenUnscented{FromTo(Multivariate, Multivariate)}(8, (8,))
        g(x, y) = [x + y[1:3]; y[5]; abs.(y[1:4]) .^ 1.2]
        mdist = mean.(dist)
        vdist = _cov.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        estimate_genut = @inferred ExpectationApproximations.approximate(method, g, dist)
        samples = rand.(dist, 100000)
        mapped_samples = map((x, y) -> g(x, y), eachcol(first(samples)), eachcol(last(samples)))
        mean_estimate_mc = mean(mapped_samples)

        @test isapprox(mean(estimate_genut), mean_estimate_mc, rtol = 1e-1)
        @test sum(getcache(method.cache, (:weights, 17))) ≈ 1
    end

    @testset "Unscented_approximate: Multiple MvNormal,Gamma, Beta trace scaled" begin
        rng = StableRNG(11)
        dist = (MvNormalMeanCovariance(randn(rng, 3), rand(rng, InverseWishart(10, 0.001diagm(ones(3))))), Gamma(3, 4), Beta(0.1))
        method = @inferred GenUnscented{FromTo(Multivariate, Matrixvariate)}(5, (3, 3))
        g(x, y, z) = z * y * x * x'
        f(x, y, z) = log(1 + y) * sin(z) * x * x'
        h(x, y, z) = tr(x * x') * log(1 + y) * sin(z) * x * x'
        mdist = mean.(dist)
        vdist = _cov.(dist)
        sdist = _skewness.(dist)
        kdist = _kurtosis.(dist)
        estimate_genut = @inferred ExpectationApproximations.approximate_expectation(method, dist, g)
        estimate_genut2 = @inferred ExpectationApproximations.approximate_expectation(method, dist, f)
        estimate_genut3 = @inferred ExpectationApproximations.approximate_expectation(method, dist, h)
        samples = rand.(dist, 1000000)
        mapped_samples = map((x, y, z) -> g(x, y, z), eachcol(first(samples)), getindex(samples, 2), last(samples))
        mean_estimate_mc = mean(mapped_samples)
        mapped_samples2 = map((x, y, z) -> f(x, y, z), eachcol(first(samples)), getindex(samples, 2), last(samples))
        mean_estimate_mc2 = mean(mapped_samples2)
        mapped_samples3 = map((x, y, z) -> h(x, y, z), eachcol(first(samples)), getindex(samples, 2), last(samples))
        mean_estimate_mc3 = mean(mapped_samples3)

        @test isapprox(estimate_genut3, mean_estimate_mc3, rtol = 1e-1)
        @test isapprox(estimate_genut2, mean_estimate_mc2, rtol = 1e-1)
        @test isapprox(estimate_genut, mean_estimate_mc, rtol = 1e-1)
        @test sum(getcache(method.cache, (:weights, 11))) ≈ 1
    end
end

@testitem "GenUnscented: Numerical stability improvements" begin
    using Test, Distributions, ExponentialFamily, LinearAlgebra, Random
    import ExpectationApproximations: GenUnscented, prepare_cache!, _cov, _skewness, _kurtosis, approximate_expectation, FromTo

    @testset "Handle ill-conditioned covariance matrix" begin
        # Create matrix with extremely small eigenvalues
        d = 4
        V = Matrix{Float64}(I, d, d)
        V[1,1] = 1e-7  # Nearly zero diagonal element
        V[2,2] = 1e-10  # Small diagonal element
        
        # Create method for testing
        method = GenUnscented{FromTo(Multivariate, Multivariate)}(d, (2,))
        m = zeros(d)
        S = zeros(d)  # Skewness vector
        K = fill(3.0, d)  # Kurtosis vector (normal distribution value)
        
        # This should not throw or produce NaNs
        ExpectationApproximations.sigma_points_weights!(method, m, V, S, K)
        
        # Check if weights are valid (sum to 1 and no NaNs)
        weights = ExpectationApproximations.getweights(method)
        @test all(isfinite, weights)
        @test sum(weights) ≈ 1.0
        
        # Check if points contain no NaNs
        points = ExpectationApproximations.getpoints(method)
        @test all(isfinite, points)
    end
    
    @testset "Handle NaN in covariance matrix" begin
        # Create matrix with NaN values
        d = 5
        V = Matrix{Float64}(I, d, d)
        V[1,3] = NaN  # NaN off-diagonal
        V[3,1] = NaN  # Symmetric NaN
        
        # Create method for testing
        method = GenUnscented{FromTo(Multivariate, Multivariate)}(d, (2,))
        m = zeros(d)
        S = zeros(d)  # Skewness vector
        K = fill(3.0, d)  # Kurtosis vector
        
        # This should not throw with NaN input
        ExpectationApproximations.sigma_points_weights!(method, m, V, S, K)
        
        # Check results are finite
        weights = ExpectationApproximations.getweights(method)
        @test all(isfinite, weights)
        @test sum(weights) ≈ 1.0
        
        points = ExpectationApproximations.getpoints(method)
        @test all(isfinite, points)
    end
    
    @testset "Handle NaN in kurtosis vector" begin
        d = 3
        V = Matrix{Float64}(I, d, d)
        m = zeros(d)
        S = zeros(d)
        K = [3.0, NaN, 5.0]  # NaN in kurtosis
        
        method = GenUnscented{FromTo(Multivariate, Multivariate)}(d, (2,))
        
        # This should not throw with NaN kurtosis
        ExpectationApproximations.sigma_points_weights!(method, m, V, S, K)
        
        # Check results
        weights = ExpectationApproximations.getweights(method)
        @test all(isfinite, weights)
        @test sum(weights) ≈ 1.0
        
        points = ExpectationApproximations.getpoints(method)
        @test all(isfinite, points)
    end
    
    @testset "Full integration test with problematic inputs" begin
        d = 4
        # Create an extremely ill-conditioned matrix (nearly singular)
        V = Matrix{Float64}(I, d, d)
        V[1,1] = 1e-14
        V[2,2] = 1e-12
        V[1,2] = 1e-13  # Off-diagonal element
        V[2,1] = 1e-13  # Ensure symmetry
        
        m = zeros(d)
        S = [0.0, 0.0, NaN, 0.0]  # NaN in skewness
        K = [3.0, NaN, 5.0, 3.0]  # NaN in kurtosis
        
        method = GenUnscented{FromTo(Multivariate, Univariate)}(d, (1,))
        
        # Non-linear function with division that could amplify numerical issues
        g(x) = sum(x) / (1.0 + abs(x[1]))
        
        # This should compute successfully without NaNs
        result = @inferred ExpectationApproximations.approximate_expectation(method, m, V, S, K, g)
        
        # Result should be finite
        @test isfinite(result)
        # Result should be close to 0.0
        @test result ≈ 0.0  atol = 1e-4
    end
end
