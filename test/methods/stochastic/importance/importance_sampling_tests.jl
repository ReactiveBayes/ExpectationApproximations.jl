@testitem "ImportanceSampling: Basic integrations/approximations with Gaussian p" begin
    include("./importance_setuptests.jl")

    irng = StableRNG(123)
    true_dist = Normal(5.0, 6.0)
    n = 10000
    sigmas = [5, 6, 7, 8, 9, 10, 11, 16]
    means = [3, 4, 7, 8]
    brtol = 0.1
    resampling = [true, false]

    for sigma in sigmas, m in means, resample in resampling
        importance_dist = convert(ExponentialFamilyDistribution, Normal(m, sigma))
        method = ImportanceSampling(irng, n, importance_dist; resampling = resample)
        @test isapprox(approximate(method, true_dist, (x) -> 1), 1.0, rtol = brtol)
        tmp = approximate(method, true_dist, (x) -> x)
        @test isapprox(tmp, mean(true_dist), rtol = brtol)
        @test isapprox(approximate(method, true_dist, (x) -> (x - tmp)^2), var(true_dist), rtol = brtol)
    end
end

@testitem "ImportanceSampling: Basic integration with unnormalized distribution: exp(-x^2)" begin
    include("./importance_setuptests.jl")

    irng = StableRNG(123)
    true_dist = x -> exp(-0.5(x - 5)^2 / 36)
    n = 10000
    sigmas = [5, 6, 7, 8, 9, 10, 11, 16]
    means = [3, 4, 7, 8]
    brtol = 0.1
    resampling = [true, false]
    for sigma in sigmas, m in means, resample in resampling
        importance_dist = Normal(m, sigma)
        method = ImportanceSampling(irng, n, importance_dist; resampling = resample)
        @test isapprox(approximate(method, true_dist, (x) -> 1), 1, rtol = brtol)
        tmp = approximate(method, true_dist, (x) -> x)
        @test isapprox(tmp, 5, rtol = brtol)
        @test isapprox(approximate(method, true_dist, (x) -> (x - tmp)^2), 36, rtol = brtol)
    end
end

@testitem "ImportanceSampling: Basic integration with normalized distribution: Gamma" begin
    include("./importance_setuptests.jl")

    irng = StableRNG(123)
    alpha = 5
    beta = 4
    true_dist = Gamma(alpha, 1 / beta)
    n = 10000
    alphas = [3, 4, 5, 6, 7]
    betas = [3, 4, 5, 6, 7]
    brtol = 1.0
    resampling = [true, false]
    for a in alphas, b in betas, resample in resampling
        importance_dist = Gamma(a, 1 / b)
        method = ImportanceSampling(irng, n, importance_dist; resampling = resample)
        @test isapprox(approximate(method, true_dist, (x) -> 1), 1, rtol = brtol)
        tmp = approximate(method, true_dist, (x) -> x)
        @test isapprox(tmp, alpha / beta, rtol = brtol)
        @test isapprox(approximate(method, true_dist, (x) -> log(x)), digamma(alpha) - log(beta), rtol = brtol)
    end
end

@testitem "ImportanceSampling: Basic integration with unnormalized distribution: Beta" begin
    include("./importance_setuptests.jl")

    irng = StableRNG(123)
    alpha = 5
    beta = 4
    true_dist = x -> x^(alpha - 1) * (1 - x)^(beta - 1)
    n = 10000
    alphas = [3, 4, 5, 6, 7]
    betas = [3, 4, 5, 6, 7]
    brtol = 0.1
    resampling = [true, false]
    for a in alphas, b in betas, resample in resampling
        importance_dist = Beta(a, b)
        method = ImportanceSampling(irng, n, importance_dist; resampling = resample)
        @test isapprox(approximate(method, true_dist, (x) -> 1), 1, rtol = brtol)
        tmp = approximate(method, true_dist, (x) -> x)
        @test isapprox(tmp, alpha / (beta + alpha), rtol = brtol)
        @test isapprox(approximate(method, true_dist, (x) -> log(x)), digamma(alpha) - digamma(alpha + beta), rtol = brtol)
    end
end

@testitem "ImportanceSampling: Basic integration with unnormalized distribution and Beta importance (function)" begin
    include("./importance_setuptests.jl")

    irng = StableRNG(123)
    true_dist = x -> x^(4) * (1 - x)^3 * exp(-x^2)
    trange = collect(0.0000001:0.00001:1.0)
    normalization_true = mapreduce(true_dist, +, trange)
    weights = mapreduce(true_dist, +, trange) / normalization_true
    m = sum(weights .* trange) / length(trange)
    logm = sum(weights .* log.(trange)) / length(trange)
    n = 10000
    alphas = [2, 3, 4, 5, 6, 7, 10, 20]
    betas = [2, 3, 4, 5, 6, 7, 10, 20]
    brtol = 0.5
    logmtol = 1.0
    resampling = [true, false]
    for a in alphas, b in betas, resample in resampling
        importance_dist = Beta(a, b)
        method = ImportanceSampling(irng, n, importance_dist; resampling = resample)
        @test isapprox(approximate(method, true_dist, (x) -> 1), 1, rtol = brtol)
        tmp = approximate(method, true_dist, (x) -> x)
        @test isapprox(tmp, m, rtol = brtol)
        @test isapprox(approximate(method, true_dist, (x) -> log(x)), logm, rtol = logmtol)
    end
end

@testitem "ImportanceSampling: Basic integration with unnormalized distribution and Beta importance (pdf)" begin
    include("./importance_setuptests.jl")

    irng = StableRNG(123)
    ##expectation of sin(2*pi*x)pdf(Normal(1,0.1),x)x
    trange = collect(0.0000001:0.00001:1.0)
    true_dist = x -> pdf(Beta(5, 3), x) * pdf(Gamma(5, 1 / 4), x)
    normalization_true = mapreduce(true_dist, +, trange)
    weights = mapreduce(true_dist, +, trange) / normalization_true
    m = sum(weights .* trange) / length(trange)
    logm = sum(weights .* log.(trange)) / length(trange)
    n = 10000
    alphas = [2, 3, 4, 5, 6, 7, 10, 20]
    betas = [2, 3, 4, 5, 6, 7, 10, 20]
    brtol = 0.5
    resampling = [true, false]
    logmtol = 1.0
    for a in alphas, b in betas, resample in resampling
        importance_dist = Beta(a, b)
        method = ImportanceSampling(irng, n, importance_dist; resampling = resample)
        @test isapprox(approximate(method, true_dist, (x) -> 1), 1, rtol = brtol)
        tmp = approximate(method, true_dist, (x) -> x)
        @test isapprox(tmp, m, rtol = brtol)
        @test isapprox(approximate(method, true_dist, (x) -> log(x)), logm, rtol = logmtol)
    end
end
