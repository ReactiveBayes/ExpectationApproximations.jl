@testitem "TransformedGaussian: a density on its domain" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    t = EA.pushforward(EA.Bounded(2.0, 5.0), Normal(0.3, 0.7))
    grid = range(2.0 + 1.0e-9, 5.0 - 1.0e-9; length = 200_000)
    @test sum(x -> pdf(t, x), grid) * step(grid) ≈ 1 rtol = 1.0e-6
    @test all(x -> 2 < x < 5, rand(StableRNG(1), t, 1000))
    # moments through the protocol, against sampling
    @test mean(t) ≈ P.expectation(EA.MonteCarlo(200_000, StableRNG(2)), identity, t) rtol = 1.0e-2
    @test P.supports_expectation(EA.GaussHermite(8), Any, typeof(t)) === true
end

@testitem "TransformedGaussian: a simplex from a multivariate Gaussian" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions, StableRNGs
    t = EA.pushforward(EA.Simplex(3), MvNormal([0.2, -0.5], [0.3 0.1; 0.1 0.2]))
    @test length(t) == 3
    x = rand(StableRNG(1), t)
    @test sum(x) ≈ 1
    @test all(>(0), x)
    m = P.expectation(EA.GaussHermite(8), identity, t)
    @test sum(m) ≈ 1
    @test m ≈ P.expectation(EA.MonteCarlo(100_000, StableRNG(3)), identity, t) rtol = 2.0e-2
end

@testitem "laplace in a domain: the Poisson–Gamma check of the domain experiment" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions
    # the posterior Gamma(a, rate b) in log coordinates has the log-density a u - b eᵘ: mode
    # log(a/b), curvature a, so the transformed Laplace approximation is LogNormal(log(a/b), 1/√a)
    a, b = 23.0, 21.0
    result = EA.laplace(x -> logpdf(Gamma(a, 1 / b), x), 1.0, EA.Positive())
    @test result.distribution isa LogNormal
    @test params(result.distribution)[1] ≈ log(a / b)
    @test params(result.distribution)[2] ≈ 1 / sqrt(a)
    @test result.converged
    # the log-normaliser is that of the density in the original coordinates: zero, up to
    # Laplace's error
    @test abs(result.logZ) < 1.0e-2
end

@testitem "laplace in a domain: a mode at the boundary" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions
    # the density of Beta(0.8, 3) is unbounded at zero; in logit coordinates it has a mode
    result = EA.laplace(x -> logpdf(Beta(0.8, 3.0), x), 0.5, EA.UnitInterval())
    @test result.converged
    @test result.distribution isa EA.TransformedGaussian
    @test 0 < mean(EA.mean_cov(EA.GaussHermite(32), identity, result.distribution)[1]) < 0.5
end

@testitem "TransformedGaussian: its entropy" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions
    # H[t] = H[g] + E_g[log |J|], against -E_t[log t] and, on the half-line, the LogNormal's
    g = Normal(0.3, 0.7)
    @test entropy(EA.TransformedGaussian(g, EA.Positive())) ≈ entropy(LogNormal(0.3, 0.7))
    for D in (EA.UnitInterval(), EA.Bounded(2.0, 5.0), EA.UpperBounded(-1.0))
        t = EA.TransformedGaussian(g, D)
        @test entropy(t) ≈ -P.expectation(EA.GaussHermite(64), x -> logpdf(t, x), t) rtol = 1.0e-10
    end
    t = EA.pushforward(EA.Simplex(3), MvNormal([0.2, -0.5], [0.3 0.1; 0.1 0.2]))
    @test entropy(t) ≈ -P.expectation(EA.GaussHermite(32), x -> logpdf(t, x), t) rtol = 1.0e-10
    c = EA.TransformedGaussian(MvNormal([0.2, -0.5], [0.3 0.1; 0.1 0.2]), EA.Componentwise(EA.UnitInterval(), EA.Positive()))
    @test entropy(c) ≈ -P.expectation(EA.GaussHermite(32), x -> logpdf(c, x), c) rtol = 1.0e-10
end
