# Moment matching by quadrature on the density's own Laplace approximation: exact for a Gaussian,
# a factor much sharper than its start resolved, and a skewed density against its exact moments.

@testitem "moment_matching: exact for a Gaussian, with its log-normaliser" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    result = EA.moment_matching(x -> logpdf(Normal(2.0, 3.0), x), 0.0)
    @test result.distribution isa Normal
    @test mean(result.distribution) ≈ 2.0 && std(result.distribution) ≈ 3.0
    @test result.logZ ≈ 0.0 atol = 1.0e-10
    @test result.converged && result.mean ≈ 2.0
    # unnormalised: the log-normaliser itself, log √(18π)
    @test EA.moment_matching(x -> -(x - 2)^2 / 18, 0.0).logZ ≈ log(sqrt(18π))

    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    result = EA.moment_matching(x -> logpdf(MvNormal(m, Σ), x), zeros(2))
    @test result.distribution isa MvNormal
    @test mean(result.distribution) ≈ m && cov(result.distribution) ≈ Σ
    @test result.logZ ≈ 0.0 atol = 1.0e-10
    # a vector start of length one keeps vectors
    @test EA.moment_matching(x -> logpdf(MvNormal([2.0], fill(9.0, 1, 1)), x), [0.0]).distribution isa MvNormal
end

@testitem "moment_matching: a factor much sharper than its cavity" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    # a wide cavity N(0, 10²) and a sharp likelihood N(3 | x, 0.01²): conjugate, so exact
    cavity, y, r = Normal(0.0, 10.0), 3.0, 1.0e-4
    logf(x) = logpdf(Normal(x, sqrt(r)), y)
    v = inv(1 / var(cavity) + 1 / r)
    m = v * y / r
    result = EA.moment_matching(x -> logf(x) + logpdf(cavity, x), 0.0)
    @test mean(result.distribution) ≈ m rtol = 1.0e-10
    @test var(result.distribution) ≈ v rtol = 1.0e-8
    @test result.logZ ≈ logpdf(Normal(0.0, sqrt(var(cavity) + r)), y) rtol = 1.0e-10
    # on the cavity, the same points miss the factor: the variance collapses
    _, _, vcavity = EA.tilted_mean_cov(EA.GaussHermite(20), logf, cavity)
    @test !isapprox(vcavity, v; rtol = 0.5)
end

@testitem "moment_matching: a skewed density, against its exact moments" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    # log of a Gumbel density, x - exp(x) / 2: normaliser 2, mean log 2 - γ, variance π²/6
    logp(x) = x - exp(x) / 2
    result = EA.moment_matching(logp, 0.0; backend = EA.GaussHermite(40))
    @test result.converged
    @test result.logZ ≈ log(2.0) rtol = 1.0e-4
    @test mean(result.distribution) ≈ log(2.0) - Base.MathConstants.eulergamma atol = 1.0e-3
    @test var(result.distribution) ≈ π^2 / 6 rtol = 1.0e-3
    # not Laplace's Gaussian: the mode is at log 2 and the mean below it
    @test mean(result.distribution) < log(2.0) - 0.1
end

@testitem "moment_matching: in a domain, a Gamma density" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    # Gamma(3, 2) in log coordinates: u = log x has density ∝ exp(3u - e^u / 2)
    result = EA.moment_matching(x -> logpdf(Gamma(3.0, 2.0), x), 1.0, EA.Positive(); backend = EA.GaussHermite(40))
    @test result.distribution isa EA.TransformedGaussian
    @test result.converged
    @test result.logZ ≈ 0.0 atol = 1.0e-6
    grid = range(-10.0, 6.0; length = 100_001)
    w = exp.(3 .* grid .- exp.(grid) ./ 2)
    mu = sum(w .* grid) / sum(w)
    @test log(result.mean) ≈ mu rtol = 1.0e-6
end

@testitem "moment_matching: moments that are not a Gaussian's fall back to its points' Gaussian" tags = [:projections] setup = [EAHelpers] begin
    using Distributions, LinearAlgebra
    # A density far sharper than the unit Gaussian the points fall back to when Laplace's method
    # does not converge (here it is not allowed a step): the weights fall on one point.
    logp(x) = -1.0e12 * sum(abs2, x .- 0.3)
    result = EA.moment_matching(logp, [5.0, 5.0]; maxiters = 0)
    @test !result.converged
    @test result.distribution isa MvNormal && isposdef(cov(result.distribution))
    result = EA.moment_matching(x -> -1.0e12 * (x - 0.3)^2, 5.0; maxiters = 0)
    @test !result.converged && result.distribution isa Normal
end
