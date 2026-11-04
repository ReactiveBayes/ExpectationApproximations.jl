# The variational Gaussian: the q minimising KL(q ‖ p), found by the natural gradient in the
# Gaussian's mean parameters. At its fixed point E_q[∇ logp] = 0 and E_q[∇² logp] = -P, which a
# fine quadrature checks independently of the iterations.

@testitem "gaussian_variational: exact for a Gaussian, with its log-normaliser" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    result = EA.gaussian_variational(x -> logpdf(Normal(2.0, 3.0), x), 0.0)
    @test result.distribution isa Normal
    @test mean(result.distribution) ≈ 2.0 && std(result.distribution) ≈ 3.0
    @test result.logZ ≈ 0.0 atol = 1.0e-10
    @test result.converged && result.mean ≈ 2.0
    # unnormalised: the bound is the log-normaliser itself, log √(18π)
    @test EA.gaussian_variational(x -> -(x - 2)^2 / 18, 0.0).logZ ≈ log(sqrt(18π))

    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    result = EA.gaussian_variational(x -> logpdf(MvNormal(m, Σ), x), zeros(2))
    @test result.distribution isa MvNormal
    @test mean(result.distribution) ≈ m && cov(result.distribution) ≈ Σ
    @test result.logZ ≈ 0.0 atol = 1.0e-10
    @test result.converged

    # a vector start of length one: vector points, and an MvNormal
    result = EA.gaussian_variational(x -> logpdf(MvNormal([2.0], fill(9.0, 1, 1)), x), [0.0])
    @test result.distribution isa MvNormal
    @test mean(result.distribution) ≈ [2.0] && cov(result.distribution) ≈ fill(9.0, 1, 1)
    @test result.converged
end

@testitem "gaussian_variational: the fixed point of a skewed density, against a fine quadrature" tags = [:projections] setup = [EAHelpers] begin
    using Distributions, ForwardDiff
    import BayesBase
    # log of a Gumbel-like density, skewed: x - exp(x) / 2
    logp(x) = x - exp(x) / 2
    result = EA.gaussian_variational(logp, 0.0)
    @test result.converged
    q = result.distribution
    fine = BayesBase.prepare_expectation(EA.GaussHermite(64), q)
    g = BayesBase.expectation(fine, x -> ForwardDiff.derivative(logp, x))
    h = BayesBase.expectation(fine, x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(logp, y), x))
    @test abs(g) < 1.0e-7
    @test h ≈ -1 / var(q) rtol = 1.0e-6
    # the bound beats Laplace's Gaussian, and both are below the true log-normaliser, log(2)
    laplace = EA.laplace(logp, 0.0).distribution
    elbo(q) = BayesBase.expectation(BayesBase.prepare_expectation(EA.GaussHermite(64), q), logp) + entropy(q)
    @test result.logZ ≈ elbo(q) rtol = 1.0e-8
    @test elbo(laplace) < result.logZ < log(2.0)
    # mass-covering beyond the mode: the variational mean sits on the heavier side, below the mode
    @test mean(q) < mean(laplace)
end

@testitem "gaussian_variational: two dimensions, a logistic regression" tags = [:projections] setup = [EAHelpers] begin
    using Distributions, ForwardDiff, LinearAlgebra, StatsFuns
    import BayesBase
    x = [-0.86, -0.30, -0.05, 0.73]
    n = [5, 5, 5, 5]
    y = [0, 1, 3, 5]
    logp(θ) = logpdf(MvNormal(zeros(2), Diagonal([16.0, 400.0])), θ) +
        sum(y[i] * log(logistic(θ[1] + θ[2] * x[i])) + (n[i] - y[i]) * log(logistic(-(θ[1] + θ[2] * x[i]))) for i in eachindex(x))
    result = EA.gaussian_variational(logp, zeros(2))
    @test result.converged
    q = result.distribution
    fine = BayesBase.prepare_expectation(EA.GaussHermite(40), q)
    g = BayesBase.expectation(fine, θ -> ForwardDiff.gradient(logp, θ))
    H = BayesBase.expectation(fine, θ -> ForwardDiff.hessian(logp, θ))
    @test norm(g) < 1.0e-5
    @test H ≈ -inv(cov(q)) rtol = 1.0e-4
    laplace = EA.laplace(logp, zeros(2)).distribution
    elbo(q) = BayesBase.expectation(BayesBase.prepare_expectation(EA.GaussHermite(40), q), logp) + entropy(q)
    @test elbo(laplace) < result.logZ
    # the posterior of β is skewed to the right: the variational mean lies above the mode
    @test mean(q)[2] > mean(laplace)[2]
end

@testitem "gaussian_variational: in a domain, a Gamma density" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    target = Gamma(3.0, 0.5)
    result = EA.gaussian_variational(x -> logpdf(target, x), 1.0, EA.Positive())
    @test result.converged
    @test result.distribution isa EA.TransformedGaussian
    @test result.distribution.domain === EA.Positive()
    @test result.mean > 0
    # the bound is below the normalised density's log-normaliser, zero, and close to it
    @test -0.1 < result.logZ < 0.0
    @test mean(result.distribution) ≈ mean(target) rtol = 0.05
end

@testitem "gaussian_variational: reports when it does not settle" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    result = EA.gaussian_variational(x -> x - exp(x) / 2, 0.0; maxiters = 1)
    @test !result.converged && result.iterations == 1
    @test_throws "not finite" EA.gaussian_variational(x -> x > 0 ? -x : -Inf, -1.0; maxiters = 3)
end
