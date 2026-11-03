@testitem "laplace: exact for a Gaussian, with its log-normaliser" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    result = EA.laplace(x -> logpdf(Normal(2.0, 3.0), x), 0.0)
    @test result.distribution isa Normal
    @test mean(result.distribution) ≈ 2.0
    @test std(result.distribution) ≈ 3.0
    @test result.logZ ≈ 0.0 atol = 1.0e-12
    @test result.converged
    # an unnormalised log-density: log ∫ exp(-(x - 2)² / 18) dx = log √(18π)
    @test EA.laplace(x -> -(x - 2)^2 / 18, 0.0).logZ ≈ log(sqrt(18π))
end

@testitem "laplace: exact for a multivariate Gaussian, static too" tags = [:projections] setup = [EAHelpers] begin
    using Distributions, LinearAlgebra, StaticArrays
    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    result = EA.laplace(x -> logpdf(MvNormal(m, Σ), x), zeros(2))
    @test mean(result.distribution) ≈ m
    @test cov(result.distribution) ≈ Σ
    @test result.logZ ≈ 0.0 atol = 1.0e-12
    Λ = inv(Σ)
    static = EA.laplace(x -> -(x - SVector(1.0, -1.0))' * SMatrix{2, 2}(Λ) * (x - SVector(1.0, -1.0)) / 2, SVector(0.0, 0.0))
    @test mean(static.distribution) ≈ m
    @test cov(static.distribution) ≈ Σ
    @test static.logZ ≈ log(2π) + logdet(Σ) / 2
end

@testitem "laplace: starts where the log-density is not concave" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    # a double well: x = 0 is a minimum, the modes are ±1 with second derivative -8
    result = EA.laplace(x -> -(x^2 - 1)^2, 0.1)
    @test result.converged
    @test abs(mean(result.distribution)) ≈ 1.0
    @test var(result.distribution) ≈ 1 / 8
end

@testitem "laplace: stays in the domain of the log-density" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    # log Gamma(3, 1) = 2 log x - x: mode 2, variance 2; the first Newton step from 5 would leave
    # the positive half-line, where the log-density is -Inf
    result = EA.laplace(x -> logpdf(Gamma(3.0, 1.0), x), 5.0)
    @test result.converged
    @test mean(result.distribution) ≈ 2.0
    @test var(result.distribution) ≈ 2.0
end

@testitem "laplace: the bioassay posterior of the logistic regression notebook" tags = [:projections] setup = [EAHelpers] begin
    using Distributions, ForwardDiff, LinearAlgebra, StatsFuns
    # BDA3 §3.7: doses, animals and deaths, with a N(0, diag(4², 20²)) prior on (α, β)
    doses, animals, deaths = [-0.86, -0.3, -0.05, 0.73], [5, 5, 5, 5], [0, 1, 3, 5]
    function logposterior(θ)
        α, β = θ
        prior = logpdf(Normal(0.0, 4.0), α) + logpdf(Normal(0.0, 20.0), β)
        return prior + sum(deaths[i] * log(logistic(α + β * doses[i])) + (animals[i] - deaths[i]) * log1p(-logistic(α + β * doses[i])) for i in 1:4)
    end
    result = EA.laplace(logposterior, zeros(2))
    mode = mean(result.distribution)
    @test norm(ForwardDiff.gradient(logposterior, mode)) < 1.0e-8
    @test cov(result.distribution) ≈ inv(-ForwardDiff.hessian(logposterior, mode))
    # the mode by brute force on a grid
    αs, βs = range(-2, 4; length = 601), range(-5, 30; length = 701)
    best = argmax(((α, β),) -> logposterior([α, β]), Iterators.product(αs, βs))
    @test abs(mode[1] - best[1]) <= step(αs)
    @test abs(mode[2] - best[2]) <= step(βs)
end

@testitem "laplace: reports when it does not converge" tags = [:projections] setup = [EAHelpers] begin
    result = EA.laplace(x -> -(x - 100)^4, 0.0; maxiters = 3)
    @test !result.converged
    @test result.iterations == 3
end

@testitem "laplace: converged when it starts at the mode" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    # a Newton step of zero cannot raise the log-density, which is no failure to converge
    result = EA.laplace(x -> logpdf(Normal(0.3, 0.7), x), 0.3)
    @test result.converged && result.mode == 0.3
    result = EA.laplace(x -> -sum(abs2, x .- [1.0, 2.0]) / 2, [1.0, 2.0])
    @test result.converged && result.mode == [1.0, 2.0]
end
