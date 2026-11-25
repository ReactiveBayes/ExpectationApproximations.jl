@testitem "domains: forward and inverse are inverses" tags = [:transforms] setup = [EAHelpers] begin
    domains = (EA.RealLine(), EA.Positive(), EA.UnitInterval(), EA.LowerBounded(2.0), EA.UpperBounded(-1.0), EA.Bounded(2.0, 5.0))
    for D in domains, u in (-3.0, -0.4, 0.0, 1.7)
        @test EA.inverse(D, EA.forward(D, u)) ≈ u
    end
    @test EA.forward(EA.Positive(), 0.3) > 0
    @test 0 < EA.forward(EA.UnitInterval(), 0.3) < 1
    @test EA.forward(EA.LowerBounded(2.0), -5.0) > 2
    @test EA.forward(EA.UpperBounded(-1.0), 5.0) < -1
    @test 2 < EA.forward(EA.Bounded(2.0, 5.0), 10.0) <= 5
    S = EA.Simplex(4)
    u = [0.3, -1.2, 0.8]
    x = EA.forward(S, u)
    @test length(x) == 4
    @test sum(x) ≈ 1
    @test all(>(0), x)
    @test EA.inverse(S, x) ≈ u
end

@testitem "domains: the log-Jacobians agree with ForwardDiff" tags = [:transforms] setup = [EAHelpers] begin
    using ForwardDiff, LinearAlgebra
    for D in (EA.RealLine(), EA.Positive(), EA.UnitInterval(), EA.LowerBounded(2.0), EA.UpperBounded(-1.0), EA.Bounded(2.0, 5.0))
        for u in (-2.0, 0.1, 1.5)
            @test EA.logjacobian(D, u) ≈ log(abs(ForwardDiff.derivative(v -> EA.forward(D, v), u))) atol = 1.0e-12
        end
    end
    # the simplex has K - 1 free coordinates, so the Jacobian is that of the first K - 1
    S = EA.Simplex(4)
    u = [0.3, -1.2, 0.8]
    J = ForwardDiff.jacobian(v -> EA.forward(S, v)[1:3], u)
    @test EA.logjacobian(S, u) ≈ logabsdet(J)[1]
end

@testitem "domain: read from the support of a distribution" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions
    @test EA.domain(Normal(0.0, 1.0)) === EA.RealLine()
    @test EA.domain(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0])) === EA.RealLine()
    @test EA.domain(Gamma(2.0, 1.0)) === EA.Positive()
    @test EA.domain(Exponential(1.0)) === EA.Positive()
    @test EA.domain(Beta(2.0, 3.0)) === EA.UnitInterval()
    @test EA.domain(Uniform(2.0, 5.0)) == EA.Bounded(2.0, 5.0)
    @test EA.domain(Pareto(3.0, 2.0)) == EA.LowerBounded(2.0)
    @test EA.domain(Dirichlet([1.0, 2.0, 3.0])) == EA.Simplex(3)
    @test EA.domain(Poisson(3.0)) === nothing
    # a distribution that does not say its support: Distributions has no default
    struct Unsaid <: ContinuousUnivariateDistribution end
    @test EA.domain(Unsaid()) === nothing
    # a product: the narrowest domain of its factors, as a prior times messages; a Gaussian
    # factor, such as a message moment-matched through a function, does not hide the others'
    @test EA.domain(P.ProductOf(Gamma(2.0, 1.0), Normal(0.0, 1.0))) === EA.Positive()
    @test EA.domain(P.ProductOf(Normal(0.0, 1.0), Gamma(2.0, 1.0))) === EA.Positive()
    @test EA.domain(P.ProductOf(Normal(0.3, 1.0), Beta(2.0, 1.0))) === EA.UnitInterval()
    @test EA.domain(P.ProductOf(Uniform(-1.0, 2.0), Gamma(2.0, 1.0))) == EA.Bounded(0.0, 2.0)
    @test EA.domain(P.ProductOf(Exponential(1.0), Uniform(-1.0, 0.5))) == EA.Bounded(0.0, 0.5)
    @test EA.domain(P.ProductOf(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]), Dirichlet([1.0, 1.0]))) == EA.Simplex(2)
    @test EA.domain(P.LinearizedProductOf([Beta(1.0, 1.0), Beta(2.0, 2.0)], 2)) === EA.UnitInterval()
    # a linearised product holds its factors in the first `length` entries of its vector only
    @test EA.domain(P.LinearizedProductOf([Normal(0.0, 1.0), Gamma(2.0, 1.0)], 1)) === EA.RealLine()
    @test EA.domain(P.LinearizedProductOf([Poisson(1.0), Gamma(2.0, 1.0)], 1)) === nothing
    # a transformed Gaussian is on its domain, a vector one as well
    @test EA.domain(EA.TransformedGaussian(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]), EA.Simplex(3))) == EA.Simplex(3)
    @test EA.domain(EA.TransformedGaussian(Normal(0.0, 1.0), EA.Bounded(2.0, 5.0))) == EA.Bounded(2.0, 5.0)
end

@testitem "domain: read from the domain of BayesBase's log-densities" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions
    DomainSets = P.DomainSets
    @test EA.domain(P.ContinuousUnivariateLogPdf(DomainSets.FullSpace(), x -> -x^2)) === EA.RealLine()
    @test EA.domain(P.ContinuousUnivariateLogPdf(DomainSets.HalfLine(), x -> -x)) === EA.Positive()
    @test EA.domain(P.ContinuousUnivariateLogPdf(DomainSets.Interval(0.0, 1.0), x -> 0.0)) === EA.UnitInterval()
    @test EA.domain(P.ContinuousUnivariateLogPdf(DomainSets.Interval(2.0, 5.0), x -> 0.0)) == EA.Bounded(2.0, 5.0)
    @test EA.domain(P.ContinuousMultivariateLogPdf(DomainSets.FullSpace()^2, x -> -sum(abs2, x))) === EA.RealLine()
    # an unspecified domain says nothing, so a product takes the domain of its next factor
    unspecified = P.ContinuousUnivariateLogPdf(P.UnspecifiedDomain(), x -> -x)
    @test EA.domain(unspecified) === nothing
    @test EA.domain(P.ProductOf(unspecified, Gamma(2.0, 1.0))) === EA.Positive()
end

@testitem "pushforward: the Gaussian itself, or a TransformedGaussian" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions
    g = Normal(0.3, 0.7)
    @test EA.pushforward(EA.RealLine(), g) === g
    # Distributions' LogNormal and LogitNormal have no E[log x] in the ecosystem, LogitNormal no
    # mean, variance or entropy either: a TransformedGaussian has them all.
    for D in (EA.Positive(), EA.UnitInterval())
        t = EA.pushforward(D, g)
        @test t isa EA.TransformedGaussian
        @test t.gaussian === g && t.domain === D
    end
    b = EA.pushforward(EA.Bounded(2.0, 5.0), g)
    @test b isa EA.TransformedGaussian
    @test minimum(b) == 2.0
    @test maximum(b) == 5.0
end

@testitem "Componentwise: a block of variables with different domains" tags = [:transforms] setup = [EAHelpers] begin
    using ForwardDiff, LinearAlgebra
    D = EA.Componentwise(EA.UnitInterval(), EA.Positive(), EA.RealLine())
    u = [0.4, -0.3, 2.0]
    x = EA.forward(D, u)
    @test 0 < x[1] < 1
    @test x[2] > 0
    @test x[3] == 2.0
    @test EA.inverse(D, x) ≈ u
    @test EA.logjacobian(D, u) ≈ logabsdet(ForwardDiff.jacobian(v -> EA.forward(D, v), u))[1]
end

@testitem "laplace in a componentwise domain: a joint Laplace of the zero-inflated Poisson model" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions, SpecialFunctions
    # p ~ Beta(1, 1), λ ~ Gamma(2, 1), y ~ p δ₀ + (1 - p) Poisson(λ), as in the notebook, with an
    # illustrative year of counts (365 days, 195 of them without a manuscript)
    counts = Dict(0 => 195, 1 => 104, 2 => 48, 3 => 15, 4 => 3)
    function zip_logpdf(p, λ, y)
        y == 0 && return log(p + (1 - p) * exp(-λ))
        return log1p(-p) + y * log(λ) - λ - loggamma(y + 1)
    end
    logposterior(θ) = logpdf(Beta(1.0, 1.0), θ[1]) + logpdf(Gamma(2.0, 1.0), θ[2]) + sum(n * zip_logpdf(θ[1], θ[2], y) for (y, n) in counts)
    D = EA.Componentwise(EA.UnitInterval(), EA.Positive())
    result = EA.laplace(logposterior, [0.5, 2.0], D)
    @test result.converged
    @test result.distribution isa EA.TransformedGaussian
    @test length(result.distribution) == 2
    p, λ = result.mode
    @test 0 < p < 1
    @test λ > 0
    m = P.expectation(EA.GaussHermite(16), identity, result.distribution)
    @test 0 < m[1] < 1
    @test m[2] > 0
end
