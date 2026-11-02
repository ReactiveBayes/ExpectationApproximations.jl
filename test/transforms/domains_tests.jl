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
    # a product: the first factor that has a support, as a prior times messages
    @test EA.domain(P.ProductOf(Gamma(2.0, 1.0), Normal(0.0, 1.0))) === EA.Positive()
    @test EA.domain(P.LinearizedProductOf([Beta(1.0, 1.0), Beta(2.0, 2.0)], 2)) === EA.UnitInterval()
end

@testitem "pushforward: named families where they exist" tags = [:transforms] setup = [EAHelpers] begin
    using Distributions
    g = Normal(0.3, 0.7)
    @test EA.pushforward(EA.RealLine(), g) === g
    @test EA.pushforward(EA.Positive(), g) == LogNormal(0.3, 0.7)
    @test EA.pushforward(EA.UnitInterval(), g) == LogitNormal(0.3, 0.7)
    t = EA.pushforward(EA.Bounded(2.0, 5.0), g)
    @test t isa EA.TransformedGaussian
    @test minimum(t) == 2.0
    @test maximum(t) == 5.0
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
