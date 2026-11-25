@testitem "UnconstrainedLegendre: E[log x] where the old default rules were off" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, SpecialFunctions
    rule = EA.UnconstrainedLegendre()
    # with 16 points, Gauss–Laguerre was off by 15.2, 6.1, 0.97 and 0.039 at these shapes
    for a in (0.05, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 1.0e4, 1.0e8), θ in (1.0e-3, 2.0, 1.0e3)
        @test P.expectation(rule, log, Gamma(a, θ)) ≈ digamma(a) + log(θ) atol = 1.0e-10 rtol = 1.0e-12
    end
    # Gauss–Jacobi with 16 points was off by 0.043 under Beta(0.5, 0.5)
    for (a, b) in ((0.05, 0.05), (0.05, 3.0), (3.0, 0.05), (0.5, 0.5), (2.0, 5.0), (100.0, 200.0), (1.0e4, 3.0))
        @test P.expectation(rule, log, Beta(a, b)) ≈ digamma(a) - digamma(a + b) atol = 1.0e-10
    end
    for (a, b) in ((2.0, 5.0), (0.05, 3.0), (3.0, 0.7), (100.0, 200.0))
        @test P.expectation(rule, x -> log1p(-x), Beta(a, b)) ≈ digamma(b) - digamma(a + b) atol = 1.0e-10
    end
    # a point near one carries `1 - x` with a relative rounding error of about eps / (1 - x), so
    # with half the mass within √δ of one the error of log(1 - x) is about √eps
    @test P.expectation(rule, x -> log1p(-x), Beta(0.5, 0.5)) ≈ digamma(0.5) - digamma(1.0) atol = 1.0e-7
    # the default router and the registered rules choose it
    for backend in (EA.EADefault(), P.ExpectationFirstOf(P.ExpectationDynamicRouter()))
        @test P.expectation(backend, log, Gamma(0.1, 1.0)) ≈ digamma(0.1) atol = 1.0e-10
        @test P.expectation(backend, log, Beta(0.5, 0.5)) ≈ digamma(0.5) - digamma(1.0) atol = 1.0e-10
    end
end

@testitem "UnconstrainedLegendre: moments and the other families" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, SpecialFunctions
    rule = EA.UnconstrainedLegendre()
    for a in (0.05, 0.3, 1.0, 10.0, 1.0e4), k in 1:3
        @test P.expectation(rule, x -> x^k, Gamma(a, 2.0)) ≈ 2.0^k * prod(a + i for i in 0:(k - 1)) rtol = 1.0e-10
    end
    @test P.expectation(rule, log, Exponential(3.0)) ≈ digamma(1.0) + log(3.0) atol = 1.0e-12
    @test P.expectation(rule, log, Chisq(3.0)) ≈ digamma(1.5) + log(2.0) atol = 1.0e-12
    @test P.expectation(rule, log, Erlang(3, 2.0)) ≈ digamma(3.0) + log(2.0) atol = 1.0e-12
    @test P.expectation(rule, log, InverseGamma(2.5, 1.5)) ≈ log(1.5) - digamma(2.5) atol = 1.0e-12
    @test P.expectation(rule, identity, Weibull(0.7, 2.0)) ≈ mean(Weibull(0.7, 2.0)) rtol = 1.0e-12
    # any other unimodal density in its unconstrained coordinate, without declared support
    q = truncated(Normal(1.0, 2.0); lower = 0.0)
    @test P.supports_expectation(rule, Any, typeof(q)) === false
    @test P.expectation(rule, identity, q) ≈ mean(q) rtol = 1.0e-10
    @test P.expectation(rule, x -> x^2, Normal(1.0, 2.0)) ≈ 5.0 rtol = 1.0e-12
    for Q in (Gamma{Float64}, Exponential{Float64}, Beta{Float64}, Chisq{Float64}, InverseGamma{Float64}, Weibull{Float64})
        @test P.supports_expectation(rule, Any, Q) === true
    end
    @test P.supports_expectation(rule, Any, Normal{Float64}) === false
    @test P.supports_expectation(rule, Any, Poisson{Float64}) === false
end

@testitem "UnconstrainedLegendre: the average energy of a Poisson rate" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, SpecialFunctions
    # -E[log Poisson(y | λ)] under λ ~ Gamma(0.3, 2): the default was 11.62 against 14.53
    y, a, θ = 7, 0.3, 2.0
    exact = -(y * (digamma(a) + log(θ)) - a * θ - loggamma(y + 1.0))
    @test -P.expectation(EA.EADefault(), λ -> logpdf(Poisson(λ), y), Gamma(a, θ)) ≈ exact rtol = 1.0e-10
end

@testitem "UnconstrainedLegendre: shapes too small for Float64" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, SpecialFunctions
    rule = EA.UnconstrainedLegendre()
    # about 8e-4 of the mass of Gamma(0.01, 1) lies below floatmin(); its points sit there
    prepared = P.prepare_expectation(rule, Gamma(0.01, 1.0))
    @test all(x -> floatmin() <= x < Inf, prepared.points)
    @test P.expectation(prepared, log) ≈ digamma(0.01) atol = 0.1
    @test P.expectation(prepared, identity) ≈ 0.01 rtol = 1.0e-10
    # and the points of a Beta never round onto 0 or 1
    prepared = P.prepare_expectation(rule, Beta(0.02, 0.02))
    @test all(x -> 0 < x < 1, prepared.points)
    @test isfinite(P.expectation(prepared, x -> log(x) + log1p(-x)))
end

@testitem "UnconstrainedLegendre: duals, allocations and points" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, ForwardDiff
    rule = EA.UnconstrainedLegendre()
    @test_throws ArgumentError EA.UnconstrainedLegendre(0)
    @test EA.UnconstrainedLegendre() == EA.UnconstrainedLegendre(8)
    # d/dθ E_{Gamma(2, θ)}[log x] = 1 / θ, through the scale
    @test ForwardDiff.derivative(θ -> P.expectation(rule, log, Gamma(2.0, θ)), 2.0) ≈ 0.5 rtol = 1.0e-12
    @test_throws ArgumentError ForwardDiff.derivative(a -> P.expectation(rule, log, Gamma(a, 1.0)), 2.0)
    prepared = P.prepare_expectation(rule, Beta(0.5, 2.0))
    @test sum(prepared.weights) ≈ 1
    @test length(prepared.points) <= 26 * length(rule)
    @test allocations(P.expectation, prepared, log) == 0
end

@testitem "conformance: UnconstrainedLegendre" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    EA.Testing.test_backend(
        EA.UnconstrainedLegendre(), [
            EA.Testing.boundary_cases();
            EA.Testing.Case(Gamma(2.0, 3.0), abs2, 2 * 3 * 9.0);
            EA.Testing.Case(Beta(2.0, 5.0), x -> x^3, (2 * 3 * 4) / (7 * 8 * 9))
        ]
    )
end
