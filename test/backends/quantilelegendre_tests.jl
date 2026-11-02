@testitem "QuantileLegendre: any continuous univariate distribution" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    @test_throws ArgumentError EA.QuantileLegendre(0)
    # quantile functions are not smooth at the ends, so convergence is algebraic
    beta_error(n) = abs(P.expectation(EA.QuantileLegendre(n), identity, Beta(2.0, 5.0)) - 2 / 7)
    @test beta_error(16) < 1.0e-3
    @test beta_error(64) < beta_error(16) / 10
    @test P.expectation(EA.QuantileLegendre(64), identity, Weibull(2.0, 1.0)) ≈ mean(Weibull(2.0, 1.0)) rtol = 1.0e-3
    @test P.supports_expectation(EA.QuantileLegendre(8), Any, Weibull{Float64}) === true
    @test P.supports_expectation(EA.QuantileLegendre(8), Any, Poisson{Float64}) === false
    @test allocations(P.expectation, P.prepare_expectation(EA.QuantileLegendre(16), Gamma(2.0, 2.0)), log) == 0
end
