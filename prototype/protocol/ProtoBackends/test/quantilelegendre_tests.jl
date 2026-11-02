@testitem "QuantileLegendre: any continuous univariate distribution" setup = [Helpers] begin
    using Distributions

    @test_throws ArgumentError B.QuantileLegendre(0)
    # the quantile functions are not smooth at the ends (like √u for this Beta near zero, like
    # -log(1 - u) for a Gamma near one), so convergence is algebraic, not spectral
    beta_error(n) = abs(P.expectation(B.QuantileLegendre(n), identity, Beta(2.0, 5.0)) - 2 / 7)
    @test beta_error(16) < 1e-3
    @test beta_error(64) < beta_error(16) / 10
    gamma_error(n) = abs(P.expectation(B.QuantileLegendre(n), identity, Gamma(2.0, 2.0)) - 4.0)
    @test gamma_error(16) < 0.05
    @test gamma_error(64) < gamma_error(16)
    @test P.supports(B.QuantileLegendre(16), Any, Gamma) === true
    @test P.supports(B.QuantileLegendre(16), Any, Poisson) === false
    @test P.supports(B.QuantileLegendre(16), Any, MvNormal) === false
    @test P.isexact(B.QuantileLegendre(16), identity, Beta(2.0, 5.0)) === false
end

@testitem "QuantileLegendre: prepared expectations do not allocate" setup = [Helpers] begin
    using Distributions

    prep = P.prepare_expectation(B.QuantileLegendre(16), Gamma(2.0, 2.0))
    @test allocations(P.expectation, prep, log) == 0
end
