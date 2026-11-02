@testitem "ClosedForms: the mean and second moment of Gaussians" setup = [Helpers] begin
    using Distributions, ExponentialFamily

    for q in (Normal(1.0, 2.0), NormalMeanVariance(1.0, 4.0), NormalMeanPrecision(1.0, 0.25))
        @test P.expectation(B.ClosedForms(), identity, q) ≈ 1.0
        @test P.expectation(B.ClosedForms(), abs2, q) ≈ 5.0
        @test P.isexact(B.ClosedForms(), abs2, q) === true
    end
end

@testitem "ClosedForms: E[log x] under Gamma and LogNormal" setup = [Helpers] begin
    using Distributions, SpecialFunctions

    @test P.expectation(B.ClosedForms(), log, Gamma(2.0, 3.0)) ≈ digamma(2.0) + log(3.0)
    # the gap found by the domain transform experiment: E[log x] under a log-normal is μ
    @test P.expectation(B.ClosedForms(), log, LogNormal(0.4, 0.7)) == 0.4
    @test P.isexact(B.ClosedForms(), log, LogNormal(0.4, 0.7)) === true
    # against an independent numerical computation
    @test P.expectation(B.ClosedForms(), log, Gamma(2.0, 3.0)) ≈ P.expectation(B.QuantileLegendre(200), log, Gamma(2.0, 3.0)) rtol = 1.0e-3
end

@testitem "ClosedForms: E_q[log p(x)] of a Gaussian under a Gaussian, through Logpdf" setup = [Helpers] begin
    using Distributions

    q, p = Normal(1.0, 2.0), Normal(-1.0, 0.5)
    expected = -log(2π * 0.25) / 2 - ((1.0 + 1.0)^2 + 4.0) / (2 * 0.25)
    @test P.expectation(B.ClosedForms(), P.Logpdf(p), q) ≈ expected
    # Gauss–Hermite with three points is exact for this quadratic integrand
    @test P.expectation(B.GaussHermite(3), P.Logpdf(p), q) ≈ expected
end

@testitem "ClosedForms: nothing else is supported" setup = [Helpers] begin
    using Distributions

    @test P.supports(B.ClosedForms(), sin, Normal(0.0, 1.0)) === false
    @test P.supports(B.ClosedForms(), Any, Normal) === false
    @test P.supports(B.ClosedForms(), log, Normal(0.0, 1.0)) === false
    @test_throws P.NoBackendError P.expectation(B.ClosedForms(), sin, Normal(0.0, 1.0))
    @test folds_to(P.supports, (B.ClosedForms, typeof(log), Gamma{Float64}), true)
end
