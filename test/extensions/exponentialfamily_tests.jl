@testitem "ExponentialFamily: the univariate Gaussians through GaussHermite" tags = [:extensions] setup = [EAHelpers] begin
    using ExponentialFamily
    μ, σ = 0.5, 1.5
    for q in (NormalMeanVariance(μ, σ^2), NormalMeanPrecision(μ, 1 / σ^2), NormalWeightedMeanPrecision(μ / σ^2, 1 / σ^2))
        @test P.supports_expectation(EA.GaussHermite(3), Any, typeof(q)) === true
        @test P.expectation(EA.GaussHermite(3), x -> x^4, q) ≈ μ^4 + 6μ^2 * σ^2 + 3σ^4
        @test allocations(P.expectation, P.prepare_expectation(EA.GaussHermite(8), q), sin) == 0
    end
end

@testitem "ExponentialFamily: the multivariate Gaussians, static points for a static mean" tags = [:extensions] setup = [EAHelpers] begin
    using ExponentialFamily, StaticArrays
    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    for q in (MvNormalMeanCovariance(m, Σ), MvNormalMeanPrecision(m, inv(Σ)))
        @test P.expectation(EA.GaussHermite(4), x -> x * x', q) ≈ Σ + m * m'
    end
    q = MvNormalMeanCovariance(SVector(1.0, -1.0), SMatrix{2, 2}(Σ))
    prep = P.prepare_expectation(EA.GaussHermite(4), q)
    @test eltype(prep.points) <: SVector{2}
    @test P.expectation(prep, x -> x * x') ≈ Σ + m * m'
    @test allocations(P.expectation, prep, x -> x * x') == 0
end

@testitem "ExponentialFamily: GammaShapeRate through GaussLaguerre" tags = [:extensions] setup = [EAHelpers] begin
    using ExponentialFamily
    q = GammaShapeRate(2.5, 4.0)
    @test P.supports_expectation(EA.GaussLaguerre(3), Any, typeof(q)) === true
    # E[x²] = a(a + 1) / b²
    @test P.expectation(EA.GaussLaguerre(3), abs2, q) ≈ 2.5 * 3.5 / 16
end

@testitem "ExponentialFamily: the mixture notebook's integrand, no allocations, duals" tags = [:extensions, :alloc] setup = [EAHelpers] begin
    using ExponentialFamily, Distributions, ForwardDiff
    # the expected log-density of y under component z, over z, the means m and log-precisions s
    loglik(y, (z, m, s)) = (s[z] - log(2π) - exp(s[z]) * (y - m[z])^2) / 2
    joint = P.FactorizedJoint(
        (
            Categorical([0.4, 0.6]),
            MvNormalMeanCovariance([-2.0, 2.0], [0.1 0.0; 0.0 0.1]),
            MvNormalMeanCovariance([0.5, -0.5], [0.05 0.0; 0.0 0.05]),
        )
    )
    prep = P.prepare_expectation(P.ExpectationFirstOf(EA.ExactSum(), EA.GaussHermite(5)), joint)
    message(p, y) = P.expectation(p, inputs -> loglik(y, inputs))
    @test allocations(message, prep, 1.0) == 0
    # the Laplace kernel differentiates the message twice
    curvature(p, y) = ForwardDiff.derivative(t -> ForwardDiff.derivative(u -> message(p, u), t), y)
    @test curvature(prep, 1.0) < 0
    @test allocations(curvature, prep, 1.0) == 0
end
