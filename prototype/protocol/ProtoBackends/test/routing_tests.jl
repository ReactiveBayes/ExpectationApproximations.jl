@testitem "a default router: closed forms first, then quadrature" tags = [:routing] setup = [Helpers] begin
    using Distributions, ExponentialFamily, JET

    router = B.default_router(16)
    @test router isa P.FirstOf
    @test P.which_backend(router, log, Gamma(2.0, 1.0)) === B.ClosedForms()
    @test P.which_backend(router, identity, NormalMeanVariance(0.0, 1.0)) === B.ClosedForms()
    @test P.which_backend(router, sin, NormalMeanVariance(0.0, 1.0)) isa B.GaussHermite
    @test P.which_backend(router, sin, Gamma(2.0, 1.0)) isa B.QuantileLegendre
    @test folds_to(P.isexact, (typeof(router), typeof(log), Gamma{Float64}), true)
    @test folds_to(P.isexact, (typeof(router), typeof(sin), Gamma{Float64}), false)
    @test_opt P.expectation(router, sin, Gamma(2.0, 1.0))
    @test_opt P.expectation(router, log, Gamma(2.0, 1.0))
end

@testitem "a default router: products route each component" tags = [:routing] setup = [Helpers] begin
    using Distributions, ExponentialFamily
    import BayesBase: FactorizedJoint, PointMass

    router = B.default_router(16)
    # the first iteration of the zero-inflated Poisson notebook: data and a Gamma prior
    joint = FactorizedJoint((PointMass(3.0), Gamma(2.0, 1.0)))
    @test P.supports(router, Any, typeof(joint)) === true
    @test P.isexact(router, sin, joint) === false
    @test P.isexact(router, sin, FactorizedJoint((PointMass(3.0), Categorical([0.5, 0.5])))) === true
    @test P.expectation(router, ((y, λ),) -> y * λ, joint) ≈ 3.0 * P.expectation(B.QuantileLegendre(16), identity, Gamma(2.0, 1.0))
end

@testitem "the mixture notebook's integrand: no allocations, duals pass" tags = [:routing, :alloc] setup = [Helpers] begin
    using Distributions, ExponentialFamily, ForwardDiff, JET
    import BayesBase: FactorizedJoint, PointMass

    # E over z, m, s of the log-density of y under component z, as a function of y
    loglik(y, (z, m, s)) = (s[z] - log(2π) - exp(s[z]) * (y - m[z])^2) / 2
    joint = FactorizedJoint(
        (
            Categorical([0.4, 0.6]),
            MvNormalMeanCovariance([-2.0, 2.0], [0.1 0.0; 0.0 0.1]),
            MvNormalMeanCovariance([0.5, -0.5], [0.05 0.0; 0.0 0.05]),
        )
    )
    prep = P.prepare_expectation(B.default_router(5), joint)
    message(p, y) = P.expectation(p, inputs -> loglik(y, inputs))
    @test allocations(message, prep, 1.0) == 0
    @test_opt message(prep, 1.0)

    # the Laplace kernel differentiates the message twice
    curvature(p, y) = ForwardDiff.derivative(t -> ForwardDiff.derivative(u -> message(p, u), t), y)
    @test curvature(prep, 1.0) < 0
    @test allocations(curvature, prep, 1.0) == 0
end

@testitem "withrng reaches a sampler inside the default router, without allocating" tags = [:routing, :alloc] setup = [Helpers] begin
    using StableRNGs

    rng = StableRNG(1)
    router = P.FirstOf(B.ClosedForms(), B.GaussHermite(16), B.MonteCarlo(100))
    @test P.getrng(P.withrng(router, rng).backends[3]) === rng
    @test allocations(P.withrng, router, rng) == 0
end
