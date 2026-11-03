@testitem "EADefault: each family goes to its rule" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    backend = EA.EADefault()
    @test backend isa P.ExpectationFirstOf
    @test P.which_expectation_backend(backend, sin, Normal(0.0, 1.0)) isa EA.GaussHermite
    @test P.which_expectation_backend(backend, sin, MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])) isa EA.GaussHermite
    @test P.which_expectation_backend(backend, sin, LogitNormal(0.0, 1.0)) isa EA.GaussHermite
    @test P.which_expectation_backend(backend, sin, Gamma(2.0, 1.0)) isa EA.GaussLaguerre
    @test P.which_expectation_backend(backend, sin, Beta(2.0, 3.0)) isa EA.GaussJacobi
    @test P.which_expectation_backend(backend, sin, Weibull(2.0, 1.0)) isa EA.QuantileLegendre
    @test P.which_expectation_backend(backend, sin, Categorical([0.5, 0.5])) isa EA.ExactSum
    @test P.which_expectation_backend(backend, sin, Poisson(3.0)) === nothing
    @test length(P.which_expectation_backend(EA.EADefault(7), sin, Normal(0.0, 1.0))) == 7
end

@testitem "EADefault: decided by inference, exact where it can be" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, JET
    backend = EA.EADefault()
    @test @inferred(P.which_expectation_backend(backend, sin, Gamma(2.0, 1.0))) isa EA.GaussLaguerre
    @test folds_to(P.is_expectation_exact, (typeof(backend), typeof(sin), Categorical{Float64, Vector{Float64}}), true)
    @test folds_to(P.is_expectation_exact, (typeof(backend), typeof(sin), Normal{Float64}), false)
    @test_opt P.expectation(backend, sin, Normal(0.0, 1.0))
    @test_opt P.expectation(backend, sin, Gamma(2.0, 1.0))
end

@testitem "EADefault: a product of mixed kinds, component by component" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    backend = EA.EADefault(8)
    # the first iteration of the zero-inflated Poisson notebook: data and two priors
    joint = P.FactorizedJoint((P.PointMass(3), Beta(1.0, 1.0), Gamma(2.0, 2.0)))
    @test P.supports_expectation(backend, Any, typeof(joint)) === true
    @test P.is_expectation_exact(backend, sin, joint) === false
    @test P.expectation(backend, ((y, p, λ),) -> y * p * λ, joint) ≈ 3 * 0.5 * 4.0
    @test allocations(P.expectation, P.prepare_expectation(backend, joint), ((y, p, λ),) -> y * p * λ) == 0
end

@testitem "EADefault: ExponentialFamily's types once it is loaded" tags = [:backends, :extensions] setup = [EAHelpers] begin
    using ExponentialFamily
    backend = EA.EADefault()
    @test P.which_expectation_backend(backend, sin, NormalMeanVariance(0.0, 1.0)) isa EA.GaussHermite
    @test P.which_expectation_backend(backend, sin, GammaShapeRate(2.0, 1.0)) isa EA.GaussLaguerre
end

@testitem "register_backends!: the session's choice is EADefault's" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    registry = P.ExpectationBackendRegistry()
    EA.register_backends!(; registry)
    @test length(registry) == 5
    # registering again changes nothing
    EA.register_backends!(; registry)
    @test length(registry) == 5
    frozen = P.ExpectationFirstOf(P.ExpectationDynamicRouter(registry))
    reference = EA.EADefault()
    for q in (Normal(0.0, 1.0), MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), LogNormal(0.0, 1.0), Gamma(2.0, 1.0), Beta(2.0, 3.0), Weibull(2.0, 1.0), Categorical([0.5, 0.5]), P.PointMass(1.0))
        @test P.which_expectation_backend(frozen, sin, q) == P.which_expectation_backend(reference, sin, q)
    end
    @test P.which_expectation_backend(frozen, sin, Poisson(3.0)) === nothing
    joint = P.FactorizedJoint((P.PointMass(3), Beta(1.0, 1.0), Gamma(2.0, 2.0)))
    f = ((y, p, λ),) -> y * p * λ
    @test P.expectation(frozen, f, joint) ≈ P.expectation(reference, f, joint)
    @test allocations(P.expectation, P.prepare_expectation(frozen, joint), f) == 0
    # loading the package registers them in the global registry
    global_frozen = P.ExpectationFirstOf(P.ExpectationDynamicRouter())
    @test P.which_expectation_backend(global_frozen, sin, Gamma(2.0, 1.0)) == EA.GaussLaguerre(16)
end
