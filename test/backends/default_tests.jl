@testitem "EADefault: each family goes to its rule" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    backend = EA.EADefault()
    @test backend isa P.FirstOf
    @test P.which_backend(backend, sin, Normal(0.0, 1.0)) isa EA.GaussHermite
    @test P.which_backend(backend, sin, MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])) isa EA.GaussHermite
    @test P.which_backend(backend, sin, LogitNormal(0.0, 1.0)) isa EA.GaussHermite
    @test P.which_backend(backend, sin, Gamma(2.0, 1.0)) isa EA.GaussLaguerre
    @test P.which_backend(backend, sin, Beta(2.0, 3.0)) isa EA.GaussJacobi
    @test P.which_backend(backend, sin, Weibull(2.0, 1.0)) isa EA.QuantileLegendre
    @test P.which_backend(backend, sin, Categorical([0.5, 0.5])) isa EA.ExactSum
    @test P.which_backend(backend, sin, Poisson(3.0)) === nothing
    @test length(P.which_backend(EA.EADefault(7), sin, Normal(0.0, 1.0))) == 7
end

@testitem "EADefault: decided by inference, exact where it can be" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, JET
    backend = EA.EADefault()
    @test @inferred(P.which_backend(backend, sin, Gamma(2.0, 1.0))) isa EA.GaussLaguerre
    @test folds_to(P.isexact, (typeof(backend), typeof(sin), Categorical{Float64, Vector{Float64}}), true)
    @test folds_to(P.isexact, (typeof(backend), typeof(sin), Normal{Float64}), false)
    @test_opt P.expectation(backend, sin, Normal(0.0, 1.0))
    @test_opt P.expectation(backend, sin, Gamma(2.0, 1.0))
end

@testitem "EADefault: a product of mixed kinds, component by component" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    backend = EA.EADefault(8)
    # the first iteration of the zero-inflated Poisson notebook: data and two priors
    joint = P.FactorizedJoint((P.PointMass(3), Beta(1.0, 1.0), Gamma(2.0, 2.0)))
    @test P.supports(backend, Any, typeof(joint)) === true
    @test P.isexact(backend, sin, joint) === false
    @test P.expectation(backend, ((y, p, λ),) -> y * p * λ, joint) ≈ 3 * 0.5 * 4.0
    @test allocations(P.expectation, P.prepare_expectation(backend, joint), ((y, p, λ),) -> y * p * λ) == 0
end

@testitem "EADefault: ExponentialFamily's types once it is loaded" tags = [:backends, :extensions] setup = [EAHelpers] begin
    using ExponentialFamily
    backend = EA.EADefault()
    @test P.which_backend(backend, sin, NormalMeanVariance(0.0, 1.0)) isa EA.GaussHermite
    @test P.which_backend(backend, sin, GammaShapeRate(2.0, 1.0)) isa EA.GaussLaguerre
end
