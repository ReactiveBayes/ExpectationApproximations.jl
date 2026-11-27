@testitem "products: a backend that supports every component, exactly where it is exact" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    joint = P.FactorizedJoint((Categorical([0.4, 0.6]), Bernoulli(0.5)))
    @test P.is_expectation_exact(EA.ExactSum(), sin, joint) === true
    @test P.expectation(EA.ExactSum(), ((z, b),) -> z + b, joint) ≈ 1.6 + 0.5
    # with a router, each component goes to its own backend
    router = P.ExpectationFirstOf(EA.ExactSum(), EA.GaussHermite(5), EA.GaussLaguerre(5))
    mixed = P.FactorizedJoint((Categorical([0.4, 0.6]), MvNormal([1.0, 2.0], [1.0 0.0; 0.0 2.0]), Gamma(2.0, 1.0)))
    f((z, m, λ)) = λ * (3.0 - m[z])^2
    # E[λ] E[(3 - m_z)²] = 2 Σ_k π_k ((3 - μ_k)² + σ_k²)
    @test P.expectation(router, f, mixed) ≈ 2 * (0.4 * (4 + 1) + 0.6 * (1 + 2))
    @test P.is_expectation_exact(router, f, mixed) === false
    @test allocations(P.expectation, P.prepare_expectation(router, mixed), f) == 0
end
