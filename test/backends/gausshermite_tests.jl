@testitem "GaussHermite: construction" tags = [:backends] setup = [EAHelpers] begin
    @test_throws ArgumentError EA.GaussHermite(0)
    @test length(EA.GaussHermite(7)) == 7
    @test sum(EA.GaussHermite(7).weights) ≈ 1
end

@testitem "GaussHermite: exact moments of a Normal up to degree 2n - 1" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    μ, σ = 0.5, 1.5
    q = Normal(μ, σ)
    @test P.expectation(EA.GaussHermite(3), x -> x^4, q) ≈ μ^4 + 6μ^2 * σ^2 + 3σ^4
    @test P.expectation(EA.GaussHermite(3), x -> x^5, q) ≈ μ^5 + 10μ^3 * σ^2 + 15μ * σ^4
    @test P.is_expectation_exact(EA.GaussHermite(3), sin, q) === false
    @test P.supports_expectation(EA.GaussHermite(3), Any, Normal{Float64}) === true
    @test P.supports_expectation(EA.GaussHermite(3), Any, Gamma{Float64}) === false
end

@testitem "GaussHermite: tensor grid for a multivariate Normal" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    prep = P.prepare_expectation(EA.GaussHermite(4), MvNormal(m, Σ))
    @test length(prep) == 16
    @test P.expectation(prep, x -> x * x') ≈ Σ + m * m'
    # Isserlis: E[x₁² x₂²] = (Σ₁₁ + m₁²)(Σ₂₂ + m₂²) + 2Σ₁₂² + 4Σ₁₂ m₁ m₂
    @test P.expectation(prep, x -> x[1]^2 * x[2]^2) ≈ 3 * 2 + 2 * 0.25 - 2
end

@testitem "GaussHermite: log-normal and logit-normal through their Gaussian" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    @test P.expectation(EA.GaussHermite(32), identity, LogNormal(0.3, 0.5)) ≈ exp(0.3 + 0.5^2 / 2)
    q = LogitNormal(0.4, 0.8)
    grid = range(1.0e-6, 1 - 1.0e-6; length = 400_000)
    reference = sum(x * pdf(q, x) for x in grid) * step(grid)
    @test P.expectation(EA.GaussHermite(32), identity, q) ≈ reference rtol = 1.0e-6
end

@testitem "GaussHermite: duals in the integrand and in the parameters" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, ForwardDiff
    prep = P.prepare_expectation(EA.GaussHermite(8), Normal(1.0, 2.0))
    derivative(p, θ) = ForwardDiff.derivative(t -> P.expectation(p, x -> (t * x)^2), θ)
    @test derivative(prep, 0.5) ≈ 2 * 0.5 * 5.0
    @test allocations(derivative, prep, 0.5) == 0
    # the points are typed from the parameters: d/dμ E_{N(μ, 1)}[x²] = 2μ
    @test ForwardDiff.derivative(μ -> P.expectation(EA.GaussHermite(8), abs2, Normal(μ, 1.0)), 0.7) ≈ 1.4
end

@testitem "GaussHermite: prepared expectations do not allocate" tags = [:backends, :alloc] setup = [EAHelpers] begin
    using Distributions
    @test allocations(P.expectation, P.prepare_expectation(EA.GaussHermite(16), Normal(0.5, 2.0)), sin) == 0
    @test allocations(P.expectation, P.prepare_expectation(EA.GaussHermite(5), MvNormal([1.0, 2.0], [1.0 0.0; 0.0 2.0])), sum) == 0
    @test @inferred(P.prepare_expectation(EA.GaussHermite(4), Normal(0.0, 1.0))) isa P.ExpectationPointSet
end
