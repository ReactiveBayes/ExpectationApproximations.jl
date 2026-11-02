@testitem "GaussHermite: construction" setup = [Helpers] begin
    @test_throws ArgumentError B.GaussHermite(0)
    @test length(B.GaussHermite(5).nodes) == 5
    @test sum(B.GaussHermite(5).weights) ≈ 1.0
end

@testitem "GaussHermite: exact moments up to degree 2n - 1 under Gaussians" setup = [Helpers] begin
    using Distributions, ExponentialFamily

    μ, σ = 0.5, 1.5
    for q in (Normal(μ, σ), NormalMeanVariance(μ, σ^2), NormalMeanPrecision(μ, 1 / σ^2))
        @test P.expectation(B.GaussHermite(3), x -> x^4, q) ≈ μ^4 + 6μ^2 * σ^2 + 3σ^4
        @test P.expectation(B.GaussHermite(3), x -> x^5, q) ≈ μ^5 + 10μ^3 * σ^2 + 15μ * σ^4
        @test P.isexact(B.GaussHermite(3), sin, q) === false
    end
end

@testitem "GaussHermite: tensor grid under multivariate Gaussians" setup = [Helpers] begin
    using Distributions, ExponentialFamily, LinearAlgebra

    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    for q in (MvNormal(m, Σ), MvNormalMeanCovariance(m, Σ))
        prep = P.prepare_expectation(B.GaussHermite(4), q)
        @test length(prep) == 16
        @test P.expectation(prep, x -> x * x') ≈ Σ + m * m'
        # Isserlis: E[x₁² x₂²] = (Σ₁₁ + m₁²)(Σ₂₂ + m₂²) + 2Σ₁₂² + 4Σ₁₂ m₁ m₂
        @test P.expectation(prep, x -> x[1]^2 * x[2]^2) ≈ (2 + 1) * (1 + 1) + 2 * 0.5^2 + 4 * 0.5 * 1 * (-1)
    end
end

@testitem "GaussHermite: log-normal and logit-normal, through the unconstrained Gaussian" setup = [Helpers] begin
    using Distributions, StatsFuns

    @test P.expectation(B.GaussHermite(32), identity, LogNormal(0.3, 0.5)) ≈ exp(0.3 + 0.5^2 / 2)
    # the mean of a logit-normal has no closed form: compare with a fine Riemann sum
    q = LogitNormal(0.4, 0.8)
    grid = range(1.0e-6, 1 - 1.0e-6; length = 400_000)
    reference = sum(x * pdf(q, x) for x in grid) * step(grid)
    @test P.expectation(B.GaussHermite(32), identity, q) ≈ reference rtol = 1.0e-6
end

@testitem "GaussHermite: exact sums for point masses and categoricals" setup = [Helpers] begin
    using Distributions
    import BayesBase: PointMass

    @test P.expectation(B.GaussHermite(5), x -> x^3, PointMass(2.0)) == 8.0
    @test P.expectation(B.GaussHermite(5), x -> [x, 1.0], PointMass(2.0)) == [2.0, 1.0]
    @test P.expectation(B.GaussHermite(5), x -> x^2, Categorical([0.2, 0.3, 0.5])) ≈ 0.2 + 1.2 + 4.5
    @test P.isexact(B.GaussHermite(5), sin, PointMass(2.0)) === true
    @test P.isexact(B.GaussHermite(5), sin, Categorical([0.2, 0.8])) === true
end

@testitem "GaussHermite: products of mixed kinds" setup = [Helpers] begin
    using Distributions, ExponentialFamily
    import BayesBase: FactorizedJoint, PointMass

    joint = FactorizedJoint((Categorical([0.4, 0.6]), MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 2.0]), PointMass(3.0)))
    f((z, m, y)) = (y - m[z])^2
    # E[(y - m_z)²] = Σ_k π_k ((y - μ_k)² + σ_k²)
    @test P.expectation(B.GaussHermite(3), f, joint) ≈ 0.4 * (4 + 1) + 0.6 * (1 + 2)
    @test P.isexact(B.GaussHermite(3), f, joint) === false
    @test P.isexact(B.GaussHermite(3), f, FactorizedJoint((Categorical([0.4, 0.6]), PointMass(3.0)))) === true
end

@testitem "GaussHermite: prepared expectations do not allocate" setup = [Helpers] begin
    using Distributions, ExponentialFamily, JET
    import BayesBase: FactorizedJoint, PointMass

    scalar = P.prepare_expectation(B.GaussHermite(16), NormalMeanVariance(0.5, 2.0))
    @test allocations(P.expectation, scalar, sin) == 0
    @test_opt P.expectation(scalar, sin)

    vector = P.prepare_expectation(B.GaussHermite(5), MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 2.0]))
    @test allocations(P.expectation, vector, sum) == 0

    joint = FactorizedJoint((Categorical([0.4, 0.6]), MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 2.0]), PointMass(3.0)))
    product = P.prepare_expectation(B.GaussHermite(5), joint)
    f((z, m, y)) = (y - m[z])^2
    @test allocations(P.expectation, product, f) == 0
    @test_opt P.expectation(product, f)
end

@testitem "GaussHermite: duals in the integrand and in the parameters" setup = [Helpers] begin
    using Distributions, ExponentialFamily, ForwardDiff

    # in the integrand: d/dθ E[(θ x)²] = 2θ E[x²]
    prep = P.prepare_expectation(B.GaussHermite(8), NormalMeanVariance(1.0, 2.0))
    derivative(p, θ) = ForwardDiff.derivative(t -> P.expectation(p, x -> (t * x)^2), θ)
    @test derivative(prep, 0.5) ≈ 2 * 0.5 * 3.0
    @test allocations(derivative, prep, 0.5) == 0

    # in the parameters: d/dμ E_{N(μ, 1)}[x²] = 2μ, the nodes are typed from the parameters
    second_moment(μ) = P.expectation(B.GaussHermite(8), abs2, NormalMeanVariance(μ, 1.0))
    @test ForwardDiff.derivative(second_moment, 0.7) ≈ 1.4
end
