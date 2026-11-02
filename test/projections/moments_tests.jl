@testitem "mean_cov: the pushforward moments of a function" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    μ, σ = 0.5, 1.5
    q = Normal(μ, σ)
    @test all(EA.mean_cov(EA.GaussHermite(8), identity, q) .≈ (μ, σ^2))
    @test all(EA.mean_cov(EA.GaussHermite(8), x -> 2x + 1, q) .≈ (2μ + 1, 4σ^2))
    # y = x²: E[y] = μ² + σ², Var[y] = 4μ²σ² + 2σ⁴
    @test all(EA.mean_cov(EA.GaussHermite(8), abs2, q) .≈ (μ^2 + σ^2, 4μ^2 * σ^2 + 2σ^4))
    # a linear map of a vector: (A m + b, A Σ Aᵀ)
    m, Σ, A, b = [1.0, -1.0], [2.0 0.5; 0.5 1.0], [1.0 2.0; 0.0 1.0; 3.0 -1.0], [0.5, 0.0, 1.0]
    mean, cov = EA.mean_cov(EA.GaussHermite(4), x -> A * x + b, MvNormal(m, Σ))
    @test mean ≈ A * m + b
    @test cov ≈ A * Σ * A'
    # prepared once, for many functions
    prepared = P.prepare_expectation(EA.GaussHermite(8), q)
    @test all(EA.mean_cov(prepared, identity) .≈ (μ, σ^2))
    @test allocations(EA.mean_cov, prepared, abs2) == 0
end

@testitem "mean_cov: the moments a logit-normal lacks in closed form" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    q = LogitNormal(0.4, 0.8)
    grid = range(1.0e-6, 1 - 1.0e-6; length = 400_000)
    m = sum(x * pdf(q, x) for x in grid) * step(grid)
    v = sum((x - m)^2 * pdf(q, x) for x in grid) * step(grid)
    mean, variance = EA.mean_cov(EA.EADefault(32), identity, q)
    @test mean ≈ m rtol = 1.0e-6
    @test variance ≈ v rtol = 1.0e-5
end

@testitem "mean_cov_cross: the covariance between input and output, in every shape" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    μ, σ = 0.5, 1.5
    # scalar to scalar: Cov[x, 3x] = 3σ²
    @test all(EA.mean_cov_cross(EA.GaussHermite(8), x -> 3x, Normal(μ, σ)) .≈ (3μ, 9σ^2, 3σ^2))
    # scalar to vector: a row, Cov[x, (x, 2x)] = (σ², 2σ²)
    _, _, C = EA.mean_cov_cross(EA.GaussHermite(8), x -> [x, 2x], Normal(μ, σ))
    @test C ≈ [σ^2 2σ^2]
    # vector to scalar and vector to vector: Σ Aᵀ
    m, Σ, A = [1.0, -1.0], [2.0 0.5; 0.5 1.0], [1.0 2.0; 0.0 1.0; 3.0 -1.0]
    _, _, C = EA.mean_cov_cross(EA.GaussHermite(4), x -> A * x, MvNormal(m, Σ))
    @test C ≈ Σ * A'
    _, _, C = EA.mean_cov_cross(EA.GaussHermite(4), x -> sum(x), MvNormal(m, Σ))
    @test C ≈ Σ * ones(2)
end

@testitem "tilted_mean_cov: a conjugate tilt is exact" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    # prior N(m₀, v₀), likelihood N(y | x, r): posterior N((m₀ r + y v₀) / (v₀ + r), v₀ r / (v₀ + r))
    # and normaliser N(y | m₀, v₀ + r)
    m₀, v₀, y, r = 0.5, 2.0, 1.7, 0.8
    logZ, m, v = EA.tilted_mean_cov(EA.GaussHermite(48), x -> logpdf(Normal(x, sqrt(r)), y), Normal(m₀, sqrt(v₀)))
    @test logZ ≈ logpdf(Normal(m₀, sqrt(v₀ + r)), y) rtol = 1.0e-10
    @test m ≈ (m₀ * r + y * v₀) / (v₀ + r) rtol = 1.0e-10
    @test v ≈ v₀ * r / (v₀ + r) rtol = 1.0e-8
end

@testitem "tilted_mean_cov: extreme log-densities neither overflow nor underflow" tags = [:projections] setup = [EAHelpers] begin
    using Distributions
    for shift in (-2000.0, 2000.0)
        logZ, m, v = EA.tilted_mean_cov(EA.EADefault(), x -> shift - (x - 1)^2, Normal(0.0, 1.0))
        reference, mr, vr = EA.tilted_mean_cov(EA.EADefault(), x -> -(x - 1)^2, Normal(0.0, 1.0))
        @test logZ ≈ shift + reference
        @test m ≈ mr
        @test v ≈ vr
    end
end

@testitem "tilted_mean_cov: through a router, a product, and with duals" tags = [:projections] setup = [EAHelpers] begin
    using Distributions, ForwardDiff
    # d/dy log N(y | m₀, v₀ + r) = -(y - m₀) / (v₀ + r)
    m₀, v₀, r = 0.5, 2.0, 0.8
    logZ(y) = first(EA.tilted_mean_cov(EA.EADefault(48), x -> logpdf(Normal(x, sqrt(r)), y), Normal(m₀, sqrt(v₀))))
    @test ForwardDiff.derivative(logZ, 1.7) ≈ -(1.7 - m₀) / (v₀ + r) rtol = 1.0e-8
    # a product: the tilt couples nothing, so the moments are those of each component
    joint = P.FactorizedJoint((Normal(0.0, 1.0), Categorical([0.25, 0.75])))
    logZ0, m, v = EA.tilted_mean_cov(EA.EADefault(), ((x, z),) -> 0.0, joint)
    @test logZ0 ≈ 0.0 atol = 1.0e-12
    @test m[1] ≈ 0.0 atol = 1.0e-12
    @test m[2] ≈ 1.75
end
