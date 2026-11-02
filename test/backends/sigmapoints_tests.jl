@testitem "Unscented: exact up to degree 3 for Gaussians" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    μ, σ = 0.5, 1.5
    q = Normal(μ, σ)
    for backend in (EA.Unscented(), EA.Unscented(α = 1.0, β = 0.0, κ = 2.0))
        @test P.expectation(backend, identity, q) ≈ μ
        @test P.expectation(backend, abs2, q) ≈ μ^2 + σ^2
        @test P.expectation(backend, x -> x^3, q) ≈ μ^3 + 3μ * σ^2
    end
    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    prepared = P.prepare_expectation(EA.Unscented(), MvNormal(m, Σ))
    @test length(prepared.points) == 5
    @test P.expectation(prepared, x -> x * x') ≈ Σ + m * m'
    @test P.isexact(EA.Unscented(), identity, q) === false
end

@testitem "Unscented: the covariance weight of the centre" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    # one dimension, α = 1, κ = 2: λ = 2, points μ and μ ± √3σ with weights 2/3, 1/6, 1/6, and
    # a centre covariance weight 2/3 + (1 - 1 + β)
    μ, σ, β = 0.5, 1.5, 2.0
    backend = EA.Unscented(α = 1.0, β = β, κ = 2.0)
    f = abs2
    points = (μ, μ - sqrt(3) * σ, μ + sqrt(3) * σ)
    wm = (2 / 3, 1 / 6, 1 / 6)
    wc = (2 / 3 + β, 1 / 6, 1 / 6)
    m = sum(wm .* f.(points))
    V = sum(wc .* (f.(points) .- m) .^ 2)
    @test all(EA.mean_cov(backend, f, Normal(μ, σ)) .≈ (m, V))
    # a linear map is exact whatever the weights
    A, b = [1.0 2.0; 0.0 1.0], [0.5, -1.0]
    mean, cov = EA.mean_cov(EA.Unscented(), x -> A * x + b, MvNormal([1.0, -1.0], [2.0 0.5; 0.5 1.0]))
    @test mean ≈ A * [1.0, -1.0] + b
    @test cov ≈ A * [2.0 0.5; 0.5 1.0] * A'
    @test_throws ArgumentError EA.tilted_mean_cov(P.prepare_expectation(backend, Normal(μ, σ)), x -> -x^2)
end

@testitem "GenUnscented: exact up to degree 4 for any univariate distribution" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    for q in (Gamma(2.0, 3.0), Beta(2.0, 5.0), Normal(0.5, 1.5), Poisson(3.0), LogNormal(0.1, 0.4))
        prepared = P.prepare_expectation(EA.GenUnscented(), q)
        @test length(prepared) == 3
        for k in 1:4
            # raw moments through the central ones: E[x^k] = Σ binomial(k, j) μ^(k-j) E[(x-μ)^j]
            μ, σ = mean(q), std(q)
            central = (1.0, 0.0, σ^2, skewness(q) * σ^3, (kurtosis(q) + 3) * σ^4)
            exact = sum(binomial(k, j) * μ^(k - j) * central[j + 1] for j in 0:k)
            @test P.expectation(prepared, x -> x^k) ≈ exact rtol = 1.0e-10
        end
    end
end

@testitem "GenUnscented: Gaussian vectors along each axis" tags = [:backends] setup = [EAHelpers] begin
    using Distributions, LinearAlgebra
    m, Σ = [1.0, -1.0], [2.0 0.5; 0.5 1.0]
    prepared = P.prepare_expectation(EA.GenUnscented(), MvNormal(m, Σ))
    @test length(prepared) == 5
    @test P.expectation(prepared, x -> x * x') ≈ Σ + m * m'
    # standardised fourth moments along an axis: 3
    L = cholesky(Σ).L
    @test P.expectation(prepared, x -> ((L \ (x - m))[1])^4) ≈ 3.0
end

@testitem "SphericalRadialCubature: exact up to degree 3, positive weights" tags = [:backends] setup = [EAHelpers] begin
    using Distributions
    m, Σ = [1.0, -1.0, 0.5], [2.0 0.5 0.0; 0.5 1.0 0.2; 0.0 0.2 1.5]
    prepared = P.prepare_expectation(EA.SphericalRadialCubature(), MvNormal(m, Σ))
    @test length(prepared) == 6
    @test all(>(0), prepared.weights)
    @test P.expectation(prepared, x -> x * x') ≈ Σ + m * m'
    @test P.expectation(prepared, x -> x[1] * x[2] * x[3]) ≈ m[1] * m[2] * m[3] + m[1] * Σ[2, 3] + m[2] * Σ[1, 3] + m[3] * Σ[1, 2]
    @test P.expectation(EA.SphericalRadialCubature(), x -> x^3, Normal(0.5, 1.5)) ≈ 0.5^3 + 3 * 0.5 * 1.5^2
end

@testitem "sigma points: static points for a static mean, no allocations" tags = [:backends, :alloc] setup = [EAHelpers] begin
    using Distributions, ExponentialFamily, StaticArrays
    q = MvNormalMeanCovariance(SVector(1.0, -1.0), SMatrix{2, 2}(2.0, 0.5, 0.5, 1.0))
    for backend in (EA.Unscented(), EA.GenUnscented(), EA.SphericalRadialCubature())
        prepared = P.prepare_expectation(backend, q)
        @test eltype(prepared.points) <: SVector{2}
        @test allocations(P.expectation, prepared, x -> x' * x) == 0
    end
    for backend in (EA.Unscented(), EA.SphericalRadialCubature())
        @test P.expectation(backend, abs2, NormalMeanVariance(0.5, 2.0)) ≈ 0.25 + 2.0
    end
end

@testitem "conformance: the sigma points" tags = [:backends, :conformance] begin
    using Distributions
    const EA = ExpectationApproximations
    cases = [
        EA.Testing.Case(Normal(0.5, 1.5), x -> x^3, 0.5^3 + 3 * 0.5 * 1.5^2),
        EA.Testing.Case(MvNormal([1.0, -1.0], [2.0 0.5; 0.5 1.0]), x -> x[1] * x[2], 0.5 - 1.0),
    ]
    EA.Testing.test_backend(EA.Unscented(), cases)
    EA.Testing.test_backend(EA.SphericalRadialCubature(), cases)
    EA.Testing.test_backend(EA.GenUnscented(), [cases; EA.Testing.Case(Gamma(2.0, 3.0), abs2, 2 * 3 * 9.0)])
end
